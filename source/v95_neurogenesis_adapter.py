"""Supermix v95 Transactional Neurogenesis Adapter & Curriculum Mixture Scheduler.

Completes the deferred v94 trainer integration blockers documented in
``docs/V94_PREPARATION.md`` and ``docs/V94_RESEARCH_REVIEW.md``:

1. **GradientUtilityCollector**: Computes normalized per-module and per-expert
   gradient-energy utility ``||grad||_2 * ||param||_2`` on frozen
   ``growth_calibration`` batches, alongside Dale-signed RigL candidate edge
   scores through ``ConnectomeCore._edge_grad_probe``.
2. **TransactionalNeurogenesisAdapter**: Wraps ``GrowthPolicy`` proposals and
   ``ConnectomeCore`` mutations in an atomic snapshot transaction (model
   parameters/buffers, AdamW moments, policy state, RNG state) gated by
   per-family ``v94_growth_policy.witness_acceptance`` with automatic rollback.
3. **CurriculumMixtureScheduler**: Implements smooth cosine/linear transition
   from anchor families (e.g. v89) to mixed v93/v95 families with cumulative
   token quota tracking (Liu & Mou 2026).
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from mimomix_core import ConnectomeCore, MiMoMixModel, SparseMoEFeedForward  # noqa: E402
from neurogenesis import zero_moments  # noqa: E402
from v94_growth_policy import (  # noqa: E402
    GrowthPolicy,
    UnitObservation,
    witness_acceptance,
)

__all__ = [
    "GradientUtilityCollector",
    "NeurogenesisStepReceipt",
    "TransactionalNeurogenesisAdapter",
    "CurriculumMixtureScheduler",
    "evaluate_witness_rows",
]


@dataclass
class NeurogenesisStepReceipt:
    """Audit receipt emitted for each transactional neurogenesis step."""

    step: int
    proposed_births: List[Dict[str, int]] = field(default_factory=list)
    proposed_prunes: List[int] = field(default_factory=list)
    grown_edges: List[Tuple[int, int]] = field(default_factory=list)
    witness_accepted: bool = True
    rolled_back: bool = False
    regressed_families: List[str] = field(default_factory=list)
    family_delta_nats: Dict[str, float] = field(default_factory=dict)
    max_delta_nats: float = 0.001
    rollback_reason: Optional[str] = None
    alive_modules_after: int = 0
    edges_after: int = 0

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["grown_edges"] = [[int(p), int(q)] for p, q in self.grown_edges]
        return payload


def evaluate_witness_rows(
    model: nn.Module,
    witness_rows: Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]],
) -> List[Dict[str, Any]]:
    """Evaluate per-row CE loss sum & token count in eval mode for witness verification."""
    if not witness_rows:
        raise ValueError("witness_rows must be non-empty")
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    results: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for row_id, family, input_ids, labels in witness_rows:
                x = torch.as_tensor(input_ids, device=device, dtype=torch.long)
                y = torch.as_tensor(labels, device=device, dtype=torch.long)
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                if y.ndim == 1:
                    y = y.unsqueeze(0)
                out = model(x, labels=y, return_mtp=False)
                logits = out.logits[:, :-1, :]
                target = y[:, 1:]
                valid = (target != -100)
                token_count = int(valid.sum().item())
                if token_count <= 0:
                    token_count = int(target.numel())
                per_tok = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                results.append({
                    "row_id": str(row_id),
                    "family": str(family),
                    "loss_sum": float(per_tok.item()),
                    "token_count": max(1, token_count),
                })
    finally:
        model.train(was_training)
    return results


class GradientUtilityCollector:
    """Computes normalized gradient-energy utilities & Dale-signed edge scores."""

    def __init__(self, model: nn.Module, *, scale_by_weight: bool = True):
        self.model = model
        self.scale_by_weight = bool(scale_by_weight)

    @property
    def core(self) -> Optional[ConnectomeCore]:
        c = getattr(self.model, "cns_core", None)
        return c if isinstance(c, ConnectomeCore) else None

    def collect_connectome_utilities(
        self,
        calibration_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        step: int = 0,
        original_slots: Optional[Sequence[bool]] = None,
    ) -> Tuple[List[UnitObservation], torch.Tensor]:
        """Run gradient calibration pass and return (UnitObservations, signed_edge_scores)."""
        core = self.core
        if core is None:
            raise ValueError("model has no ConnectomeCore (`cns_core`)")
        if not calibration_batches:
            raise ValueError("calibration_batches must be non-empty")

        n = core.n_nodes
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        edge_probe = torch.zeros_like(core.edge_logit, requires_grad=True)
        core._edge_grad_probe = edge_probe
        try:
            n_batches = float(len(calibration_batches))
            for x_raw, y_raw in calibration_batches:
                x = torch.as_tensor(x_raw, device=device, dtype=torch.long)
                y = torch.as_tensor(y_raw, device=device, dtype=torch.long)
                out = self.model(x, labels=y, return_mtp=False)
                loss = out.lm_loss if out.lm_loss is not None else out.loss
                if loss is None or not torch.isfinite(loss):
                    raise ValueError("non-finite loss during growth calibration probe")
                (loss / n_batches).backward()

            scores = core.probe_signed_edge_gradients(
                pre_activity=edge_probe.new_zeros((1, n)),
                post_grad=edge_probe.new_zeros((1, n)),
                edge_grad_override=edge_probe.grad if edge_probe.grad is not None else torch.zeros_like(edge_probe),
            )
            utilities = self._slot_gradient_energy(core)
        finally:
            core._edge_grad_probe = None
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

        alive_vec = core.alive.detach().cpu().tolist()
        born_vec = core.born_step.detach().cpu().tolist()
        observations: List[UnitObservation] = []
        for slot in range(n):
            alive = bool(alive_vec[slot])
            born_step = int(born_vec[slot]) if alive else 0
            if original_slots is not None:
                orig = bool(original_slots[slot]) and alive and born_step == 0
            else:
                orig = alive and born_step == 0
            util = float(utilities[slot]) if alive else 0.0
            observations.append(
                UnitObservation(
                    slot=slot,
                    alive=alive,
                    born_step=born_step,
                    utility=util,
                    original=orig,
                )
            )
        return observations, scores

    def _slot_gradient_energy(self, core: ConnectomeCore) -> List[float]:
        """Compute per-module ``sum(|grad * param|)`` (or ``|grad|`` when weights near 0)."""
        n = core.n_nodes
        energy = torch.zeros(n, device=core.alive.device, dtype=torch.float32)

        def _accum_param_1d(param: Optional[torch.Tensor]) -> None:
            if param is None or param.grad is None:
                return
            g = param.grad.detach().float().abs()
            w = param.detach().float().abs()
            energy.add_(g * (w + 1e-4) if self.scale_by_weight else g)

        # Node vector parameters
        _accum_param_1d(core.node_bias)
        _accum_param_1d(core.leak_logit)
        _accum_param_1d(getattr(core, "slow_leak_logit", None))
        _accum_param_1d(getattr(core, "slow_coupling", None))
        _accum_param_1d(getattr(core, "glial_strength", None))

        # Afferent rows & efferent columns
        for lin in [core.read_in, *core.extra_read_in]:
            if lin.weight.grad is not None:
                g = lin.weight.grad.detach().float()
                w = lin.weight.detach().float()
                term = (g * w).abs().sum(dim=1) + 1e-4 * g.abs().sum(dim=1) if self.scale_by_weight else g.abs().sum(dim=1)
                energy.add_(term)

        out_linears = [core.read_out, *core.extra_read_out]
        if core.to_thinking is not None:
            out_linears.append(core.to_thinking)
        for lin in out_linears:
            if lin.weight.grad is not None:
                g = lin.weight.grad.detach().float()
                w = lin.weight.detach().float()
                term = (g * w).abs().sum(dim=0) + 1e-4 * g.abs().sum(dim=0) if self.scale_by_weight else g.abs().sum(dim=0)
                energy.add_(term)

        if core.edge_logit.grad is not None:
            g = core.edge_logit.grad.detach().float().abs()
            w = core.edge_logit.detach().float().abs()
            term = g * (w + 1e-4) if self.scale_by_weight else g
            energy.add_(0.5 * (term.sum(dim=0) + term.sum(dim=1)))

        return [float(v) for v in energy.detach().cpu().tolist()]

    @staticmethod
    def collect_moe_utilities(model: nn.Module) -> List[List[UnitObservation]]:
        """Collect current gradient energy per expert for all MoE layers after backward()."""
        per_layer: List[List[UnitObservation]] = []
        for module in model.modules():
            if not isinstance(module, SparseMoEFeedForward):
                continue
            alive_vec = module.expert_alive.detach().cpu().tolist()
            obs: List[UnitObservation] = []
            for idx, expert in enumerate(module.experts):
                alive = bool(alive_vec[idx])
                util = 0.0
                if alive:
                    for p in expert.parameters():
                        if p.grad is not None:
                            util += float((p.grad.detach().float() * p.detach().float()).abs().sum().item())
                            util += 1e-4 * float(p.grad.detach().float().abs().sum().item())
                obs.append(
                    UnitObservation(
                        slot=idx,
                        alive=alive,
                        born_step=0,
                        utility=util if alive else 0.0,
                        original=alive,
                    )
                )
            per_layer.append(obs)
        return per_layer


class TransactionalNeurogenesisAdapter:
    """Atomic neurogenesis mutation driver gated by per-family witness acceptance."""

    def __init__(
        self,
        model: MiMoMixModel,
        policy: GrowthPolicy,
        *,
        membership_sha256: str,
        total_steps: int,
        optimiser: Optional[torch.optim.Optimizer] = None,
        eval_witness_fn: Optional[Callable[[nn.Module], List[Dict[str, Any]]]] = None,
        max_delta_nats: float = 0.001,
        edge_quota_per_event: int = 0,
        edge_logit: float = -7.0,
        cross_quota: float = 0.5,
    ):
        self.model = model
        self.policy = policy
        self.membership_sha256 = str(membership_sha256)
        self.total_steps = int(total_steps)
        self.optimiser = optimiser
        self.eval_witness_fn = eval_witness_fn
        self.max_delta_nats = float(max_delta_nats)
        self.edge_quota_per_event = int(edge_quota_per_event)
        self.edge_logit = float(edge_logit)
        self.cross_quota = float(cross_quota)
        self.collector = GradientUtilityCollector(model)
        self.history: List[NeurogenesisStepReceipt] = []
        self.rollbacks: int = 0
        self.commits: int = 0

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture an atomic, deep-copied snapshot of model, optimizer, policy, and RNG."""
        core = self.model.cns_core
        model_state = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        opt_state = copy.deepcopy(self.optimiser.state_dict()) if self.optimiser is not None else None
        return {
            "model_state": model_state,
            "core_last_event": copy.deepcopy(core.last_event) if core is not None else {},
            "opt_state": opt_state,
            "policy_state": self.policy.state_dict(),
            "cpu_rng_state": torch.get_rng_state().clone(),
        }

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore exact pre-mutation weights, buffers, optimizer moments, policy, and RNG."""
        self.model.load_state_dict(snapshot["model_state"], strict=True)
        core = self.model.cns_core
        if core is not None:
            core.last_event = copy.deepcopy(snapshot.get("core_last_event", {}))
        if self.optimiser is not None and snapshot.get("opt_state") is not None:
            self.optimiser.load_state_dict(snapshot["opt_state"])
        self.policy = GrowthPolicy.from_state_dict(snapshot["policy_state"])
        torch.set_rng_state(snapshot["cpu_rng_state"])

    def _measure_witness(
        self,
        witness_rows: Optional[Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]]],
    ) -> List[Dict[str, Any]]:
        if self.eval_witness_fn is not None:
            return self.eval_witness_fn(self.model)
        if witness_rows is None:
            raise ValueError("Provide either eval_witness_fn or witness_rows")
        return evaluate_witness_rows(self.model, witness_rows)

    def _zero_connectome_slices(
        self,
        core: ConnectomeCore,
        slots: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> None:
        if self.optimiser is None:
            return
        unique_slots = sorted({int(s) for s in slots})
        if unique_slots:
            idx_tensor = torch.tensor(unique_slots, dtype=torch.long, device=core.alive.device)
            for param in (
                core.node_bias,
                core.leak_logit,
                getattr(core, "slow_leak_logit", None),
                getattr(core, "slow_coupling", None),
                getattr(core, "glial_strength", None),
            ):
                if param is not None:
                    zero_moments(self.optimiser, param, (0, idx_tensor))
            for lin in [core.read_in, *core.extra_read_in]:
                zero_moments(self.optimiser, lin.weight, (0, idx_tensor))
            out_linears = [core.read_out, *core.extra_read_out]
            if core.to_thinking is not None:
                out_linears.append(core.to_thinking)
            for lin in out_linears:
                zero_moments(self.optimiser, lin.weight, (1, idx_tensor))
            zero_moments(self.optimiser, core.edge_logit, (0, idx_tensor))
            zero_moments(self.optimiser, core.edge_logit, (1, idx_tensor))
        if edges:
            edge_mask = torch.zeros_like(core.edge_logit, dtype=torch.bool)
            for post, pre in edges:
                edge_mask[int(post), int(pre)] = True
            zero_moments(self.optimiser, core.edge_logit, edge_mask)

    def step_neurogenesis(
        self,
        step: int,
        calibration_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        *,
        witness_rows: Optional[Sequence[Tuple[str, str, torch.Tensor, torch.Tensor]]] = None,
        original_slots: Optional[Sequence[bool]] = None,
    ) -> NeurogenesisStepReceipt:
        """Collect calibration utility, propose structural edits, and commit iff witness accepts."""
        core = self.model.cns_core
        if core is None:
            raise ValueError("TransactionalNeurogenesisAdapter requires a ConnectomeCore")

        snapshot = self.capture_snapshot()
        observations, edge_scores = self.collector.collect_connectome_utilities(
            calibration_batches, step=step, original_slots=original_slots
        )
        proposal = self.policy.propose(
            observations,
            step=step,
            total_steps=self.total_steps,
            split_role="growth_calibration",
            membership_sha256=self.membership_sha256,
        )
        births: List[Dict[str, int]] = list(proposal["births"])
        prunes: List[int] = list(proposal["prune_slots"])

        if not births and not prunes and self.edge_quota_per_event <= 0:
            receipt = NeurogenesisStepReceipt(
                step=step,
                alive_modules_after=int(core.alive.sum().item()),
                edges_after=int(core.mask.sum().item()),
            )
            self.history.append(receipt)
            return receipt

        before_witness = self._measure_witness(witness_rows)
        touched_slots: List[int] = []
        grown_edges: List[Tuple[int, int]] = []
        applied_births: List[Dict[str, int]] = []
        applied_prunes: List[int] = []

        try:
            for slot in prunes:
                if core.kill_module(int(slot)):
                    applied_prunes.append(int(slot))
                    touched_slots.append(int(slot))
            for birth in births:
                parent = int(birth["parent"])
                child = core.split_module(parent, step=step)
                if child is not None:
                    applied_births.append({"parent": parent, "child": int(child)})
                    touched_slots.extend([parent, int(child)])
            if self.edge_quota_per_event > 0:
                grown_edges = core.grow_edges_from_scores(
                    edge_scores,
                    count=self.edge_quota_per_event,
                    step=step,
                    logit=self.edge_logit,
                    cross_quota=self.cross_quota,
                )
            self._zero_connectome_slices(core, touched_slots, grown_edges)
            after_witness = self._measure_witness(witness_rows)
            gate = witness_acceptance(
                before_witness,
                after_witness,
                split_role="growth_calibration",
                membership_sha256=self.membership_sha256,
                max_delta_nats=self.max_delta_nats,
            )
        except Exception as exc:
            self.restore_snapshot(snapshot)
            self.rollbacks += 1
            receipt = NeurogenesisStepReceipt(
                step=step,
                proposed_births=applied_births or births,
                proposed_prunes=applied_prunes or prunes,
                grown_edges=grown_edges,
                witness_accepted=False,
                rolled_back=True,
                max_delta_nats=self.max_delta_nats,
                rollback_reason=f"exception:{type(exc).__name__}: {exc}",
                alive_modules_after=int(core.alive.sum().item()),
                edges_after=int(core.mask.sum().item()),
            )
            self.history.append(receipt)
            return receipt

        if not gate["accepted"]:
            self.restore_snapshot(snapshot)
            self.rollbacks += 1
            receipt = NeurogenesisStepReceipt(
                step=step,
                proposed_births=applied_births,
                proposed_prunes=applied_prunes,
                grown_edges=grown_edges,
                witness_accepted=False,
                rolled_back=True,
                regressed_families=list(gate["regressed_families"]),
                family_delta_nats=dict(gate["family_delta_nats"]),
                max_delta_nats=self.max_delta_nats,
                rollback_reason=f"witness_regression:{','.join(gate['regressed_families'])}",
                alive_modules_after=int(core.alive.sum().item()),
                edges_after=int(core.mask.sum().item()),
            )
            self.history.append(receipt)
            return receipt

        self.commits += 1
        receipt = NeurogenesisStepReceipt(
            step=step,
            proposed_births=applied_births,
            proposed_prunes=applied_prunes,
            grown_edges=grown_edges,
            witness_accepted=True,
            rolled_back=False,
            regressed_families=[],
            family_delta_nats=dict(gate["family_delta_nats"]),
            max_delta_nats=self.max_delta_nats,
            alive_modules_after=int(core.alive.sum().item()),
            edges_after=int(core.mask.sum().item()),
        )
        self.history.append(receipt)
        return receipt


class CurriculumMixtureScheduler:
    """Smooth anchor-to-target family mixture scheduler (Liu & Mou 2026)."""

    def __init__(
        self,
        anchor_weights: Dict[str, float],
        target_weights: Dict[str, float],
        *,
        warmup_steps: int = 100,
        ramp_steps: int = 400,
        schedule: str = "cosine",
    ):
        if schedule not in {"cosine", "linear"}:
            raise ValueError(f"unknown schedule {schedule!r}")
        if warmup_steps < 0 or ramp_steps < 1:
            raise ValueError("warmup_steps must be >= 0 and ramp_steps >= 1")
        self.anchor_weights = self._normalise(anchor_weights)
        self.target_weights = self._normalise(target_weights)
        self.all_families = sorted(set(self.anchor_weights) | set(self.target_weights))
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.schedule = schedule
        self.cumulative_tokens: Dict[str, int] = {fam: 0 for fam in self.all_families}

    @staticmethod
    def _normalise(weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            raise ValueError("weights must be non-empty")
        total = float(sum(float(v) for v in weights.values()))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("weights must sum to a positive finite number")
        return {str(k): max(0.0, float(v)) / total for k, v in weights.items()}

    def blend_alpha(self, step: int) -> float:
        s = int(step)
        if s <= self.warmup_steps:
            return 0.0
        if s >= self.warmup_steps + self.ramp_steps:
            return 1.0
        progress = (s - self.warmup_steps) / float(self.ramp_steps)
        if self.schedule == "linear":
            return progress
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def weights_at(self, step: int) -> Dict[str, float]:
        alpha = self.blend_alpha(step)
        blended = {
            fam: (1.0 - alpha) * self.anchor_weights.get(fam, 0.0) + alpha * self.target_weights.get(fam, 0.0)
            for fam in self.all_families
        }
        return self._normalise(blended)

    def record_tokens(self, family_tokens: Dict[str, int]) -> None:
        for fam, count in family_tokens.items():
            key = str(fam)
            self.cumulative_tokens[key] = self.cumulative_tokens.get(key, 0) + max(0, int(count))

    def quota_report(self, step: int) -> Dict[str, Any]:
        total = max(1, sum(self.cumulative_tokens.values()))
        expected = self.weights_at(step)
        actual = {fam: round(self.cumulative_tokens.get(fam, 0) / total, 6) for fam in expected}
        drift = {fam: round(actual[fam] - expected[fam], 6) for fam in expected}
        return {
            "step": int(step),
            "blend_alpha": round(self.blend_alpha(step), 6),
            "total_tokens": int(sum(self.cumulative_tokens.values())),
            "expected_weights": {k: round(v, 6) for k, v in expected.items()},
            "actual_shares": actual,
            "quota_drift": drift,
        }
