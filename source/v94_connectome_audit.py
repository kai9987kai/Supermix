"""Offline v94 mechanism audits. Never imported by the active trainer.

Use a separately loaded model in eval mode, without concurrent inference.
All-off includes every residual write and the thinking bond. Removing grown
components measures deletion sensitivity, NOT reversal of training or mitosis.
Paired intervals resample semantic groups when group_id is supplied.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch


MODES = ("baseline", "all_off", "commissure_off", "left_off", "right_off",
         "grown_components_off")


@contextmanager
def connectome_mode(model: torch.nn.Module, mode: str):
    """Temporarily set one ablation; restore gates, flags and buffers exactly.

Requires an offline eval model. Buffer restoration includes nonpersistent
router/statistics telemetry. Parameter identities and optimizer references
are preserved; weights other than the gates are never modified here.
"""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if any(module.training for module in model.modules()):
        raise ValueError("an offline model in eval mode is required")
    core = getattr(model, "cns_core", None)
    if core is None:
        raise ValueError("a connectome core is required")
    if mode in {"commissure_off", "left_off", "right_off"}:
        sides = set(core.hemisphere[core.alive.bool()].tolist())
        if sides != {0, 1}:
            raise ValueError("hemisphere ablations require live modules on both sides")
    gates = [core.gate, *core.extra_gates]
    if core.thinking_gate is not None:
        gates.append(core.thinking_gate)
    saved_gates = [(gate, gate.detach().clone()) for gate in gates]
    saved_flags = [(core, name, getattr(core, name)) for name in
                   ("ablate_cross", "ablate_side", "ablate_grown")]
    saved_buffers = []
    missing = object()
    saved_runtime = []
    for module in model.modules():
        if hasattr(module, "collect_stats"):
            saved_flags.append((module, "collect_stats", module.collect_stats))
        for name, buffer in module._buffers.items():
            if buffer is not None:
                saved_buffers.append((module, name, buffer, buffer.detach().clone()))
        # MiMoMix forward replaces these plain telemetry/auxiliary attributes.
        # They are not registered buffers and must be restored separately.
        names = {name for name in module.__dict__ if name.startswith("last_")}
        names.update(("_aux_loss", "_last_quality_logits"))
        for name in names:
            saved_runtime.append((module, name, module.__dict__.get(name, missing)))
    try:
        core.ablate_cross = mode == "commissure_off"
        core.ablate_side = {"left_off": 0, "right_off": 1}.get(mode)
        core.ablate_grown = mode == "grown_components_off"
        for module, name, _ in saved_flags:
            if name == "collect_stats":
                setattr(module, name, False)
        if mode == "all_off":
            with torch.no_grad():
                for gate in gates:
                    gate.zero_()
        yield model
    finally:
        with torch.no_grad():
            for gate, value in saved_gates:
                gate.copy_(value)
            for module, name, original, value in saved_buffers:
                original.copy_(value)
                module._buffers[name] = original
        for module, name, value in saved_flags:
            setattr(module, name, value)
        for module, name, value in saved_runtime:
            if value is missing:
                module.__dict__.pop(name, None)
            else:
                setattr(module, name, value)


@dataclass(frozen=True)
class RowLoss:
    row_id: str
    family: str
    loss_sum: float
    token_count: int
    group_id: str = ""


def _index(rows: Iterable[RowLoss]) -> dict[str, RowLoss]:
    result = {}
    group_families = {}
    for row in rows:
        if not isinstance(row, RowLoss):
            raise ValueError("expected RowLoss records")
        if not isinstance(row.row_id, str) or not row.row_id.strip():
            raise ValueError("row_id must be nonempty")
        if not isinstance(row.family, str) or not row.family.strip():
            raise ValueError("family must be nonempty")
        if not isinstance(row.group_id, str):
            raise ValueError("group_id must be a string")
        if row.row_id in result:
            raise ValueError(f"duplicate row_id: {row.row_id}")
        if type(row.token_count) is not int or row.token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        if isinstance(row.loss_sum, bool) or not math.isfinite(row.loss_sum) or row.loss_sum < 0:
            raise ValueError("loss_sum must be finite and nonnegative")
        # Tag explicit group keys to avoid collisions with fallback row IDs.
        group = ("group", row.group_id) if row.group_id else ("row", row.row_id)
        if group in group_families and group_families[group] != row.family:
            raise ValueError("one semantic group must belong to one family")
        group_families[group] = row.family
        result[row.row_id] = row
    if not result:
        raise ValueError("paired comparison requires rows")
    return result


def compare_losses(before: Iterable[RowLoss], after: Iterable[RowLoss], *,
                   seed: int = 94, bootstrap_samples: int = 2000) -> dict:
    """Token-weighted loss delta with a paired group percentile bootstrap.

Positive delta means the ablation worsened loss. Family intervals are
descriptive, not multiplicity-adjusted promotion tests. A single independent
group has no interval. Row pairing is by ID, never by file order.
"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 100000:
        raise ValueError("bootstrap_samples must be an integer in [100, 100000]")
    left, right = _index(before), _index(after)
    if left.keys() != right.keys():
        raise ValueError("paired row IDs differ")
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if (a.family, a.group_id, a.token_count) != (b.family, b.group_id, b.token_count):
            raise ValueError(f"paired family, group or token count differs: {key}")
        pairs.append((a, b))

    def summarize(selected, rng):
        groups = {}
        for a, b in selected:
            key = ("group", a.group_id) if a.group_id else ("row", a.row_id)
            totals = groups.setdefault(key, [0.0, 0.0, 0.0])
            totals[0] += a.loss_sum
            totals[1] += b.loss_sum
            totals[2] += a.token_count
        values = np.asarray([groups[key] for key in sorted(groups)], dtype=np.float64)
        sums = values.sum(axis=0)
        if not np.isfinite(sums).all():
            raise ValueError("loss totals overflowed")
        ci = None
        if len(values) >= 2:
            deltas = np.empty(bootstrap_samples)
            for i in range(bootstrap_samples):
                draw = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
                deltas[i] = (draw[1] - draw[0]) / draw[2]
            if not np.isfinite(deltas).all():
                raise ValueError("bootstrap totals overflowed")
            ci = np.quantile(deltas, [0.025, 0.975]).tolist()
        return {"rows": len(selected), "independent_groups": len(groups),
                "tokens": int(sums[2]), "before_nats": float(sums[0] / sums[2]),
                "after_nats": float(sums[1] / sums[2]),
                "delta_nats": float((sums[1] - sums[0]) / sums[2]),
                "ci95": ci}

    rng = np.random.default_rng(seed)
    return {"schema": "supermix-v94-paired-loss-v1", "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "interpretation": "paired deletion sensitivity; not a training counterfactual",
            "overall": summarize(pairs, rng),
            "families": {family: summarize([(a, b) for a, b in pairs if a.family == family], rng)
                         for family in sorted({a.family for a, _ in pairs})}}
