"""Compatibility adapters and unified registry for Supermix model families."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "source"


def _ensure_source_path() -> None:
    src = str(SOURCE_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_source_path()

from scf.config import SCFFeatureFlags, resolve_decoding_policy  # noqa: E402

__all__ = [
    "ModelFamilyStatus",
    "BaseModelAdapter",
    "ChampionNetAdapter",
    "CognitiveLeapAdapter",
    "QwenAdapter",
    "OmniCollectiveAdapter",
    "MiMoMixAdapter",
    "NexusMindAdapter",
    "SCFModelRegistry",
]


@dataclass
class ModelFamilyStatus:
    family: str
    available_in_source: bool
    available_weights: List[str] = field(default_factory=list)
    maturity_status: str = "experimental_isolated"
    experimental_isolated: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseModelAdapter:
    family_name: str = "base"

    def status(self) -> ModelFamilyStatus:
        raise NotImplementedError


class ChampionNetAdapter(BaseModelAdapter):
    """Compatibility wrapper for ChampionNet v1-v29 backbones and expert heads."""

    family_name = "ChampionNet"
    PRESERVED_HEADS = (
        "HierarchicalMoEClassifierHead",
        "DeepExpertClassifierHead",
        "ExpertChoiceClassifierHead",
        "SmarterExpertClassifierHead",
        "ThoughtExpertClassifierHead",
        "RecursiveThoughtExpertHead",
        "ReflexiveThoughtExpertHead",
        "MetaCognitiveExpertHead",
        "TreeOfThoughtExpertHead",
        "ConsensusExpertHead",
        "DeliberativeAlignmentExpertHead",
        "NeurogenesisExpertHead",
        "CognitiveSingularityExpertHead",
        "TranscendentArchitectExpertHead",
        "ActiveInferenceExpertHead",
        "TestTimeDifferentialExpertHead",
        "HolographicStateSpaceExpertHead",
        "PaperFusionExpertHead",
    )
    EXPERIMENTAL_ISOLATED_HEADS = (
        "LiquidSpikingTensorExpertHead",
        "OmniversalQuantumExpertHead",
        "FractalGenesisExpertHead",
    )

    def status(self) -> ModelFamilyStatus:
        weights = [
            rel
            for rel in (
                "artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth",
                "artifacts/v52_initialization/champion_model_chat_v52_initialized.pth",
                "output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth",
            )
            if (ROOT / rel).is_file()
        ]
        try:
            import model_variants as mv

            available = hasattr(mv, "build_model")
        except Exception:
            available = False
        return ModelFamilyStatus(
            family=self.family_name,
            available_in_source=available,
            available_weights=weights,
            maturity_status="mixed_trained_and_experimental",
            experimental_isolated=True,
            details={
                "preserved_heads_count": len(self.PRESERVED_HEADS),
                "experimental_heads": list(self.EXPERIMENTAL_ISOLATED_HEADS),
            },
        )

    def build(
        self,
        *,
        model_size: str = "hierarchical_moe",
        feature_dim: int = 256,
        num_classes: int = 10,
        allow_experimental: bool = False,
    ) -> Dict[str, Any]:
        import model_variants as mv

        _ = num_classes  # Preserved in signature for adapter compatibility
        size_alias = {
            "hierarchical_moe": "hierarchical_expert",
            "liquid_spiking": "liquid_spiking_expert",
            "omniversal_quantum": "omniversal_expert",
            "fractal_genesis": "fractal_expert",
        }
        resolved_size = size_alias.get(model_size, model_size)
        is_experimental = resolved_size in {
            "liquid_spiking_expert",
            "omniversal_expert",
            "fractal_expert",
        }
        if is_experimental and not allow_experimental:
            return {
                "built": False,
                "blocked_reason": "experimental_component_isolated_set_allow_experimental",
                "model_size": model_size,
                "resolved_model_size": resolved_size,
                "experimental_isolated": True,
            }
        model = mv.build_model(model_size=resolved_size, expansion_dim=feature_dim)
        return {
            "built": True,
            "model_size": model_size,
            "resolved_model_size": resolved_size,
            "model_class": type(model).__name__,
            "model": model,
            "experimental_isolated": is_experimental,
            "production_ready": False if is_experimental else None,
        }


class CognitiveLeapAdapter(BaseModelAdapter):
    """Compatibility wrapper for Cognitive Leap v50-v52 adaptive compute & stability."""

    family_name = "CognitiveLeap"

    def status(self) -> ModelFamilyStatus:
        weights = [
            rel
            for rel in (
                "artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth",
                "artifacts/v52_initialization/champion_model_chat_v52_initialized.pth",
                "output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth",
            )
            if (ROOT / rel).is_file()
        ]
        try:
            import chat_app

            available = hasattr(chat_app, "resolve_runtime_compute_cycles")
        except Exception:
            available = False
        return ModelFamilyStatus(
            family=self.family_name,
            available_in_source=available,
            available_weights=weights,
            maturity_status="trained_checkpoint_present" if weights else "source_only",
            experimental_isolated=False,
        )

    def plan_compute_budget(
        self,
        prompt: str,
        *,
        mode: str = "auto",
        explicit_cycles: Optional[int] = None,
    ) -> Dict[str, Any]:
        import chat_app

        est = chat_app.estimate_auto_reasoning_cycles(prompt)
        est_cycles = int(est.get("cycles", 1)) if isinstance(est, dict) else int(est)
        target = explicit_cycles if explicit_cycles is not None else est_cycles
        resolved = chat_app.resolve_runtime_compute_cycles([target])
        chosen = int(resolved[0]) if resolved else est_cycles
        return {
            "cycles": chosen,
            "mode": mode,
            "estimate": est_cycles,
        }


class QwenAdapter(BaseModelAdapter):
    """Compatibility wrapper for Qwen LoRA adapters with strict promotion enforcement."""

    family_name = "QwenLoRA"

    def discover_adapters(self) -> List[str]:
        roots = [ROOT / "artifacts", ROOT / "output" / "training_candidates"]
        found: List[str] = []
        for base in roots:
            if not base.exists():
                continue
            for p in sorted(base.rglob("adapter_model.safetensors")):
                found.append(p.relative_to(ROOT).as_posix())
        return found

    def status(self) -> ModelFamilyStatus:
        adapters = self.discover_adapters()
        try:
            import qwen_adapter_promotion as qap

            available = hasattr(qap, "validate_promoted_adapter")
        except Exception:
            available = False
        return ModelFamilyStatus(
            family=self.family_name,
            available_in_source=available,
            available_weights=adapters,
            maturity_status="adapter_checkpoints_present_promotion_gated",
            experimental_isolated=False,
            details={"automatic_promotion_allowed": False},
        )

    def check_adapter_promotion(self, adapter_dir: Path) -> Dict[str, Any]:
        import qwen_adapter_promotion as qap

        target = Path(adapter_dir)
        if not target.exists():
            return {"valid": False, "promoted": False, "reason": "adapter_directory_missing"}
        try:
            result = qap.validate_promoted_adapter(target)
            return {"valid": bool(result.get("valid", True)), "promoted": bool(result.get("promoted", False)), "detail": result}
        except Exception as exc:
            return {"valid": False, "promoted": False, "reason": f"{type(exc).__name__}: {exc}"}


class OmniCollectiveAdapter(BaseModelAdapter):
    """Compatibility wrapper for Omni Collective v1-v8, v41, v42 and catalog."""

    family_name = "OmniCollective"

    def status(self) -> ModelFamilyStatus:
        try:
            import multimodel_catalog as mc_cat

            specs = [s.key for s in mc_cat.MODEL_SPECS if s.family in {"fusion", "math", "protein", "materials", "3d", "vision", "gan"}]
            available = True
        except Exception:
            specs = []
            available = False
        weights = [
            p.relative_to(ROOT).as_posix()
            for p in sorted(ROOT.rglob("omni_collective*.pth"))
            if ".git" not in p.parts
        ]
        return ModelFamilyStatus(
            family=self.family_name,
            available_in_source=available,
            available_weights=weights,
            maturity_status="trained_weights_present" if weights else "source_present_external_weights_optional",
            experimental_isolated=False,
            details={"registered_specs": specs},
        )

    def route_hint(self, prompt: str) -> Dict[str, Any]:
        import omni_collective_v42_model as omni_v42

        budget = omni_v42._budget_hint_v42(prompt)
        return {"family": self.family_name, "budget_hint": budget}


class MiMoMixAdapter(BaseModelAdapter):
    """Compatibility wrapper for MiMoMix v53/v56 LatentStateReasoner and v57-v97 MiMoMixModel."""

    family_name = "MiMoMix"
    _TALK_CACHE: Dict[str, Any] = {}

    def discover_talk_checkpoints(self) -> List[str]:
        from mimomix_checkpoints import discover_talk_models

        return [path.as_posix() for path in discover_talk_models(ROOT).values()]

    def status(self) -> ModelFamilyStatus:
        talk_checkpoints = self.discover_talk_checkpoints()
        weights = [
            rel
            for rel in (
                "output/hf_v56_publish/v56b_randslots_entropy.pt",
                "output/v96_physio_neurogenesis/v96_physio_neurogenesis.pt",
            )
            if (ROOT / rel).is_file()
        ]
        weights.extend(talk_checkpoints)
        try:
            import mimomix_core as core_mod
            import mimomix_decoding as dec_mod
            import mimomix_reasoner as reasoner_mod

            available = all(
                (
                    hasattr(core_mod, "MiMoMixModel"),
                    hasattr(dec_mod, "speculative_generate"),
                    hasattr(reasoner_mod, "LatentStateReasoner"),
                )
            )
        except Exception:
            available = False
        return ModelFamilyStatus(
            family=self.family_name,
            available_in_source=available,
            available_weights=weights,
            maturity_status="trained_checkpoint_present" if weights else "source_only",
            experimental_isolated=False,
            details={
                "supports_latent_state_reasoner": True,
                "supports_mtp_speculative": True,
                "supports_connectome_core": True,
                "talk_checkpoints_available": talk_checkpoints,
            },
        )

    def infer_text(
        self,
        prompt: str,
        *,
        flags: Optional[SCFFeatureFlags] = None,
        checkpoint_path: Optional[str] = None,
        max_new_tokens: int = 96,
        adaptive_thinking: bool = False,
        prefer_speculative: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate a bounded text candidate from a local MiMoMix talk checkpoint.

        Always tagged ``verified=False`` and ``answer_authority=False`` until an
        independent verifier confirms exact claims.
        """
        import train_mimomix_talk as tmt

        rel_list = self.discover_talk_checkpoints()
        resolved_path: Optional[Path] = None
        if checkpoint_path:
            raw_cand = Path(checkpoint_path)
            if raw_cand.is_file():
                resolved_path = raw_cand
            elif (ROOT / raw_cand).is_file():
                resolved_path = ROOT / raw_cand
        elif rel_list:
            resolved_path = ROOT / rel_list[0]
        if resolved_path is None or not resolved_path.is_file():
            return {
                "generated": False,
                "answer": "",
                "reason": "no_local_talk_checkpoint_found",
                "verified": False,
                "answer_authority": False,
                "model_conditional": True,
            }
        cache_key = str(resolved_path.resolve())
        if cache_key not in self._TALK_CACHE:
            model, tokenizer, payload = tmt.load_talk_checkpoint(resolved_path, map_location="cpu")
            self._TALK_CACHE[cache_key] = (model, tokenizer, dict(payload.get("extra") or {}))
        cached_entry = self._TALK_CACHE[cache_key]
        if len(cached_entry) == 2:
            model, tokenizer = cached_entry
            extra = {}
        else:
            model, tokenizer, extra = cached_entry

        active_flags = flags or SCFFeatureFlags(SCF_ENABLED=True)
        policy = resolve_decoding_policy(
            active_flags,
            adaptive_thinking=adaptive_thinking,
            prefer_speculative=prefer_speculative,
        )
        gen = tmt.generate_reply(
            model,
            tokenizer,
            str(prompt or ""),
            max_new_tokens=max(1, int(max_new_tokens)),
            speculative=bool(policy["speculative_allowed"]),
        )
        try:
            arch_tele = model.telemetry()
        except Exception:
            arch_tele = {}
        return {
            "generated": True,
            "family": self.family_name,
            "answer": str(gen.get("reply") or ""),
            "checkpoint": str(resolved_path),
            "run_name": str(extra.get("run_name") or resolved_path.stem),
            "policy": policy,
            "tokens": int(gen.get("tokens", 0)),
            "latency_ms": float(gen.get("latency_ms", 0.0)),
            "acceptance_length": float(gen.get("acceptance_length", 1.0)),
            "architecture_telemetry": arch_tele,
            "verified": False,
            "answer_authority": False,
            "model_conditional": True,
        }

    def generate_tokens(
        self,
        model: Any,
        prompt_ids: Sequence[int],
        *,
        flags: SCFFeatureFlags,
        max_new_tokens: int = 8,
        adaptive_thinking: bool = False,
        prefer_speculative: Optional[bool] = None,
    ) -> Dict[str, Any]:
        import torch
        import mimomix_decoding as md

        policy = resolve_decoding_policy(
            flags,
            adaptive_thinking=adaptive_thinking,
            prefer_speculative=prefer_speculative,
        )
        ids_tensor = torch.tensor([list(prompt_ids)], dtype=torch.long)
        if policy["speculative_allowed"]:
            result = md.speculative_generate(
                model, ids_tensor, max_new_tokens=max_new_tokens
            )
        else:
            result = md.greedy_generate(
                model,
                ids_tensor,
                max_new_tokens=max_new_tokens,
                adaptive_thinking=bool(policy["adaptive_thinking"]),
            )
        drafted = int(getattr(result.stats, "drafted_tokens", 0))
        accepted = int(getattr(result.stats, "accepted_draft_tokens", 0))
        return {
            "policy": policy,
            "tokens": result.sequences[0].tolist(),
            "new_tokens": result.new_tokens[0].tolist(),
            "stats": {
                "mode": result.stats.mode,
                "new_tokens": int(result.stats.new_tokens),
                "drafted_tokens": drafted,
                "accepted_draft_tokens": accepted,
                "acceptance_rate": round(accepted / max(1, drafted), 4) if drafted else 1.0,
            },
        }


class NexusMindAdapter(BaseModelAdapter):
    """Compatibility wrapper for NexusMind v71-v95 symbolic, causal, and epistemic stack."""

    family_name = "NexusMind"

    def status(self) -> ModelFamilyStatus:
        try:
            import nexus_engine as ne
            import nexus_solver as ns

            available = hasattr(ne, "NexusEngine") and hasattr(ns, "NexusSolver")
        except Exception:
            available = False
        return ModelFamilyStatus(
            family=self.family_name,
            available_in_source=available,
            available_weights=["built_in_no_external_weights_required"],
            maturity_status="production_verified_deterministic_core_plus_experimental_diagnostics",
            experimental_isolated=False,
            details={"v95_forensic_engines_answer_authority": False},
        )

    def solve_or_analyze(self, query: str) -> Dict[str, Any]:
        import nexus_solver as ns

        solver = ns.NexusSolver()
        res = solver.solve(query)
        return {
            "solved": bool(res is not None and res.solved),
            "display_answer": res.display_answer if res is not None else None,
            "unit": res.unit if res is not None else "",
            "formula_id": res.formula_id if res is not None else None,
            "receipt_sha256": res.receipt.receipt_sha256 if res is not None and res.receipt is not None else None,
        }


class SCFModelRegistry:
    """Unified discovery and health registry across all six Supermix model families."""

    def __init__(self) -> None:
        self._adapters: Dict[str, BaseModelAdapter] = {
            "ChampionNet": ChampionNetAdapter(),
            "CognitiveLeap": CognitiveLeapAdapter(),
            "QwenLoRA": QwenAdapter(),
            "OmniCollective": OmniCollectiveAdapter(),
            "MiMoMix": MiMoMixAdapter(),
            "NexusMind": NexusMindAdapter(),
        }

    def list_families(self) -> List[str]:
        return list(self._adapters.keys())

    def get_adapter(self, family: str) -> BaseModelAdapter:
        if family not in self._adapters:
            raise KeyError(f"Unknown SCF model family: {family!r}")
        return self._adapters[family]

    def summary(self) -> Dict[str, Any]:
        return {
            family: adapter.status().to_dict()
            for family, adapter in self._adapters.items()
        }
