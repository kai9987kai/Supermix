"""Versioned configuration and feature flags for Supermix v91 Cognitive Fabric."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


SCF_CONFIG_SCHEMA_VERSION = "supermix-scf91-config-v1"

FLAG_NAMES = (
    "SCF_ENABLED",
    "SCF_ROUTER_ENABLED",
    "SCF_HIERARCHICAL_MOE_ENABLED",
    "SCF_COLLECTIVE_REASONING_ENABLED",
    "SCF_LATENT_MEMORY_ENABLED",
    "SCF_ADAPTIVE_COMPUTE_ENABLED",
    "SCF_SPECULATIVE_DECODING_ENABLED",
    "SCF_VERIFICATION_ENABLED",
    "SCF_MULTIMODAL_ENABLED",
    "SCF_ACTIVE_INFERENCE_ENABLED",
)

__all__ = [
    "SCF_CONFIG_SCHEMA_VERSION",
    "FLAG_NAMES",
    "SCFFeatureFlags",
    "SCFConfig",
    "resolve_decoding_policy",
]


def _parse_env_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    norm = str(value).strip().lower()
    if norm in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if norm in {"0", "false", "no", "off", "disable", "disabled", ""}:
        return False
    return default


@dataclass
class SCFFeatureFlags:
    """Independent feature flags for SCF subsystems. All default False (safe fallback)."""

    SCF_ENABLED: bool = False
    SCF_ROUTER_ENABLED: bool = False
    SCF_HIERARCHICAL_MOE_ENABLED: bool = False
    SCF_COLLECTIVE_REASONING_ENABLED: bool = False
    SCF_LATENT_MEMORY_ENABLED: bool = False
    SCF_ADAPTIVE_COMPUTE_ENABLED: bool = False
    SCF_SPECULATIVE_DECODING_ENABLED: bool = False
    SCF_VERIFICATION_ENABLED: bool = False
    SCF_MULTIMODAL_ENABLED: bool = False
    SCF_ACTIVE_INFERENCE_ENABLED: bool = False

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "SCFFeatureFlags":
        source = os.environ if env is None else env
        return cls(**{name: _parse_env_bool(source.get(name), False) for name in FLAG_NAMES})

    @classmethod
    def all_enabled(cls, *, speculative_decoding: bool = True) -> "SCFFeatureFlags":
        return cls(
            SCF_ENABLED=True,
            SCF_ROUTER_ENABLED=True,
            SCF_HIERARCHICAL_MOE_ENABLED=True,
            SCF_COLLECTIVE_REASONING_ENABLED=True,
            SCF_LATENT_MEMORY_ENABLED=True,
            SCF_ADAPTIVE_COMPUTE_ENABLED=True,
            SCF_SPECULATIVE_DECODING_ENABLED=bool(speculative_decoding),
            SCF_VERIFICATION_ENABLED=True,
            SCF_MULTIMODAL_ENABLED=True,
            SCF_ACTIVE_INFERENCE_ENABLED=True,
        )

    def enabled_list(self) -> list[str]:
        return [name for name in FLAG_NAMES if bool(getattr(self, name))]

    def to_dict(self) -> Dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in FLAG_NAMES}


def resolve_decoding_policy(
    flags: SCFFeatureFlags,
    *,
    adaptive_thinking: bool = False,
    prefer_speculative: Optional[bool] = None,
) -> Dict[str, Any]:
    """Enforce MiMoMix invariant: never run speculative decoding with adaptive thinking."""

    want_spec = (
        bool(prefer_speculative)
        if prefer_speculative is not None
        else bool(flags.SCF_SPECULATIVE_DECODING_ENABLED)
    )
    want_adaptive = bool(adaptive_thinking) or bool(flags.SCF_ADAPTIVE_COMPUTE_ENABLED and adaptive_thinking)
    if want_spec and want_adaptive:
        return {
            "mode": "greedy",
            "speculative_allowed": False,
            "adaptive_thinking": True,
            "reason": "speculative_decoding_blocked_by_incompatible_adaptive_thinking_path",
        }
    if want_spec:
        return {
            "mode": "speculative_mtp",
            "speculative_allowed": True,
            "adaptive_thinking": False,
            "reason": "speculative_mtp_enabled_with_deterministic_block_verification",
        }
    return {
        "mode": "greedy",
        "speculative_allowed": False,
        "adaptive_thinking": bool(want_adaptive),
        "reason": "standard_autoregressive_fallback",
    }


@dataclass
class SCFConfig:
    """Top-level configuration container for Supermix v91 Cognitive Fabric."""

    schema_version: str = SCF_CONFIG_SCHEMA_VERSION
    flags: SCFFeatureFlags = field(default_factory=SCFFeatureFlags)
    max_collective_candidates: int = 3
    max_reasoning_steps: int = 8
    conformal_alpha: float = 0.05
    allow_experimental_isolated: bool = False
    allow_automatic_promotion: bool = False  # Enforces Rule 9
    preferred_compute_budget: str = "auto"

    def __post_init__(self) -> None:
        if int(self.max_collective_candidates) < 1:
            raise ValueError("max_collective_candidates must be >= 1")
        if int(self.max_reasoning_steps) < 1:
            raise ValueError("max_reasoning_steps must be >= 1")
        if not (0.0 < float(self.conformal_alpha) < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1)")
        if self.allow_automatic_promotion:
            raise ValueError("automatic promotion is prohibited without passing evaluation gates")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["flags"] = self.flags.to_dict()
        return data
