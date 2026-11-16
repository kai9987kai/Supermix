"""Supermix v91 Cognitive Fabric (SCF) — Additive Cognitive Meta-Architecture.

Zero-regression, feature-flagged meta-architecture integrating historical
ChampionNet, Cognitive Leap, Omni Collective, Qwen, MiMoMix, and NexusMind
subsystems through compatibility adapters and verifiable evidence gates.
"""

from __future__ import annotations

from scf.config import SCF_CONFIG_SCHEMA_VERSION, SCFConfig, SCFFeatureFlags, resolve_decoding_policy
from scf.core import CognitiveFabricRuntime
from scf.models import SCFModelRegistry

SCF_VERSION = "91.0.0-scf"
SCF_INTERNAL_NAME = "Cognitive Fabric"

__all__ = [
    "SCF_VERSION",
    "SCF_INTERNAL_NAME",
    "SCF_CONFIG_SCHEMA_VERSION",
    "SCFFeatureFlags",
    "SCFConfig",
    "SCFModelRegistry",
    "CognitiveFabricRuntime",
    "resolve_decoding_policy",
]
