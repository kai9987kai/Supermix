"""Task understanding, complexity estimation, and specialist routing for SCF."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scf.config import SCFFeatureFlags  # noqa: E402
from scf.models import SCFModelRegistry  # noqa: E402

__all__ = ["SCFRouter"]


class SCFRouter:
    """Routes tasks across Supermix model families and specialist engines."""

    _MATH_SCIENCE_RE = re.compile(
        r"\b(force|acceleration|momentum|kinetic|impulse|voltage|current|ohm|"
        r"resistance|spring|velocity|work|power|permutation|combination|solve|"
        r"calculate|compute|molarity|wave|frequency|wavelength)\b|[\d\)]\s*[\+\-\*/=]",
        re.IGNORECASE,
    )
    _CODE_RE = re.compile(
        r"\b(python|range\(|nums\.|code|debug|trace|function|regex|script|ast)\b",
        re.IGNORECASE,
    )
    _CAUSAL_EPISTEMIC_RE = re.compile(
        r"\b(do\(|causal|interventional|counterfactual|epistemic|active inference|"
        r"fisher-rao|shapley|barycenter|faultline|crosswire|mnemorph|verify proof)\b",
        re.IGNORECASE,
    )
    _CONNECTOME_RE = re.compile(
        r"\b(male cns|connectome|hemisphere|commissural|synapse count|neuron count|"
        r"dual soma|glial|dale)\b",
        re.IGNORECASE,
    )
    _MULTIMODAL_RE = re.compile(
        r"\b(image|photo|picture|vision|video|3d|mesh|stl|obj|openscad|protein|"
        r"folding|crystal|cif|mattergen)\b",
        re.IGNORECASE,
    )

    def __init__(self, registry: Optional[SCFModelRegistry] = None) -> None:
        self.registry = registry or SCFModelRegistry()

    def assess_task(self, prompt: Any) -> Dict[str, Any]:
        """Understand objectives, normalize symbolic templates, and score complexity."""

        import nexus_complexity as nc
        import prompt_normaliser as pn
        import prompt_understanding as pu

        text = str(prompt or "").strip()
        profile = pu.analyze_prompt(text)
        diag = pu.prompt_understanding_diagnostics(profile)
        norm = pn.normalise(text)
        comp = nc.AlgorithmicComplexityAnalyzer().analyze_sequence(text).to_dict()

        is_math = bool(self._MATH_SCIENCE_RE.search(text)) or (
            norm.rule is not None and norm.rule not in {"causal_intervention", "diffusion_thought", "conformal_stopping"}
        )
        is_code = bool(self._CODE_RE.search(text)) or (norm.rule or "").startswith("code_")
        is_causal = bool(self._CAUSAL_EPISTEMIC_RE.search(text)) or (norm.rule == "causal_intervention")
        is_connectome = bool(self._CONNECTOME_RE.search(text)) or (norm.rule or "").startswith("cns_")
        is_multimodal = bool(self._MULTIMODAL_RE.search(text))

        domain = "general"
        if is_connectome:
            domain = "connectome"
        elif is_code:
            domain = "code"
        elif is_causal:
            domain = "causal_epistemic"
        elif is_math:
            domain = "math_science"
        elif is_multimodal:
            domain = "multimodal"

        token_count = len(text.split())
        complexity = (
            "high"
            if (token_count >= 25 or is_causal or (is_math and is_code))
            else "medium"
            if (token_count >= 10 or is_math or is_code or is_connectome or is_multimodal)
            else "low"
        )
        return {
            "domain": domain,
            "complexity": complexity,
            "normalised_prompt": norm.prompt,
            "normalisation_rule": norm.rule,
            "prompt_diagnostics": diag,
            "complexity_profile": comp,
            "flags_detected": {
                "math_science": is_math,
                "code": is_code,
                "causal_epistemic": is_causal,
                "connectome": is_connectome,
                "multimodal": is_multimodal,
            },
        }

    def route(
        self,
        prompt: Any,
        *,
        flags: Optional[SCFFeatureFlags] = None,
        available_families: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Select optimal specialist/model family and hierarchical MoE policy."""

        active_flags = flags or SCFFeatureFlags()
        assessment = self.assess_task(prompt)
        status_map = self.registry.summary()
        if available_families is not None:
            allowed = set(available_families)
        else:
            allowed = {k for k, v in status_map.items() if v.get("available_in_source")}

        domain = assessment["domain"]
        if not active_flags.SCF_ENABLED or not active_flags.SCF_ROUTER_ENABLED:
            # Legacy safe baseline preserves direct MiMoMix/Nexus routing
            selected = "NexusMind" if domain in {"math_science", "code", "causal_epistemic"} else "MiMoMix"
            if selected not in allowed and allowed:
                selected = sorted(allowed)[0]
            return {
                "router_active": False,
                "selected_family": selected,
                "fallback_family": "MiMoMix" if "MiMoMix" in allowed else selected,
                "hierarchical_moe": False,
                "specialist_candidates": [selected],
                "assessment": assessment,
            }

        domain_preferences = {
            "math_science": ["NexusMind", "MiMoMix", "CognitiveLeap"],
            "code": ["NexusMind", "MiMoMix", "OmniCollective"],
            "causal_epistemic": ["NexusMind", "ChampionNet", "MiMoMix"],
            "connectome": ["MiMoMix", "NexusMind"],
            "multimodal": ["OmniCollective", "MiMoMix", "ChampionNet"],
            "general": ["MiMoMix", "OmniCollective", "CognitiveLeap", "NexusMind"],
        }
        prefs = domain_preferences.get(domain, ["MiMoMix", "NexusMind"])
        candidates = [fam for fam in prefs if fam in allowed]
        if not candidates:
            candidates = sorted(allowed) or ["NexusMind"]

        # Hierarchical MoE option (ChampionNet HierarchicalMoEClassifierHead + MiMoMix SparseMoE)
        hmoe_active = bool(active_flags.SCF_HIERARCHICAL_MOE_ENABLED)
        selected = candidates[0]
        fallback = candidates[1] if len(candidates) > 1 else candidates[0]
        return {
            "router_active": True,
            "selected_family": selected,
            "fallback_family": fallback,
            "hierarchical_moe": hmoe_active,
            "specialist_candidates": candidates,
            "assessment": assessment,
        }
