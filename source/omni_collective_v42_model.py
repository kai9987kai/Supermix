from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from omni_collective_v41_model import OmniCollectiveEngineV41, OmniCollectiveNetV41, OmniPredictionV41, _prompt_variants_v41
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
except ImportError:  # pragma: no cover
    from .omni_collective_v41_model import OmniCollectiveEngineV41, OmniCollectiveNetV41, OmniPredictionV41, _prompt_variants_v41
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES


OmniCollectiveNetV42 = OmniCollectiveNetV41


def _budget_hint_v42(prompt: str, *, response_confidence: float = 0.0, domain_confidence: float = 0.0) -> str:
    lowered = str(prompt or "").lower()
    deep_markers = (
        "benchmark",
        "prove",
        "derive",
        "why does",
        "gsm8k",
        "mmlu",
        "arc",
        "hellaswag",
        "debug",
        "refactor",
        "traceback",
        "test failure",
        "compare",
        "choose",
    )
    medium_markers = (
        "summarize",
        "rewrite",
        "plan",
        "route",
        "which model",
        "auto mode",
        "collective",
        "image",
        "video",
        "audio",
        "ocr",
    )
    if response_confidence < 0.50 or domain_confidence < 0.44 or any(token in lowered for token in deep_markers):
        return "deep"
    if any(token in lowered for token in medium_markers):
        return "medium"
    return "short"


def _prompt_variants_v42(
    prompt: str,
    draft_answer: str = "",
    *,
    response_confidence: float = 0.0,
    domain_confidence: float = 0.0,
) -> List[str]:
    variants = list(
        _prompt_variants_v41(
            prompt,
            draft_answer,
            response_confidence=response_confidence,
            domain_confidence=domain_confidence,
        )
    )
    prompt_text = " ".join(str(prompt or "").split())
    lowered = prompt_text.lower()
    budget = _budget_hint_v42(
        prompt_text,
        response_confidence=response_confidence,
        domain_confidence=domain_confidence,
    )
    extras = [
        (
            f"Choose a {budget} reasoning budget before answering.\n"
            f"Request: {prompt_text}\n"
            "Keep the answer direct, but spend more thinking only if the request is actually hard."
        ),
        (
            "Decide whether you need a verifier pass before finalizing.\n"
            f"Request: {prompt_text}\n"
            "If the first answer could fail on correctness, route choice, or code accuracy, mentally verify it before returning the final answer."
        ),
        (
            "Prefer the smallest evidence set that still justifies the answer.\n"
            f"Request: {prompt_text}\n"
            "Do not overuse context or explanation length when a shorter grounded answer is enough."
        ),
    ]
    if any(token in lowered for token in ("benchmark", "boolq", "piqa", "hellaswag", "mmlu", "arc", "gsm8k")):
        extras.append(
            (
                "This is benchmark-style reasoning.\n"
                f"Request: {prompt_text}\n"
                "Use benchmark-specialist discipline first: identify the exact task type, eliminate weak options, then answer without filler."
            )
        )
    if any(token in lowered for token in ("python", "code", "bug", "traceback", "refactor", "patch", "test")):
        extras.append(
            (
                "This is an agentic coding request.\n"
                f"Request: {prompt_text}\n"
                "Plan the edit, simulate a verifier or failing test, then return the repair-oriented answer."
            )
        )
    if any(token in lowered for token in ("image", "video", "audio", "ocr", "diagram", "photo", "caption")):
        extras.append(
            (
                "This is a multimodal or omni-style request.\n"
                f"Request: {prompt_text}\n"
                "Think about the best grounded modality route first, then answer in a concise, evidence-aware form."
            )
        )
    if any(token in lowered for token in ("long context", "many files", "huge", "large", "full log", "full transcript")):
        extras.append(
            (
                "This request may tempt overthinking.\n"
                f"Request: {prompt_text}\n"
                "Select only the most relevant evidence and stop once the answer is supported."
            )
        )

    deduped: List[str] = []
    seen = set()
    for item in variants + extras:
        cooked = str(item).strip()
        if cooked and cooked not in seen:
            deduped.append(cooked)
            seen.add(cooked)
    return deduped


@dataclass
class OmniPredictionV42(OmniPredictionV41):
    reasoning_mode: str = "budgeted_route_first_verifier_polish_v42"
    planned_budget: str = "short"


class OmniCollectiveEngineV42(OmniCollectiveEngineV41):
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        super().__init__(weights_path=weights_path, meta_path=meta_path, device=device)
        self.deliberation_passes = int(self.meta.get("deliberation_passes") or 14)
        self.minimum_passes = int(self.meta.get("minimum_passes") or 7)
        self.grounding_threshold = float(self.meta.get("grounding_threshold") or 0.60)

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPredictionV42:
        prompt_text = str(prompt or "").strip()
        image_tensor, has_image = self._encode_image(image_path)
        raw_outputs = self._run_prompt(prompt_text, image_tensor, has_image)
        response_conf, domain_conf = self._confidence_from_outputs(raw_outputs)
        draft_idx = int(raw_outputs["response"].argmax(dim=0).item()) if self.responses else 0
        draft_answer = self.responses[draft_idx] if self.responses else ""
        planned_budget = _budget_hint_v42(
            prompt_text,
            response_confidence=response_conf,
            domain_confidence=domain_conf,
        )
        variants = _prompt_variants_v42(
            prompt_text,
            draft_answer,
            response_confidence=response_conf,
            domain_confidence=domain_conf,
        )

        weighted_outputs: List[tuple[float, Dict[str, torch.Tensor]]] = []
        combined = raw_outputs
        for index, variant in enumerate(variants, start=1):
            lowered = variant.lower()
            weight = 1.20 if index == 1 else 1.0
            if "deep reasoning budget" in lowered:
                weight = 1.10
            elif "medium reasoning budget" in lowered:
                weight = 1.07
            elif "verifier pass" in lowered:
                weight = 1.10
            elif "benchmark-style reasoning" in lowered:
                weight = 1.09
            elif "agentic coding request" in lowered:
                weight = 1.10
            elif "multimodal or omni-style request" in lowered:
                weight = 1.08
            weighted_outputs.append((weight, self._run_prompt(variant, image_tensor, has_image)))
            combined = self._combine_outputs(weighted_outputs)
            response_conf, domain_conf = self._confidence_from_outputs(combined)
            if index >= self.minimum_passes and response_conf >= self.grounding_threshold and domain_conf >= 0.50:
                break
            if index >= self.deliberation_passes:
                break

        outputs = combined
        intent_probs = torch.softmax(outputs["intent"], dim=0)
        response_probs = torch.softmax(outputs["response"], dim=0) if self.responses else torch.ones(1)
        vision_probs = torch.softmax(outputs["vision"], dim=0)
        domain_probs = torch.softmax(outputs["domain"], dim=0)

        intent_idx = int(intent_probs.argmax(dim=0).item()) if intent_probs.numel() else 0
        response_idx = int(response_probs.argmax(dim=0).item()) if response_probs.numel() else 0
        domain_idx = int(domain_probs.argmax(dim=0).item()) if domain_probs.numel() else 0
        values, indices = torch.topk(vision_probs, k=min(3, len(SCIENCE_IMAGE_CLASSES)))
        vision_key = SCIENCE_IMAGE_CLASSES[int(indices[0].item())] if SCIENCE_IMAGE_CLASSES else ""
        top_vision = [
            {"concept": SCIENCE_IMAGE_CLASSES[int(index.item())], "confidence": round(float(value.item()), 4)}
            for value, index in zip(values, indices)
            if SCIENCE_IMAGE_CLASSES
        ]
        return OmniPredictionV42(
            intent=self.intent_labels[min(intent_idx, len(self.intent_labels) - 1)] if self.intent_labels else "general",
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()) if values.numel() else 0.0,
            top_vision=top_vision,
            domain=self.domain_labels[min(domain_idx, len(self.domain_labels) - 1)] if self.domain_labels else "general",
            domain_confidence=float(domain_probs[domain_idx].item()) if domain_probs.numel() else 0.0,
            response_confidence=float(response_probs[response_idx].item()) if response_probs.numel() else 0.0,
            reasoning_passes=len(weighted_outputs),
            planned_budget=planned_budget,
        )


__all__ = [
    "OmniCollectiveEngineV42",
    "OmniCollectiveNetV42",
    "OmniPredictionV42",
    "_prompt_variants_v42",
]
