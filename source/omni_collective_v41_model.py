from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from omni_collective_v8_model import OmniCollectiveEngineV8, OmniCollectiveNetV8, OmniPredictionV8, _prompt_variants_v8
    from math_equation_model import heuristic_intent, solve_intent
    from image_feature_utils import describe_image_for_text
    from image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES
except ImportError:  # pragma: no cover
    from .omni_collective_v8_model import OmniCollectiveEngineV8, OmniCollectiveNetV8, OmniPredictionV8, _prompt_variants_v8
    from .math_equation_model import heuristic_intent, solve_intent
    from .image_feature_utils import describe_image_for_text
    from .image_recognition_model import CLASS_LABELS, SCIENCE_IMAGE_CLASSES


OmniCollectiveNetV41 = OmniCollectiveNetV8


def _prompt_variants_v41(
    prompt: str,
    draft_answer: str = "",
    *,
    response_confidence: float = 0.0,
    domain_confidence: float = 0.0,
) -> List[str]:
    variants = list(
        _prompt_variants_v8(
            prompt,
            draft_answer,
            response_confidence=response_confidence,
            domain_confidence=domain_confidence,
        )
    )
    prompt_text = " ".join(str(prompt or "").split())
    lowered = prompt_text.lower()
    extras = [
        (
            "Create a short hidden plan before answering.\n"
            f"Request: {prompt_text}\n"
            "Identify the route, the evidence you actually have, the likely failure mode, and the clearest final form of the answer."
        ),
        (
            "Run a reflection pass before finalizing the answer.\n"
            f"Request: {prompt_text}\n"
            "Look for the most likely mistake in your first instinct, then keep the supported core and repair the weak parts."
        ),
        (
            "Optimize the answer for a human reader.\n"
            f"Request: {prompt_text}\n"
            "Prefer directness, clarity, and useful structure while preserving the grounded substance."
        ),
    ]
    if any(token in lowered for token in ("python", "code", "bug", "traceback", "test", "stack trace", "refactor", "patch")):
        extras.append(
            (
                "This is a coding or debugging request.\n"
                f"Request: {prompt_text}\n"
                "Simulate a code-review and repair pass before answering. Prefer exact fixes, concrete reasoning, and reproducible guidance."
            )
        )
    if any(token in lowered for token in ("creative", "story", "poem", "vivid", "rewrite", "tone", "paragraph")):
        extras.append(
            (
                "This answer should be more vivid and human-friendly without becoming ungrounded.\n"
                f"Request: {prompt_text}\n"
                "Prefer controlled creativity over bland filler or invented specifics."
            )
        )
    if any(token in lowered for token in ("which model", "auto mode", "collective", "agent mode", "route", "selector")):
        extras.append(
            (
                "Choose the route before answering.\n"
                f"Request: {prompt_text}\n"
                "Pick the best local specialist or generalist in your head first, then answer from that perspective."
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
class OmniPredictionV41(OmniPredictionV8):
    reasoning_mode: str = "latent_plan_reflective_multitype_consensus_v41"


class OmniCollectiveEngineV41(OmniCollectiveEngineV8):
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        super().__init__(weights_path=weights_path, meta_path=meta_path, device=device)
        self.deliberation_passes = int(self.meta.get("deliberation_passes") or 12)
        self.minimum_passes = int(self.meta.get("minimum_passes") or 6)
        self.grounding_threshold = float(self.meta.get("grounding_threshold") or 0.56)

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPredictionV41:
        prompt_text = str(prompt or "").strip()
        image_tensor, has_image = self._encode_image(image_path)
        raw_outputs = self._run_prompt(prompt_text, image_tensor, has_image)
        response_conf, domain_conf = self._confidence_from_outputs(raw_outputs)
        draft_idx = int(raw_outputs["response"].argmax(dim=0).item()) if self.responses else 0
        draft_answer = self.responses[draft_idx] if self.responses else ""
        variants = _prompt_variants_v41(
            prompt_text,
            draft_answer,
            response_confidence=response_conf,
            domain_confidence=domain_conf,
        )

        weighted_outputs: List[tuple[float, Dict[str, torch.Tensor]]] = []
        combined = raw_outputs
        for index, variant in enumerate(variants, start=1):
            weight = 1.22 if index == 1 else 1.0
            lowered = variant.lower()
            if "hidden plan" in lowered:
                weight = 1.08
            elif "reflection pass" in lowered:
                weight = 1.09
            elif "human reader" in lowered:
                weight = 1.06
            elif "code-review and repair" in lowered:
                weight = 1.08
            elif "choose the route before answering" in lowered:
                weight = 1.07
            weighted_outputs.append((weight, self._run_prompt(variant, image_tensor, has_image)))
            combined = self._combine_outputs(weighted_outputs)
            response_conf, domain_conf = self._confidence_from_outputs(combined)
            if index >= self.minimum_passes and response_conf >= self.grounding_threshold and domain_conf >= 0.48:
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
        return OmniPredictionV41(
            intent=self.intent_labels[min(intent_idx, len(self.intent_labels) - 1)] if self.intent_labels else "general",
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()) if values.numel() else 0.0,
            top_vision=top_vision,
            domain=self.domain_labels[min(domain_idx, len(self.domain_labels) - 1)] if self.domain_labels else "general",
            domain_confidence=float(domain_probs[domain_idx].item()) if domain_probs.numel() else 0.0,
            response_confidence=float(response_probs[response_idx].item()) if response_probs.numel() else 0.0,
            reasoning_passes=len(weighted_outputs),
        )

    def answer(self, prompt: str, image_path: Optional[str] = None) -> str:
        prompt_text = str(prompt or "").strip()
        math_intent = heuristic_intent(prompt_text)
        prediction = self.predict(prompt_text, image_path=image_path)
        if math_intent:
            try:
                solved = solve_intent(prompt_text, math_intent)
                return str(solved.get("response") or prediction.response_text or "Math intent detected.")
            except Exception:
                return prediction.response_text or "This looks like a math problem, but I could not solve it cleanly."
        if image_path:
            info = self.class_info.get(prediction.vision_concept, {})
            descriptor = describe_image_for_text(str(image_path))
            lines = [
                f"Image read: {CLASS_LABELS.get(prediction.vision_concept, prediction.vision_concept.replace('_', ' '))} ({prediction.vision_confidence * 100:.1f}% confidence).",
            ]
            if str(info.get("caption") or "").strip():
                lines.append(f"Likely concept: {str(info.get('caption')).strip()}.")
            lines.append(f"Visual descriptor: {descriptor}.")
            if prediction.response_text:
                lines.append(prediction.response_text)
            return "\n".join(lines)
        if prediction.response_confidence < self.grounding_threshold and prediction.domain in {"knowledge", "general", "planning", "language", "model_selection"}:
            if prediction.response_text:
                return f"Best grounded answer from my local training: {prediction.response_text}"
            return "I do not have enough grounded evidence in my local training for a confident answer."
        return prediction.response_text or "I did not have a strong fused answer for that prompt."


__all__ = [
    "OmniCollectiveEngineV41",
    "OmniCollectiveNetV41",
    "OmniPredictionV41",
    "_prompt_variants_v41",
]
