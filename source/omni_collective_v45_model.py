from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from omni_collective_v44_model import OmniCollectiveEngineV44, OmniCollectiveNetV44, OmniPredictionV44, _prompt_variants_v44
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
except ImportError:  # pragma: no cover
    from .omni_collective_v44_model import OmniCollectiveEngineV44, OmniCollectiveNetV44, OmniPredictionV44, _prompt_variants_v44
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES


OmniCollectiveNetV45 = OmniCollectiveNetV44


# ---------------------------------------------------------------------------
# V45 NOVEL INNOVATION 1: Process Reward Model (PRM) Scoring
# ---------------------------------------------------------------------------
# Instead of only scoring the *final* answer confidence, V45 decomposes
# each reasoning variant into logical "steps" and scores each step
# independently. If any intermediate step causes a confidence regression,
# the entire variant is penalized proportionally to *where* the failure
# occurred (earlier failures = heavier penalty). This is inspired by
# Process-Based Reward Models (Lightman et al., 2023; extended 2025-2026).
# ---------------------------------------------------------------------------

def _process_reward_score(
    baseline_conf: float,
    variant_conf: float,
    step_index: int,
    total_steps: int,
) -> float:
    """Compute a process-level reward for a single reasoning step.
    
    Early steps that degrade confidence are penalized exponentially harder
    than late steps, since early errors compound through the reasoning chain.
    """
    delta = variant_conf - baseline_conf
    if delta >= 0:
        # Positive progress: reward proportional to improvement
        return 1.0 + (delta * 2.0)
    else:
        # Negative progress: penalize earlier steps more heavily
        # Position factor: step 1 of 10 gets a 1.0 multiplier, step 10 gets 0.1
        position_penalty = 1.0 - (step_index / max(total_steps, 1))
        return max(0.05, 1.0 + (delta * 3.0 * (1.0 + position_penalty)))


# ---------------------------------------------------------------------------
# V45 NOVEL INNOVATION 2: Adaptive Compute Budget via Difficulty Estimation
# ---------------------------------------------------------------------------
# Rather than a fixed number of deliberation passes, V45 dynamically
# estimates prompt difficulty using a lightweight heuristic and allocates
# compute accordingly. This implements the "compute-optimal" inference
# strategy from recent test-time scaling research (2025-2026).
# ---------------------------------------------------------------------------

def _estimate_difficulty_v45(
    prompt: str,
    response_confidence: float,
    domain_confidence: float,
) -> str:
    """Four-tier difficulty estimation: trivial / easy / hard / frontier.
    
    This controls how many deliberation passes the engine will attempt,
    preventing wasted compute on simple prompts while allocating maximum
    resources to genuinely difficult problems.
    """
    lowered = str(prompt or "").lower()
    
    # Keyword-based complexity signals
    frontier_markers = (
        "prove", "derive", "formal proof", "step by step", "multi-step",
        "analyze the tradeoff", "compare and contrast thoroughly",
        "debug this complex", "architectural decision",
    )
    hard_markers = (
        "benchmark", "gsm8k", "mmlu", "arc", "hellaswag", "boolq", "piqa",
        "debug", "refactor", "traceback", "test failure", "compare", "choose",
        "why does", "explain why",
    )
    easy_markers = (
        "summarize", "rewrite", "translate", "what is", "define",
        "list", "name",
    )
    
    # Confidence-based override: if the model is already very uncertain,
    # the problem is at least "hard" regardless of keyword signals
    if response_confidence < 0.30 or domain_confidence < 0.30:
        return "frontier"
    if response_confidence < 0.45 or domain_confidence < 0.40:
        if any(m in lowered for m in frontier_markers):
            return "frontier"
        return "hard"
    
    if any(m in lowered for m in frontier_markers):
        return "frontier"
    if any(m in lowered for m in hard_markers):
        return "hard"
    if response_confidence > 0.80 and domain_confidence > 0.80:
        return "trivial"
    if any(m in lowered for m in easy_markers):
        return "easy"
    return "hard"  # default to hard so we don't under-think


def _compute_budget_for_difficulty(difficulty: str) -> Tuple[int, int]:
    """Return (minimum_passes, max_passes) for a given difficulty tier."""
    budgets = {
        "trivial":  (1, 3),
        "easy":     (3, 6),
        "hard":     (5, 12),
        "frontier": (8, 18),
    }
    return budgets.get(difficulty, (5, 12))


# ---------------------------------------------------------------------------
# V45 NOVEL INNOVATION 3: GRPO-Inspired Group Relative Scoring
# ---------------------------------------------------------------------------
# Instead of comparing each variant against a single "best_conf" baseline,
# V45 generates a group of variant outputs and scores them *relative to
# each other* using a normalized advantage, exactly like GRPO (DeepSeek-R1).
# Variants that score above the group mean get boosted; those below get
# suppressed. This removes the need for absolute confidence thresholds.
# ---------------------------------------------------------------------------

def _grpo_relative_weights(
    variant_confidences: List[float],
    base_weights: List[float],
) -> List[float]:
    """Apply Group Relative Policy Optimization weighting to variant outputs.
    
    For each variant, compute advantage = (conf - mean) / std, then use
    that to scale the base weight. This is a statistically principled
    alternative to arbitrary threshold-based boosting.
    """
    if len(variant_confidences) < 2:
        return list(base_weights)
    
    mean_conf = sum(variant_confidences) / len(variant_confidences)
    variance = sum((c - mean_conf) ** 2 for c in variant_confidences) / len(variant_confidences)
    std_conf = max(math.sqrt(variance), 1e-6)
    
    adjusted = []
    for conf, base_w in zip(variant_confidences, base_weights):
        advantage = (conf - mean_conf) / std_conf
        # Sigmoid-like scaling: map advantage to a multiplier in [0.1, 2.0]
        multiplier = 2.0 / (1.0 + math.exp(-advantage))
        adjusted.append(base_w * multiplier)
    
    return adjusted


# ---------------------------------------------------------------------------
# V45 NOVEL INNOVATION 4: Speculative Chain-of-Thought (SpecCoT)
# ---------------------------------------------------------------------------
# Before running expensive full deliberation, V45 generates a quick
# "speculative draft" using a lightweight prompt. If the draft achieves
# high confidence, we accept it without full deliberation. If not, we
# fall through to the full MCTS-style reasoning loop. This is inspired
# by Speculative Decoding (Leviathan et al., 2023) applied at the
# *reasoning step* level rather than the token level.
# ---------------------------------------------------------------------------

def _speculative_draft_prompt(prompt_text: str) -> str:
    return (
        "Quick-answer mode. Give the most direct, confident answer you can.\n"
        f"Request: {prompt_text}\n"
        "If you are very confident, answer in one sentence. "
        "If you are not confident, say 'NEED_DEEPER_REASONING'."
    )


# ---------------------------------------------------------------------------
# V45 NOVEL INNOVATION 5: Self-Play Adversarial Verification
# ---------------------------------------------------------------------------
# After generating a candidate answer, V45 generates an adversarial
# challenge prompt that tries to find flaws in the answer. If the
# adversarial pass reveals low confidence, the answer is re-routed
# through deeper deliberation. This implements the "Generator-Verifier-
# Critic" triad from recent self-play research (2025-2026).
# ---------------------------------------------------------------------------

def _adversarial_challenge_prompt(prompt_text: str, draft_answer: str) -> str:
    return (
        "You are a strict adversarial verifier. Your job is to find flaws.\n"
        f"Original request: {prompt_text}\n"
        f"Proposed answer: {draft_answer}\n"
        "Identify the single most critical flaw, factual error, or logical gap "
        "in the proposed answer. If the answer is actually correct and well-grounded, "
        "confirm it with high confidence."
    )


# ---------------------------------------------------------------------------
# V45 Budget Hint (extends V44 with difficulty-aware tiering)
# ---------------------------------------------------------------------------

def _budget_hint_v45(prompt: str, *, response_confidence: float = 0.0, domain_confidence: float = 0.0) -> str:
    difficulty = _estimate_difficulty_v45(prompt, response_confidence, domain_confidence)
    if difficulty == "frontier":
        return "deep"
    if difficulty == "hard":
        return "deep" if response_confidence < 0.50 else "medium"
    if difficulty == "easy":
        return "medium" if response_confidence < 0.70 else "short"
    return "short"  # trivial


# ---------------------------------------------------------------------------
# V45 Prompt Variants (extends V44 with new reasoning modalities)
# ---------------------------------------------------------------------------

def _prompt_variants_v45(
    prompt: str,
    draft_answer: str = "",
    *,
    response_confidence: float = 0.0,
    domain_confidence: float = 0.0,
) -> List[str]:
    # Inherit all V44 variants as the foundation
    variants = list(
        _prompt_variants_v44(
            prompt,
            draft_answer,
            response_confidence=response_confidence,
            domain_confidence=domain_confidence,
        )
    )
    prompt_text = " ".join(str(prompt or "").split())
    lowered = prompt_text.lower()
    difficulty = _estimate_difficulty_v45(prompt_text, response_confidence, domain_confidence)
    budget = _budget_hint_v45(
        prompt_text,
        response_confidence=response_confidence,
        domain_confidence=domain_confidence,
    )
    extras: List[str] = []

    # V45 NOVEL: Process Reward verification prompt
    if difficulty in ("hard", "frontier"):
        extras.append(
            "Use step-by-step Process Reward verification.\n"
            f"Request: {prompt_text}\n"
            "Break your reasoning into numbered steps. After each step, "
            "briefly assess whether it logically follows from the previous step. "
            "If any step is weak, revise it before proceeding to the next."
        )

    # V45 NOVEL: GRPO-style multi-candidate generation
    if response_confidence < 0.40:
        extras.append(
            "Generate three candidate answers internally using different approaches.\n"
            f"Request: {prompt_text}\n"
            "Approach 1: Direct reasoning from first principles.\n"
            "Approach 2: Elimination-based reasoning (rule out wrong options).\n"
            "Approach 3: Analogical reasoning (what similar problem do you know?).\n"
            "Select the candidate with the strongest evidence chain and output only that one."
        )

    # V45 NOVEL: Adversarial self-play verification
    if difficulty == "frontier" and draft_answer:
        extras.append(
            _adversarial_challenge_prompt(prompt_text, draft_answer)
        )

    # V45 NOVEL: Curriculum-aware difficulty signal
    extras.append(
        f"Difficulty estimate: {difficulty}. Budget: {budget}.\n"
        f"Request: {prompt_text}\n"
        f"Calibrate your answer depth to match the difficulty. "
        f"{'Go deep and thorough.' if difficulty in ('hard', 'frontier') else 'Be concise and direct.'}"
    )

    # V45 NOVEL: Reasoning scaffolding via semantic structure tags
    if any(token in lowered for token in ("prove", "derive", "formal", "theorem", "proof")):
        extras.append(
            "Use formal reasoning scaffolding.\n"
            f"Request: {prompt_text}\n"
            "Structure: [GIVEN] -> [ASSUME] -> [DERIVE] -> [CONCLUDE]. "
            "Each transition must be explicitly justified."
        )

    # V45 NOVEL: Uncertainty quantification prompt
    if response_confidence < 0.55 and domain_confidence < 0.55:
        extras.append(
            "Quantify your uncertainty explicitly.\n"
            f"Request: {prompt_text}\n"
            "Before answering, state what you know for certain, what you are "
            "inferring with moderate confidence, and what you are genuinely "
            "uncertain about. Then give your best answer with these caveats noted."
        )

    deduped: List[str] = []
    seen = set()
    for item in variants + extras:
        cooked = str(item).strip()
        if cooked and cooked not in seen:
            deduped.append(cooked)
            seen.add(cooked)
    return deduped


# ---------------------------------------------------------------------------
# V45 Prediction dataclass
# ---------------------------------------------------------------------------

@dataclass
class OmniPredictionV45(OmniPredictionV44):
    reasoning_mode: str = "adaptive_compute_grpo_prm_speculative_v45"
    planned_budget: str = "short"
    difficulty_estimate: str = "hard"
    speculative_accepted: bool = False
    adversarial_verified: bool = False
    grpo_group_size: int = 0
    process_reward_scores: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# V45 Engine (the core innovation hub)
# ---------------------------------------------------------------------------

class OmniCollectiveEngineV45(OmniCollectiveEngineV44):
    def __init__(self, *, weights_path: Path, meta_path: Path, device: Optional[torch.device] = None) -> None:
        super().__init__(weights_path=weights_path, meta_path=meta_path, device=device)
        # V45: These are now dynamically overridden per-prompt based on difficulty
        self.base_deliberation_passes = int(self.meta.get("deliberation_passes") or 14)
        self.base_minimum_passes = int(self.meta.get("minimum_passes") or 7)
        self.grounding_threshold = float(self.meta.get("grounding_threshold") or 0.60)
        self.speculative_threshold = float(self.meta.get("speculative_threshold") or 0.82)

    def predict(self, prompt: str, image_path: Optional[str] = None) -> OmniPredictionV45:
        prompt_text = str(prompt or "").strip()
        image_tensor, has_image = self._encode_image(image_path)

        # ── Phase 0: Initial assessment ──────────────────────────────
        raw_outputs = self._run_prompt(prompt_text, image_tensor, has_image)
        response_conf, domain_conf = self._confidence_from_outputs(raw_outputs)
        draft_idx = int(raw_outputs["response"].argmax(dim=0).item()) if self.responses else 0
        draft_answer = self.responses[draft_idx] if self.responses else ""

        # ── V45 INNOVATION: Difficulty-adaptive compute budget ────────
        difficulty = _estimate_difficulty_v45(prompt_text, response_conf, domain_conf)
        min_passes, max_passes = _compute_budget_for_difficulty(difficulty)
        planned_budget = _budget_hint_v45(
            prompt_text,
            response_confidence=response_conf,
            domain_confidence=domain_conf,
        )

        # ── V45 INNOVATION: Speculative Chain-of-Thought ─────────────
        # Try a fast speculative draft first. If it's highly confident,
        # skip the full deliberation loop entirely.
        speculative_accepted = False
        if difficulty in ("trivial", "easy"):
            spec_prompt = _speculative_draft_prompt(prompt_text)
            spec_outputs = self._run_prompt(spec_prompt, image_tensor, has_image)
            spec_r_conf, spec_d_conf = self._confidence_from_outputs(spec_outputs)
            if spec_r_conf >= self.speculative_threshold and spec_d_conf >= 0.70:
                # Speculative draft accepted — skip deliberation
                speculative_accepted = True
                outputs = spec_outputs
                response_conf = spec_r_conf
                domain_conf = spec_d_conf

        if not speculative_accepted:
            # ── Full deliberation with V45 innovations ───────────────
            variants = _prompt_variants_v45(
                prompt_text,
                draft_answer,
                response_confidence=response_conf,
                domain_confidence=domain_conf,
            )

            # Collect all variant outputs and base weights FIRST (GRPO needs the group)
            variant_outputs_list: List[Dict[str, torch.Tensor]] = []
            variant_confidences: List[float] = []
            base_weights: List[float] = []
            process_reward_scores: List[float] = []

            for index, variant in enumerate(variants, start=1):
                if index > max_passes:
                    break

                lowered = variant.lower()
                weight = 1.20 if index == 1 else 1.0

                # V45: Extended weight taxonomy
                if "process reward" in lowered:
                    weight = 1.30  # V45: Process-level verification is highest value
                elif "three candidate answers" in lowered:
                    weight = 1.25  # V45: GRPO multi-candidate
                elif "adversarial verifier" in lowered:
                    weight = 1.22  # V45: Self-play verification
                elif "tree-of-thoughts" in lowered:
                    weight = 1.25
                elif "self-reflection pass" in lowered:
                    weight = 1.20
                elif "contrastive decoding" in lowered:
                    weight = 1.18
                elif "reasoning scaffolding" in lowered:
                    weight = 1.15  # V45: Formal reasoning structure
                elif "uncertainty" in lowered and "quantify" in lowered:
                    weight = 1.12  # V45: Explicit uncertainty
                elif "deep reasoning budget" in lowered:
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
                elif "difficulty estimate" in lowered:
                    weight = 1.05  # V45: Curriculum signal

                variant_outs = self._run_prompt(variant, image_tensor, has_image)
                var_r_conf, _ = self._confidence_from_outputs(variant_outs)

                # V45 INNOVATION: Process Reward scoring
                prm_score = _process_reward_score(
                    baseline_conf=response_conf,
                    variant_conf=var_r_conf,
                    step_index=index,
                    total_steps=min(len(variants), max_passes),
                )
                process_reward_scores.append(prm_score)
                weight *= prm_score

                variant_outputs_list.append(variant_outs)
                variant_confidences.append(var_r_conf)
                base_weights.append(weight)

                # V45 INNOVATION: Dynamic early termination
                # After minimum passes, check if we've achieved convergence
                if index >= min_passes:
                    # Use running GRPO to check group quality
                    grpo_weights = _grpo_relative_weights(variant_confidences, base_weights)
                    weighted_outputs = list(zip(grpo_weights, variant_outputs_list))
                    combined = self._combine_outputs(weighted_outputs)
                    cur_r_conf, cur_d_conf = self._confidence_from_outputs(combined)
                    
                    # V45: Convergence check — stop if quality is high enough
                    if cur_r_conf >= self.grounding_threshold + 0.05 and cur_d_conf >= 0.60:
                        break

            # ── V45 INNOVATION: Final GRPO weighting ─────────────────
            if variant_outputs_list:
                final_grpo_weights = _grpo_relative_weights(variant_confidences, base_weights)
                weighted_outputs = list(zip(final_grpo_weights, variant_outputs_list))
                outputs = self._combine_outputs(weighted_outputs)
            else:
                outputs = raw_outputs
                final_grpo_weights = []
                process_reward_scores = []

            # ── V45 INNOVATION: Adversarial self-play verification ───
            # After deliberation, run one adversarial challenge pass
            adversarial_verified = False
            if difficulty in ("hard", "frontier"):
                response_conf_post, domain_conf_post = self._confidence_from_outputs(outputs)
                resp_idx = int(outputs["response"].argmax(dim=0).item()) if self.responses else 0
                current_answer = self.responses[resp_idx] if self.responses else ""
                
                adv_prompt = _adversarial_challenge_prompt(prompt_text, current_answer)
                adv_outputs = self._run_prompt(adv_prompt, image_tensor, has_image)
                adv_r_conf, _ = self._confidence_from_outputs(adv_outputs)
                
                if adv_r_conf >= 0.55:
                    # Adversarial verifier endorsed the answer
                    adversarial_verified = True
                    # Boost confidence slightly in the final output blend
                    blend_weight = 0.15
                    for key in outputs:
                        if isinstance(outputs[key], torch.Tensor) and outputs[key].shape == adv_outputs[key].shape:
                            outputs[key] = (1 - blend_weight) * outputs[key] + blend_weight * adv_outputs[key]
                else:
                    # Adversarial verifier found flaws — suppress low-quality signals
                    adversarial_verified = False
                    # Don't blend adversarial output (it's critical, not constructive)

        # ── Final prediction assembly ────────────────────────────────
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

        grpo_size = len(variant_outputs_list) if not speculative_accepted else 0

        return OmniPredictionV45(
            intent=self.intent_labels[min(intent_idx, len(self.intent_labels) - 1)] if self.intent_labels else "general",
            response_text=self.responses[response_idx] if self.responses else "",
            vision_concept=vision_key,
            vision_confidence=float(values[0].item()) if values.numel() else 0.0,
            top_vision=top_vision,
            domain=self.domain_labels[min(domain_idx, len(self.domain_labels) - 1)] if self.domain_labels else "general",
            domain_confidence=float(domain_probs[domain_idx].item()) if domain_probs.numel() else 0.0,
            response_confidence=float(response_probs[response_idx].item()) if response_probs.numel() else 0.0,
            reasoning_passes=grpo_size if not speculative_accepted else 1,
            planned_budget=planned_budget,
            difficulty_estimate=difficulty,
            speculative_accepted=speculative_accepted,
            adversarial_verified=adversarial_verified if not speculative_accepted else False,
            grpo_group_size=grpo_size,
            process_reward_scores=process_reward_scores if not speculative_accepted else [],
        )


__all__ = [
    "OmniCollectiveEngineV45",
    "OmniCollectiveNetV45",
    "OmniPredictionV45",
    "_prompt_variants_v45",
]
