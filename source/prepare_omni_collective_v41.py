from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "omni_collective_v41_blueprint.json"


RECENT_METHOD_REFERENCES: List[Dict[str, str]] = [
    {
        "name": "DeepSeek-V3 Technical Report",
        "url": "https://arxiv.org/abs/2412.19437",
        "applied_as": "multi_token_prediction_auxiliary_head",
        "why": "Add a compact next-span prediction loss so the model learns denser future-planning structure without a large parameter jump.",
    },
    {
        "name": "Better & Faster Large Language Models via Multi-token Prediction",
        "url": "https://arxiv.org/abs/2404.19737",
        "applied_as": "multi_token_prediction_training_recipe",
        "why": "Use extra future-token heads during training to improve coding and algorithmic reasoning sample efficiency.",
    },
    {
        "name": "s1: Simple test-time scaling",
        "url": "https://arxiv.org/abs/2501.19393",
        "applied_as": "reasoning_budget_curriculum",
        "why": "Train with short, medium, and long budget targets so v41 can spend more compute on hard prompts without bloating every answer.",
    },
    {
        "name": "LIMO: Less Is More for Reasoning",
        "url": "https://arxiv.org/abs/2502.03387",
        "applied_as": "high_density_reasoning_core_set",
        "why": "Shift part of the reasoning mix away from sheer volume and toward a smaller, harder, cleaner curriculum.",
    },
    {
        "name": "RefineCoder",
        "url": "https://arxiv.org/abs/2502.09183",
        "applied_as": "code_critique_and_repair_loop",
        "why": "Use critique-plus-repair traces for bug fixing, test repair, and patch refinement instead of one-shot code answers.",
    },
    {
        "name": "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "url": "https://arxiv.org/abs/2303.11366",
        "applied_as": "verbal_reflection_and_repair_memory",
        "why": "Store concise failure reflections for debugging and coding tasks instead of only training on final answers.",
    },
    {
        "name": "Self-Play Fine-Tuning (SPIN)",
        "url": "https://arxiv.org/abs/2401.01335",
        "applied_as": "iterative_self_play_preference_loop",
        "why": "Generate self-play comparisons from v8, v40, and qwen teachers to improve helpfulness and problem-solving without requiring a giant new human-labeled set.",
    },
    {
        "name": "Self-Refine",
        "url": "https://arxiv.org/abs/2303.17651",
        "applied_as": "communication_polish_phase",
        "why": "Teach v41 to critique and rewrite its own drafts into clearer human-facing answers.",
    },
    {
        "name": "Large Language Models have Intrinsic Self-Correction Ability",
        "url": "https://arxiv.org/abs/2406.15673",
        "applied_as": "fair_prompt_zero_temp_self_correction",
        "why": "Use a dedicated self-correction pass with fair prompts and deterministic decoding to reduce hallucinations and repair weak drafts.",
    },
    {
        "name": "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback",
        "url": "https://arxiv.org/abs/2309.00267",
        "applied_as": "ai_feedback_preference_labels",
        "why": "Use strong local teachers as AI preference labelers for communication, helpfulness, and harmlessness style comparisons.",
    },
]


def latest_v8_summary_path(repo_root: Path = REPO_ROOT) -> Path:
    candidates = sorted(
        repo_root.glob("output/supermix_omni_collective_v8_frontier_*/omni_collective_v8_frontier_summary.json"),
        key=lambda item: (item.parent.name, item.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No omni_collective_v8 frontier summary found under output/.")
    return candidates[0].resolve()


def latest_common_benchmark_summary_path(repo_root: Path = REPO_ROOT) -> Optional[Path]:
    candidates = sorted(
        repo_root.glob("output/benchmark_all_models_common_plus_summary_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _round_to_million(value: int) -> int:
    return int(round(float(value) / 1_000_000.0) * 1_000_000)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _stage_metrics(summary: Dict[str, Any], stage_name: str) -> Dict[str, float]:
    metrics = summary.get(stage_name, {}).get("val_metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {str(key): _safe_float(value) for key, value in metrics.items()}


def _recommended_parameter_target(current_parameter_count: int) -> int:
    growth_floor = current_parameter_count + 8_000_000
    growth_soft = current_parameter_count + max(12_000_000, int(current_parameter_count * 0.125))
    growth_cap = current_parameter_count + 20_000_000
    cooked = min(max(growth_soft, growth_floor), growth_cap)
    return _round_to_million(cooked)


def _extract_frontier_scores(repo_root: Path) -> Dict[str, Any]:
    benchmark_path = latest_common_benchmark_summary_path(repo_root)
    if benchmark_path is None:
        return {
            "benchmark_summary_path": None,
            "best_overall": None,
            "best_fusion": None,
        }
    payload = load_summary(benchmark_path)
    rows = list(payload.get("summary_rows") or [])
    if not rows:
        return {
            "benchmark_summary_path": str(benchmark_path),
            "best_overall": None,
            "best_fusion": None,
        }
    best_overall = max(rows, key=lambda item: float(item.get("overall_exact") or 0.0))
    fusion_rows = [row for row in rows if str(row.get("family") or "") == "fusion"]
    best_fusion = max(fusion_rows, key=lambda item: float(item.get("overall_exact") or 0.0)) if fusion_rows else None
    return {
        "benchmark_summary_path": str(benchmark_path),
        "best_overall": {
            "model": str(best_overall.get("model") or ""),
            "family": str(best_overall.get("family") or ""),
            "overall_exact": _safe_float(best_overall.get("overall_exact") or 0.0),
        },
        "best_fusion": (
            {
                "model": str(best_fusion.get("model") or ""),
                "family": str(best_fusion.get("family") or ""),
                "overall_exact": _safe_float(best_fusion.get("overall_exact") or 0.0),
            }
            if best_fusion is not None
            else None
        ),
    }


def build_v41_blueprint(summary: Dict[str, Any], summary_path: Optional[Path] = None) -> Dict[str, Any]:
    parameter_count = int(summary.get("parameter_count") or 0)
    stage1_rows = int(summary.get("dataset_summary", {}).get("stage1_rows") or 0)
    stage2_rows = int(summary.get("dataset_summary", {}).get("stage2_rows") or 0)
    stage2_metrics = _stage_metrics(summary, "stage2")
    stage1_metrics = _stage_metrics(summary, "stage1")
    response_accuracy = stage2_metrics.get("response_accuracy", 0.0)
    intent_accuracy = stage2_metrics.get("intent_accuracy", 0.0)
    vision_accuracy = stage2_metrics.get("vision_accuracy", 0.0)
    domain_accuracy = stage2_metrics.get("domain_accuracy", 0.0)
    teacher_keys = list(summary.get("dataset_summary", {}).get("teacher_league", {}).get("teacher_keys", []))
    frontier_scores = _extract_frontier_scores(REPO_ROOT)

    target_parameter_count = _recommended_parameter_target(parameter_count)
    current_text_hidden = 272
    current_fusion_hidden = 1088
    current_expert_count = 10
    current_expert_hidden = 1792

    return {
        "family": "omni_collective_v41",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "derived_from": {
            "artifact": str(summary.get("artifact") or ""),
            "summary_path": str(summary_path.resolve()) if summary_path else "",
            "parameter_count": parameter_count,
            "stage1_rows": stage1_rows,
            "stage2_rows": stage2_rows,
            "stage1_best_score": _safe_float(summary.get("stage1", {}).get("best_score")),
            "stage2_best_score": _safe_float(summary.get("stage2", {}).get("best_score")),
        },
        "observed_gaps": [
            {
                "name": "response_quality",
                "current": response_accuracy,
                "target": max(0.145, round(response_accuracy + 0.05, 4)),
                "why": "v8 still underperforms most on response selection and human-facing answer quality.",
            },
            {
                "name": "grounded_tool_and_model_choice",
                "current": domain_accuracy,
                "target": max(0.72, round(domain_accuracy + 0.04, 4)),
                "why": "Sample outputs show routing failures on benchmark, materials, and grounded-fact prompts.",
            },
            {
                "name": "creative_communication",
                "current": intent_accuracy,
                "target": max(0.72, round(intent_accuracy + 0.03, 4)),
                "why": "v41 should explain, compare, and reassure more naturally instead of defaulting to template fragments.",
            },
            {
                "name": "vision_and_specialist_fusion",
                "current": vision_accuracy,
                "target": max(0.73, round(vision_accuracy + 0.04, 4)),
                "why": "v8 improved multimodal accuracy, but v41 still needs cleaner cross-domain fusion between text, vision, materials, protein, and 3D prompts.",
            },
        ],
        "teacher_strategy": {
            "primary_teachers": [
                "omni_collective_v8",
                "v40_benchmax",
                "qwen_v28",
                "qwen_v30",
                "omni_collective_v7",
            ],
            "secondary_teachers": [
                key
                for key in teacher_keys
                if key not in {"v40_benchmax", "omni_collective_v7", "qwen_v28", "qwen_v30"}
            ][:8],
            "gemma4_slice": {
                "status": "include_small_grounded_slice_if_local_weights_exist",
                "target_rows": 24,
                "note": "Keep Gemma 4 bounded and quality-gated rather than letting it dominate the teacher mix.",
            },
            "disagreement_mining": {
                "enabled": True,
                "focus": [
                    "benchmark_reasoning",
                    "coding_repairs",
                    "materials_grounding",
                    "human_communication",
                ],
                "note": "Save prompts where v8, v40, and qwen disagree, then build compare-and-justify rows instead of taking only the winner.",
            },
        },
        "current_frontier": frontier_scores,
        "architecture": {
            "parameter_count_target": target_parameter_count,
            "current_parameter_count": parameter_count,
            "growth_vs_v8": target_parameter_count - parameter_count,
            "text_hidden": {"from": current_text_hidden, "to": 288},
            "fusion_hidden": {"from": current_fusion_hidden, "to": 1152},
            "expert_count": {"from": current_expert_count, "to": 12},
            "expert_hidden": {"from": current_expert_hidden, "to": 1920},
            "expert_top_k": 2,
            "deliberation_passes": {"from": 10, "to": 12},
            "min_deliberation_passes": 6,
            "new_heads": [
                {
                    "name": "latent_plan_slots",
                    "purpose": "Distill short hidden plan states from explicit draft-plan traces, then keep the plans internal at inference time.",
                },
                {
                    "name": "multi_token_prediction_aux",
                    "purpose": "Predict short future spans to increase information density and long-range coherence.",
                },
                {
                    "name": "communication_polish_head",
                    "purpose": "Separate answer correctness from answer presentation so the model learns concise, human-friendly rewrites.",
                },
                {
                    "name": "uncertainty_anchor_head",
                    "purpose": "Teach the model to declare uncertainty and missing evidence instead of inventing specifics.",
                },
            ],
            "novel_v41_ideas": [
                "Route-first then answer: supervise a hidden route/tool choice before the final answer for better Auto and collective behavior.",
                "Disagreement harvest replay: convert teacher disagreement into compare-and-justify rows instead of only winner-take-all distillation.",
                "Two-pass communication finish: solve first, then learn a short rewrite pass that preserves facts while improving tone and clarity.",
                "Creativity rescue pairs: train on correct-but-flat vs vivid-and-still-grounded answer pairs so creativity does not come from hallucination.",
                "Reflection memory shards: keep tiny verbal post-mortems for failed code, reasoning, and route decisions, then train on improved retries.",
                "Reasoning-budget labels: supervise when the model should think short, medium, or long instead of only increasing deliberation globally.",
            ],
        },
        "data_program": [
            {
                "name": "latent_plan_bootstrap",
                "target_rows": 1800,
                "source_style": "plan-first then answer rows with hidden plan targets",
                "goal": "Teach v41 to infer structure and route choice before generating the final answer.",
            },
            {
                "name": "high_density_reasoning_core",
                "target_rows": 3200,
                "source_style": "LIMO-style compact hard reasoning set",
                "goal": "Raise problem-solving quality without flooding the mix with repetitive low-signal reasoning rows.",
            },
            {
                "name": "code_critique_repair",
                "target_rows": 2800,
                "source_style": "traceback, failing-test, patch, and critique->repair examples",
                "goal": "Improve coding, debugging, and repository-grounded problem solving.",
            },
            {
                "name": "reflection_memory_rows",
                "target_rows": 1500,
                "source_style": "bad attempt, short reflection, repaired answer",
                "goal": "Turn failures into compact reusable repair knowledge instead of only supervising the winning answer.",
            },
            {
                "name": "human_communication_polish",
                "target_rows": 6200,
                "source_style": "rewrite, summarize, compare, explain, reassure, and uncertainty-calibrated replies",
                "goal": "Make v41 more useful and easier to talk to without losing directness.",
            },
            {
                "name": "teacher_disagreement_hardcases",
                "target_rows": 1800,
                "source_style": "v8/v40/qwen disagreement mining",
                "goal": "Expose the model to exactly the prompts where current strong local teachers diverge.",
            },
            {
                "name": "grounded_uncertainty_rows",
                "target_rows": 900,
                "source_style": "current-fact uncertainty, partial evidence, and missing-tool disclaimers",
                "goal": "Reduce hallucinated specifics and improve transparent communication.",
            },
            {
                "name": "creative_constraint_pairs",
                "target_rows": 2200,
                "source_style": "creative answer pairs scored for vividness, usefulness, and factual discipline",
                "goal": "Boost creativity without dropping answer discipline.",
            },
            {
                "name": "route_then_answer_supervision",
                "target_rows": 1200,
                "source_style": "model selection, tool choice, specialist handoff, then final answer",
                "goal": "Make Auto and collective mode smarter because the base model reasons about routing explicitly.",
            },
            {
                "name": "reasoning_budget_curriculum",
                "target_rows": 1400,
                "source_style": "same task solved under short, medium, and deep reasoning budgets",
                "goal": "Teach the model when extra deliberation is worthwhile instead of overthinking every prompt.",
            },
            {
                "name": "specialist_dense_topups",
                "target_rows": 1600,
                "source_style": "materials, protein, 3D, native image, and science vision deltas",
                "goal": "Keep the specialist fusion edge while shifting more capacity toward coding and human communication.",
            },
        ],
        "training_program": [
            {
                "stage": "stage0_latent_plan_bootstrap",
                "recipe": "Distill short explicit plans, route choices, and critique stubs into latent slots before normal SFT.",
                "inspiration": "DeepSeek-V3 + quiet planning style",
            },
            {
                "stage": "stage1_teacher_sft",
                "recipe": "Teacher-heavy mixed SFT with harder coding/problem-solving weighting and a smaller, denser reasoning core.",
                "inspiration": "LIMO-style compact reasoning curriculum",
            },
            {
                "stage": "stage2_self_play_preference",
                "recipe": "Generate self-play pairs from v8/v40/qwen-style answers, then train chosen-vs-rejected preference objectives.",
                "inspiration": "SPIN",
            },
            {
                "stage": "stage3_code_refine",
                "recipe": "Run critique->repair loops on code, tests, and debugging prompts and keep only verifiably improved completions.",
                "inspiration": "RefineCoder, Reflexion, and code self-reflection",
            },
            {
                "stage": "stage4_communication_polish",
                "recipe": "Short final polish phase on human communication, grounded uncertainty, and compare/explain prompts using self-refine and AI feedback labels.",
                "inspiration": "Self-Refine + RLAIF",
            },
            {
                "stage": "stage5_self_correction_finish",
                "recipe": "Run deterministic self-correction on weak drafts with fair prompts and keep only improved final answers.",
                "inspiration": "Intrinsic Self-Correction",
            },
        ],
        "runtime_recommendations": {
            "device": "prefer_cuda_or_dml_never_cpu_if_avoidable",
            "amp": "auto",
            "amp_dtype": "bfloat16",
            "compile_model": True,
            "compile_mode": "reduce-overhead",
            "grad_accum_steps": 4,
            "ema_decay": 0.9993,
            "warmup_ratio": 0.06,
            "min_lr_scale": 0.03,
            "checkpoint_every_optimizer_steps": 100,
            "keep_last_stage_checkpoints": 3,
            "save_stage_complete_weights": True,
        },
        "success_gates": {
            "response_accuracy_min": max(0.145, round(response_accuracy + 0.05, 4)),
            "stage2_score_min": max(0.455, round(_safe_float(summary.get("stage2", {}).get("best_score")) + 0.03, 4)),
            "common_benchmark_exact_min": 0.25,
            "must_beat": ["omni_collective_v8_preview", "omni_collective_v7"],
            "stretch_goal": "challenge_v40_benchmax_on_benchmark_reasoning_without losing multimodal utility",
        },
        "promotion_protocol": {
            "required_preview_benchmarks": 3,
            "preview_selection_rule": "Promote only the best preview checkpoint by weighted score across benchmark exact, coding repair, communication, and grounded uncertainty packs.",
            "must_not_regress": [
                "stage2 vision_accuracy vs v8 by more than 0.02",
                "stage2 domain_accuracy vs v8 by more than 0.02",
            ],
            "must_beat_frontier_if_available": frontier_scores.get("best_fusion"),
            "human_style_gate": {
                "required": True,
                "description": "v41 must beat v8 on the communication-polish eval pack before promotion, even if the raw benchmark score is close.",
            },
        },
        "recent_method_references": RECENT_METHOD_REFERENCES,
        "implementation_checklist": [
            "Freeze the latest v8 summary, benchmark outputs, and hard-failure prompts as v41 design inputs.",
            "Create new dataset builders for disagreement mining, route-then-answer rows, communication polish pairs, latent-plan rows, and code-repair rows.",
            "Add latent plan slots, multi-token prediction auxiliary loss, and uncertainty-anchor head to the v41 model.",
            "Reuse the improved v8 checkpoint/resume path from the start so long v41 runs survive reboots and disk pressure.",
            "Benchmark preview checkpoints during stage2 rather than waiting until the end of the full run.",
            "Run the new communication, coding-repair, and grounded-uncertainty eval packs before any promotion decision.",
        ],
    }


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a machine-readable blueprint for omni_collective_v41.")
    parser.add_argument("--summary", default="", help="Optional path to a v8 frontier summary JSON. Defaults to the latest local v8 summary.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Where to write the v41 blueprint JSON.")
    parser.add_argument("--stdout", action="store_true", help="Print the blueprint to stdout as well.")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve() if str(args.summary).strip() else latest_v8_summary_path(REPO_ROOT)
    output_path = Path(args.output).resolve()

    summary = load_summary(summary_path)
    blueprint = build_v41_blueprint(summary, summary_path=summary_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.stdout:
        print(json.dumps(blueprint, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
