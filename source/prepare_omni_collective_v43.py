from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from prepare_omni_collective_v42 import latest_common_benchmark_summary_path, load_summary
except ImportError:  # pragma: no cover
    from .prepare_omni_collective_v42 import latest_common_benchmark_summary_path, load_summary


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "omni_collective_v43_blueprint.json"
V41_MANIFEST_PATH = Path(__file__).resolve().parent / "v41_benchmax_manifest.json"


RECENT_METHOD_REFERENCES_V43: List[Dict[str, str]] = [
    {
        "name": "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
        "url": "https://arxiv.org/abs/2504.19874",
        "applied_as": "teacher_generation_kv_cache_compression",
        "why": "Treat TurboQuant as a runtime enabler for longer teacher traces and cheaper verifier passes, not as a weight-training substitute.",
    },
    {
        "name": "Small Language Models Need Strong Verifiers to Self-Correct Reasoning",
        "url": "https://arxiv.org/abs/2404.17140",
        "applied_as": "verifier_then_repair_rows",
        "why": "Add explicit critique and repair supervision instead of assuming the student will self-correct from final answers alone.",
    },
    {
        "name": "Self-Enhanced Reasoning Training",
        "url": "https://arxiv.org/abs/2502.12744",
        "applied_as": "latent_reasoning_bootstrap",
        "why": "Use self-generated reasoning paths selectively to activate latent reasoning without requiring giant chain-of-thought corpora everywhere.",
    },
    {
        "name": "Lost at the Beginning of Reasoning",
        "url": "https://arxiv.org/abs/2506.22058",
        "applied_as": "opening_step_quality_filtering",
        "why": "Prefer high-quality first steps and route choices so the model does not waste budget on weak starts.",
    },
    {
        "name": "Learning from Partial Chain-of-Thought via Truncated-Reasoning Self-Distillation",
        "url": "https://arxiv.org/abs/2603.13274",
        "applied_as": "partial_reasoning_distillation",
        "why": "Distill useful partial traces so v43 can stop earlier on easy problems without losing answer quality.",
    },
    {
        "name": "google/gemma-4-31B-it",
        "url": "https://huggingface.co/google/gemma-4-31B-it",
        "applied_as": "grounded_multimodal_teacher",
        "why": "Use Gemma 4 as a bounded teacher for grounded multimodal answers, image reasoning, and careful presentation.",
    },
    {
        "name": "Qwen/Qwen3.5-397B-A17B",
        "url": "https://huggingface.co/Qwen/Qwen3.5-397B-A17B",
        "applied_as": "long_context_tool_use_teacher",
        "why": "Use Qwen 3.5 for long-context, tool-use, and reasoning traces, especially where v42 is weak against v41_benchmax.",
    },
    {
        "name": "Qwen/Qwen3-Coder-Next",
        "url": "https://huggingface.co/Qwen/Qwen3-Coder-Next",
        "applied_as": "agentic_coding_teacher",
        "why": "Use Coder-Next for code-edit, test-repair, and long-horizon debugging traces instead of only one-shot code answers.",
    },
    {
        "name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "url": "https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "applied_as": "omni_audio_video_teacher",
        "why": "Use the omni model for audio-video alignment, captioning, OCR-style prompts, and broader multimodal route supervision.",
    },
]


def latest_v42_summary_path(repo_root: Path = REPO_ROOT) -> Path:
    candidates = sorted(
        repo_root.glob("output/supermix_omni_collective_v42_frontier_*/omni_collective_v42_frontier_summary.json"),
        key=lambda item: (item.parent.name, item.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No omni_collective_v42 frontier summary found under output/.")
    return candidates[0].resolve()


def latest_v42_zip_path(repo_root: Path = REPO_ROOT) -> Path:
    candidates = sorted(
        repo_root.glob("output/supermix_omni_collective_v42_frontier_*.zip"),
        key=lambda item: (item.name, item.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No omni_collective_v42 frontier zip found under output/.")
    return candidates[0].resolve()


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


def _round_to_million(value: int) -> int:
    return int(round(float(value) / 1_000_000.0) * 1_000_000)


def _recommended_parameter_target(current_parameter_count: int) -> int:
    growth_floor = current_parameter_count + 10_000_000
    growth_soft = current_parameter_count + max(16_000_000, int(current_parameter_count * 0.14))
    growth_cap = current_parameter_count + 24_000_000
    return _round_to_million(min(max(growth_soft, growth_floor), growth_cap))


def _extract_benchmark_context(repo_root: Path) -> Dict[str, Any]:
    benchmark_path = latest_common_benchmark_summary_path(repo_root)
    if benchmark_path is None:
        return {
            "benchmark_summary_path": None,
            "v41_benchmax": None,
            "omni_collective_v42": None,
            "best_overall": None,
        }
    payload = load_summary(benchmark_path)
    rows = list(payload.get("summary_rows") or [])
    indexed = {str(row.get("model") or ""): row for row in rows}
    best_overall = max(rows, key=lambda item: float(item.get("overall_exact") or 0.0)) if rows else None

    def _shape(model_key: str) -> Optional[Dict[str, Any]]:
        row = indexed.get(model_key)
        if row is None:
            return None
        return {
            "model": model_key,
            "family": str(row.get("family") or ""),
            "overall_exact": _safe_float(row.get("overall_exact") or 0.0),
        }

    return {
        "benchmark_summary_path": str(benchmark_path.resolve()),
        "v41_benchmax": _shape("v41_benchmax"),
        "omni_collective_v42": _shape("omni_collective_v42"),
        "best_overall": (
            {
                "model": str(best_overall.get("model") or ""),
                "family": str(best_overall.get("family") or ""),
                "overall_exact": _safe_float(best_overall.get("overall_exact") or 0.0),
            }
            if best_overall is not None
            else None
        ),
    }


def _load_v41_manifest(path: Path = V41_MANIFEST_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_v43_blueprint(summary: Dict[str, Any], summary_path: Optional[Path] = None) -> Dict[str, Any]:
    parameter_count = int(summary.get("parameter_count") or 0)
    stage1_rows = int(summary.get("dataset_summary", {}).get("stage1_rows") or 0)
    stage2_rows = int(summary.get("dataset_summary", {}).get("stage2_rows") or 0)
    stage1_metrics = _stage_metrics(summary, "stage1")
    stage2_metrics = _stage_metrics(summary, "stage2")
    response_accuracy = stage2_metrics.get("response_accuracy", 0.0)
    intent_accuracy = stage2_metrics.get("intent_accuracy", 0.0)
    vision_accuracy = stage2_metrics.get("vision_accuracy", 0.0)
    domain_accuracy = stage2_metrics.get("domain_accuracy", 0.0)
    current_benchmark = _extract_benchmark_context(REPO_ROOT)
    v41_manifest = _load_v41_manifest()
    v41_exact = _safe_float((current_benchmark.get("v41_benchmax") or {}).get("overall_exact"))
    v42_exact = _safe_float((current_benchmark.get("omni_collective_v42") or {}).get("overall_exact"))
    benchmark_gap = round(max(0.0, v41_exact - v42_exact), 4)
    target_parameter_count = _recommended_parameter_target(parameter_count)

    return {
        "family": "omni_collective_v43",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": "2026-04-10",
        "derived_from": {
            "artifact": str(summary.get("artifact") or ""),
            "summary_path": str(summary_path.resolve()) if summary_path else "",
            "parameter_count": parameter_count,
            "stage1_rows": stage1_rows,
            "stage2_rows": stage2_rows,
            "stage1_best_score": _safe_float(summary.get("stage1", {}).get("best_score")),
            "stage2_best_score": _safe_float(summary.get("stage2", {}).get("best_score")),
        },
        "baseline_findings": {
            "common_benchmark_exact": {
                "v41_benchmax": v41_exact,
                "omni_collective_v42": v42_exact,
                "gap_to_close": benchmark_gap,
            },
            "v42_strengths_to_keep": [
                "route-then-answer behavior",
                "communication polish",
                "coding repair supervision",
                "grounded uncertainty responses",
            ],
            "v41_strengths_to_import": [
                "failure-cluster curriculum",
                "benchmark hard-example replay",
                "budget-aware verifier prompts",
                "consensus-teacher sampling",
            ],
        },
        "observed_gaps": [
            {
                "name": "benchmark_reasoning",
                "current": v42_exact,
                "target": max(0.25, round(v42_exact + 0.08, 4)),
                "why": "v43 needs to recover the benchmark-specialist edge from v41_benchmax while keeping v42's broader multimodal utility.",
            },
            {
                "name": "response_quality",
                "current": response_accuracy,
                "target": max(0.16, round(response_accuracy + 0.05, 4)),
                "why": "v42 improved presentation but still underperforms on response selection and final answer quality.",
            },
            {
                "name": "agentic_coding_and_verification",
                "current": domain_accuracy,
                "target": max(0.73, round(domain_accuracy + 0.04, 4)),
                "why": "v43 should route to better verifier-first coding behavior and retain stronger repair traces from Qwen3-Coder-Next style supervision.",
            },
            {
                "name": "multimodal_grounding",
                "current": vision_accuracy,
                "target": max(0.48, round(vision_accuracy + 0.08, 4)),
                "why": "Gemma 4 and Qwen3-Omni expose better grounded multimodal behaviors than the current v42 prep mix.",
            },
            {
                "name": "diversity_and_global_coverage",
                "current": intent_accuracy,
                "target": max(0.64, round(intent_accuracy + 0.05, 4)),
                "why": "Qwen3.5 explicitly broadens long-context, multilingual, and tool-use coverage that the current v42 dataset underrepresents.",
            },
        ],
        "teacher_strategy": {
            "primary_teachers": [
                "omni_collective_v42",
                "v41_benchmax",
                "google/gemma-4-31B-it",
                "Qwen/Qwen3.5-397B-A17B",
                "Qwen/Qwen3-Coder-Next",
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            ],
            "bounded_teachers": [
                {
                    "model": "google/gemma-4-E4B-it",
                    "role": "small grounded multimodal comparisons",
                    "row_budget": 160,
                },
                {
                    "model": "google/gemma-4-26B-A4B-it",
                    "role": "mid-sized multimodal and instruction style comparisons",
                    "row_budget": 220,
                },
            ],
            "teacher_roles": [
                {
                    "model": "v41_benchmax",
                    "role": "benchmark hard-case teacher",
                    "why": "Use it for benchmark replay, verifier rows, and failure-cluster curriculum.",
                },
                {
                    "model": "google/gemma-4-31B-it",
                    "role": "grounded multimodal teacher",
                    "why": "Use it for image-grounded comparisons, safer multimodal phrasing, and controlled explanation style.",
                },
                {
                    "model": "Qwen/Qwen3.5-397B-A17B",
                    "role": "long-context reasoning and tool-use teacher",
                    "why": "Use it for long-context summaries, tool traces, reasoning budgets, and route-then-answer rows.",
                },
                {
                    "model": "Qwen/Qwen3-Coder-Next",
                    "role": "agentic coding teacher",
                    "why": "Use it for code-edit planning, test repair, and repository-grounded debugging traces.",
                },
                {
                    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    "role": "audio-video multimodal teacher",
                    "why": "Use it for caption, OCR-style, video, and cross-modal route supervision.",
                },
            ],
            "teacher_mixture_rules": [
                "Never let any single hosted teacher dominate the distilled mix.",
                "Keep benchmark replay anchored to v41_benchmax plus local verifier rows.",
                "Keep Gemma 4 bounded to grounded multimodal slices instead of generic chat domination.",
                "Use Qwen3-Coder-Next only on coding-heavy prompts so the rest of the mix stays broad.",
            ],
            "turboquant_runtime_note": {
                "enabled": True,
                "purpose": "Use TurboQuant-style KV compression or equivalent when generating long teacher traces so the data program can afford deeper verifier passes.",
                "training_implication": "Supervise budget choice and evidence compression, not the quantization algorithm itself.",
            },
        },
        "architecture": {
            "parameter_count_target": target_parameter_count,
            "current_parameter_count": parameter_count,
            "growth_vs_v42": target_parameter_count - parameter_count,
            "text_hidden": {"from": 288, "to": 304},
            "fusion_hidden": {"from": 1152, "to": 1216},
            "expert_count": {"from": 12, "to": 14},
            "expert_hidden": {"from": 1920, "to": 2048},
            "expert_top_k": 2,
            "deliberation_passes": {"from": 12, "to": 14},
            "min_deliberation_passes": 7,
            "new_heads": [
                {
                    "name": "budget_router_head",
                    "purpose": "Predict whether a prompt needs short, medium, or deep reasoning before the model spends extra compute.",
                },
                {
                    "name": "teacher_consensus_head",
                    "purpose": "Separate answer confidence from agreement confidence so the model can learn when to distrust a single teacher.",
                },
                {
                    "name": "cache_budget_head",
                    "purpose": "Teach the model how much evidence to keep active and when compressed context is enough.",
                },
                {
                    "name": "verifier_gate_head",
                    "purpose": "Learn when a second-pass verifier or repair step is worth the latency on coding and reasoning prompts.",
                },
            ],
            "novel_v43_ideas": [
                "Benchmark bridge curriculum: replay v41-style hard clusters directly inside the broader v42 omni scaffold.",
                "Teacher role specialization: use Gemma 4, Qwen 3.5, Qwen3-Coder-Next, and Qwen3-Omni for sharply different data slices instead of generic all-purpose distillation.",
                "TurboQuant-inspired context economy: train explicit evidence-budget labels so long-context behavior improves without inflating every answer.",
                "Partial reasoning distillation: keep good early and mid reasoning segments, not only full-length traces.",
                "Verifier-first coding: train code repair rows where a verifier note or failing test arrives before the final patch.",
                "Diversity expansion by scenario, modality, and user goal rather than only adding more generic chat volume.",
            ],
        },
        "data_program": [
            {
                "name": "v41_benchmark_bridge",
                "target_rows": 4200,
                "source_style": "v41 hard-example replay, failure clusters, and verifier-guided benchmark repair",
                "goal": "Close the measured v41-v42 benchmark gap without collapsing the omni model into a narrow benchmark specialist.",
            },
            {
                "name": "gemma4_grounded_multimodal",
                "target_rows": 2600,
                "source_style": "bounded Gemma 4 image-text and grounded explanation slices",
                "goal": "Lift multimodal grounding and safer, clearer explanation quality.",
            },
            {
                "name": "qwen35_long_context_and_tool_use",
                "target_rows": 3800,
                "source_style": "Qwen3.5 long-context, tool-use, and reasoning-budget traces",
                "goal": "Improve route choice, long-context stability, and tool-aware reasoning.",
            },
            {
                "name": "qwen3_coder_next_agentic_repairs",
                "target_rows": 3400,
                "source_style": "agentic coding, repository-grounded edits, failing tests, and patch repair traces",
                "goal": "Make coding behavior more deliberate and more verifiable than v42's mostly answer-level repair mix.",
            },
            {
                "name": "qwen3_omni_audio_video_alignment",
                "target_rows": 1800,
                "source_style": "audio, video, OCR-style, and cross-modal route supervision",
                "goal": "Improve multimodal specialist routing and richer omni prompt understanding.",
            },
            {
                "name": "verifier_guided_self_correction",
                "target_rows": 2600,
                "source_style": "draft, verifier note, repaired answer",
                "goal": "Teach v43 to repair weak drafts with targeted feedback rather than only rerolling responses.",
            },
            {
                "name": "partial_reasoning_distill",
                "target_rows": 2100,
                "source_style": "high-quality partial chains, truncated reasoning, and early-stop successful traces",
                "goal": "Keep strong reasoning while reducing wasteful overthinking on easy or medium tasks.",
            },
            {
                "name": "global_diversity_and_style",
                "target_rows": 4800,
                "source_style": "broader users, locales, tones, professions, and request styles",
                "goal": "Make the dataset larger and more diverse in a meaningful way instead of only duplicating similar prompts.",
            },
            {
                "name": "cache_budget_and_context_selection",
                "target_rows": 1600,
                "source_style": "compressed evidence selection, context budget labels, and when-to-stop traces",
                "goal": "Translate TurboQuant-style efficiency lessons into trainable budget and evidence behavior.",
            },
            {
                "name": "specialist_dense_topups_v43",
                "target_rows": 2400,
                "source_style": "materials, protein, 3D, science vision, and model-selection top-ups",
                "goal": "Preserve the specialist fusion edge while expanding coding, benchmark, and multimodal coverage.",
            },
        ],
        "training_program": [
            {
                "stage": "stage0_route_budget_bootstrap",
                "recipe": "Train route choice, verifier choice, and short-medium-deep budget prediction before heavier SFT.",
                "inspiration": "reasoning budget control plus route-first supervision",
            },
            {
                "stage": "stage1_teacher_sft",
                "recipe": "Mix v42 carryover rows with v41 hard examples and the new Gemma/Qwen role-specific teacher slices.",
                "inspiration": "teacher specialization instead of flat all-model distillation",
            },
            {
                "stage": "stage2_verifier_and_repair",
                "recipe": "Keep only repairs that beat the draft under verifier heuristics or benchmark-style exact checks.",
                "inspiration": "strong verifier self-correction",
            },
            {
                "stage": "stage3_partial_reasoning_distill",
                "recipe": "Distill good partial traces and explicit early-stop examples so v43 learns when enough reasoning is enough.",
                "inspiration": "TRSD and early-step quality filtering",
            },
            {
                "stage": "stage4_agentic_coding_finish",
                "recipe": "Run code-edit, test-repair, and repo-grounded critique->patch loops with Coder-Next style traces.",
                "inspiration": "agentic coding teachers and repair-first data",
            },
            {
                "stage": "stage5_multimodal_omni_finish",
                "recipe": "Short final phase on Gemma 4 and Qwen3-Omni slices for grounded image/audio/video route supervision.",
                "inspiration": "teacher-role specialization",
            },
        ],
        "runtime_recommendations": {
            "device": "prefer_cuda_or_dml_never_cpu_if_avoidable",
            "amp": "auto",
            "amp_dtype": "bfloat16",
            "compile_model": True,
            "compile_mode": "reduce-overhead",
            "grad_accum_steps": 6,
            "ema_decay": 0.9994,
            "warmup_ratio": 0.07,
            "min_lr_scale": 0.025,
            "checkpoint_every_optimizer_steps": 100,
            "keep_last_stage_checkpoints": 4,
            "save_stage_complete_weights": True,
            "teacher_generation_acceleration": "TurboQuant-style KV compression or equivalent cache reduction for long teacher passes",
        },
        "success_gates": {
            "response_accuracy_min": max(0.16, round(response_accuracy + 0.05, 4)),
            "stage2_score_min": max(0.39, round(_safe_float(summary.get("stage2", {}).get("best_score")) + 0.03, 4)),
            "common_benchmark_exact_min": max(0.245, round(v41_exact + 0.002, 4)),
            "must_beat": ["omni_collective_v42"],
            "stretch_goal": "beat_v41_benchmax_on_common_benchmark_exact_while_preserving_omni_utility",
        },
        "promotion_protocol": {
            "required_preview_benchmarks": 4,
            "preview_selection_rule": "Promote only checkpoints that beat v42 overall and stay within multimodal regression limits while closing the v41 benchmark gap.",
            "must_not_regress": [
                "stage2 vision_accuracy vs v42 by more than 0.02",
                "stage2 domain_accuracy vs v42 by more than 0.02",
                "coding repair pack vs v42 by more than 0.01",
            ],
            "hard_gate": "No promotion if common benchmark exact remains below v41_benchmax.",
        },
        "recent_method_references": RECENT_METHOD_REFERENCES_V43,
        "v41_manifest_snapshot": {
            "objective": str(v41_manifest.get("objective") or ""),
            "innovation_flags": dict(v41_manifest.get("innovation_flags") or {}),
            "novel_features": list(v41_manifest.get("novel_features") or []),
        },
        "current_benchmark_context": current_benchmark,
        "implementation_checklist": [
            "Freeze the latest v42 frontier summary and v41 benchmark summary as the v43 design inputs.",
            "Add new v43 dataset builders for benchmark bridge, teacher-role routing, verifier repair, TurboQuant-style budget rows, and diversity expansion.",
            "Keep the v42 data pipeline as the stable base, then add v43 row groups on top instead of rewriting the whole training loop.",
            "Use Gemma 4 and Qwen 3.5 family models only in bounded, role-specific slices.",
            "Evaluate preview checkpoints on the common benchmark plus coding-repair and multimodal route packs before promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a machine-readable blueprint for omni_collective_v43.")
    parser.add_argument("--summary", default="", help="Optional path to a v42 frontier summary JSON. Defaults to the latest local v42 summary.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Where to write the v43 blueprint JSON.")
    parser.add_argument("--stdout", action="store_true", help="Print the blueprint to stdout as well.")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve() if str(args.summary).strip() else latest_v42_summary_path(REPO_ROOT)
    output_path = Path(args.output).resolve()

    summary = load_summary(summary_path)
    blueprint = build_v43_blueprint(summary, summary_path=summary_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.stdout:
        print(json.dumps(blueprint, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
