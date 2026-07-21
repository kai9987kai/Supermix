"""Materialize and verify chat-demo metadata for the v51 ultra checkpoint.

The v51 checkpoint produced by ``benchmark_cognitive_leap_ultra_v51.py`` is a
synthetic reasoning benchmark model, not a general chat fine-tune. This script
creates a small retrieval metadata file so the existing chat UI can load the
checkpoint and expose runtime-compute controls consistently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
PROJECT_ROOT = SOURCE_DIR.parent

from chat_pipeline import featurize_text
from chat_web_app import Engine


DEFAULT_ARTIFACT_DIR = SOURCE_DIR.parent / "output" / "benchmark_v51_cognitive_leap_ultra"
DEFAULT_WEIGHTS = DEFAULT_ARTIFACT_DIR / "cognitive_leap_ultra_v51_trained.pth"
DEFAULT_METRICS = DEFAULT_ARTIFACT_DIR / "benchmark_results.json"
DEFAULT_META = DEFAULT_ARTIFACT_DIR / "chat_demo_meta.json"


PROMPT_RESPONSE_ROWS: Tuple[Tuple[str, str], ...] = (
    (
        "v51 ultra demo checkpoint status and limitations",
        "This is the v51 Cognitive Leap Ultra demo checkpoint. It was trained on a small synthetic chained-arithmetic task, so treat this chat surface as a runtime demo rather than a polished assistant.",
    ),
    (
        "training results metrics checkpoint reload",
        "Training finished successfully on CPU and produced a reloadable checkpoint plus metrics under output/benchmark_v51_cognitive_leap_ultra.",
    ),
    (
        "runtime compute reasoning cycles adaptive exit entropy",
        "Runtime compute controls are available: set reasoning cycles, adaptive compute, exit tolerance, and entropy threshold, then compare cycles used and latency.",
    ),
    (
        "synthetic chained modular arithmetic task",
        "The training task applies chained modular arithmetic operations. It is useful for testing recursive latent reasoning, not broad natural-language capability.",
    ),
    (
        "how to test the model in chat interface",
        "Use this interface to verify loading, inference, candidate retrieval, and compute diagnostics. For serious chat quality, run a real chat fine-tune with conversation data.",
    ),
    (
        "explain confidence uncertainty for this demo",
        "Uncertainty is high for broad chat prompts because this checkpoint was not trained on conversation data. Prefer asking about the run, metrics, or compute controls.",
    ),
    (
        "debug model response and inspect candidates",
        "For debugging, enable top candidates and compare timing, cycles used, and selected responses. If output is poor, the limitation is training data, not the web interface.",
    ),
    (
        "next training step for real chat fine tune",
        "The next useful training step is a bounded chat fine-tune from a compatible base checkpoint with real conversation manifests and recursive-head auxiliary losses enabled.",
    ),
    (
        "short concise answer mode",
        "Short answer: the v51 model is loaded for demo inference, but it is not a full chat model yet.",
    ),
    (
        "technical analyst answer mode",
        "Technical readout: v51 ultra uses a recurrent MoE latent core, cross-latent attention, ACT-style halting, and runtime-compute diagnostics.",
    ),
)


def _load_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _portable_project_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_demo_metadata(weights: Path, metrics_path: Path) -> Dict[str, Any]:
    metrics = _load_metrics(metrics_path)
    stability_metrics_path = metrics_path.with_name("prediction_stability_results.json")
    stability_metrics = _load_metrics(stability_metrics_path)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for label, (prompt, response) in enumerate(PROMPT_RESPONSE_ROWS):
        buckets[str(label)] = [
            {
                "text": response,
                "vec": featurize_text(response).tolist(),
                "ctx_vec": featurize_text(prompt).tolist(),
                "count": 1,
            }
        ]

    return {
        "created_by": "materialize_v51_chat_demo.py",
        "purpose": "demo metadata for v51 cognitive_leap_ultra_expert chat_web_app",
        "model_size": "cognitive_leap_ultra_expert",
        "feature_mode": "context_mix_v4",
        "num_classes": 10,
        "training_task": "synthetic chained modular arithmetic benchmark",
        "fine_tuned_weights": str(weights.name),
        "checkpoint_path": _portable_project_path(weights),
        "benchmark_metrics": _portable_project_path(metrics_path) if metrics else None,
        "prediction_stability_metrics": (
            _portable_project_path(stability_metrics_path) if stability_metrics else None
        ),
        "benchmark_summary": {
            "train_seconds": metrics.get("train_seconds"),
            "eval_default": metrics.get("eval_default"),
            "test_time_scaling": metrics.get("test_time_scaling"),
            "adaptive_compute": metrics.get("adaptive_compute"),
            "prediction_stability": stability_metrics.get("prediction_stability"),
            "prediction_stability_comparison": stability_metrics.get("comparison"),
        },
        "runtime_defaults": {
            "reasoning_cycles": 3,
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.001,
            "adaptive_exit_entropy": 0.2,
            "prediction_stability_patience": 2,
            "prediction_stability_tol": 0.005,
        },
        "buckets": buckets,
        "label_priors": {str(i): 0.1 for i in range(10)},
    }


def materialize(weights: Path, metrics: Path, meta: Path) -> Path:
    payload = build_demo_metadata(weights=weights, metrics_path=metrics)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return meta


def check_load(weights: Path, meta: Path) -> Dict[str, Any]:
    engine = Engine(
        torch.device("cpu"),
        {"resolved": "cpu"},
        {
            "model_size": "auto",
            "pool_mode": "all",
            "reasoning_cycles": 3,
            "adaptive_compute": True,
            "adaptive_exit_entropy": 0.2,
            "prediction_stability_patience": 2,
            "prediction_stability_tol": 0.005,
        },
    )
    status = engine.load(str(weights), str(meta))
    chat = engine.chat(
        session_id="v51-check",
        user_text="how should I test this v51 demo?",
        show_top_responses=3,
        reasoning_cycles=3,
        adaptive_compute=True,
        adaptive_exit_entropy=0.2,
        prediction_stability_patience=2,
        prediction_stability_tol=0.005,
    )
    return {
        "loaded": status,
        "chat_response": chat.get("response"),
        "compute": chat.get("compute"),
        "top_candidates": chat.get("top_candidates"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS))
    ap.add_argument("--meta", default=str(DEFAULT_META))
    ap.add_argument("--check", action="store_true", help="Load the checkpoint and run a one-turn smoke chat.")
    args = ap.parse_args()

    weights = Path(args.weights)
    metrics = Path(args.metrics)
    meta = Path(args.meta)
    if not weights.exists():
        raise FileNotFoundError(f"Missing v51 checkpoint: {weights}")

    materialize(weights=weights, metrics=metrics, meta=meta)
    print(f"Wrote demo metadata: {meta}")
    if args.check:
        result = check_load(weights=weights, meta=meta)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
