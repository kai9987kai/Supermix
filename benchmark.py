import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

source_dir = Path(__file__).parent
if str(source_dir) not in sys.path:
    sys.path.append(str(source_dir))

from chat_pipeline import assign_labels, build_training_tensors, load_conversation_examples, resolve_feature_mode
from model_variants import build_model, detect_model_size_from_state_dict, load_weights_for_model


def load_metadata(meta_path: str) -> Dict[str, object]:
    path = Path(meta_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_model_size(meta: Dict[str, object], state_dict: Dict[str, object]) -> str:
    model_size = str(meta.get("model_size", "")).strip()
    if model_size:
        return model_size
    return detect_model_size_from_state_dict(state_dict)


def resolve_eval_feature_mode(meta: Dict[str, object], override: Optional[str]) -> str:
    if override:
        return resolve_feature_mode(str(override), smarter_auto=True)
    return resolve_feature_mode(str(meta.get("feature_mode", "context_v2")), smarter_auto=True)


def build_eval_cache(
    examples,
    labels: torch.Tensor,
    feature_modes: Sequence[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for feature_mode in sorted(set(feature_modes)):
        x, y = build_training_tensors(examples, labels, feature_mode=feature_mode)
        cache[feature_mode] = (x, y)
    return cache


def evaluate_model(
    *,
    name: str,
    weights_path: str,
    meta_path: str,
    feature_mode: str,
    eval_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> float:
    x, y = eval_cache[feature_mode]
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    print(f"Evaluating {name}: {weights_path}")
    print(f"  Feature mode: {feature_mode}")
    meta = load_metadata(meta_path)
    sd = torch.load(weights_path, map_location=device)
    model_size = resolve_model_size(meta, sd)
    print(f"  Detected size: {model_size}")

    model = build_model(model_size=model_size).to(device).eval()
    missing, unexpected = load_weights_for_model(model, sd, model_size=model_size)
    missing = [key for key in missing if key and not key.startswith("layers.10.")]
    unexpected = [key for key in unexpected if key]
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch for {name}. Missing={missing}, Unexpected={unexpected}")

    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if xb.dtype != torch.float32:
                xb = xb.float()
            logits = model(xb).squeeze(1)
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

    accuracy = correct / total if total > 0 else 0.0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def run_compare(
    weights_a: str,
    meta_a: str,
    weights_b: Optional[str] = None,
    meta_b: Optional[str] = None,
    data_path: Optional[str] = None,
    device: str = "cpu",
    feature_mode_a: Optional[str] = None,
    feature_mode_b: Optional[str] = None,
) -> Dict[str, float]:
    if not data_path:
        raise ValueError("data_path is required")

    print(f"Loading data from {data_path}...")
    examples = load_conversation_examples(data_path)
    print(f"Loaded {len(examples)} examples.")

    labels, _ = assign_labels(examples, seed=42)
    meta_a_dict = load_metadata(meta_a)
    meta_b_dict = load_metadata(meta_b) if weights_b and meta_b else {}

    resolved_mode_a = resolve_eval_feature_mode(meta_a_dict, feature_mode_a)
    resolved_mode_b = resolve_eval_feature_mode(meta_b_dict, feature_mode_b) if weights_b and meta_b else None
    eval_cache = build_eval_cache(
        examples,
        labels,
        [mode for mode in (resolved_mode_a, resolved_mode_b) if mode],
    )

    results = {
        "Model A": evaluate_model(
            name="Model A",
            weights_path=weights_a,
            meta_path=meta_a,
            feature_mode=resolved_mode_a,
            eval_cache=eval_cache,
            device=device,
        )
    }
    if weights_b and meta_b and resolved_mode_b:
        results["Model B"] = evaluate_model(
            name="Model B",
            weights_path=weights_b,
            meta_path=meta_b,
            feature_mode=resolved_mode_b,
            eval_cache=eval_cache,
            device=device,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_a", required=True)
    parser.add_argument("--meta_a", required=True)
    parser.add_argument("--weights_b", default=None)
    parser.add_argument("--meta_b", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature_mode_a", default=None)
    parser.add_argument("--feature_mode_b", default=None)
    args = parser.parse_args()

    run_compare(
        weights_a=args.weights_a,
        meta_a=args.meta_a,
        weights_b=args.weights_b,
        meta_b=args.meta_b,
        data_path=args.data,
        device=args.device,
        feature_mode_a=args.feature_mode_a,
        feature_mode_b=args.feature_mode_b,
    )
