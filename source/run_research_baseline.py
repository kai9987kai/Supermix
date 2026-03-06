import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

import qwen_supermix_pipeline as qp


def _load_eval_pairs_from_jsonl(path: Path) -> List[qp.ChatPair]:
    pairs: List[qp.ChatPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            user = str(row.get("user", "")).strip()
            assistant = str(row.get("assistant", "")).strip()
            if not user or not assistant:
                continue
            pairs.append(
                qp.ChatPair(
                    user=user,
                    assistant=assistant,
                    source=str(row.get("source", "eval")),
                )
            )
    return pairs


def _numeric_deltas(base: Dict[str, float], tuned: Dict[str, float]) -> Dict[str, float]:
    keys = set(base.keys()) & set(tuned.keys())
    out: Dict[str, float] = {}
    for key in sorted(keys):
        b = base.get(key)
        t = tuned.get(key)
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            out[key] = float(t) - float(b)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run reproducible Week-1 baseline benchmarks.")
    ap.add_argument("--base_model", required=True, help="HF model id or local model path")
    ap.add_argument("--device", default="cpu", choices=["cpu"])
    ap.add_argument("--adapter_dir", default="", help="Optional LoRA adapter dir for tuned comparison")
    ap.add_argument("--eval_jsonl", default="", help="Optional fixed eval set JSONL")
    ap.add_argument(
        "--data",
        nargs="*",
        default=[],
        help="Optional training/eval data JSONL(s). Used only when --eval_jsonl is not provided.",
    )
    ap.add_argument("--max_records", type=int, default=480)
    ap.add_argument("--eval_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_eval_samples", type=int, default=0, help="0 keeps all eval rows")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--output_root", default="artifacts/research_baselines")
    ap.add_argument("--run_name", default="", help="Optional run folder name")
    ap.add_argument("--benchmark_type", default="week1_baseline")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / args.output_root).resolve()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name.strip() or f"{args.benchmark_type}_{run_stamp}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_pairs: Sequence[qp.ChatPair]
    eval_source: Optional[Path] = None
    if args.eval_jsonl.strip():
        eval_source = Path(args.eval_jsonl).expanduser().resolve()
        if not eval_source.exists():
            raise FileNotFoundError(f"Eval JSONL not found: {eval_source}")
        eval_pairs = _load_eval_pairs_from_jsonl(eval_source)
    elif args.data:
        data_paths = [str(Path(p).expanduser()) for p in args.data]
        all_pairs = qp.load_jsonl_pairs(
            paths=data_paths,
            max_records=max(2, int(args.max_records)),
        )
        _, eval_pairs = qp.split_train_eval(
            pairs=all_pairs,
            eval_size=max(1, int(args.eval_size)),
            seed=int(args.seed),
        )
    else:
        raise ValueError("Provide either --eval_jsonl or --data.")

    if int(args.max_eval_samples) > 0 and len(eval_pairs) > int(args.max_eval_samples):
        rng = random.Random(int(args.seed) + 101)
        eval_pairs = rng.sample(list(eval_pairs), int(args.max_eval_samples))

    eval_out = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(eval_out, list(eval_pairs))
    print(f"[eval] samples={len(eval_pairs)}")
    print("[benchmark] base...")
    base_metrics = qp.evaluate_model(
        base_model=args.base_model,
        eval_pairs=eval_pairs,
        device=device,
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        adapter_dir=None,
    )

    tuned_metrics: Dict[str, float] = {}
    adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir.strip() else None
    if adapter_dir is not None:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        print("[benchmark] tuned...")
        tuned_metrics = qp.evaluate_model(
            base_model=args.base_model,
            eval_pairs=eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=adapter_dir,
        )

    results = {
        "config": {
            "benchmark_type": args.benchmark_type,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model": str(args.base_model),
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
            "eval_source": str(eval_source) if eval_source is not None else "split_from_data",
            "eval_samples": int(len(eval_pairs)),
            "seed": int(args.seed),
            "max_length": int(args.max_length),
            "max_new_tokens": int(args.max_new_tokens),
            "max_records": int(args.max_records),
            "eval_size": int(args.eval_size),
            "max_eval_samples": int(args.max_eval_samples),
        },
        "base": base_metrics,
    }
    if tuned_metrics:
        results["tuned"] = tuned_metrics
        results["delta_tuned_minus_base"] = _numeric_deltas(base_metrics, tuned_metrics)

    out_json = run_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[done] {out_json}")
    if tuned_metrics:
        out_png = run_dir / "benchmark_comparison.png"
        qp.plot_benchmark({"base": base_metrics, "tuned": tuned_metrics}, out_png)
        print(f"[done] {out_png}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
