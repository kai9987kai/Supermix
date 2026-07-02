import json
from datetime import datetime, timezone
from pathlib import Path

import qwen_supermix_pipeline as qp


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    base_model = (
        r"C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
        r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    )
    out_dir = (project_root / "artifacts" / "qwen_supermix_enhanced_v10_smarter").resolve()
    eval_path = out_dir / "eval_pairs.jsonl"
    adapter_dir = out_dir / "adapter"
    out_json = out_dir / "benchmark_quick_v10_20tokens.json"

    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval set: {eval_path}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    eval_pairs = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            eval_pairs.append(
                qp.ChatPair(
                    user=str(row.get("user", "")),
                    assistant=str(row.get("assistant", "")),
                    source=str(row.get("source", "eval")),
                )
            )

    max_length = 256
    max_new_tokens = 20
    device = "cpu"

    print(f"[benchmark] loaded eval pairs: {len(eval_pairs)}")
    print("[benchmark] base...")
    base_metrics = qp.evaluate_model(
        base_model=base_model,
        eval_pairs=eval_pairs,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        adapter_dir=None,
    )
    print("[benchmark] tuned...")
    tuned_metrics = qp.evaluate_model(
        base_model=base_model,
        eval_pairs=eval_pairs,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        adapter_dir=adapter_dir,
    )

    out = {
        "config": {
            "benchmark_type": "quick",
            "max_length": int(max_length),
            "max_new_tokens": int(max_new_tokens),
            "eval_samples": int(len(eval_pairs)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model": base_model,
        },
        "base": base_metrics,
        "tuned": tuned_metrics,
    }
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[done]")
    print(out_json)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
