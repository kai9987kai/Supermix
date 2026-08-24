import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

import qwen_supermix_pipeline as qp
from qwen_adapter_promotion import BENCHMARK_SCHEMA_VERSION, evaluation_code_hashes, sha256_file
from qwen_paired_evidence import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    PAIRED_EVIDENCE_SCHEMA_VERSION,
    build_paired_evidence,
    paired_evidence_sha256,
    validate_reported_metrics,
)


def _load_eval_pairs_from_jsonl(path: Path) -> List[qp.ChatPair]:
    return qp.load_saved_chat_pairs(path)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _resolved_base_revision(base_model: str, explicit_revision: str = "") -> str:
    explicit = str(explicit_revision or "").strip()
    if explicit:
        return explicit
    candidate = Path(str(base_model or "")).expanduser()
    if candidate.is_dir() and candidate.parent.name.lower() == "snapshots":
        return candidate.resolve().name
    return ""


def _resolve_base_model_for_evaluation(
    base_model: str,
    explicit_revision: str = "",
) -> tuple[str, str]:
    """Resolve a model identity to the exact immutable snapshot that will be loaded."""

    requested = str(base_model or "").strip()
    if not requested:
        raise ValueError("Base model must be non-empty.")
    explicit = str(explicit_revision or "").strip()
    candidate = Path(requested).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
        if resolved.parent.name.lower() != "snapshots" or not resolved.name:
            raise ValueError(
                "Content-bound evaluation requires a local Hugging Face snapshots/<revision> path."
            )
        actual_revision = resolved.name
        if explicit and explicit != actual_revision:
            raise ValueError("Explicit base model revision does not match the local snapshot path.")
        return str(resolved), actual_revision

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - transformers installs this dependency
        raise RuntimeError("huggingface_hub is required to resolve an immutable model snapshot.") from exc
    resolved = Path(snapshot_download(repo_id=requested, revision=explicit or None)).resolve()
    if resolved.parent.name.lower() != "snapshots" or not resolved.name:
        raise RuntimeError("Hugging Face did not return an immutable snapshots/<revision> path.")
    actual_revision = resolved.name
    if (
        explicit
        and len(explicit) == 40
        and all(char in "0123456789abcdefABCDEF" for char in explicit)
        and explicit.lower() != actual_revision.lower()
    ):
        raise RuntimeError("Resolved model snapshot does not match the requested commit revision.")
    return str(resolved), actual_revision


def _adapter_provenance(adapter_dir: Optional[Path]) -> Dict[str, str]:
    if adapter_dir is None:
        return {"adapter_sha256": "", "adapter_config_sha256": ""}
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Adapter weights or configuration are missing: {adapter_dir}")
    return {
        "adapter_sha256": _sha256_file(weights_path),
        "adapter_config_sha256": _sha256_file(config_path),
    }


def _curriculum_provenance(
    curriculum_manifest_path: Optional[Path],
    *,
    eval_source: Optional[Path],
) -> Dict[str, str]:
    if curriculum_manifest_path is None:
        return {"curriculum_manifest_sha256": "", "curriculum_eval_sha256": ""}
    manifest_path = curriculum_manifest_path.expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    payload = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("eval"), dict):
        raise ValueError("Curriculum manifest is missing its eval artifact metadata.")
    eval_meta = payload["eval"]
    eval_name = str(eval_meta.get("file") or "").strip()
    if not eval_name or Path(eval_name).is_absolute():
        raise ValueError("Curriculum manifest has an invalid eval artifact path.")
    eval_path = (manifest_path.parent / eval_name).resolve()
    try:
        eval_path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Curriculum eval artifact escapes the manifest directory.") from exc
    if not eval_path.is_file():
        raise FileNotFoundError(f"Curriculum eval artifact not found: {eval_path}")
    actual_eval_sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    declared_eval_sha = str(eval_meta.get("sha256") or "").strip().lower()
    if declared_eval_sha != actual_eval_sha:
        raise ValueError("Curriculum eval artifact hash does not match its manifest.")
    if eval_source is None or eval_source.resolve() != eval_path:
        raise ValueError("Evaluation source does not match the curriculum manifest eval artifact.")
    return {
        "curriculum_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "curriculum_eval_sha256": actual_eval_sha,
    }


def _load_jsonl_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if not cooked:
                continue
            payload = json.loads(cooked)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _load_reusable_reference(
    benchmark_path: Path,
    *,
    selected_eval_path: Path,
    seed: int,
    samples_per_family: int,
    max_eval_samples: int,
    max_length: int,
    max_new_tokens: int,
    base_model: str,
    base_model_revision: str,
    curriculum_provenance: Mapping[str, str],
    code_hashes: Mapping[str, object],
) -> tuple[Dict[str, float], List[Dict[str, object]], str]:
    benchmark_bytes = benchmark_path.read_bytes()
    benchmark_sha256 = hashlib.sha256(benchmark_bytes).hexdigest()
    payload = json.loads(benchmark_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reusable reference benchmark must be a JSON object.")
    if payload.get("schema") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("Reusable reference benchmark uses an unsupported evidence schema.")
    config = payload.get("config")
    provenance = payload.get("provenance")
    metrics = payload.get("base")
    if not isinstance(config, dict) or not isinstance(provenance, dict) or not isinstance(metrics, dict):
        raise ValueError("Reusable reference benchmark is missing config/provenance/base metrics.")
    expected = {
        "seed": int(seed),
        "samples_per_family": int(samples_per_family),
        "max_eval_samples": int(max_eval_samples),
        "max_length": int(max_length),
        "max_new_tokens": int(max_new_tokens),
    }
    for key, value in expected.items():
        configured = config.get(key)
        # Benchmarks written before stratified sampling existed omitted this
        # field; omission meant the same thing as today's disabled value.
        if configured is None and key == "samples_per_family":
            configured = 0
        if int(configured if configured is not None else -1) != value:
            raise ValueError(f"Reusable reference config mismatch for {key}.")
    if str(config.get("base_model") or "") != str(base_model):
        raise ValueError("Reusable reference config mismatch for base_model.")
    if str(config.get("base_model_revision") or "") != str(base_model_revision):
        raise ValueError("Reusable reference config mismatch for base_model_revision.")
    resolved_base_model_raw = str(config.get("resolved_base_model_path") or "").strip()
    resolved_base_model = Path(resolved_base_model_raw).expanduser()
    if not resolved_base_model_raw or not resolved_base_model.is_absolute() or not resolved_base_model.is_dir():
        raise ValueError(
            "Reusable reference config mismatch for resolved_base_model_path."
        )
    resolved_base_model = resolved_base_model.resolve()
    if (
        resolved_base_model.parent.name.lower() != "snapshots"
        or resolved_base_model.name != str(base_model_revision)
    ):
        raise ValueError(
            "Reusable reference resolved_base_model_path is not the claimed immutable snapshot."
        )
    if str(config.get("adapter_dir") or "").strip() or str(config.get("reference_adapter_dir") or "").strip():
        raise ValueError("Reusable reference must represent the untouched base model.")
    prior_eval_path = benchmark_path.parent / "eval_pairs.jsonl"
    if not prior_eval_path.is_file():
        raise FileNotFoundError(f"Reusable reference eval artifact is missing: {prior_eval_path}")
    selected_eval_sha = _sha256_file(selected_eval_path)
    if _sha256_file(prior_eval_path) != selected_eval_sha:
        raise ValueError("Reusable reference evaluated a different held-out sample.")
    expected_provenance = {
        "base_model": str(base_model),
        "base_model_revision": str(base_model_revision),
        "selected_eval_sha256": selected_eval_sha,
        "curriculum_manifest_sha256": str(curriculum_provenance.get("curriculum_manifest_sha256") or ""),
        "curriculum_eval_sha256": str(curriculum_provenance.get("curriculum_eval_sha256") or ""),
        "verifier_schema": str(getattr(qp, "_VERIFIER_SCHEMA", "")),
    }
    for key, value in expected_provenance.items():
        if str(provenance.get(key) or "") != value:
            raise ValueError(f"Reusable reference provenance mismatch for {key}.")
    if provenance.get("code_hashes") != dict(code_hashes):
        raise ValueError("Reusable reference evaluator/verifier hashes do not match current code.")
    if str(provenance.get("adapter_sha256") or "") or str(
        provenance.get("adapter_config_sha256") or ""
    ):
        raise ValueError("Reusable reference provenance unexpectedly binds an adapter.")
    artifacts = payload.get("artifacts")
    artifact_hashes = payload.get("artifact_hashes")
    samples: List[Dict[str, object]] = []
    if isinstance(artifacts, dict):
        sample_path_raw = str(artifacts.get("base_samples_jsonl") or "").strip()
        if sample_path_raw:
            sample_path = Path(sample_path_raw).expanduser().resolve()
            if not sample_path.is_file():
                raise FileNotFoundError(
                    f"Reusable reference base sample artifact is missing: {sample_path}"
                )
            if not isinstance(artifact_hashes, Mapping):
                raise ValueError("Reusable reference is missing sample artifact hashes.")
            declared_sample_sha = str(artifact_hashes.get("base_samples_sha256") or "")
            actual_sample_sha = _sha256_file(sample_path)
            if declared_sample_sha != actual_sample_sha:
                raise ValueError("Reusable reference base sample artifact hash does not match.")
            if str(provenance.get("base_samples_sha256") or "") != actual_sample_sha:
                raise ValueError("Reusable reference provenance does not bind its base samples.")
            samples = _load_jsonl_records(sample_path)
    numeric_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"Reusable reference contains a non-finite metric: {key}.")
        numeric_metrics[str(key)] = parsed
    return numeric_metrics, samples, benchmark_sha256


def _numeric_deltas(base: Dict[str, float], tuned: Dict[str, float]) -> Dict[str, float]:
    keys = set(base.keys()) & set(tuned.keys())
    out: Dict[str, float] = {}
    for key in sorted(keys):
        b = base.get(key)
        t = tuned.get(key)
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            out[key] = float(t) - float(b)
    return out


def _stratified_eval_sample(
    pairs: Sequence[qp.ChatPair],
    *,
    samples_per_family: int,
    seed: int,
) -> List[qp.ChatPair]:
    per_family = max(0, int(samples_per_family))
    if per_family <= 0:
        return list(pairs)
    grouped: Dict[str, List[qp.ChatPair]] = {}
    for pair in pairs:
        metadata = pair.metadata if isinstance(pair.metadata, dict) else {}
        family = str(metadata.get("problem_family") or pair.source or "unknown").strip()
        grouped.setdefault(family, []).append(pair)
    selected: List[qp.ChatPair] = []
    for family in sorted(grouped):
        family_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{family}".encode("utf-8")).digest()[:8],
            "big",
        )
        rng = random.Random(family_seed)
        rows = list(grouped[family])
        rng.shuffle(rows)
        selected.extend(rows[:per_family])
    return selected


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run reproducible Week-1 baseline benchmarks.")
    ap.add_argument("--base_model", required=True, help="HF model id or local model path")
    ap.add_argument(
        "--base_model_revision",
        default="",
        help="Immutable model revision; inferred from a local Hugging Face snapshots/<revision> path when possible.",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu"])
    ap.add_argument("--adapter_dir", default="", help="Optional LoRA adapter dir for tuned comparison")
    ap.add_argument(
        "--reference_adapter_dir",
        default="",
        help="Optional adapter loaded for the reference side instead of the untouched base.",
    )
    ap.add_argument(
        "--reference_benchmark_json",
        default="",
        help="Reuse a prior reference result after exact held-out/config hash validation.",
    )
    ap.add_argument("--eval_jsonl", default="", help="Optional fixed eval set JSONL")
    ap.add_argument(
        "--curriculum_manifest",
        default="",
        help="Optional curriculum manifest whose eval artifact must exactly match --eval_jsonl.",
    )
    ap.add_argument(
        "--data",
        nargs="*",
        default=[],
        help="Optional training/eval data JSONL(s). Used only when --eval_jsonl is not provided.",
    )
    ap.add_argument("--max_records", type=int, default=480)
    ap.add_argument("--eval_size", type=int, default=64)
    ap.add_argument("--eval_split_mode", choices=["auto", "random"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_eval_samples", type=int, default=0, help="0 keeps all eval rows")
    ap.add_argument(
        "--samples_per_family",
        type=int,
        default=0,
        help="Deterministically sample up to N rows from every problem_family (0 disables).",
    )
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--paired_bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument(
        "--paired_bootstrap_resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    ap.add_argument("--output_root", default="artifacts/research_baselines")
    ap.add_argument("--run_name", default="", help="Optional run folder name")
    ap.add_argument("--benchmark_type", default="week1_baseline")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    evaluation_base_model, base_model_revision = _resolve_base_model_for_evaluation(
        str(args.base_model),
        str(args.base_model_revision),
    )
    code_hashes = evaluation_code_hashes(Path(__file__).resolve().parent)

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
            split_mode=str(args.eval_split_mode),
        )
    else:
        raise ValueError("Provide either --eval_jsonl or --data.")

    curriculum_manifest_path = (
        Path(args.curriculum_manifest).expanduser().resolve()
        if str(args.curriculum_manifest or "").strip()
        else None
    )
    curriculum_provenance = _curriculum_provenance(
        curriculum_manifest_path,
        eval_source=eval_source,
    )

    if int(args.samples_per_family) > 0 and int(args.max_eval_samples) > 0:
        raise ValueError("Use either --samples_per_family or --max_eval_samples, not both.")
    eval_pairs = _stratified_eval_sample(
        eval_pairs,
        samples_per_family=int(args.samples_per_family),
        seed=int(args.seed),
    )
    if int(args.max_eval_samples) > 0 and len(eval_pairs) > int(args.max_eval_samples):
        rng = random.Random(int(args.seed) + 101)
        eval_pairs = rng.sample(list(eval_pairs), int(args.max_eval_samples))

    eval_out = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(eval_out, list(eval_pairs))
    selected_eval_sha256 = _sha256_file(eval_out)
    print(f"[eval] samples={len(eval_pairs)}")
    reference_adapter_dir = (
        Path(args.reference_adapter_dir).expanduser().resolve()
        if args.reference_adapter_dir.strip()
        else None
    )
    if reference_adapter_dir is not None and not reference_adapter_dir.exists():
        raise FileNotFoundError(f"Reference adapter directory not found: {reference_adapter_dir}")
    reusable_reference_path = (
        Path(args.reference_benchmark_json).expanduser().resolve()
        if args.reference_benchmark_json.strip()
        else None
    )
    if reusable_reference_path is not None and reference_adapter_dir is not None:
        raise ValueError("Do not combine --reference_benchmark_json and --reference_adapter_dir.")
    reference_benchmark_sha256 = ""
    if reusable_reference_path is not None:
        if not reusable_reference_path.is_file():
            raise FileNotFoundError(f"Reusable reference benchmark not found: {reusable_reference_path}")
        print("[benchmark] reusing content-validated reference metrics...")
        base_metrics, base_samples, reference_benchmark_sha256 = _load_reusable_reference(
            reusable_reference_path,
            selected_eval_path=eval_out,
            seed=int(args.seed),
            samples_per_family=int(args.samples_per_family),
            max_eval_samples=int(args.max_eval_samples),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            base_model=str(args.base_model),
            base_model_revision=base_model_revision,
            curriculum_provenance=curriculum_provenance,
            code_hashes=code_hashes,
        )
    else:
        print("[benchmark] reference..." if reference_adapter_dir is not None else "[benchmark] base...")
        base_metrics, base_samples = qp.evaluate_model_detailed(
            base_model=evaluation_base_model,
            eval_pairs=eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=reference_adapter_dir,
        )

    tuned_metrics: Dict[str, float] = {}
    tuned_samples: List[Dict[str, object]] = []
    adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir.strip() else None
    adapter_provenance = _adapter_provenance(adapter_dir)
    if adapter_dir is not None:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
        print("[benchmark] tuned...")
        tuned_metrics, tuned_samples = qp.evaluate_model_detailed(
            base_model=evaluation_base_model,
            eval_pairs=eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=adapter_dir,
        )
    if _adapter_provenance(adapter_dir) != adapter_provenance:
        raise RuntimeError("Adapter weights or configuration changed during evaluation.")
    if _sha256_file(eval_out) != selected_eval_sha256:
        raise RuntimeError("Selected eval artifact changed during evaluation.")
    if curriculum_manifest_path is not None and _curriculum_provenance(
        curriculum_manifest_path,
        eval_source=eval_source,
    ) != curriculum_provenance:
        raise RuntimeError("Curriculum manifest or eval artifact changed during evaluation.")
    if evaluation_code_hashes(Path(__file__).resolve().parent) != code_hashes:
        raise RuntimeError("Evaluator or verifier code changed during evaluation.")
    if reusable_reference_path is not None and _sha256_file(
        reusable_reference_path
    ) != reference_benchmark_sha256:
        raise RuntimeError("Reusable reference benchmark changed during evaluation.")

    artifact_paths, sample_summary = qp.save_benchmark_sample_artifacts(
        output_dir=run_dir,
        base_samples=base_samples,
        tuned_samples=tuned_samples,
    )
    artifact_hashes: Dict[str, str] = {}
    for artifact_key, hash_key in (
        ("base_samples_jsonl", "base_samples_sha256"),
        ("tuned_samples_jsonl", "tuned_samples_sha256"),
        ("sample_comparison_jsonl", "sample_comparison_sha256"),
    ):
        raw_path = str(artifact_paths.get(artifact_key) or "").strip()
        if raw_path:
            artifact_hashes[hash_key] = _sha256_file(Path(raw_path))

    paired_evidence: Dict[str, object] = {}
    paired_evidence_digest = ""
    if tuned_samples:
        paired_evidence = build_paired_evidence(
            base_samples,
            tuned_samples,
            eval_pairs,
            artifact_hashes=artifact_hashes,
            bootstrap_seed=int(args.paired_bootstrap_seed),
            bootstrap_resamples=int(args.paired_bootstrap_resamples),
        )
        recomputed_metrics = paired_evidence.get("recomputed_metrics")
        if not isinstance(recomputed_metrics, Mapping):  # pragma: no cover - module contract
            raise RuntimeError("Paired evidence omitted recomputed metrics.")
        recomputed_base = recomputed_metrics.get("base")
        recomputed_tuned = recomputed_metrics.get("tuned")
        if not isinstance(recomputed_base, Mapping) or not isinstance(recomputed_tuned, Mapping):
            raise RuntimeError("Paired evidence omitted base or tuned recomputed metrics.")
        validate_reported_metrics(base_metrics, recomputed_base, side="base")
        validate_reported_metrics(tuned_metrics, recomputed_tuned, side="tuned")
        paired_evidence_digest = paired_evidence_sha256(paired_evidence)

    results = {
        "schema": BENCHMARK_SCHEMA_VERSION,
        "config": {
            "benchmark_type": args.benchmark_type,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model": str(args.base_model),
            "base_model_revision": base_model_revision,
            "resolved_base_model_path": evaluation_base_model,
            "adapter_dir": str(adapter_dir) if adapter_dir is not None else "",
            "reference_adapter_dir": (
                str(reference_adapter_dir) if reference_adapter_dir is not None else ""
            ),
            "reference_benchmark_json": (
                str(reusable_reference_path) if reusable_reference_path is not None else ""
            ),
            "reference_benchmark_sha256": reference_benchmark_sha256,
            "eval_source": str(eval_source) if eval_source is not None else "split_from_data",
            "eval_samples": int(len(eval_pairs)),
            "seed": int(args.seed),
            "eval_split_mode": str(args.eval_split_mode),
            "max_length": int(args.max_length),
            "max_new_tokens": int(args.max_new_tokens),
            "max_records": int(args.max_records),
            "eval_size": int(args.eval_size),
            "max_eval_samples": int(args.max_eval_samples),
            "samples_per_family": int(args.samples_per_family),
            "paired_bootstrap_seed": int(args.paired_bootstrap_seed),
            "paired_bootstrap_resamples": int(args.paired_bootstrap_resamples),
            "curriculum_manifest": (
                str(curriculum_manifest_path) if curriculum_manifest_path is not None else ""
            ),
        },
        "provenance": {
            "base_model": str(args.base_model),
            "base_model_revision": base_model_revision,
            "selected_eval_sha256": selected_eval_sha256,
            "curriculum_manifest_sha256": curriculum_provenance["curriculum_manifest_sha256"],
            "curriculum_eval_sha256": curriculum_provenance["curriculum_eval_sha256"],
            "adapter_sha256": adapter_provenance["adapter_sha256"],
            "adapter_config_sha256": adapter_provenance["adapter_config_sha256"],
            "code_hashes": code_hashes,
            "verifier_schema": str(getattr(qp, "_VERIFIER_SCHEMA", "")),
            "base_samples_sha256": artifact_hashes.get("base_samples_sha256", ""),
            "tuned_samples_sha256": artifact_hashes.get("tuned_samples_sha256", ""),
            "sample_comparison_sha256": artifact_hashes.get(
                "sample_comparison_sha256", ""
            ),
            "paired_evidence_schema": (
                PAIRED_EVIDENCE_SCHEMA_VERSION if paired_evidence else ""
            ),
            "paired_evidence_sha256": paired_evidence_digest,
        },
        "artifacts": artifact_paths,
        "artifact_hashes": artifact_hashes,
        "sample_summary": sample_summary,
        "base": base_metrics,
    }
    if tuned_metrics:
        results["tuned"] = tuned_metrics
        results["delta_tuned_minus_base"] = _numeric_deltas(base_metrics, tuned_metrics)
        results["paired_evidence"] = paired_evidence

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
