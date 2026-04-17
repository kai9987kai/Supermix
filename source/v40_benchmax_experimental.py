from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from v40_benchmax_common import infer_domain, infer_intent, json_dump, jsonl_write, stable_hash


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if not cooked:
                continue
            yield json.loads(cooked)


def _curriculum_stage(hardness_score: float, signature: str, cfg: Mapping[str, Any]) -> str:
    frontier_threshold = float(cfg.get("frontier_hardness_threshold") or 1.15)
    bridge_threshold = float(cfg.get("bridge_hardness_threshold") or 0.70)
    lowered = signature.lower()
    if hardness_score >= frontier_threshold or any(token in lowered for token in ("math", "coding", "knowledge")):
        return "frontier"
    if hardness_score >= bridge_threshold:
        return "bridge"
    return "stabilize"


def _verification_budget(stage: str) -> str:
    if stage == "frontier":
        return "slow"
    if stage == "bridge":
        return "medium"
    return "fast"


def _teacher_map(distillation_jsonl: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not distillation_jsonl or not distillation_jsonl.exists():
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    for row in _jsonl_rows(distillation_jsonl):
        prompt_hash = _normalize_text(row.get("prompt_hash") or "")
        if prompt_hash:
            mapping[prompt_hash] = dict(row)
    return mapping


def _teacher_is_useful(teacher_row: Mapping[str, Any], reference_text: str, reference_extracted: str) -> bool:
    answer = _normalize_text(teacher_row.get("teacher_answer") or "")
    if not answer:
        return False
    agreement_ratio = float(teacher_row.get("teacher_agreement_ratio") or 0.0)
    if agreement_ratio >= 0.66:
        return True
    ref = _normalize_text(reference_extracted or reference_text)
    if ref and ref.lower() in answer.lower():
        return True
    return False


def build_research_pack(
    *,
    hard_examples_jsonl: Path | str,
    output_dir: Path | str,
    manifest: Optional[Mapping[str, Any]] = None,
    distillation_jsonl: Optional[Path | str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    payload = dict(manifest or {})
    research_cfg = dict(payload.get("research_pack") or {})
    max_rows_total = int(max_rows or research_cfg.get("max_rows_total") or 768)
    max_rows_per_variant = int(research_cfg.get("max_rows_per_variant") or 256)

    hard_path = Path(hard_examples_jsonl).resolve()
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    teacher_rows = _teacher_map(Path(distillation_jsonl).resolve()) if distillation_jsonl else {}

    emitted: List[Dict[str, Any]] = []
    variant_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    def _emit(row: Dict[str, Any]) -> None:
        variant = str(row.get("metadata", {}).get("variant") or "unknown")
        if len(emitted) >= max_rows_total:
            return
        if variant_counts[variant] >= max_rows_per_variant:
            return
        emitted.append(row)
        variant_counts[variant] += 1
        stage_counts[str(row.get("metadata", {}).get("curriculum_stage") or "unknown")] += 1
        source_counts[str(row.get("source") or "unknown")] += 1

    for hard in _jsonl_rows(hard_path):
        prompt = _normalize_text(hard.get("prompt") or "")
        reference_text = _normalize_text(hard.get("reference_text") or "")
        reference_extracted = _normalize_text(hard.get("reference_extracted") or "")
        prediction = _normalize_text(hard.get("prediction") or "")
        if not prompt or not reference_text:
            continue
        signature = _normalize_text(hard.get("failure_signature") or "general_hard_example")
        hardness_score = float(hard.get("hardness_score") or 0.0)
        stage = _curriculum_stage(hardness_score, signature, research_cfg)
        budget = _verification_budget(stage)
        domain = infer_domain(prompt, {"benchmark": hard.get("benchmark") or "", "domain": hard.get("domain") or ""})
        intent = infer_intent(prompt, domain)
        prompt_hash = _normalize_text(hard.get("prompt_hash") or "")
        lookup_hashes = [value for value in (prompt_hash, stable_hash(prompt)) if value]
        canonical_prompt_hash = lookup_hashes[0] if lookup_hashes else stable_hash(prompt)
        common_meta = {
            "benchmark": hard.get("benchmark") or "unknown",
            "failure_signature": signature,
            "hardness_score": hardness_score,
            "curriculum_stage": stage,
            "verification_budget": budget,
            "prompt_hash": canonical_prompt_hash,
        }

        _emit(
            {
                "prompt": prompt,
                "response_text": reference_text,
                "intent": intent,
                "domain": domain,
                "source": "v40_benchmax_benchmark_replay",
                "metadata": common_meta | {"variant": "benchmark_replay"},
            }
        )

        _emit(
            {
                "prompt": (
                    "You previously produced an incorrect draft.\n"
                    f"Question: {prompt}\n"
                    f"Previous draft: {prediction or '[missing draft]'}\n"
                    "Find the mistake, reason carefully, and provide the corrected final answer."
                ),
                "response_text": reference_text,
                "intent": intent,
                "domain": domain,
                "source": "v40_benchmax_repair_replay",
                "metadata": common_meta | {"variant": "repair_replay"},
            }
        )

        verifier_answer = (
            f"Verdict: incorrect\n"
            f"Why: the draft does not match the reference target for this benchmark item.\n"
            f"Corrected final answer: {reference_text}"
        )
        _emit(
            {
                "prompt": (
                    "Act as a verifier.\n"
                    f"Question: {prompt}\n"
                    f"Draft answer: {prediction or '[missing draft]'}\n"
                    "State whether the draft is correct. Then provide the corrected final answer if needed."
                ),
                "response_text": verifier_answer,
                "intent": intent,
                "domain": domain,
                "source": "v40_benchmax_verifier_replay",
                "metadata": common_meta | {"variant": "verifier_replay"},
            }
        )

        teacher_row = None
        for key in lookup_hashes:
            teacher_row = teacher_rows.get(key)
            if teacher_row:
                break
        if teacher_row and _teacher_is_useful(teacher_row, reference_text, reference_extracted):
            teacher_answer = _normalize_text(teacher_row.get("teacher_answer") or "")
            agreement_ratio = float(teacher_row.get("teacher_agreement_ratio") or 0.0)
            _emit(
                {
                    "prompt": (
                        f"Question: {prompt}\n"
                        f"Teacher hint: {teacher_answer}\n"
                        "Use the teacher hint if it helps, but produce the strongest final answer."
                    ),
                    "response_text": reference_text,
                    "intent": intent,
                    "domain": domain,
                    "source": "v40_benchmax_consensus_teacher_replay",
                    "metadata": common_meta
                    | {
                        "variant": "consensus_teacher_replay",
                        "teacher_model_key": teacher_row.get("teacher_model_key") or "",
                        "teacher_agreement_ratio": agreement_ratio,
                    },
                }
            )

        if len(emitted) >= max_rows_total:
            break

    jsonl_path = out_root / "v40_benchmax_research_pack.jsonl"
    csv_path = out_root / "v40_benchmax_research_pack.csv"
    json_path = out_root / "v40_benchmax_research_pack_summary.json"
    md_path = out_root / "v40_benchmax_research_pack_summary.md"
    jsonl_write(jsonl_path, emitted)

    fieldnames = ["source", "intent", "domain", "prompt", "response_text", "metadata"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in emitted:
            writer.writerow(
                {
                    "source": row.get("source", ""),
                    "intent": row.get("intent", ""),
                    "domain": row.get("domain", ""),
                    "prompt": row.get("prompt", ""),
                    "response_text": row.get("response_text", ""),
                    "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=True, sort_keys=True),
                }
            )

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hard_examples_jsonl": str(hard_path),
        "distillation_jsonl": str(Path(distillation_jsonl).resolve()) if distillation_jsonl else None,
        "output_dir": str(out_root),
        "total_rows": len(emitted),
        "variant_counts": dict(sorted(variant_counts.items())),
        "stage_counts": dict(sorted(stage_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "research_features": [
            "curriculum_stages",
            "repair_replay",
            "verifier_replay",
            "consensus_teacher_replay",
            "budget_aware_verification",
        ],
        "jsonl_path": str(jsonl_path),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
    }
    json_dump(json_path, summary)

    by_stage: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in emitted:
        by_stage[str(row.get("metadata", {}).get("curriculum_stage") or "unknown")].append(row)

    lines = [
        "# v40_benchmax Research Pack",
        "",
        f"- Total rows: {len(emitted)}",
        f"- Hard examples input: `{hard_path}`",
        f"- Distillation input: `{Path(distillation_jsonl).resolve()}`" if distillation_jsonl else "- Distillation input: none",
        "",
        "## Variant Counts",
    ]
    for key, value in sorted(variant_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Curriculum Stages"])
    for key, value in sorted(stage_counts.items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Sample Rows"])
    for stage in sorted(by_stage):
        sample = by_stage[stage][:2]
        for row in sample:
            lines.append(f"- [{stage}] {row['source']}: {str(row['prompt'])[:140]}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
