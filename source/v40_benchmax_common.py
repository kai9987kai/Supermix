from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SOURCE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SOURCE_DIR / "v40_benchmax_manifest.json"


def load_manifest(path: Path | None = None) -> Dict[str, Any]:
    manifest_path = Path(path or MANIFEST_PATH).resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def output_root(repo_root: Path) -> Path:
    return Path(repo_root).resolve() / "output" / "v40_benchmax"


def ablation_root(repo_root: Path, ablation_id: str) -> Path:
    return output_root(repo_root) / "ablations" / ablation_id


def manifest_ablation_map(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    payload = manifest or load_manifest()
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("ablation_matrix", []) or []:
        if not isinstance(item, dict):
            continue
        cooked = dict(item)
        key = _normalize_text(cooked.get("ablation_id") or cooked.get("id"))
        if not key:
            continue
        cooked.setdefault("ablation_id", key)
        cooked.setdefault("id", key)
        mapping[key] = cooked
    return mapping


def _recipe_section(manifest: Dict[str, Any], recipe_key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(defaults)
    sections = manifest.get("data_recipes")
    if isinstance(sections, dict):
        payload.update(dict(sections.get(recipe_key) or {}))
    return payload


def json_dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def csv_write(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def infer_domain(prompt: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    meta = metadata or {}
    benchmark = str(meta.get("benchmark") or "").lower()
    task_mode = str(meta.get("task_mode") or "").lower()
    lowered = str(prompt or "").lower()
    if benchmark in {"gsm8k"} or "math" in task_mode:
        return "math"
    if benchmark in {"arc_challenge", "boolq", "hellaswag", "mmlu", "piqa"}:
        return "knowledge"
    if "openscad" in lowered or ".scad" in lowered:
        return "spatial_3d"
    if any(token in lowered for token in ("video", "contact sheet", "frame", "clip")) or task_mode == "video":
        return "video"
    if any(token in lowered for token in ("image", "photo", "diagram", "recognize", "identify", "visual")):
        return "vision"
    if task_mode == "coding" or any(token in lowered for token in ("python", "javascript", "code", "bug", "traceback", "sql", "regex")):
        return "coding"
    if task_mode == "creative":
        return "creative"
    if "plan" in lowered or "compare" in lowered or "tradeoff" in lowered:
        return "planning"
    return str(meta.get("domain") or task_mode or "general") or "general"


def infer_intent(prompt: str, domain: str = "") -> str:
    lowered = str(prompt or "").lower()
    if domain == "math":
        return "math"
    if domain == "coding":
        return "coding"
    if domain == "vision":
        return "vision"
    if domain == "spatial_3d":
        return "spatial_3d"
    if domain == "video":
        return "video"
    if any(token in lowered for token in ("latest", "current", "recent", "news", "web")):
        return "current_info"
    if any(token in lowered for token in ("plan", "steps", "roadmap")):
        return "planning"
    if any(token in lowered for token in ("compare", "tradeoff")):
        return "comparison"
    return "general"


def prompt_row_to_omni(row: Any, *, source: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if hasattr(row, "user") and hasattr(row, "assistant"):
        user = _normalize_text(getattr(row, "user"))
        assistant = _normalize_text(getattr(row, "assistant"))
        meta = dict(metadata or {})
        src = str(getattr(row, "source", source) or source)
        meta.update(dict(getattr(row, "metadata", {}) or {}))
    else:
        user = _normalize_text(row.get("user") or row.get("prompt") or row.get("question"))
        assistant = _normalize_text(row.get("assistant") or row.get("response_text") or row.get("response") or row.get("answer"))
        meta = dict(metadata or {})
        src = str(row.get("source") or source)
        meta.update(dict(row.get("metadata") or {}))
    domain = infer_domain(user, meta)
    return {
        "prompt": user,
        "intent": infer_intent(user, domain),
        "response_text": assistant,
        "domain": domain,
        "source": src,
        "metadata": meta,
    }


def dict_row_to_omni(row: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    meta = dict(row.get("metadata") or {})
    user = _normalize_text(row.get("user") or row.get("prompt") or row.get("question"))
    assistant = _normalize_text(row.get("assistant") or row.get("response") or row.get("answer") or row.get("completion"))
    domain = infer_domain(user, meta)
    return {
        "prompt": user,
        "intent": infer_intent(user, domain),
        "response_text": assistant,
        "domain": domain,
        "source": str(row.get("source") or source),
        "metadata": meta,
    }


def _sample_rows(rows: Sequence[Any], limit: int, seed: int) -> List[Any]:
    items = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    if limit > 0:
        return items[: min(len(items), limit)]
    return items


def _resolve_repo_paths(repo_root: Path, rel_paths: Sequence[str]) -> List[Path]:
    root = Path(repo_root).resolve()
    resolved: List[Path] = []
    for rel in rel_paths:
        cooked = Path(str(rel))
        candidates = []
        if cooked.is_absolute():
            candidates.append(cooked)
        else:
            candidates.append(root / cooked)
            candidates.append(root / "datasets" / cooked)
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists():
                resolved.append(candidate)
                break
    return resolved


def _fallback_load_prompt_rows_safe(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(dict_row_to_omni(payload, source="v33_style_fallback"))
    return rows


def _fallback_merge_unique_rows(prompt_rows: Sequence[Dict[str, Any]], paper_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in list(prompt_rows) + list(paper_rows):
        prompt = _normalize_text(row.get("prompt") or "")
        response = _normalize_text(row.get("response_text") or row.get("assistant") or "")
        key = (prompt.lower(), response.lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(row))
    return merged


def _fallback_select_reference_rows(rows: Sequence[Dict[str, Any]], *, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    return _sample_rows(rows, sample_size, seed)


PROTEIN_FOLDING_CONCEPTS: Tuple[Dict[str, str], ...] = (
    {
        "concept": "secondary_structure",
        "prompt": "Explain the difference between an alpha helix and a beta sheet in protein folding.",
        "answer": "An alpha helix is a coiled local structure stabilized by backbone hydrogen bonds every few residues, while a beta sheet forms when stretched strands align and hydrogen-bond side by side.",
    },
    {
        "concept": "hydrophobic_collapse",
        "prompt": "What role does the hydrophobic effect play in protein folding?",
        "answer": "Hydrophobic side chains tend to bury themselves away from water, which drives early collapse of many proteins and helps create a stable packed core.",
    },
    {
        "concept": "chaperones",
        "prompt": "Why do molecular chaperones matter for protein folding?",
        "answer": "Molecular chaperones reduce misfolding and aggregation by shielding exposed hydrophobic regions, giving proteins more chances to reach a productive folded state.",
    },
    {
        "concept": "disulfide_bonds",
        "prompt": "How do disulfide bonds affect protein stability?",
        "answer": "Disulfide bonds covalently link cysteines, which can stabilize the folded structure and reduce the number of conformations the chain can explore.",
    },
    {
        "concept": "glycine_proline",
        "prompt": "Why can glycine and proline strongly influence local protein structure?",
        "answer": "Glycine is unusually flexible, while proline is conformationally restricted and can break or kink helices, so both residues strongly affect local backbone geometry.",
    },
    {
        "concept": "folding_funnel",
        "prompt": "What is meant by a protein folding funnel?",
        "answer": "A folding funnel describes how many high-energy conformations progressively collapse toward fewer, lower-energy native-like states during folding.",
    },
    {
        "concept": "contact_map",
        "prompt": "What does a protein contact map represent?",
        "answer": "A protein contact map marks which residue pairs are close in three-dimensional space, which helps summarize tertiary structure constraints.",
    },
    {
        "concept": "alphafold_confidence",
        "prompt": "What does pLDDT indicate in protein structure prediction?",
        "answer": "pLDDT is a per-residue confidence estimate that indicates how reliable a predicted local structure is, with higher values meaning greater confidence.",
    },
    {
        "concept": "msa_signal",
        "prompt": "Why do multiple-sequence alignments help protein structure prediction?",
        "answer": "Multiple-sequence alignments expose correlated mutations across homologs, which reveal likely residue contacts and structural constraints.",
    },
    {
        "concept": "intrinsically_disordered",
        "prompt": "What makes intrinsically disordered proteins difficult for structure prediction?",
        "answer": "Intrinsically disordered proteins do not settle into one rigid native structure, so they are better described by conformational ensembles than a single fold.",
    },
    {
        "concept": "membrane_proteins",
        "prompt": "Why are membrane proteins a special case for folding and modeling?",
        "answer": "Membrane proteins fold in a hydrophobic lipid environment and often rely on transmembrane helices or beta barrels, so their constraints differ from soluble proteins.",
    },
    {
        "concept": "rmsd_tm",
        "prompt": "How are RMSD and TM-score different when comparing protein structures?",
        "answer": "RMSD measures average coordinate deviation and is sensitive to outliers, while TM-score is length-normalized and focuses more on overall fold similarity.",
    },
)


def _protein_row(prompt: str, answer: str, *, concept: str, variant: str, reasoning_budget: str) -> Dict[str, Any]:
    return {
        "prompt": _normalize_text(prompt),
        "intent": "knowledge",
        "response_text": str(answer).strip(),
        "domain": "knowledge",
        "source": "v40_benchmax_protein_folding",
        "metadata": {
            "task_mode": "knowledge",
            "domain": "knowledge",
            "subdomain": "protein_folding",
            "concept": concept,
            "variant": variant,
            "reasoning_budget": reasoning_budget,
            "research_tags": ["v40", "protein_folding"],
        },
    }


def build_protein_folding_rows(*, seed: int, max_rows: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in PROTEIN_FOLDING_CONCEPTS:
        concept = item["concept"]
        prompt = item["prompt"]
        answer = item["answer"]
        rows.extend(
            [
                _protein_row(prompt, answer, concept=concept, variant="concept_explain", reasoning_budget="medium"),
                _protein_row(
                    f"Give a concise benchmark-style answer: {prompt}",
                    answer,
                    concept=concept,
                    variant="concise_answer",
                    reasoning_budget="medium",
                ),
                _protein_row(
                    f"Correct this mistake about protein folding: {prompt} Someone claims the opposite.",
                    answer,
                    concept=concept,
                    variant="error_correction",
                    reasoning_budget="deep",
                ),
                _protein_row(
                    f"Connect this idea to structure prediction practice: {prompt}",
                    f"{answer} In practice, this matters because structure prediction systems need these constraints to rank native-like states over decoys.",
                    concept=concept,
                    variant="structure_prediction_link",
                    reasoning_budget="deep",
                ),
                _protein_row(
                    f"Relate this protein-folding concept to 3D reasoning: {prompt}",
                    f"{answer} Thinking in 3D matters because residue packing, contacts, and long-range geometry determine whether the fold is physically plausible.",
                    concept=concept,
                    variant="spatial_reasoning",
                    reasoning_budget="deep",
                ),
                _protein_row(
                    f"Answer in one careful paragraph for a biology student: {prompt}",
                    answer,
                    concept=concept,
                    variant="student_paragraph",
                    reasoning_budget="medium",
                ),
                _protein_row(
                    f"You are debugging a protein model. Why does this concept matter? {prompt}",
                    f"{answer} Ignoring it can make a predicted fold look numerically plausible while violating core physical or evolutionary constraints.",
                    concept=concept,
                    variant="model_debugging",
                    reasoning_budget="deep",
                ),
                _protein_row(
                    f"Turn this into a benchmark-style correction task: {prompt}",
                    f"{answer} A strong correction should name the concept directly and explain how it affects folded structure or prediction quality.",
                    concept=concept,
                    variant="benchmark_correction",
                    reasoning_budget="deep",
                ),
                _protein_row(
                    f"What failure mode shows up if a model ignores this concept in protein folding? {prompt}",
                    f"{answer} If the concept is ignored, the system can favor decoy structures, miss long-range constraints, or overstate confidence in the wrong fold.",
                    concept=concept,
                    variant="failure_mode",
                    reasoning_budget="deep",
                ),
                _protein_row(
                    f"Relate this to structure ranking and confidence scoring: {prompt}",
                    f"{answer} It also matters for ranking because better fold models should satisfy these constraints more consistently than weaker decoys.",
                    concept=concept,
                    variant="ranking_and_confidence",
                    reasoning_budget="deep",
                ),
            ]
        )

    rng = random.Random(int(seed))
    rng.shuffle(rows)
    limited = rows[: max_rows] if max_rows > 0 else rows
    summary = {
        "source": "protein_folding_pack",
        "selected_rows": len(limited),
        "source_rows": len(rows),
        "concepts": len(PROTEIN_FOLDING_CONCEPTS),
    }
    return limited, summary


def _fallback_make_paper_prompt_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(manifest.get("research_influences") or []):
        if not isinstance(item, dict):
            continue
        title = _normalize_text(item.get("title") or "recent research")
        idea = _normalize_text(item.get("idea") or "practical improvement")
        rows.append(
            {
                "user": f"Summarize the core idea of {title} in one sentence for a compact model builder.",
                "assistant": idea or title,
                "source": "v40_benchmax_research_fallback",
                "metadata": {"kind": "research_influence", "title": title},
            }
        )
    for feature in list(manifest.get("novel_features") or []):
        feature_text = _normalize_text(feature)
        if not feature_text:
            continue
        rows.append(
            {
                "user": f"Explain how this benchmark-maximization feature helps training: {feature_text}",
                "assistant": feature_text,
                "source": "v40_benchmax_feature_fallback",
                "metadata": {"kind": "novel_feature"},
            }
        )
    return rows


def build_v33_style_rows(repo_root: Path, *, seed: int, sample_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = load_manifest()
    data_recipe = _recipe_section(
        manifest,
        "v33",
        {
            "sample_size": 960,
            "paper_rows": True,
            "prompt_sources": [
                "conversation_data.supermix_plus_v27_500k.jsonl",
                "conversation_data.delta_anchor_mix_2026_03_26.jsonl",
                "conversation_data.delta_official_refresh_2026_03_26.jsonl",
                "conversation_data.coding_knowledge_2026_02_19.jsonl",
                "conversation_data.hybrid_v6_live_knowledge.jsonl",
                "conversation_data.mega_reasoning_creative_v25_75582.jsonl",
                "conversation_data.quality_anchor_v2.jsonl",
            ],
            "protein_pack_count": 0,
        },
    )
    prompt_paths = _resolve_repo_paths(repo_root, list(data_recipe.get("prompt_sources", []) or []))
    try:
        from build_v33_frontier_dataset import make_paper_prompt_rows, merge_unique_rows, load_prompt_rows_safe, select_reference_rows

        prompt_rows = load_prompt_rows_safe(prompt_paths) if prompt_paths else []
        paper_rows = make_paper_prompt_rows() if data_recipe.get("paper_rows", True) else []
        merged = merge_unique_rows(prompt_rows, paper_rows)
        total_target = int(sample_size or data_recipe.get("sample_size") or 0)
        protein_target = int(data_recipe.get("protein_pack_count") or 0)
        base_target = max(total_target - protein_target, 0)
        selected = select_reference_rows(merged, sample_size=base_target or total_target, seed=int(seed))
    except Exception:
        prompt_rows = _fallback_load_prompt_rows_safe(prompt_paths) if prompt_paths else []
        paper_rows = _fallback_make_paper_prompt_rows(manifest) if data_recipe.get("paper_rows", True) else []
        merged = _fallback_merge_unique_rows(prompt_rows, paper_rows)
        total_target = int(sample_size or data_recipe.get("sample_size") or 0)
        protein_target = int(data_recipe.get("protein_pack_count") or 0)
        base_target = max(total_target - protein_target, 0)
        selected = _fallback_select_reference_rows(merged, sample_size=base_target or total_target, seed=int(seed))
    protein_rows, protein_summary = build_protein_folding_rows(
        seed=int(seed) + 41,
        max_rows=int(data_recipe.get("protein_pack_count") or 0),
    )
    out = [prompt_row_to_omni(row, source="v33_style") for row in selected]
    out.extend(protein_rows)
    summary = {
        "recipe": "v33",
        "selected_rows": len(out),
        "source_rows": len(merged),
        "prompt_sources": [str(path) for path in prompt_paths],
        "paper_rows": len(paper_rows),
        "protein_rows": len(protein_rows),
        "protein_summary": protein_summary,
    }
    return out, summary


def build_v39_style_rows(repo_root: Path, *, seed: int, sample_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = load_manifest()
    data_recipe = _recipe_section(
        manifest,
        "v39",
        {
            "sample_size": 960,
            "counts": {
                "arc_count": 1119,
                "boolq_count": 3000,
                "gsm8k_count": 2500,
                "hellaswag_count": 4000,
                "mmlu_count": 5000,
                "piqa_count": 4000,
            },
            "support_prompt_sources": [],
            "support_sample_size": 0,
            "protein_pack_count": 0,
        },
    )
    counts = dict(data_recipe.get("counts") or {})
    support_paths = _resolve_repo_paths(repo_root, list(data_recipe.get("support_prompt_sources", []) or []))
    support_rows_loaded = _fallback_load_prompt_rows_safe(support_paths) if support_paths else []
    support_count = int(data_recipe.get("support_sample_size") or 0)
    protein_rows, protein_summary = build_protein_folding_rows(
        seed=int(seed) + 67,
        max_rows=int(data_recipe.get("protein_pack_count") or 0),
    )
    try:
        from build_reasoning_benchmix_v39 import build_dataset

        rows = build_dataset(argparse_namespace_from_counts(counts, seed=int(seed)))
        total_target = int(sample_size or data_recipe.get("sample_size") or len(rows))
        benchmark_target = max(total_target - support_count - len(protein_rows), 0)
        rows = _sample_rows(rows, benchmark_target or len(rows), seed=int(seed))
        out = [dict_row_to_omni(row, source="v39_style") for row in rows]
    except Exception:
        template_rows = [
            {
                "user": "Answer the multiple-choice science question. End with 'Final answer: <letter>'.\nQuestion: Which object is most likely a planet in our solar system?\nA. Moon\nB. Comet\nC. Earth\nD. Asteroid",
                "assistant": "Final answer: C. Earth",
                "source": "v39_style_fallback_arc",
                "metadata": {"benchmark": "arc_challenge", "task_mode": "reasoning", "reasoning_budget": "medium"},
            },
            {
                "user": "Read the passage and answer the yes/no question. End with 'Final answer: yes' or 'Final answer: no'.\nPassage: The library opens at 9 AM.\nQuestion: Does the library open before noon?",
                "assistant": "Final answer: yes",
                "source": "v39_style_fallback_boolq",
                "metadata": {"benchmark": "boolq", "task_mode": "knowledge", "reasoning_budget": "medium"},
            },
            {
                "user": "Solve the math word problem carefully. End with 'Final answer: <number>'.\nQuestion: A box has 12 apples and you add 7 more. How many apples are there?",
                "assistant": "Solution: 12 + 7 = 19.\nFinal answer: 19",
                "source": "v39_style_fallback_gsm8k",
                "metadata": {"benchmark": "gsm8k", "task_mode": "reasoning", "reasoning_budget": "deep"},
            },
            {
                "user": "Choose the most plausible next sentence for the situation. End with 'Final answer: <letter>'.\nActivity: opening a package\nContext: The person carefully cut the tape and lifted the lid.\nA. They examined the contents.\nB. They teleported away.\nC. The package vanished.\nD. The room became underwater.",
                "assistant": "Final answer: A. They examined the contents.",
                "source": "v39_style_fallback_hellaswag",
                "metadata": {"benchmark": "hellaswag", "task_mode": "reasoning", "reasoning_budget": "medium"},
            },
            {
                "user": "Answer the multiple-choice knowledge question. End with 'Final answer: <letter>'.\nSubject: biology\nQuestion: Which organ pumps blood through the body?\nA. Liver\nB. Heart\nC. Lung\nD. Kidney",
                "assistant": "Final answer: B. Heart",
                "source": "v39_style_fallback_mmlu",
                "metadata": {"benchmark": "mmlu", "task_mode": "knowledge", "reasoning_budget": "medium"},
            },
            {
                "user": "Choose the better physical commonsense solution. End with 'Final answer: <letter>'.\nGoal: carry groceries without dropping them.\nA. Use both hands to hold the bag.\nB. Put the bag on the floor and kick it.\n",
                "assistant": "Final answer: A. Use both hands to hold the bag.",
                "source": "v39_style_fallback_piqa",
                "metadata": {"benchmark": "piqa", "task_mode": "reasoning", "reasoning_budget": "medium"},
            },
        ]
        total_target = int(sample_size or data_recipe.get("sample_size") or len(template_rows))
        benchmark_target = max(total_target - support_count - len(protein_rows), 0)
        rows = _sample_rows(template_rows, benchmark_target or len(template_rows), seed=int(seed))
        out = [dict_row_to_omni(row, source="v39_style") for row in rows]
    support_rows = _sample_rows(support_rows_loaded, support_count, seed=int(seed) + 11)
    out.extend(prompt_row_to_omni(row, source="v40_v39_support") for row in support_rows)
    out.extend(protein_rows)
    summary = {
        "recipe": "v39",
        "selected_rows": len(out),
        "source_rows": len(rows),
        "counts": counts,
        "support_prompt_sources": [str(path) for path in support_paths],
        "support_rows": len(support_rows),
        "protein_rows": len(protein_rows),
        "protein_summary": protein_summary,
    }
    return out, summary


def argparse_namespace_from_counts(counts: Dict[str, Any], *, seed: int) -> Any:
    class _Args:
        pass

    args = _Args()
    for key, value in counts.items():
        setattr(args, key, value)
    setattr(args, "seed", seed)
    return args


def build_ablation_pack(repo_root: Path, ablation_id: str, *, seed: int, sample_size: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = load_manifest()
    ablations = manifest_ablation_map(manifest)
    if ablation_id not in ablations:
        raise KeyError(f"Unknown ablation_id: {ablation_id}")
    ablation = ablations[ablation_id]
    data_recipe = _normalize_text(ablation.get("data_recipe") or ablation.get("data_family") or "").lower()
    recipe_family = _normalize_text(ablation.get("head_recipe") or ablation.get("recipe_family") or data_recipe).lower()
    v33_defaults = _recipe_section(manifest, "v33", {"sample_size": 960})
    v39_defaults = _recipe_section(manifest, "v39", {"sample_size": 960})
    if "v33" in data_recipe:
        rows, summary = build_v33_style_rows(repo_root, seed=seed, sample_size=int(sample_size or v33_defaults.get("sample_size") or 960))
    elif "v39" in data_recipe:
        rows, summary = build_v39_style_rows(repo_root, seed=seed, sample_size=int(sample_size or v39_defaults.get("sample_size") or 960))
    else:
        raise ValueError(f"Unsupported data recipe: {data_recipe}")
    summary.update(
        {
            "ablation_id": ablation_id,
            "head_recipe": recipe_family or data_recipe,
            "data_family": str(ablation.get("data_family") or data_recipe),
            "recipe_family": str(ablation.get("recipe_family") or recipe_family or data_recipe),
        }
    )
    return rows, summary


def hard_example_signature(row: Dict[str, Any]) -> str:
    benchmark = str(row.get("benchmark") or "unknown")
    prompt = str(row.get("prompt") or "")
    if benchmark == "gsm8k":
        kind = "math"
    elif benchmark in {"arc_challenge", "mmlu", "hellaswag", "piqa"}:
        kind = "multiple_choice"
    elif benchmark == "boolq":
        kind = "yes_no"
    else:
        lowered = prompt.lower()
        if any(token in lowered for token in ("python", "code", "bug", "traceback", "sql", "regex")):
            kind = "coding"
        elif any(token in lowered for token in ("image", "photo", "diagram", "visual")):
            kind = "vision"
        else:
            kind = "general"
    return f"{benchmark}:{kind}"


def build_hard_example_pack(details_rows: Sequence[Dict[str, Any]], *, max_examples: int, max_examples_per_group: int, leader_models: Sequence[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    leader_models = tuple(str(item) for item in leader_models)

    for row in details_rows:
        model = str(row.get("model") or "")
        if leader_models and model not in leader_models:
            continue
        exact = float(row.get("exact") or 0.0)
        token_f1 = float(row.get("token_f1") or 0.0)
        char_similarity = float(row.get("char_similarity") or 0.0)
        if exact >= 1.0 and token_f1 >= 0.85 and char_similarity >= 0.85:
            continue
        signature = hard_example_signature(row)
        enriched = dict(row)
        enriched["failure_signature"] = signature
        enriched["hardness_score"] = round(float(1.0 - exact + (0.55 - token_f1) + (0.62 - char_similarity)), 6)
        groups[signature].append(enriched)

    for signature in sorted(groups.keys()):
        bucket = sorted(groups[signature], key=lambda item: float(item.get("hardness_score") or 0.0), reverse=True)
        count = 0
        for row in bucket:
            key = (
                _normalize_text(row.get("prompt") or "").lower(),
                _normalize_text(row.get("reference_extracted") or "").lower(),
                signature,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(row)
            count += 1
            if len(selected) >= max_examples:
                break
            if count >= max_examples_per_group:
                break
        if len(selected) >= max_examples:
            break

    summary = {
        "selected_examples": len(selected),
        "group_counts": {key: len(value) for key, value in groups.items()},
        "leader_models": list(leader_models),
        "max_examples": int(max_examples),
        "max_examples_per_group": int(max_examples_per_group),
    }
    return selected, summary


def compare_baselines(candidate: Dict[str, Any], baselines: Sequence[Dict[str, Any]], *, promotion_threshold: float, max_allowed_drop: float) -> Dict[str, Any]:
    cand_score = float(candidate.get("overall_exact") or 0.0)
    baseline_rows = [dict(row) for row in baselines]
    deltas: List[Dict[str, Any]] = []
    best_baseline = None
    best_delta = None
    for baseline in baseline_rows:
        score = float(baseline.get("overall_exact") or 0.0)
        delta = round(cand_score - score, 6)
        deltas.append({"baseline": baseline.get("model"), "delta": delta, "overall_exact": score})
        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_baseline = baseline
    benchmark_delta = {}
    candidate_benchmarks = dict(candidate.get("benchmarks") or {})
    for baseline in baseline_rows:
        bname = str(baseline.get("model"))
        benchmark_delta[bname] = {
            name: round(float(candidate_benchmarks.get(name, 0.0)) - float((baseline.get("benchmarks") or {}).get(name, 0.0)), 6)
            for name in sorted(set(candidate_benchmarks) | set(baseline.get("benchmarks") or {}))
        }
    better_than_leader = bool(best_delta is not None and best_delta >= float(promotion_threshold))
    severe_regressions = []
    if baseline_rows:
        leader = max(baseline_rows, key=lambda row: float(row.get("overall_exact") or 0.0))
        for name, value in candidate_benchmarks.items():
            drop = float((leader.get("benchmarks") or {}).get(name, 0.0)) - float(value)
            if drop > float(max_allowed_drop):
                severe_regressions.append({"benchmark": name, "drop": round(drop, 6)})
    promoted = better_than_leader and not severe_regressions
    return {
        "candidate": candidate.get("model"),
        "candidate_overall_exact": cand_score,
        "baselines": deltas,
        "best_baseline": best_baseline.get("model") if best_baseline else None,
        "best_delta": round(float(best_delta or 0.0), 6),
        "promoted": promoted,
        "promotion_threshold": float(promotion_threshold),
        "max_allowed_drop": float(max_allowed_drop),
        "severe_regressions": severe_regressions,
        "benchmark_deltas": benchmark_delta,
    }


def _same_shape(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        va = a[key]
        vb = b[key]
        if getattr(va, "shape", None) != getattr(vb, "shape", None):
            return False
    return True


def compatibility_report(state_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not state_dicts:
        return {"ok": False, "reason": "no_state_dicts"}
    ref = state_dicts[0]
    incompatible: List[Dict[str, Any]] = []
    for index, state in enumerate(state_dicts[1:], start=2):
        if ref.keys() != state.keys():
            incompatible.append({"checkpoint_index": index, "reason": "key_mismatch", "missing": sorted(set(ref) - set(state)), "unexpected": sorted(set(state) - set(ref))})
            continue
        shape_mismatch = []
        for key in ref:
            if getattr(ref[key], "shape", None) != getattr(state[key], "shape", None):
                shape_mismatch.append(key)
        if shape_mismatch:
            incompatible.append({"checkpoint_index": index, "reason": "shape_mismatch", "keys": shape_mismatch[:32]})
    return {"ok": not incompatible, "incompatible": incompatible, "checkpoint_count": len(state_dicts)}


def average_state_dicts(state_dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not state_dicts:
        raise ValueError("No checkpoints provided for averaging.")
    report = compatibility_report(state_dicts)
    if not report["ok"]:
        raise ValueError(f"Incompatible checkpoints: {report['incompatible']}")
    merged: Dict[str, Any] = {}
    keys = state_dicts[0].keys()
    for key in keys:
        values = [sd[key].float() for sd in state_dicts]
        merged[key] = sum(values) / float(len(values))
    return merged


def stable_hash(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()[:16]
