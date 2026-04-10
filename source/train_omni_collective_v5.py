from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v5_model import OmniCollectiveEngineV5, OmniCollectiveNetV5
    from train_image_recognition_model import ensure_base_images
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from train_omni_collective_v4 import (
        DEFAULT_QWEN_ADAPTER_ZIP,
        _build_large_rows_v4,
        _load_expanded_state_from_zip,
        _teacher_league_distill_rows,
        _train_stage,
    )
except ImportError:  # pragma: no cover
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v5_model import OmniCollectiveEngineV5, OmniCollectiveNetV5
    from .train_image_recognition_model import ensure_base_images
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from .train_omni_collective_v4 import (
        DEFAULT_QWEN_ADAPTER_ZIP,
        _build_large_rows_v4,
        _load_expanded_state_from_zip,
        _teacher_league_distill_rows,
        _train_stage,
    )


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v4_frontier_20260329.zip")


def _coding_delta_rows_v5() -> List[OmniRow]:
    items = [
        ("How do I debug a Python traceback that ends with KeyError on 'id'?", "coding", "Check the dictionary or JSON payload before reading 'id', print the incoming keys, and guard with `.get('id')` only if a missing key is acceptable."),
        ("My async JavaScript function returns before fetch data is ready. What should I change?", "coding", "Return or await the fetch promise inside the async function so callers wait for the resolved data instead of an unfinished promise chain."),
        ("Why does SQL LEFT JOIN still duplicate rows?", "coding", "The joined table likely has multiple matches per key. Pre-aggregate it or join against a subquery that reduces each key to one row first."),
        ("Best way to add tests before refactoring a Python parser?", "coding", "Freeze behavior with small fixture-based tests first, then refactor in slices while rerunning the same fixtures after each change."),
        ("A regex is too greedy and eats the rest of the line. What should I use?", "coding", "Replace `.*` with a non-greedy form like `.*?`, or anchor the pattern more tightly so it only captures the intended span."),
        ("How should I explain a segmentation of bug, fix, and regression risk in a code review?", "coding", "State the root cause, the exact fix, and the remaining regression risks separately so reviewers can validate behavior instead of reading one blended paragraph."),
        ("What is the practical use of git bisect?", "coding", "Use `git bisect` to narrow a regression by marking known good and bad commits so Git binary-searches the history for the first broken change."),
        ("Why does my PowerShell script fail on paths with spaces?", "coding", "Quote the path, prefer `-LiteralPath` when available, and avoid building shell strings when you can pass the path as a direct argument."),
        ("How do I fix Python circular imports without hiding the structure?", "coding", "Move shared types or helpers into a third module, or delay one import inside the function that actually needs it."),
        ("My REST API pagination keeps skipping rows when data changes. What is safer?", "coding", "Use cursor or keyset pagination based on a stable ordered column instead of offset pagination when the dataset is changing during reads."),
        ("What should a minimal code review answer say when a function has too many responsibilities?", "coding", "Recommend splitting the function by responsibility, name the current mixed concerns, and point to the first extraction boundary instead of giving vague style feedback."),
        ("How do I explain a memory leak from event listeners in simple terms?", "coding", "The code keeps references alive because listeners are attached but never removed, so objects that should be released stay reachable."),
        ("When should I use Python dataclasses over plain dicts?", "coding", "Use dataclasses when the shape is stable and meaningful, because named fields, defaults, and type hints make state changes easier to reason about than ad hoc dict keys."),
        ("How do I make a shell command safer when arguments come from users?", "coding", "Avoid string-built shell commands, pass arguments as structured parameters, and validate or restrict user input before invoking the tool."),
    ]
    return [
        OmniRow(prompt=prompt, intent="coding", response_text=response, domain=domain, source="coding_delta_v5")
        for prompt, domain, response in items
    ]


def _openscad_rows_v5() -> List[OmniRow]:
    items = [
        (
            "Write a simple OpenSCAD example for a hollow box with 2 mm walls.",
            "module hollow_box(size=[40,30,20], wall=2){difference(){cube(size);translate([wall,wall,wall])cube([size[0]-2*wall,size[1]-2*wall,size[2]-2*wall]);}} hollow_box();",
        ),
        (
            "How do I model a rounded plate with four mounting holes in OpenSCAD?",
            "Use `minkowski()` or an offset-style hull for the rounded plate, then subtract four cylinders at the mounting coordinates with `difference()`.",
        ),
        (
            "Give me an OpenSCAD example for a parametric phone stand.",
            "angle=68; thickness=4; module stand(){rotate([0,0,0])linear_extrude(height=thickness)polygon([[0,0],[80,0],[48,55],[18,55]]);} stand();",
        ),
        (
            "What is the difference between union, difference, and intersection in OpenSCAD?",
            "`union()` combines solids, `difference()` subtracts later solids from the first, and `intersection()` keeps only the overlapping volume shared by the solids.",
        ),
        (
            "Show a concise OpenSCAD snippet for a cylinder with a centered hole.",
            "difference(){cylinder(h=20, r=12, $fn=64); translate([0,0,-1]) cylinder(h=22, r=4, $fn=48);}",
        ),
        (
            "How do I repeat spokes evenly around a circle in OpenSCAD?",
            "Loop over angles with `for(angle=[0:30:330]) rotate([0,0,angle]) translate([radius,0,0])` and place the spoke geometry inside that transform chain.",
        ),
        (
            "Write a tiny OpenSCAD module for a countersunk screw hole.",
            "module countersunk_hole(depth=8, shaft_r=1.7, head_r=3.6, head_h=2){cylinder(h=depth, r=shaft_r, $fn=36); cylinder(h=head_h, r1=head_r, r2=shaft_r, $fn=36);} countersunk_hole();",
        ),
        (
            "Best practice for keeping OpenSCAD models editable?",
            "Use small reusable modules, expose the key dimensions as parameters, and separate primitive generation from transforms so edits stay local and predictable.",
        ),
        (
            "How do I mirror a feature onto both sides of a part in OpenSCAD?",
            "Wrap the feature in a module and call it twice, once normally and once with `mirror([1,0,0])` or the axis that matches your symmetry plane.",
        ),
        (
            "Give an OpenSCAD example for a gear-like decorative star without importing libraries.",
            "points=[for(i=[0:11]) let(r=(i%2==0)?24:12,a=i*30) [r*cos(a), r*sin(a)]]; linear_extrude(height=4) polygon(points);",
        ),
    ]
    return [OmniRow(prompt=prompt, intent="coding", response_text=_normalize(response, 420), domain="coding", source="openscad_v5") for prompt, response in items]


def _prompt_understanding_rows_v5() -> List[OmniRow]:
    items = [
        ("Reply with exactly three bullet points explaining why tests fail intermittently.", "- Hidden shared state can leak between tests.\n- Time or network dependencies can make outcomes nondeterministic.\n- Tests may rely on ordering that is not guaranteed.", "planning"),
        ("Give me one sentence only: why should I use keyset pagination?", "Keyset pagination is safer for changing datasets because it follows a stable cursor instead of skipping or duplicating rows with shifting offsets.", "knowledge"),
        ("Answer in JSON with keys cause and fix for a Python import error.", "{\"cause\":\"The package or module path is missing or mismatched.\",\"fix\":\"Install the package in the active environment and verify the interpreter and import path are aligned.\"}", "coding"),
        ("Explain OpenSCAD difference() in plain language, no code block.", "It cuts later shapes out of the first shape, so you can make holes, cavities, and trimmed parts without rewriting the base geometry.", "coding"),
        ("Summarize why a user might want command prompt access in two short clauses.", "They may want to inspect the system quickly; they may also need to run manual tools outside the chat UI.", "general"),
        ("Give a concise answer and do not mention unrelated topics: how do I rotate a part in OpenSCAD?", "Use `rotate([x,y,z])` around the part or the module that builds it, then translate afterward if you need to reposition the rotated result.", "coding"),
        ("I asked for coding help, not current news. How should the model respond?", "It should stay on the coding task, answer the technical request directly, and avoid drifting into current-events or unrelated canned advice.", "coding"),
        ("Return a numbered list with two steps for debugging a failing API call.", "1. Inspect the exact request, status code, and response body.\n2. Reproduce the call with known-good inputs and compare headers, auth, and payload shape.", "planning"),
    ]
    return [OmniRow(prompt=prompt, intent="general" if domain == "general" else ("coding" if domain == "coding" else "planning"), response_text=response, domain=domain if domain in OMNI_DOMAIN_LABELS_V2 else "general", source="prompt_understanding_v5") for prompt, response, domain in items]


def _build_rows_v5(
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    seed: int,
    allowed_model_keys: Optional[Sequence[str]] = None,
) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows, source_counts = _build_large_rows_v4(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
        allowed_model_keys=allowed_model_keys,
    )
    coding_path = repo_root / "datasets" / "conversation_data.coding_knowledge_2026_02_19.jsonl"
    if coding_path.exists():
        extra_coding = _rows_from_jsonl(coding_path, limit=180, seed=seed + 401, domain="coding", source_tag="coding_delta_jsonl_v5")
        rows.extend(extra_coding)
        source_counts["coding_delta_jsonl_v5"] = len(extra_coding)
    coding_rows = _coding_delta_rows_v5()
    openscad_rows = _openscad_rows_v5()
    prompt_rows = _prompt_understanding_rows_v5()
    rows.extend(coding_rows)
    rows.extend(openscad_rows)
    rows.extend(prompt_rows)
    source_counts["coding_delta_v5"] = len(coding_rows)
    source_counts["openscad_v5"] = len(openscad_rows)
    source_counts["prompt_understanding_v5"] = len(prompt_rows)
    return rows, dict(sorted(source_counts.items()))


def build_training_rows(*, repo_root: Path, models_dir: Path, images_dir: Path, adapter_zip: Path, seed: int, qwen_distill_limit: int) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    full_rows, source_counts = _build_rows_v5(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, seed=seed)
    qwen_rows, qwen_summary = _teacher_league_distill_rows(full_rows, adapter_zip=adapter_zip, seed=seed + 311, limit=qwen_distill_limit)
    full_rows.extend(qwen_rows)
    source_counts["qwen_teacher_league_total"] = len(qwen_rows)
    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    return stage1_rows, list(full_rows), {"stage1_rows": len(stage1_rows), "stage2_rows": len(full_rows), "source_counts": source_counts, "teacher_league": qwen_summary}


def train_model(*, repo_root: Path, output_dir: Path, models_dir: Path, base_zip: Path, adapter_zip: Path, images_dir: Path, image_size: int, batch_size: int, stage1_epochs: int, stage2_epochs: int, stage1_lr: float, stage2_lr: float, seed: int, qwen_distill_limit: int) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    _stage1_rows, full_rows, dataset_summary = build_training_rows(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, adapter_zip=adapter_zip, seed=seed, qwen_distill_limit=qwen_distill_limit)
    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    train_rows, val_rows = split_rows(full_rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    vocab = build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in full_rows if row.response_text})
    print(json.dumps({"event": "label_space", "train_rows": len(train_rows), "val_rows": len(val_rows), "response_bank": len(response_bank), "vocab_size": len(vocab)}, ensure_ascii=True), flush=True)

    max_len = 320
    max_words = 72
    word_buckets = 16384
    device = torch.device("cpu")
    model = OmniCollectiveNetV5(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=96,
        text_hidden=184,
        image_channels=40,
        word_buckets=word_buckets,
        word_embed_dim=84,
        deep_text_channels=224,
        deep_image_channels=76,
        fusion_hidden=640,
        memory_slots=14,
        depth_steps=6,
        expert_count=5,
        expert_hidden=1024,
        context_top_k=3,
        expert_top_k=2,
    ).to(device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)

    stage1 = _train_stage(
        model=model,
        train_rows=train_stage1,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=batch_size,
        learning_rate=stage1_lr,
        epochs=stage1_epochs,
        seed=seed + 101,
        device=device,
        loss_weights={"intent": 0.60, "response": 1.00, "domain": 0.74, "vision": 0.72},
        balance_weight=0.028,
    )
    stage2 = _train_stage(
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=max(16, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        device=device,
        loss_weights={"intent": 0.52, "response": 1.00, "domain": 0.70, "vision": 1.08},
        balance_weight=0.035,
    )

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v5_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v5_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v5_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v5_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v5_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "architecture_version": 5,
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 96,
        "text_hidden": 184,
        "image_channels": 40,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 84,
        "deep_text_channels": 224,
        "deep_image_channels": 76,
        "fusion_hidden": 640,
        "memory_slots": 14,
        "depth_steps": 6,
        "expert_count": 5,
        "expert_hidden": 1024,
        "context_top_k": 3,
        "expert_top_k": 2,
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
        "deliberation_passes": 4,
        "prompt_understanding_mode": "arq_consensus_repair",
        "notes": [
            "v5 continues from v4 with extra coding, OpenSCAD, and prompt-understanding rows.",
            "Inference uses prompt-variant consensus and a repair-style pass so every prompt gets more internal compute before answer selection.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    engine = OmniCollectiveEngineV5(weights_path=weights_path, meta_path=meta_path, device=device)
    sample_prompts = [
        "Debug this Python traceback and explain the likely root cause.",
        "Write a simple OpenSCAD example for a hollow box with 2 mm walls.",
        "Reply with exactly three bullet points explaining why tests fail intermittently.",
        "Make a photorealistic image prompt for a storm-battered lighthouse.",
    ]
    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "sample_outputs": [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts],
        "notes": [
            "v5 grows the v4 network slightly and adds a deliberative consensus inference path for better prompt understanding.",
            "The continuation adds small coding and OpenSCAD deltas rather than rebalancing the whole recipe around code.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    return {
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path),
        "artifact_dir": str(artifact_dir),
        "parameter_count": parameter_count,
        "stage1_val": stage1["val_metrics"],
        "stage2_val": stage2["val_metrics"],
        "warm_start": warm_start,
        "dataset_summary": dataset_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the omni_collective_v5 continuation model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--qwen_adapter_zip", default=str(DEFAULT_QWEN_ADAPTER_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=0.00042)
    parser.add_argument("--stage2_lr", type=float, default=0.00020)
    parser.add_argument("--seed", type=int, default=173)
    parser.add_argument("--qwen_distill_limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        base_zip=Path(args.base_zip).resolve(),
        adapter_zip=Path(args.qwen_adapter_zip).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        seed=int(args.seed),
        qwen_distill_limit=int(args.qwen_distill_limit),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
