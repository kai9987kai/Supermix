from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.join(os.getcwd(), "source"))

from three_d_generation_model import (
    THREE_D_CONCEPT_LABELS,
    ThreeDGenerationMiniNet,
    build_vocab,
    choose_answer_variant,
    encode_text,
    encode_words,
    heuristic_3d_concept,
    looks_like_3d_generation_prompt,
)
from train_three_d_generation_model import build_rows, split_rows


def test_3d_prompt_detection_and_heuristics() -> None:
    prompt = "Write a tiny OpenSCAD snippet for a centered cylinder with a hole."
    assert looks_like_3d_generation_prompt(prompt)
    assert heuristic_3d_concept(prompt) == "cylinder_center_hole"


def test_3d_rows_cover_all_concepts() -> None:
    rows, summary, answer_bank = build_rows(seed=11)
    concepts = {row.concept for row in rows}
    assert concepts == set(THREE_D_CONCEPT_LABELS)
    assert summary["source_rows"] == len(rows)
    assert all(answer_bank.get(label) for label in THREE_D_CONCEPT_LABELS)


def test_3d_split_keeps_validation_coverage() -> None:
    rows, _summary, _answer_bank = build_rows(seed=19)
    train_rows, val_rows = split_rows(rows, seed=23)
    assert train_rows
    assert val_rows
    assert {row.concept for row in val_rows} == set(THREE_D_CONCEPT_LABELS)


def test_3d_network_builds_and_runs() -> None:
    prompts = [
        "Generate OpenSCAD for a square pyramid.",
        "How do I mirror a feature onto both sides of a part in OpenSCAD?",
    ]
    vocab = build_vocab(prompts)
    char_ids = torch.tensor([encode_text(prompt, vocab, 96) for prompt in prompts], dtype=torch.long)
    word_ids = torch.tensor([encode_words(prompt, word_buckets=128, max_words=16) for prompt in prompts], dtype=torch.long)
    model = ThreeDGenerationMiniNet(
        vocab_size=len(vocab),
        num_concepts=len(THREE_D_CONCEPT_LABELS),
        char_embed_dim=16,
        conv_channels=24,
        word_buckets=128,
        word_embed_dim=12,
        hidden_dim=32,
    )
    outputs = model(char_ids, word_ids)
    assert outputs.shape == (2, len(THREE_D_CONCEPT_LABELS))


def test_variant_selection_tracks_prompt_style() -> None:
    assert choose_answer_variant("Give a concise OpenSCAD answer for a rounded plate.") == "concise_answer"
    assert choose_answer_variant("Explain the difference between union and difference in OpenSCAD.") == "explain_then_code"
    assert choose_answer_variant("Write code only for a tetrahedron mesh.") == "code"


if __name__ == "__main__":
    tests = [
        test_3d_prompt_detection_and_heuristics,
        test_3d_rows_cover_all_concepts,
        test_3d_split_keeps_validation_coverage,
        test_3d_network_builds_and_runs,
        test_variant_selection_tracks_prompt_style,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
