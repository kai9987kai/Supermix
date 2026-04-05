from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.join(os.getcwd(), "source"))

from mattergen_generation_model import (
    MATTERGEN_CONCEPT_LABELS,
    MatterGenMicroNet,
    build_vocab,
    choose_answer_variant,
    encode_text,
    encode_words,
    extract_property_targets,
    heuristic_mattergen_concept,
    looks_like_mattergen_prompt,
)
from train_mattergen_generation_model import build_rows, split_rows


def test_mattergen_prompt_detection_and_heuristics() -> None:
    prompt = "Generate a CIF-style perovskite absorber candidate with a 1.8 eV band gap."
    assert looks_like_mattergen_prompt(prompt)
    assert heuristic_mattergen_concept(prompt) == "perovskite_photovoltaic"


def test_mattergen_rows_cover_all_concepts() -> None:
    rows, summary = build_rows(seed=11)
    concepts = {row.concept for row in rows}
    assert concepts == set(MATTERGEN_CONCEPT_LABELS)
    assert summary["source_rows"] == len(rows)


def test_mattergen_split_keeps_validation_coverage() -> None:
    rows, _summary = build_rows(seed=19)
    train_rows, val_rows = split_rows(rows, seed=23)
    assert train_rows
    assert val_rows
    assert {row.concept for row in val_rows} == set(MATTERGEN_CONCEPT_LABELS)


def test_mattergen_network_builds_and_runs() -> None:
    prompts = [
        "Generate a solid electrolyte candidate with ionic conductivity near 1e-3 S/cm.",
        "Suggest a transparent conductor with a 3.5 eV band gap.",
    ]
    vocab = build_vocab(prompts)
    char_ids = torch.tensor([encode_text(prompt, vocab, 128) for prompt in prompts], dtype=torch.long)
    word_ids = torch.tensor([encode_words(prompt, word_buckets=128, max_words=16) for prompt in prompts], dtype=torch.long)
    cond = torch.tensor([[1.0] * 8, [0.0] * 8], dtype=torch.float32)
    model = MatterGenMicroNet(
        vocab_size=len(vocab),
        num_concepts=len(MATTERGEN_CONCEPT_LABELS),
        char_embed_dim=16,
        conv_channels=24,
        word_buckets=128,
        word_embed_dim=12,
        condition_dim=8,
        hidden_dim=32,
    )
    outputs = model(char_ids, word_ids, cond)
    assert outputs.shape == (2, len(MATTERGEN_CONCEPT_LABELS))


def test_variant_selection_and_property_extraction() -> None:
    assert choose_answer_variant("Give a CIF seed for a thermoelectric material.") == "cif_seed"
    assert choose_answer_variant("How should I validate a hard magnet candidate?") == "validation_plan"
    targets = extract_property_targets("Generate a perovskite with a 1.9 eV band gap and energy above hull below 25 meV/atom.")
    assert abs(targets["band_gap_eV"] - 1.9) < 1e-9
    assert abs(targets["energy_above_hull_meV_atom"] - 25.0) < 1e-9


if __name__ == "__main__":
    tests = [
        test_mattergen_prompt_detection_and_heuristics,
        test_mattergen_rows_cover_all_concepts,
        test_mattergen_split_keeps_validation_coverage,
        test_mattergen_network_builds_and_runs,
        test_variant_selection_and_property_extraction,
    ]
    for test in tests:
        print(f"Running {test.__name__}...")
        test()
