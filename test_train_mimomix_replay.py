"""Safety and sampling contracts for bounded checkpoint-vocabulary repair."""

from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import train_mimomix_replay as replay
from mimomix_core import MiMoMixConfig, MiMoMixModel
from mimomix_text import WordTokenizer
from train_mimomix_talk import load_talk_checkpoint, save_talk_checkpoint


def row(task, prompt, group):
    return {"task": task, "user": prompt, "response": "total 3", "group_id": group}


def test_partition_rejects_semantic_leakage_and_relabelled_prompt():
    train = [row("math", "1 + 2", "a")]
    with pytest.raises(ValueError, match="overlap"):
        replay.validate_partitions(train, [row("math", "2 + 1", "a")])
    with pytest.raises(ValueError, match="overlap"):
        replay.validate_partitions(train, [row("other", "1 + 2", "b")])
    with pytest.raises(ValueError, match="every training task"):
        replay.validate_partitions(train, [row("other", "2 + 1", "b")])


def test_task_sampling_mass_ignores_duplicated_row_count():
    rows = [row("repair", "1 + 2", "a")] + [row("replay", "2 + 1", "b")] * 99
    probabilities, masses = replay.task_probabilities(rows, ["repair=3"])
    assert masses == {"repair": .75, "replay": .25}
    assert probabilities[0].item() == .75
    assert probabilities[1:].sum().item() == pytest.approx(.25)
    generator = torch.Generator().manual_seed(1)
    draws = torch.multinomial(probabilities, 10000, replacement=True, generator=generator)
    assert .72 < float((draws == 0).float().mean()) < .78


@pytest.mark.parametrize("weight", ["repair=nan", "repair=inf", "repair=0", "repair=-1", "absent=2"])
def test_invalid_weights_fail(weight):
    with pytest.raises(ValueError):
        replay.task_probabilities([row("repair", "1 + 2", "a")], [weight])


def test_unknown_and_overlength_turns_are_rejected_without_truncation():
    tokenizer = WordTokenizer.build(["1 + 2", "total 3"], digit_tokens=True)
    with pytest.raises(ValueError, match="unknown"):
        replay.encode_rows([row("math", "newword", "a")], tokenizer, 32)
    with pytest.raises(ValueError, match="context"):
        replay.encode_rows([row("math", "1 + 2", "a")], tokenizer, 5)
    x, y, stats = replay.encode_rows([row("math", "1 + 2", "a")], tokenizer, 32)
    _, prompt_length = tokenizer.encode_turn("1 + 2", "total 3")
    assert bool((y[0, :prompt_length] == -100).all())
    assert bool((y[0, prompt_length:stats["max_turn_tokens"]] != -100).all())
    assert bool((y[0, stats["max_turn_tokens"]:] == -100).all())
    assert x.shape == (1, 32)


def test_tiny_actual_training_preserves_checkpoint_and_tokenizer(tmp_path):
    torch.set_num_threads(1)
    tokenizer = WordTokenizer.build(["1 + 2", "2 + 1", "total 3"], digit_tokens=True)
    config = MiMoMixConfig(vocab_size=tokenizer.vocab_size, hidden_size=16, n_layers=1,
                          n_heads=2, n_kv_heads=1, head_dim=8, intermediate_size=24,
                          use_moe=False, use_thinking_core=False, n_mtp_layers=0,
                          max_position_embeddings=24, native_context=24)
    checkpoint = tmp_path / "base.pt"
    save_talk_checkpoint(checkpoint, MiMoMixModel(config), tokenizer)
    original_hash = replay.digest_file(checkpoint)
    for name, data in [("train", row("math", "1 + 2", "a")),
                       ("dev", row("math", "2 + 1", "b"))]:
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(data) + "\n")
    heldout = tmp_path / "evaluation.json"
    heldout.write_text('{"purpose": "unread heldout"}')
    args = Namespace(checkpoint=str(checkpoint), train_jsonl=str(tmp_path / "train.jsonl"),
                     dev_jsonl=str(tmp_path / "dev.jsonl"), evaluation_manifest=str(heldout),
                     output_dir=str(tmp_path / "run"), steps=2, batch_size=1,
                     eval_every=1, torch_threads=1, seed=123, lr=1e-3,
                     weight_decay=.01, task_weight=[])
    report = replay.run(args)
    assert report["status"] == "completed" and report["weights_changed"]
    assert replay.digest_file(checkpoint) == original_hash
    _, candidate_tokenizer, candidate = load_talk_checkpoint(tmp_path / "run" / "candidate.pt")
    assert candidate_tokenizer.to_dict() == tokenizer.to_dict()
    assert candidate["config"] == config.to_dict()
    assert report["selection"]["best_dev_loss"] <= report["history"][0]["dev"]["loss"]
    assert report["sampled_rows_by_task"] == {"math": 2}
    assert report["heldout_evaluated"] is False and report["pointer_written"] is False
    with pytest.raises(ValueError, match="already exists"):
        replay.run(args)
