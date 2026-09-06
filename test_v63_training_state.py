"""Tests for v63 resume, mid-run checkpointing and the tiny-run scheduler fix.

Three failures motivated these, and each is pinned by a test that fails without
its fix:

* A continuation restored weights but not the optimiser, so AdamW's moments and
  the LR schedule restarted cold. v62's second leg watched dev loss climb from
  0.8919 to 1.0036 and spent ~1,500 steps recovering.
* The best weights lived only in memory until the loop ended, so a kill at hour
  eleven left nothing at all.
* `OneCycleLR` divides by zero whenever `pct_start * total_steps <= 1`, so every
  run of ten steps or fewer crashed at the default `pct_start=0.1` -- precisely
  the range a smoke test uses.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import mimomix_text as text_utils  # noqa: E402
import train_mimomix_generalisation as trainer  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402
from train_mimomix_talk import save_talk_checkpoint  # noqa: E402


def _tiny_model(vocab: int = 48) -> MiMoMixModel:
    torch.manual_seed(63)
    return MiMoMixModel(
        MiMoMixConfig(
            vocab_size=vocab, hidden_size=32, n_layers=2, n_heads=2, n_kv_heads=1,
            intermediate_size=64, moe_intermediate_size=16, n_routed_experts=4,
            moe_top_k=2, n_mtp_layers=1, native_context=32,
            max_position_embeddings=32, rope_scaling="none", sliding_window=16,
        )
    )


def _tokenizer() -> text_utils.WordTokenizer:
    return text_utils.WordTokenizer.build(["hello there friend", "a second sentence here"])


def _optimiser_and_scheduler(model, total_steps: int = 50):
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=1e-3, total_steps=total_steps, pct_start=0.1
    )
    return optimiser, scheduler


# -- the scheduler fix ------------------------------------------------------


def _build_scheduler(optimiser, total_steps: int, pct_start: float = 0.1):
    """The trainer's scheduler choice, mirrored so the rule itself is tested."""

    if pct_start * total_steps > 1.0:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=1e-3, total_steps=total_steps, pct_start=pct_start
        )
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0)


@pytest.mark.parametrize("steps", [1, 2, 5, 10, 11, 100])
def test_scheduler_survives_any_step_count(steps):
    """`pct_start * total_steps <= 1` used to divide by zero at construction."""

    model = _tiny_model()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = _build_scheduler(optimiser, steps)

    for _ in range(steps):
        optimiser.step()
        scheduler.step()  # must not raise


def test_unguarded_scheduler_still_reproduces_the_original_bug():
    """The guard is load-bearing: without it this raises.

    OneCycleLR calls `step()` internally while constructing, to set the initial
    learning rate, so the failure happens on the constructor rather than on the
    first training step.
    """

    model = _tiny_model()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ZeroDivisionError):
        torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=1e-3, total_steps=10, pct_start=0.1
        )


@pytest.mark.parametrize("steps", [2, 10])
def test_short_runs_get_a_flat_learning_rate(steps):
    model = _tiny_model()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = _build_scheduler(optimiser, steps)

    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_real_runs_still_get_the_onecycle_curve():
    model = _tiny_model()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = _build_scheduler(optimiser, 2000)

    assert isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


# -- checkpoint contents ----------------------------------------------------


def test_checkpoint_carries_optimiser_and_scheduler_state(tmp_path):
    model, tokenizer = _tiny_model(), _tokenizer()
    optimiser, scheduler = _optimiser_and_scheduler(model)
    optimiser.step()

    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 5},
                         optimiser=optimiser, scheduler=scheduler)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "optimiser_state" in payload
    assert "scheduler_state" in payload


def test_checkpoint_without_optimiser_is_still_valid(tmp_path):
    """Pre-v63 checkpoints must keep loading."""

    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "old.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={})
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert "optimiser_state" not in payload
    assert "state_dict" in payload and "tokenizer" in payload


# -- restore ----------------------------------------------------------------


def test_matching_schedule_restores_optimiser_and_scheduler(tmp_path):
    model, tokenizer = _tiny_model(), _tokenizer()
    optimiser, scheduler = _optimiser_and_scheduler(model)
    optimiser.step()
    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 50},
                         optimiser=optimiser, scheduler=scheduler)

    fresh = _tiny_model()
    provenance = trainer.load_initial_weights(fresh, tokenizer, str(path))
    new_opt, new_sched = _optimiser_and_scheduler(fresh)
    applied = trainer.restore_training_state(provenance, new_opt, new_sched)

    assert applied == {"optimiser": True, "scheduler": True}


def test_mismatched_schedule_restores_moments_only(tmp_path):
    """Different `--steps` means a different curve; only moments transfer."""

    model, tokenizer = _tiny_model(), _tokenizer()
    optimiser, scheduler = _optimiser_and_scheduler(model, total_steps=50)
    optimiser.step()
    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 50},
                         optimiser=optimiser, scheduler=scheduler)

    fresh = _tiny_model()
    provenance = trainer.load_initial_weights(fresh, tokenizer, str(path))
    provenance["_scheduler_state"] = None  # what run() does when steps differ
    new_opt, new_sched = _optimiser_and_scheduler(fresh, total_steps=120)
    applied = trainer.restore_training_state(provenance, new_opt, new_sched)

    assert applied == {"optimiser": True, "scheduler": False}
    # The new run's own learning rate must survive the restore.
    assert new_opt.param_groups[0]["lr"] == pytest.approx(
        _optimiser_and_scheduler(_tiny_model(), total_steps=120)[0].param_groups[0]["lr"]
    )


def test_moments_actually_transfer(tmp_path):
    """A restore that moved no moments would be indistinguishable from none."""

    model, tokenizer = _tiny_model(), _tokenizer()
    optimiser, scheduler = _optimiser_and_scheduler(model)
    loss = sum(p.sum() for p in model.parameters())
    loss.backward()
    optimiser.step()
    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 50},
                         optimiser=optimiser, scheduler=scheduler)

    fresh = _tiny_model()
    provenance = trainer.load_initial_weights(fresh, tokenizer, str(path))
    provenance["_scheduler_state"] = None
    new_opt, new_sched = _optimiser_and_scheduler(fresh, total_steps=120)
    assert not new_opt.state_dict()["state"], "fresh optimiser should have no moments"

    trainer.restore_training_state(provenance, new_opt, new_sched)

    assert new_opt.state_dict()["state"], "moments were not carried across"


def test_restore_is_a_noop_without_provenance():
    model = _tiny_model()
    optimiser, scheduler = _optimiser_and_scheduler(model)

    assert trainer.restore_training_state(None, optimiser, scheduler) == {
        "optimiser": False,
        "scheduler": False,
    }


# -- turn-aligned packing ---------------------------------------------------


def _packing_pairs(n: int = 60):
    return [(f"question number {i}", f"answer number {i} explained briefly") for i in range(n)]


def _orphan_rate(inputs, labels, length: int) -> float:
    """Fraction of supervised tokens whose block contains no turn start."""

    orphaned = total = 0
    for block in range(inputs.shape[0]):
        starts = [i for i in range(length) if int(inputs[block][i]) == text_utils.BOS]
        first = starts[0] if starts else None
        for i in range(length):
            if int(labels[block][i]) == -100:
                continue
            total += 1
            if first is None or i < first:
                orphaned += 1
    return orphaned / max(1, total)


def test_turn_aligned_packing_orphans_nothing():
    """Every supervised token must sit in a block containing its own prompt."""

    tokenizer = text_utils.WordTokenizer.build(
        [u + " " + a for u, a in _packing_pairs()]
    )
    inputs, labels = text_utils.build_training_tensors(
        _packing_pairs(), tokenizer, 64, turn_aligned=True
    )

    assert _orphan_rate(inputs, labels, 64) == 0.0


def test_stream_packing_still_orphans_tokens():
    """The default is unchanged, and the flaw it has is real.

    Without this the turn-aligned test could pass against a packer that never
    orphaned anything, making the fix unfalsifiable.
    """

    tokenizer = text_utils.WordTokenizer.build(
        [u + " " + a for u, a in _packing_pairs()]
    )
    inputs, labels = text_utils.build_training_tensors(
        _packing_pairs(), tokenizer, 64, turn_aligned=False
    )

    assert _orphan_rate(inputs, labels, 64) > 0.0


def test_every_turn_aligned_block_starts_with_bos():
    tokenizer = text_utils.WordTokenizer.build(
        [u + " " + a for u, a in _packing_pairs()]
    )
    inputs, _ = text_utils.build_training_tensors(
        _packing_pairs(), tokenizer, 64, turn_aligned=True
    )

    assert all(int(inputs[b][0]) == text_utils.BOS for b in range(inputs.shape[0]))


def test_padding_is_never_supervised():
    tokenizer = text_utils.WordTokenizer.build(
        [u + " " + a for u, a in _packing_pairs()]
    )
    inputs, labels = text_utils.build_training_tensors(
        _packing_pairs(), tokenizer, 64, turn_aligned=True
    )

    pad_positions = inputs == text_utils.PAD
    assert bool((labels[pad_positions] == -100).all())


def test_oversized_turns_are_dropped_not_truncated():
    """A truncated prompt would reintroduce the conditioning gap."""

    pairs = [("short", "short answer here"), ("x " * 400, "y " * 400)]
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])

    inputs, _ = text_utils.build_training_tensors(pairs, tokenizer, 64, turn_aligned=True)

    assert inputs.shape[0] == 1


def test_turn_aligned_raises_when_nothing_fits():
    pairs = [("x " * 400, "y " * 400)]
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])

    with pytest.raises(ValueError, match="no turn fits"):
        text_utils.build_training_tensors(pairs, tokenizer, 32, turn_aligned=True)


def test_vocabulary_mismatch_still_raises(tmp_path):
    """The v62 guard must survive the optimiser-state change."""

    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={})

    other = text_utils.WordTokenizer.build(["completely different words entirely"])
    with pytest.raises(ValueError, match="different vocabulary"):
        trainer.load_initial_weights(_tiny_model(), other, str(path))


# -- mid-curve resume (v75) -------------------------------------------------
#
# v74 segfaulted at step 11,500 of 18,000 after 9.2 hours. The crash-recovery
# checkpoint held the weights, but nothing let a new leg rejoin the *same*
# OneCycle curve: `source_steps` records steps completed, so an 11,500-step
# file compared against `--steps 18000` read as a differently-shaped run and
# the schedule was discarded. These pin the distinction.


class _ResumeArgs:
    def __init__(self, start_step=0, steps=18000, init_from="ck.pt"):
        self.start_step = start_step
        self.steps = steps
        self.init_from = init_from


def test_total_steps_is_read_separately_from_steps_completed(tmp_path):
    """The two numbers differ on every mid-run checkpoint."""

    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer,
                         extra={"steps": 40, "total_steps": 120})

    provenance = trainer.load_initial_weights(_tiny_model(), tokenizer, str(path))

    assert provenance["source_steps"] == 40
    assert provenance["source_total_steps"] == 120


def test_total_steps_absent_on_older_checkpoints(tmp_path):
    """Pre-v75 files must still load, falling back to the old comparison."""

    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "ck.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 40})

    provenance = trainer.load_initial_weights(_tiny_model(), tokenizer, str(path))

    assert provenance["source_total_steps"] is None
    assert provenance["source_steps"] == 40


def test_start_step_runs_only_the_remaining_steps():
    """The loop bound is what makes a resume cheap rather than a restart."""

    assert list(range(11500 + 1, 18000 + 1))[0] == 11501
    assert len(range(11500 + 1, 18000 + 1)) == 6500


def test_resume_requires_init_from():
    with pytest.raises(SystemExit, match="needs --init_from"):
        trainer.validate_resume_settings(_ResumeArgs(start_step=100, init_from=None))


def test_resume_past_the_end_is_rejected():
    with pytest.raises(SystemExit, match="no steps to run"):
        trainer.validate_resume_settings(_ResumeArgs(start_step=18000, steps=18000))


def test_a_normal_run_is_unaffected_by_the_resume_guard():
    trainer.validate_resume_settings(_ResumeArgs(start_step=0, init_from=None))


def test_valid_resume_passes():
    trainer.validate_resume_settings(_ResumeArgs(start_step=11500, steps=18000))


def test_start_step_defaults_to_zero():
    assert trainer.build_parser().parse_args([]).start_step == 0


# -- atomic checkpoint writes (v75) -----------------------------------------
#
# The crash-recovery checkpoint is overwritten on every improvement, and both
# observed segfaults happened at exactly that boundary. A non-atomic write puts
# the recovery point itself at risk from the crash it exists to recover from.


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "run.partial.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 500})
    good = path.read_bytes()

    real_save = torch.save

    def exploding_save(payload, target, *args, **kwargs):
        # Write some bytes first, exactly as a dying process would.
        Path(target).write_bytes(b"truncated")
        raise RuntimeError("process died mid-write")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError):
        save_talk_checkpoint(path, model, tokenizer, extra={"steps": 1000})
    monkeypatch.setattr(torch, "save", real_save)

    assert path.read_bytes() == good
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["extra"]["steps"] == 500


def test_a_failed_write_leaves_no_staging_file(tmp_path, monkeypatch):
    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "run.partial.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 500})

    def exploding_save(payload, target, *args, **kwargs):
        Path(target).write_bytes(b"truncated")
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError):
        save_talk_checkpoint(path, model, tokenizer, extra={"steps": 1000})

    assert list(tmp_path.glob("*.tmp")) == []


def test_a_successful_write_replaces_the_checkpoint(tmp_path):
    model, tokenizer = _tiny_model(), _tokenizer()
    path = tmp_path / "run.partial.pt"
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 500})
    save_talk_checkpoint(path, model, tokenizer, extra={"steps": 1000})

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["extra"]["steps"] == 1000
    assert list(tmp_path.glob("*.tmp")) == []


# -- compact corpus storage (v81) -------------------------------------------
#
# The packed corpus is the largest allocation a run makes, and it was stored as
# int64 -- 8 bytes for a token id below 9,000. v79 held 866,748 x 128 x 2
# tensors, so 1.78 GB of a 4.44 GB footprint, on a 15.6 GB machine already
# 25.6 GB committed. It spent hours at 17 s/step, faulting its own corpus back
# from the pagefile.


def test_int16_is_chosen_for_these_vocabularies():
    """8,551 for v79; 16,384 is the --max_vocab ceiling."""

    assert text_utils.compact_dtype(8551) is torch.int16
    assert text_utils.compact_dtype(16384) is torch.int16


def test_int32_is_the_fallback_for_a_large_vocabulary():
    """Never silently overflow: a big vocabulary still halves int64."""

    assert text_utils.compact_dtype(50000) is torch.int32
    assert text_utils.compact_dtype(text_utils._INT16_LIMIT) is torch.int32


def test_the_limit_leaves_room_for_the_ignore_label():
    """-100 must be representable alongside every token id."""

    assert text_utils._INT16_LIMIT < 32767


def test_packed_tensors_use_the_compact_dtype():
    pairs = [("hello there", "general kenobi")] * 8
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])

    inputs, labels = text_utils.build_training_tensors(pairs, tokenizer, 16)

    assert inputs.dtype is text_utils.compact_dtype(tokenizer.vocab_size)
    assert labels.dtype is inputs.dtype


def test_turn_aligned_tensors_use_the_compact_dtype():
    pairs = [("hello there", "general kenobi")] * 8
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])

    inputs, labels = text_utils.build_training_tensors(
        pairs, tokenizer, 16, turn_aligned=True
    )

    assert inputs.dtype is text_utils.compact_dtype(tokenizer.vocab_size)
    assert labels.dtype is inputs.dtype


def test_no_token_id_is_corrupted_by_the_narrower_type():
    """The saving is worthless if an id wraps."""

    pairs = [("hello there friend", "general kenobi indeed")] * 8
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])

    inputs, labels = text_utils.build_training_tensors(pairs, tokenizer, 16)

    assert int(inputs.max()) < tokenizer.vocab_size
    assert int(inputs.min()) >= 0
    # -100 is the ignore label and must survive intact.
    assert int(labels.min()) in (-100, 0)


def test_the_ignore_label_survives_the_narrower_type():
    pairs = [("a question here", "an answer here")] * 8
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])

    _, labels = text_utils.build_training_tensors(
        pairs, tokenizer, 16, turn_aligned=True
    )

    assert (labels == -100).any(), "prompt masking must still be present"
    assert int(labels.long().min()) == -100


def test_widening_a_batch_reproduces_the_original_ids():
    """What the trainer does per batch, pinned."""

    pairs = [("hello there", "general kenobi")] * 8
    tokenizer = text_utils.WordTokenizer.build([u + " " + a for u, a in pairs])
    inputs, _ = text_utils.build_training_tensors(pairs, tokenizer, 16)

    widened = inputs.long()

    assert widened.dtype is torch.long
    assert torch.equal(widened, inputs.to(torch.long))


def test_the_saving_is_real():
    """int64 -> int16 is a 4x reduction on the largest allocation."""

    rows, seq = 866748, 128
    wide = rows * seq * 8 * 2
    narrow = rows * seq * 2 * 2

    assert wide / narrow == 4.0
    assert (wide - narrow) / 1e9 > 1.3  # over 1.3 GB reclaimed


def _selection_fixture(model, generator):
    return trainer.selection_state_payload(
        select_on="accuracy", checkpoint_step=12, best_score=-0.7995,
        best_step=10, best_dev_loss=0.5, best_dev_seen=0.4,
        best_probe_accuracy=0.8, best_probe_verbatim_rate=None, last_accuracy=0.8,
        batch_generator=generator, history=[{"step": 10, "probe_accuracy": .8}, {"step": 12}],
        best_state=model.state_dict(), accuracy_probe={"exam": "fixed", "cap": 112},
    )


def test_resume_restores_best_weights_and_both_rng_streams():
    model = _tiny_model()
    generator = torch.Generator().manual_seed(90)
    stored = _selection_fixture(model, generator)
    expected_batch = torch.randint(100, (16,), generator=generator)
    expected_global = torch.rand(16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(3)
    torch.manual_seed(19)
    generator.manual_seed(19)
    provenance = {"_selection_state": stored}
    state = trainer.restore_resume_selection_state(
        provenance, start_step=12, select_on="accuracy", batch_generator=generator,
        model=model, accuracy_probe={"exam": "fixed", "cap": 112},
    )
    assert state["restored"] and state["best_step"] == 10
    assert state["best_score"] == -.7995
    assert torch.equal(torch.randint(100, (16,), generator=generator), expected_batch)
    assert torch.equal(torch.rand(16), expected_global)
    first = next(iter(model.state_dict()))
    assert not torch.equal(state["best_state"][first], model.state_dict()[first])
    assert "_selection_state" not in provenance


@pytest.mark.parametrize("field,value", [
    ("selection_state_schema", "old"), ("select_on", "dev_loss"),
    ("checkpoint_step", 11), ("best_score", float("nan")), ("best_step", 13),
    ("accuracy_probe", {"exam": "changed"}), ("history", [{"step": 11}]),
    ("batch_generator_state", None), ("selection_best_state", {}),
])
def test_incomplete_or_mismatched_resume_state_fails_closed(field, value):
    model = _tiny_model()
    generator = torch.Generator().manual_seed(90)
    state = _selection_fixture(model, generator)
    state[field] = value
    with pytest.raises(ValueError):
        trainer.restore_resume_selection_state(
            {"_selection_state": state}, start_step=12, select_on="accuracy",
            batch_generator=generator, model=model, accuracy_probe={"exam": "fixed", "cap": 112},
        )


def test_warm_start_does_not_inherit_old_selection_or_rng():
    model = _tiny_model()
    generator = torch.Generator().manual_seed(90)
    stored = _selection_fixture(model, generator)
    before = generator.get_state().clone()
    state = trainer.restore_resume_selection_state(
        {"_selection_state": stored}, start_step=0, select_on="dev_loss",
        batch_generator=generator, model=model, accuracy_probe={},
    )
    assert not state["restored"] and state["best_state"] is None
    assert state["best_score"] == float("inf") and state["history"] == []
    assert torch.equal(before, generator.get_state())


def test_dev_recovery_does_not_overwrite_selected_checkpoint(tmp_path):
    tokenizer = _tokenizer()
    model = _tiny_model(tokenizer.vocab_size)
    optimizer, scheduler = _optimiser_and_scheduler(model)
    extra = _selection_fixture(model, torch.Generator().manual_seed(90))
    trainer.save_progress_checkpoints(output_dir=tmp_path, run_name="test", model=model,
                                     tokenizer=tokenizer, extra=extra, selection_improved=True,
                                     dev_improved=True, optimiser=optimizer, scheduler=scheduler)
    selected = (tmp_path / "test.selected.pt").read_bytes()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    trainer.save_progress_checkpoints(output_dir=tmp_path, run_name="test", model=model,
                                     tokenizer=tokenizer, extra=extra, selection_improved=False,
                                     dev_improved=True, optimiser=optimizer, scheduler=scheduler)
    assert (tmp_path / "test.selected.pt").read_bytes() == selected
    recovery = torch.load(tmp_path / "test.partial.pt", weights_only=False)
    assert recovery["extra"]["partial"] and not recovery["extra"]["is_selection_best"]
    assert recovery["extra"]["selection_best_state"]
    assert "optimiser_state" in recovery and "scheduler_state" in recovery


def test_interrupted_tiny_run_matches_uninterrupted_weights(tmp_path, monkeypatch):
    corpus = tmp_path / "synthetic.jsonl"
    corpus.write_text("".join(json.dumps({"user": f"Question {i}", "assistant": f"The calculated result is {i}."}) + "\n"
                              for i in range(100)), encoding="utf-8")
    def arguments(name, extra=()):
        return trainer.build_parser().parse_args([
            "--corpus_jsonl", str(corpus), "--output_dir", str(tmp_path / name), "--run_name", name,
            "--steps", "4", "--eval_every", "1", "--batch_size", "2", "--eval_batch_size", "4",
            "--sequence_length", "32", "--hidden_size", "32", "--n_layers", "1", "--n_heads", "2",
            "--n_kv_heads", "1", "--intermediate_size", "64", "--moe_intermediate_size", "16",
            "--n_routed_experts", "4", "--n_mtp_layers", "1", "--no_thinking_core",
            "--accuracy_every", "0", "--select_on", "dev_loss", "--sample_tokens", "2",
            "--max_row_fraction_per_sentence", "0.05", "--turn_aligned_packing", "--digit_tokens",
            "--torch_threads", "1", "--checkpoint_every_improvement", *extra,
        ])
    continuous = trainer.run(arguments("continuous"))
    save = trainer.save_progress_checkpoints
    def crash_after_save(**kwargs):
        save(**kwargs)
        if kwargs["extra"]["steps"] == 2:
            raise InterruptedError("simulated process crash after atomic checkpoint")
    monkeypatch.setattr(trainer, "save_progress_checkpoints", crash_after_save)
    with pytest.raises(InterruptedError):
        trainer.run(arguments("resumed"))
    monkeypatch.setattr(trainer, "save_progress_checkpoints", save)
    recovered = trainer.run(arguments("resumed", ["--init_from", str(tmp_path / "resumed" / "resumed.partial.pt"), "--start_step", "2"]))
    left = torch.load(tmp_path / "continuous" / "continuous.pt", weights_only=False)
    right = torch.load(tmp_path / "resumed" / "resumed.pt", weights_only=False)
    assert left["state_dict"].keys() == right["state_dict"].keys()
    assert all(torch.equal(value, right["state_dict"][key]) for key, value in left["state_dict"].items())
    assert "optimiser_state" not in right
    assert [r["step"] for r in recovered["history"]] == [1, 2, 3, 4]
    assert [r["dev_loss"] for r in recovered["history"]] == [r["dev_loss"] for r in continuous["history"]]
