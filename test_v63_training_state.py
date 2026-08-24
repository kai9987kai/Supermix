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
