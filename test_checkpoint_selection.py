"""Tests for what a training run chooses its best checkpoint on.

V64 measured the problem this exists to fix. Between step 5,500 and step 10,000
of one run, dev loss improved from 1.0762 to 0.9910 while the mean verbatim rate
of generated replies rose from 0.14 to 0.76 and degenerate replies doubled.
Selecting on dev loss picked the worse model, and it will always do so, because
verbatim reproduction of training text is the lowest-loss behaviour available.

The default stays `dev_loss` so every published run reproduces. These tests pin
that default *and* prove the alternatives actually change the choice -- a
criterion that always picked the same checkpoint would be decoration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import train_mimomix_generalisation as trainer  # noqa: E402


# The v64 observation, as two candidate checkpoints.
EARLY = {"dev_loss": 1.0762, "verbatim": 0.14}
LATE = {"dev_loss": 0.9910, "verbatim": 0.76}


def _score(criterion, candidate):
    return trainer.selection_score(criterion, candidate["dev_loss"], candidate["verbatim"])


def test_dev_loss_prefers_the_more_memorised_checkpoint():
    """The documented failure, pinned so it cannot be mistaken for a bug later."""

    assert _score("dev_loss", LATE) < _score("dev_loss", EARLY)


def test_novelty_prefers_the_less_memorised_checkpoint():
    assert _score("novelty", EARLY) < _score("novelty", LATE)


def test_balanced_prefers_the_less_memorised_checkpoint_on_v64_numbers():
    """0.0852 nats of dev improvement does not buy 0.62 of extra recitation."""

    assert _score("balanced", EARLY) < _score("balanced", LATE)


def test_balanced_still_prefers_a_genuinely_better_checkpoint():
    """It must not become a novelty criterion in disguise."""

    much_better = {"dev_loss": 0.20, "verbatim": 0.30}
    barely_novel = {"dev_loss": 1.00, "verbatim": 0.25}

    assert _score("balanced", much_better) < _score("balanced", barely_novel)


def test_dev_loss_is_the_default():
    parser = trainer.build_parser()
    args = parser.parse_args([])

    assert args.select_on == "dev_loss"


def test_missing_verbatim_falls_back_to_dev_loss():
    """No corpus index means no measurement; it must not score as 'novel'."""

    for criterion in ("dev_loss", "novelty", "balanced"):
        assert trainer.selection_score(criterion, 0.5, None) == 0.5


def test_unknown_criterion_raises():
    with pytest.raises(ValueError, match="unknown selection criterion"):
        trainer.selection_score("vibes", 0.5, 0.1)


def test_lower_is_better_for_every_criterion():
    """The loop compares with `<`; a criterion where higher was better would
    silently invert selection."""

    worse = {"dev_loss": 2.0, "verbatim": 0.9}
    better = {"dev_loss": 0.2, "verbatim": 0.1}
    for criterion in ("dev_loss", "novelty", "balanced"):
        assert _score(criterion, better) < _score(criterion, worse)


def test_probe_verbatim_rate_returns_none_without_an_index():
    """'not measured' must be distinguishable from 'nothing was recited'."""

    assert trainer.probe_verbatim_rate(object(), object(), None) is None


def test_balanced_weight_is_declared_not_buried():
    assert 0.0 < trainer.BALANCED_VERBATIM_WEIGHT <= 1.0


# -- selection guards (v74) -------------------------------------------------


class _Args:
    """Minimal stand-in for the parsed namespace the guard inspects."""

    def __init__(self, select_on="dev_loss", accuracy_every=0, accuracy_problems=20):
        self.select_on = select_on
        self.accuracy_every = accuracy_every
        self.accuracy_problems = accuracy_problems


def test_accuracy_selection_refuses_a_probe_too_small_to_select_on():
    """v73 selected on 20 problems; that probe read 0.15 where 60 read 0.467."""

    with pytest.raises(SystemExit, match="accuracy_problems >="):
        trainer.validate_selection_settings(
            _Args("accuracy", accuracy_every=2000, accuracy_problems=20)
        )


def test_accuracy_selection_refuses_when_no_probe_runs():
    with pytest.raises(SystemExit, match="accuracy_every > 0"):
        trainer.validate_selection_settings(
            _Args("accuracy", accuracy_every=0, accuracy_problems=200)
        )


def test_accuracy_selection_accepts_an_adequate_probe():
    trainer.validate_selection_settings(
        _Args("accuracy", accuracy_every=2000,
              accuracy_problems=trainer.MIN_SELECTION_PROBLEMS)
    )


@pytest.mark.parametrize("criterion", ["dev_loss", "novelty", "balanced"])
def test_other_criteria_are_not_constrained_by_the_probe_size(criterion):
    """A small probe stays legal for monitoring; only selection is guarded."""

    trainer.validate_selection_settings(
        _Args(criterion, accuracy_every=2000, accuracy_problems=20)
    )


def test_minimum_is_large_enough_to_be_worth_having():
    """At n=100 the 95% interval near p=0.5 is about +-10 points, not +-22."""

    import math

    interval = 1.96 * math.sqrt(0.25 / trainer.MIN_SELECTION_PROBLEMS) * 100

    assert interval < 12.0


def test_accuracy_criterion_prefers_higher_accuracy_over_lower_loss():
    """The whole point: a better-fitting checkpoint must not win on loss alone."""

    accurate_but_worse_loss = trainer.selection_score("accuracy", 0.90, None, 0.75)
    inaccurate_but_better_loss = trainer.selection_score("accuracy", 0.05, None, 0.40)

    assert accurate_but_worse_loss < inaccurate_but_better_loss


def test_accuracy_criterion_breaks_ties_on_loss():
    tie_better_loss = trainer.selection_score("accuracy", 0.10, None, 0.60)
    tie_worse_loss = trainer.selection_score("accuracy", 0.90, None, 0.60)

    assert tie_better_loss < tie_worse_loss


def test_accuracy_criterion_falls_back_when_unmeasured():
    """Before the first probe there is no accuracy; loss must still order runs."""

    assert trainer.selection_score("accuracy", 0.42, None, None) == 0.42


# -- reporting the criterion honestly (v75) ---------------------------------
#
# v74 selected step 18,000 on a 0.89 accuracy probe and printed
# "selected step 18000 on dev (dev loss 0.0651)". The JSON was right and the
# console line was not, which is the direction that misleads: dev loss is the
# thing v64 proved should not be trusted as a criterion.


def test_accuracy_selection_says_accuracy_and_shows_the_probe():
    line = trainer.describe_selection({
        "selected_on": "accuracy",
        "best_dev_loss": 0.065107,
        "best_probe_accuracy": 0.89,
    })

    assert line.startswith("accuracy")
    assert "0.89" in line
    assert "0.0651" in line  # the loss is still shown, as the tie-break


def test_dev_loss_selection_still_reads_as_before():
    line = trainer.describe_selection({
        "selected_on": "dev_loss",
        "best_dev_loss": 0.5,
    })

    assert line.startswith("dev_loss")
    assert "0.5000" in line


def test_novelty_selection_shows_the_verbatim_rate():
    line = trainer.describe_selection({
        "selected_on": "novelty",
        "best_dev_loss": 1.0,
        "best_probe_verbatim_rate": 0.76,
    })

    assert line.startswith("novelty")
    assert "0.76" in line


def test_balanced_selection_shows_the_verbatim_rate():
    line = trainer.describe_selection({
        "selected_on": "balanced",
        "best_dev_loss": 1.0,
        "best_probe_verbatim_rate": 0.14,
    })

    assert line.startswith("balanced")
    assert "0.14" in line


def test_accuracy_selection_without_a_probe_does_not_invent_one():
    line = trainer.describe_selection({
        "selected_on": "accuracy",
        "best_dev_loss": 0.4,
    })

    assert "probe" not in line
    assert "0.4000" in line


def test_missing_criterion_defaults_to_dev_loss():
    assert trainer.describe_selection({"best_dev_loss": 0.25}).startswith("dev_loss")


def test_unmeasured_dev_loss_is_said_not_formatted():
    line = trainer.describe_selection({"selected_on": "dev_loss"})

    assert "unmeasured" in line
