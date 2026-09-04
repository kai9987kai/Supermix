"""v85 release contract.

Four properties that are cheap to assert and expensive to lose. Each pins a
mistake that has actually happened in this repository rather than a hypothetical
one.

1. ``v80``'s recorded command line must still rebuild ``v80``'s stored config.
   v85 gave a flag to 29 previously unreachable ``MiMoMixConfig`` fields (15 of
   51 were reachable before, 52 of 64 after); the whole change is additive only
   if the shipped run still resolves to exactly the config it shipped with.

2. The probe token budget guard must name a task whose replies cannot finish
   inside the cap. The mid-run probe capped generation at 64 tokens while every
   ``arithmetic_series`` reply is 81-84 tokens, so ``--select_on accuracy`` read
   0.00 on that task whatever the model learned. That is the same failure as
   V67 losing the ``average`` rows.

3. Every ``test_*.py`` at the repository root must be referenced by the CI
   workflow, or be listed here with a reason. CI had drifted to the point of
   running none of the 13 NexusMind suites that the README calls the focused
   suite.

4. Telemetry and receipts must report what happened, not what was requested or
   what a default left behind. An adversarial review of v85 found three numbers
   that read as a measurement of one thing while measuring another: a sink mass
   averaged over layers that have no sink, a decode-time diagnostic computed
   only during training, and a corpus flag recorded as applied on a path that
   never applies it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

WORKFLOW = ROOT / ".github" / "workflows" / "runtime-quality-gates.yml"
V80_RUN = ROOT / "output" / "v80_omni" / "supervised_run.json"
V80_CHECKPOINT = ROOT / "output" / "v80_omni" / "v80_omni.pt"


# ---------------------------------------------------------------------------
# 1. The shipped run must still resolve to the config it shipped with
# ---------------------------------------------------------------------------

#: Fields added after v80 was trained. Each must be absent from the stored v80
#: config AND must default to the pre-v85 behaviour, which is what makes the
#: wiring additive. Adding a field here without a behaviour-preserving default
#: is the mistake this list exists to make visible.
V85_ADDED_FIELDS = {
    "attention_output_gate": False,
    "attention_sink_kinds": "all",
    "differential_lambda_init": 0.8,
    "differential_noise_ratio": 1,
    "differential_output_norm": True,
    "global_layers": None,
    "mla_global_only": True,
    "mla_latent_dim": 32,
    "mla_pe_dim": 16,
    "mod_capacity_ratio": 0.5,
    "mod_causal_predictor": True,
    "mod_predictor_loss_coef": 0.01,
    "moe_balance_scope": "batch",
    "qk_norm": False,
    "rotary_dim": None,
    "use_differential_attention": False,
    "use_mla": False,
    "use_mod": False,
}


@pytest.mark.skipif(not V80_RUN.exists() or not V80_CHECKPOINT.exists(),
                    reason="v80 artifacts are not present in this checkout")
def test_v80_command_line_still_rebuilds_v80_config():
    import torch

    import train_mimomix_generalisation as generalisation
    from train_mimomix_talk import build_config

    recorded = json.loads(V80_RUN.read_text(encoding="utf-8"))["train_args"]
    args = generalisation.build_parser().parse_args(recorded)

    payload = torch.load(V80_CHECKPOINT, map_location="cpu", weights_only=False)
    stored = payload["config"]
    rebuilt = build_config(args, stored["vocab_size"]).to_dict()

    differing = {
        key: (stored[key], rebuilt[key])
        for key in stored
        if key in rebuilt and stored[key] != rebuilt[key]
    }
    assert not differing, (
        "v80's own command line no longer rebuilds v80's config. Every entry "
        f"below would silently change the shipped run: {differing}"
    )

    dropped = sorted(set(stored) - set(rebuilt))
    assert not dropped, f"config fields disappeared since v80: {dropped}"


@pytest.mark.skipif(not V80_CHECKPOINT.exists(),
                    reason="v80 artifacts are not present in this checkout")
def test_fields_added_since_v80_default_to_v80_behaviour():
    import torch

    from mimomix_core import MiMoMixConfig

    stored = torch.load(V80_CHECKPOINT, map_location="cpu",
                        weights_only=False)["config"]
    fresh = MiMoMixConfig(vocab_size=stored["vocab_size"]).to_dict()

    for field, expected_default in V85_ADDED_FIELDS.items():
        assert field not in stored, (
            f"{field} is recorded in the v80 checkpoint, so it is not a v85 "
            "addition and does not belong in V85_ADDED_FIELDS"
        )
        assert field in fresh, f"{field} vanished from MiMoMixConfig"
        assert fresh[field] == expected_default, (
            f"{field} defaults to {fresh[field]!r}, not {expected_default!r}. "
            "A new field whose default changes behaviour makes v80 "
            "irreproducible and every arm run against it unpaired."
        )


# ---------------------------------------------------------------------------
# 2. The probe must refuse to score a task it cannot see
# ---------------------------------------------------------------------------

#: Measured with the v80 tokenizer over the post-c7041897 generators.
MEASURED_REPLY_TOKENS = {
    "arithmetic_series": {"median": 81, "p95": 84, "max": 84},
    "combination": {"median": 60, "p95": 65, "max": 65},
    "kinetic_energy": {"median": 52, "p95": 56, "max": 56},
    "force": {"median": 34, "p95": 34, "max": 34},
}


def test_probe_budget_guard_names_the_blind_task_at_the_old_cap():
    from train_mimomix_talk import check_probe_token_budget

    result = check_probe_token_budget(MEASURED_REPLY_TOKENS, cap=64)

    assert result["ok"] is False
    assert "arithmetic_series" in result["tasks_truncated_at_median"], (
        "the guard must name arithmetic_series as blind at a 64-token cap; "
        "its replies are 81 tokens at the median"
    )
    assert "combination" in result["tasks_truncated_at_p95"]


def test_probe_budget_guard_passes_at_the_v85_default():
    from train_mimomix_talk import (DEFAULT_PROBE_MAX_NEW_TOKENS,
                                    check_probe_token_budget)

    result = check_probe_token_budget(MEASURED_REPLY_TOKENS,
                                      cap=DEFAULT_PROBE_MAX_NEW_TOKENS)
    assert result["ok"] is True, (
        f"the shipped default {DEFAULT_PROBE_MAX_NEW_TOKENS} must clear every "
        f"task's median reply length: {result['tasks_truncated_at_median']}"
    )


def test_strict_mode_refuses_a_blind_cap_before_the_run_starts():
    from train_mimomix_talk import check_probe_token_budget

    with pytest.raises(SystemExit) as excinfo:
        check_probe_token_budget(MEASURED_REPLY_TOKENS, cap=64, strict=True)
    assert "arithmetic_series" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. CI must reference every root test file
# ---------------------------------------------------------------------------

#: Suites that must always be in CI because a published claim rests on them.
#: Unlike the ratchet below, this list is absolute: dropping any of these is a
#: failure however good overall coverage looks.
CI_REQUIRED = [
    # The README calls these the focused suite. CI ran none of them until v85.
    "test_nexus_proof.py",
    "test_nexus_epistemics.py",
    "test_nexus_engine.py",
    "test_nexus_api.py",
    "test_nexus_studio_contract.py",
    # The architecture work v85 fixed four confirmed bugs in.
    "test_mimomix_core.py",
    "test_mimomix_decoding.py",
    # The benchmark and the answer checker every accuracy number comes from.
    "test_problem_solving_eval.py",
    "test_answer_check.py",
    "test_prompt_normaliser.py",
]

#: Measured on 2026-09-02: 76 of 183 root test files were referenced before
#: v85, 80 after its first architecture/v82/release additions, and 87 after the
#: existing v83/v84 multimodal, compare, quantum, resonance, and Studio suites
#: were restored to the gate. 96 files remain outside CI; that is the honest
#: state, and this ratchet prevents silently losing the recovered coverage.
#:
#: This is a ratchet, not a target. It fails when coverage drops and never
#: demands it be perfect. Raise it when you add suites; never lower it without
#: saying why in the commit.
CI_COVERAGE_FLOOR = 87


def _root_test_files():
    return sorted(p.name for p in ROOT.glob("test_*.py"))


def _referenced_by_ci():
    # The lookbehind matters: without it "dark_test_audit.py" also matches as
    # "test_audit.py" and the workflow appears to run a file that never existed.
    return set(re.findall(r"(?<![A-Za-z0-9_])test_[A-Za-z0-9_]+\.py",
                          WORKFLOW.read_text(encoding="utf-8")))


@pytest.mark.skipif(not WORKFLOW.exists(), reason="workflow file absent")
def test_ci_runs_the_suites_published_claims_rest_on():
    referenced = _referenced_by_ci()
    missing = [name for name in CI_REQUIRED if name not in referenced]
    assert not missing, (
        f"CI does not run {missing}. Until v85 the workflow ran none of the 13 "
        "NexusMind suites while the README called them the focused suite, so a "
        "green CI meant less than it appeared to."
    )


@pytest.mark.skipif(not WORKFLOW.exists(), reason="workflow file absent")
def test_ci_coverage_does_not_regress():
    present = set(_root_test_files())
    covered = present & _referenced_by_ci()
    assert len(covered) >= CI_COVERAGE_FLOOR, (
        f"CI now references {len(covered)} of {len(present)} root test files, "
        f"below the {CI_COVERAGE_FLOOR} recorded when this ratchet was set. "
        "Add the suites back, or lower the floor deliberately and say why."
    )


@pytest.mark.skipif(not WORKFLOW.exists(), reason="workflow file absent")
def test_ci_does_not_name_test_files_that_do_not_exist():
    present = set(_root_test_files())
    phantom = sorted(name for name in _referenced_by_ci() if name not in present)
    # A workflow line naming a deleted file makes the whole pytest invocation
    # error out, so every other suite on that line stops running too.
    assert not phantom, (
        f"the workflow runs test files that do not exist: {phantom}. pytest "
        "exits on a missing path, so every suite sharing that line is skipped."
    )


@pytest.mark.skipif(not WORKFLOW.exists(), reason="workflow file absent")
def test_ci_installs_what_the_referenced_tests_import():
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    # test_nexus_hybrid_advancements.py imports starlette.testclient at module
    # scope, so without these the suite fails to import rather than skipping.
    for package in ("fastapi", "httpx", "ruff"):
        assert re.search(rf"pip install[^\n]*\b{package}\b", workflow_text), (
            f"CI references tests that need {package} but never installs it"
        )


# ---------------------------------------------------------------------------
# 4. Telemetry and receipts must not report what did not happen
#
# Three defects found by an adversarial review of v85, each a number that read
# as a measurement of one thing while measuring another.
# ---------------------------------------------------------------------------


def test_sink_mass_is_averaged_over_sink_bearing_layers_only():
    """A layer with no sink reports 0.0, which used to dilute the mean.

    Under `attention_sink_kinds="swa"` the global layers carry no sink and
    report a real-looking 0.0. Averaging those in made the telemetry show a
    large fall in sink usage when the per-layer usage had not moved, which is
    the opposite of what the metric exists to show.
    """
    import torch

    import mimomix_core as mc

    def snapshot(kinds):
        torch.manual_seed(0)
        cfg = mc.MiMoMixConfig(attention_sink_kinds=kinds)
        model = mc.MiMoMixModel(cfg)
        model.eval()
        with torch.no_grad():
            model(torch.randint(0, cfg.vocab_size, (1, 16)))
        return model.telemetry()

    every = snapshot("all")
    swa_only = snapshot("swa")

    # The default must be untouched: with every layer sink-bearing the filtered
    # mean is the unfiltered one.
    assert every["sink_bearing_layers"] == every["attention_modules"]
    assert every["mean_sink_mass"] == pytest.approx(
        every["mean_sink_mass_all_layers"], abs=1e-9
    )

    # With sinks on local layers only, fewer layers bear one...
    assert swa_only["sink_bearing_layers"] < swa_only["attention_modules"]
    # ...the diluted mean falls sharply...
    assert swa_only["mean_sink_mass_all_layers"] < every["mean_sink_mass"] * 0.8
    # ...but the honest mean does not, because per-layer usage barely moved.
    assert swa_only["mean_sink_mass"] == pytest.approx(
        every["mean_sink_mass"], rel=0.15
    )


def test_mod_predictor_agreement_is_reported_under_eval():
    """The decode-time diagnostic must be computed at decode time.

    `mod_predictor_agreement` says whether the causal predictor is a usable
    stand-in for top-k selection, which is precisely the question when the
    model is not training. It used to be computed only inside the training
    branch, so every eval and benchmark snapshot reported the 0.0 it was
    initialised with.
    """
    import torch

    import mimomix_core as mc

    torch.manual_seed(0)
    cfg = mc.MiMoMixConfig(use_mod=True)
    model = mc.MiMoMixModel(cfg)
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab_size, (1, 16)))

    agreement = model.telemetry().get("mod_predictor_agreement")
    assert agreement, "no MoD routers reported agreement under eval()"
    assert not all(value == 0.0 for value in agreement), (
        "every router reports 0.0 under eval(), which is the initialised "
        "buffer rather than a measurement"
    )
    assert all(0.0 <= value <= 1.0 for value in agreement)


def test_balanced_operands_is_not_reported_when_it_did_not_run():
    """A receipt records what happened, not what was requested.

    Operand balancing is implemented only on the repeating draw path. Under
    `--unique` it silently did nothing while the report still recorded
    `balanced_operands: true`, so an A/B of the flag produced byte-identical
    corpora and a receipt claiming the flag was on.
    """
    import build_omni_corpus as omni

    _, skipped = omni.build(30, 79, ["force"], repeat=False, balanced_operands=True)
    assert skipped["options"]["balanced_operands"] is False
    assert "balanced_operands_skipped" in skipped

    _, applied = omni.build(30, 79, ["force"], repeat=True, balanced_operands=True)
    assert applied["options"]["balanced_operands"] is True
    assert applied.get("operand_balance")
    assert "balanced_operands_skipped" not in applied


def test_the_cli_refuses_balanced_operands_with_unique():
    """Refusing beats silently dropping: the arm would measure an exact null."""
    import build_omni_corpus as omni

    with pytest.raises(SystemExit):
        omni.main(["--per_task", "5", "--task", "force", "--unique",
                   "--balanced_operands", "--output", "unused.jsonl"])
