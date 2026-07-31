"""Tests for the Dem-Lab observatory.

These check the statistics against values that are known independently -- text-
book chi-square critical points, the JSD bound of ln 2, the CHSH classical
bound, replicator fixed points -- rather than against the implementation's own
output. A measurement harness that only agrees with itself measures nothing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_observatory as ob  # noqa: E402


# ---------------------------------------------------------------------------
# Special functions and uniformity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statistic,dof",
    [(3.841, 1), (5.991, 2), (7.815, 3), (16.919, 9), (18.307, 10)],
)
def test_chi_square_survival_matches_textbook_five_percent_points(statistic, dof):
    assert ob.chi_square_survival(statistic, dof) == pytest.approx(0.05, abs=2e-3)


def test_chi_square_survival_endpoints():
    assert ob.chi_square_survival(0.0, 4) == pytest.approx(1.0)
    assert ob.chi_square_survival(1000.0, 4) == pytest.approx(0.0, abs=1e-12)
    assert ob.chi_square_survival(5.0, 0) == 1.0


def test_perfectly_uniform_samples_score_a_zero_statistic():
    report = ob.chi_square_uniformity([i % 5 for i in range(500)], 5)
    assert report["statistic"] == pytest.approx(0.0)
    assert report["p_value"] == pytest.approx(1.0)
    assert report["approximation_valid"] is True


def test_skewed_samples_are_rejected():
    report = ob.chi_square_uniformity([0] * 400 + [1, 2, 3], 4)
    assert report["p_value"] < 1e-6


def test_uniformity_flags_a_too_small_sample():
    report = ob.chi_square_uniformity([0, 1, 2, 3], 4)
    assert report["approximation_valid"] is False


def test_uniformity_rejects_out_of_range_samples():
    with pytest.raises(ValueError):
        ob.chi_square_uniformity([0, 1, 9], 4)
    with pytest.raises(ValueError):
        ob.chi_square_uniformity([0], 1)


def test_monobit_and_runs_behave_on_known_streams():
    balanced = ob.monobit_test([0, 1] * 250)
    assert balanced["statistic"] == pytest.approx(0.0)
    assert balanced["p_value"] == pytest.approx(1.0)

    skewed = ob.monobit_test([1] * 190 + [0] * 10)
    assert skewed["p_value"] < 1e-6

    # perfect alternation is maximally non-random in the runs sense
    assert ob.runs_test([0, 1] * 100)["p_value"] < 1e-6
    # a constant stream fails the monobit prerequisite, so runs cannot apply
    assert ob.runs_test([1] * 100)["prerequisite_met"] is False

    with pytest.raises(ValueError):
        ob.monobit_test([0, 1, 2])


# ---------------------------------------------------------------------------
# Entropy family
# ---------------------------------------------------------------------------


def test_entropy_of_a_uniform_distribution_is_log_n():
    assert ob.shannon_entropy([1] * 8) == pytest.approx(math.log(8))
    assert ob.shannon_entropy([1] * 8, base=2.0) == pytest.approx(3.0)
    assert ob.perplexity([1] * 8) == pytest.approx(8.0)


def test_min_entropy_never_exceeds_shannon_entropy():
    for distribution in ([1, 1, 1, 1], [10, 1, 1, 1], [100, 1], [1, 1]):
        assert ob.min_entropy(distribution) <= ob.shannon_entropy(distribution) + 1e-9


def test_entropy_of_a_point_mass_is_zero():
    assert ob.shannon_entropy([1, 0, 0]) == pytest.approx(0.0)
    assert ob.min_entropy([1, 0, 0]) == pytest.approx(0.0)


def test_jsd_is_zero_for_identical_and_bounded_by_ln2():
    assert ob.jensen_shannon_divergence([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    disjoint = ob.jensen_shannon_divergence([1.0, 1e-12], [1e-12, 1.0])
    assert disjoint == pytest.approx(math.log(2), abs=1e-6)
    assert disjoint <= math.log(2) + 1e-9
    with pytest.raises(ValueError):
        ob.jensen_shannon_divergence([1, 2], [1, 2, 3])


def test_randomness_report_is_json_safe():
    payload = json.loads(json.dumps(ob.randomness_report([i % 4 for i in range(200)], 4)))
    assert payload["max_entropy_bits"] == pytest.approx(2.0)
    assert payload["uniformity"]["p_value"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Harness self-tests
# ---------------------------------------------------------------------------


def test_chsh_flags_anything_above_the_classical_bound():
    classical = ob.chsh_value({(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0})
    assert classical["s_value"] == pytest.approx(2.0)
    assert classical["within_classical_bound"] is True

    over = ob.chsh_value({(0, 0): 0.8, (0, 1): 0.8, (1, 0): 0.8, (1, 1): -0.8})
    assert over["s_value"] == pytest.approx(3.2)
    assert over["within_classical_bound"] is False

    with pytest.raises(ValueError):
        ob.chsh_value({(0, 0): 1.0})
    with pytest.raises(ValueError):
        ob.chsh_value({(0, 0): 2.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0})


def test_sequential_evidence_penalises_repeated_looks():
    single = ob.sequential_evidence(5200, 10000, looks=1)
    many = ob.sequential_evidence(5200, 10000, looks=1000)
    assert single["log_likelihood_ratio"] == pytest.approx(many["log_likelihood_ratio"])
    assert many["penalised_log_likelihood_ratio"] < single["penalised_log_likelihood_ratio"]
    assert many["optional_stopping_penalty"] == pytest.approx(math.log(1000))


def test_evidence_at_the_null_rate_is_zero_and_effect_size_is_reported():
    exact = ob.sequential_evidence(500, 1000, null_rate=0.5)
    assert exact["log_likelihood_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert exact["effect_size"] == pytest.approx(0.0)
    tiny = ob.sequential_evidence(50100, 100000, null_rate=0.5)
    assert tiny["log_likelihood_ratio"] > 0.0
    assert abs(tiny["effect_size"]) < 0.01  # decisive LR, trivial effect


def test_evidence_validates_its_inputs():
    with pytest.raises(ValueError):
        ob.sequential_evidence(5, 0)
    with pytest.raises(ValueError):
        ob.sequential_evidence(11, 10)
    with pytest.raises(ValueError):
        ob.sequential_evidence(5, 10, null_rate=0.0)
    with pytest.raises(ValueError):
        ob.sequential_evidence(5, 10, looks=0)


# ---------------------------------------------------------------------------
# RSI meters
# ---------------------------------------------------------------------------


def test_novelty_is_bounded_and_reflects_history():
    assert ob.novelty_score([1, 2, 3], [[1, 2, 3]], n=3)["novelty"] == 0.0
    assert ob.novelty_score([7, 8, 9], [[1, 2, 3]], n=3)["novelty"] == 1.0
    partial = ob.novelty_score([1, 2, 3, 4], [[1, 2, 3]], n=3)
    assert 0.0 < partial["novelty"] < 1.0
    assert ob.novelty_score([1], [], n=3)["ngrams"] == 0


def test_stability_is_one_for_a_constant_series_and_falls_with_spread():
    assert ob.stability_score([3.0] * 10)["stability"] == pytest.approx(1.0)
    assert ob.stability_score([1.0])["stability"] == 1.0
    noisy = ob.stability_score([1.0, 9.0, 2.0, 8.0])
    steady = ob.stability_score([5.0, 5.1, 4.9, 5.0])
    assert noisy["stability"] < steady["stability"]
    # a near-zero mean must not manufacture instability
    assert 0.0 < ob.stability_score([0.0, 0.0, 1e-9])["stability"] <= 1.0


def test_rsi_index_is_bounded_and_needs_all_three_ingredients():
    high = ob.recursive_improvement_index(0.8, 0.9, quality_delta=1.0, cost_delta=0.0)
    assert 0.0 <= high["index"] <= 1.0
    assert ob.recursive_improvement_index(0.0, 1.0, 1.0, 0.0)["index"] == 0.0
    assert ob.recursive_improvement_index(1.0, 0.0, 1.0, 0.0)["index"] == 0.0
    costly = ob.recursive_improvement_index(0.8, 0.9, quality_delta=1.0, cost_delta=10.0)
    assert costly["index"] < high["index"]


# ---------------------------------------------------------------------------
# Resonance and attribution
# ---------------------------------------------------------------------------


def test_resonance_finds_the_right_clusters_deterministically():
    states = torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    first = ob.semantic_resonance(states, threshold=0.9)
    second = ob.semantic_resonance(states, threshold=0.9)
    assert first["clusters"] == [[0, 1], [2, 3]]
    assert first == second
    assert first["n_clusters"] == 2


def test_orthogonal_states_do_not_resonate():
    report = ob.semantic_resonance(torch.eye(5), threshold=0.5)
    assert report["mean_similarity"] == pytest.approx(0.0)
    assert report["resonance_density"] == pytest.approx(0.0)
    assert report["n_clusters"] == 5


def test_resonance_validates_shape_and_handles_degenerate_input():
    with pytest.raises(ValueError):
        ob.semantic_resonance(torch.zeros(2, 3, 4))
    assert ob.semantic_resonance(torch.zeros(1, 4))["pairs"] == 0


def test_routing_attribution_reads_a_telemetry_snapshot():
    telemetry = {
        "expert_load": [[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]],
        "per_layer_sink_mass": [0.1, 0.2],
        "attention_layout": ["swa", "global"],
    }
    report = ob.routing_attribution(telemetry)
    balanced, collapsed = report["moe_layers"]
    assert balanced["normalised_entropy"] == pytest.approx(1.0)
    assert balanced["herfindahl_index"] == pytest.approx(0.25)
    assert balanced["starved_experts"] == []
    assert collapsed["normalised_entropy"] == pytest.approx(0.0)
    assert collapsed["starved_experts"] == [1, 2, 3]
    assert report["any_starved_expert"] is True
    assert report["mean_attention_sink_mass"] == pytest.approx(0.15)


def test_routing_attribution_survives_empty_telemetry():
    report = ob.routing_attribution({})
    assert report["moe_layers"] == []
    assert report["any_starved_expert"] is False


# ---------------------------------------------------------------------------
# Anomalies and ecosystem
# ---------------------------------------------------------------------------


def test_robust_anomaly_detection_finds_a_spike_the_mean_would_hide():
    report = ob.robust_anomalies([1.0] * 20 + [50.0])
    assert [a["index"] for a in report["anomalies"]] == [20]


def test_anomaly_detection_uses_mad_when_it_can():
    report = ob.robust_anomalies([1.0, 2.0, 3.0, 4.0, 100.0])
    assert report["estimator"] == "mad"
    assert report["anomalies"]


def test_a_constant_series_has_no_anomalies():
    report = ob.robust_anomalies([7.0] * 10)
    assert report["anomalies"] == []
    assert report["estimator"] == "degenerate_constant"
    assert ob.robust_anomalies([])["samples"] == 0


def test_replicator_dynamics_preserve_mass_and_favour_the_fittest():
    updated = ob.replicator_step([1.0, 1.0, 1.0], [0.5, 1.0, 0.2])
    assert sum(updated) == pytest.approx(1.0)
    assert updated[1] > updated[0] > updated[2]

    # equal payoffs are a fixed point
    assert ob.replicator_step([0.2, 0.8], [1.0, 1.0]) == pytest.approx([0.2, 0.8])

    with pytest.raises(ValueError):
        ob.replicator_step([1.0, 1.0], [1.0, -1.0])
    with pytest.raises(ValueError):
        ob.replicator_step([1.0], [1.0, 1.0])


def test_ecosystem_converges_on_the_dominant_strategy():
    report = ob.run_ecosystem([1.0, 1.0, 1.0], [0.5, 1.0, 0.2], steps=60)
    assert report["dominant_strategy"] == 1
    assert report["dominant_share"] > 0.99
    assert report["converged"] is True
    assert len(report["trajectory"]) == 61


# ---------------------------------------------------------------------------
# Q-learning feedback
# ---------------------------------------------------------------------------


def test_learner_prefers_the_cheaper_budget_at_equal_fidelity():
    learner = ob.BudgetPolicyLearner(budgets=(1, 2, 4))
    for _ in range(5):
        learner.observe(0.1, 0.1, budget=1, decision_matched_ceiling=True, cycles_spent=1, ceiling_cycles=4)
        learner.observe(0.1, 0.1, budget=4, decision_matched_ceiling=True, cycles_spent=7, ceiling_cycles=4)
    assert learner.suggest(0.1, 0.1) == 1


def test_learner_abandons_a_cheap_budget_that_loses_fidelity():
    learner = ob.BudgetPolicyLearner(budgets=(1, 4))
    for _ in range(6):
        learner.observe(0.9, 0.9, budget=1, decision_matched_ceiling=False, cycles_spent=1, ceiling_cycles=4)
        learner.observe(0.9, 0.9, budget=4, decision_matched_ceiling=True, cycles_spent=4, ceiling_cycles=4)
    assert learner.suggest(0.9, 0.9) == 4


def test_learner_returns_none_for_an_unproven_bucket():
    learner = ob.BudgetPolicyLearner()
    assert learner.suggest(0.5, 0.5) is None
    learner.observe(0.5, 0.5, budget=2, decision_matched_ceiling=True, cycles_spent=2, ceiling_cycles=4)
    assert learner.suggest(0.5, 0.5, min_visits=3) is None
    assert learner.suggest(0.5, 0.5, min_visits=1) == 2


def test_learner_rejects_an_unknown_budget_and_buckets_are_bounded():
    learner = ob.BudgetPolicyLearner()
    with pytest.raises(ValueError):
        learner.observe(0.1, 0.1, budget=3, decision_matched_ceiling=True, cycles_spent=3, ceiling_cycles=4)
    assert learner.bucket(-5.0) == 0
    assert learner.bucket(5.0) == learner.buckets - 1


def test_learner_updates_are_deterministic():
    def run():
        learner = ob.BudgetPolicyLearner()
        for index in range(10):
            learner.observe(0.2, 0.3, budget=2, decision_matched_ceiling=index % 2 == 0,
                            cycles_spent=3, ceiling_cycles=4)
        return learner.to_dict()

    assert run() == run()


# ---------------------------------------------------------------------------
# The observatory front end
# ---------------------------------------------------------------------------


def _telemetry(sink: float, cycles: float, quality: float) -> dict:
    return {
        "mean_sink_mass": sink,
        "per_layer_sink_mass": [sink],
        "expert_load": [[0.5, 0.5]],
        "attention_layout": ["global"],
        "thinking": {
            "cycles_used": cycles,
            "quality_probability": quality,
            "continue_probability": 1.0 - quality,
            "exit_reason": "budget_exhausted",
        },
    }


def test_observatory_reports_nothing_before_it_observes():
    assert ob.Observatory().report() == {"turns": 0}


def test_observatory_report_is_deterministic_and_json_safe():
    def run():
        observatory = ob.Observatory()
        for index in range(4):
            observatory.record(
                _telemetry(0.1 + 0.01 * index, 2.0 + index, 0.4 + 0.05 * index),
                tokens=[index % 4 for _ in range(12)],
            )
        return observatory.report(vocab_size=4)

    first, second = run(), run()
    assert first == second
    json.dumps(first)
    assert first["turns"] == 4
    assert "novelty" in first and "randomness" in first and "rsi" in first
    assert 0.0 <= first["rsi"]["index"] <= 1.0


def test_observatory_surfaces_an_anomalous_turn():
    observatory = ob.Observatory()
    for _ in range(8):
        observatory.record(_telemetry(0.1, 2.0, 0.5))
    observatory.record(_telemetry(0.95, 2.0, 0.5))
    report = observatory.report()
    assert report["anomalies"]["attention_sink_mass"]["anomalies"]
