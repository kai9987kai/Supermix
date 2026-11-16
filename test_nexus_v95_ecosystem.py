"""Tests for the NexusMind v95 Unified Frontier Ecosystem, Engine & REST API."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import nexus_api as api
import nexus_engine as ne
from nexus_v95_ecosystem import (
    CrosswireInterfaceEngine,
    FaultlineEnsembleEngine,
    MnemorphArchaeologyEngine,
    TetherForensicsEngine,
)


def test_tether_forensics_riemannian_sandwich_shapley_pid_and_spoof():
    tether = TetherForensicsEngine()
    probs = [
        [0.80, 0.15, 0.05],
        [0.60, 0.30, 0.10],
        [0.75, 0.10, 0.15],
    ]
    res = tether.analyze(probs)
    assert res["answer_authority"] is False

    # Distance symmetry and triangle inequality
    dmat = np.asarray(res["pairwise_fisher_rao_distances"], dtype=float)
    assert np.allclose(dmat, dmat.T)
    assert np.allclose(np.diag(dmat), 0.0)
    assert dmat[0, 2] <= dmat[0, 1] + dmat[1, 2] + 1e-9

    # Karcher barycenter achieves Fréchet variance <= arithmetic centroid
    assert res["frechet_variance"] <= res["arithmetic_frechet_variance"] + 1e-9
    assert res["geodesic_variance_suppression"] >= 0.0

    # Exact Robustness Sandwich: 0 <= r_tv_cert <= delta_break_tv <= 1
    sw = res["robustness_sandwich"]
    assert 0.0 <= sw["r_tv_cert"] <= sw["delta_break_tv"] <= 1.0
    assert sw["sandwich_valid"] is True
    assert sw["r_logit_cert"] > 0.0

    # Even strongly peaked simplex satisfies sandwich inequality <= 1
    peaked = tether.analyze([[0.92, 0.06, 0.02], [0.90, 0.07, 0.03]])
    sw_p = peaked["robustness_sandwich"]
    assert 0.0 <= sw_p["r_tv_cert"] <= sw_p["delta_break_tv"] <= 1.0

    # Shapley efficiency & PID non-negativity
    attr = res["attribution"]
    assert attr["efficiency_gap"] < 1e-6
    pid = res["partial_information_decomposition"]
    assert pid["redundancy_bits"] >= 0.0
    assert pid["synergy_bits"] >= 0.0
    assert all(u >= 0.0 for u in pid["unique_bits"])

    # Mimicry spoof detection: collapsed dark-knowledge variance vs diverse reference
    ref_probs = [
        [0.70, 0.25, 0.05],
        [0.70, 0.05, 0.25],
        [0.70, 0.18, 0.12],
    ]
    spoofed = [
        [0.70, 0.15, 0.15],
        [0.70, 0.15, 0.15],
        [0.70, 0.15, 0.15],
    ]
    dk_res = tether.analyze(spoofed, reference_probs=ref_probs)
    assert dk_res["dark_knowledge_forensics"]["spoof_verdict"] == "mimicry_spoof"

    # Orthogonal drift detection
    drift_cand = [
        [0.40, 0.55, 0.05],
        [0.40, 0.56, 0.04],
    ]
    drift_res = tether.analyze(drift_cand, reference_probs=[[0.70, 0.05, 0.25], [0.72, 0.04, 0.24]])
    assert drift_res["dark_knowledge_forensics"]["spoof_verdict"] in ("orthogonal_drift", "mimicry_spoof")


def test_faultline_ensemble_beats_accuracy_on_correlated_failures():
    faultline = FaultlineEnsembleEngine()
    # Experts 0, 1, 2 have lowest individual error (10%) on cond 1..9, but ALL fail simultaneously on cond 0!
    # Experts 3, 4, 5 have slightly higher individual error (20%), yet never fail on the same condition together.
    mat = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # E0: err 0.1, fails cond 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # E1: err 0.1, fails cond 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # E2: err 0.1, fails cond 0
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # E3: err 0.2, fails cond 1,2
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # E4: err 0.2, fails cond 3,4
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # E5: err 0.2, fails cond 5,6
    ]
    res = faultline.select_panel(mat, panel_size=3)
    assert res["answer_authority"] is False
    acc = res["strategies"]["accuracy"]
    joint = res["strategies"]["joint"]
    assert acc["expert_indices"] == [0, 1, 2]
    assert acc["worst_condition_failure_rate"] == pytest.approx(1.0)
    assert joint["worst_condition_failure_rate"] < acc["worst_condition_failure_rate"]
    assert joint["expected_joint_failure_rate"] < acc["expected_joint_failure_rate"]
    assert joint["worst_condition_failure_rate"] == pytest.approx(0.0)
    assert len(res["receipt"]["receipt_sha256"]) == 64


def test_crosswire_interface_version_space_probe_and_safety_shield():
    cw = CrosswireInterfaceEngine()
    # Before probing, all 3 default hypotheses survive, including unsafe EXEC hypotheses -> ABSTAIN_UNSAFE
    initial = cw.verify_and_probe(observed_trace=[], candidate_action="EXEC")
    assert initial["answer_authority"] is False
    assert initial["surviving_count"] == 3
    assert initial["shield_verdict"] == "ABSTAIN_UNSAFE"
    assert initial["certified_safe"] is False
    assert len(initial["counterexample_hypotheses"]) == 2
    assert initial["probe_ranking"][0]["expected_info_gain_bits"] > 0.0

    # Trace that authenticates and eliminates h_permissive_unsafe, then probes PING:
    # Step 1: PING -> PONG_STRICT eliminates h_delayed_fault;
    # Step 2: AUTH -> ACK_AUTH (arming);
    # Step 3: EXEC -> COMMIT_OK eliminates h_permissive_unsafe leaving only h_strict_guard!
    verified = cw.verify_and_probe(
        observed_trace=[
            {"action": "PING", "output": "PONG_STRICT"},
            {"action": "AUTH", "output": "ACK_AUTH"},
            {"action": "EXEC", "output": "COMMIT_OK"},
        ],
        candidate_action="EXEC",
    )
    assert verified["surviving_hypotheses"] == ["h_strict_guard"]
    assert verified["shield_verdict"] == "CERTIFIED_SAFE"
    assert verified["certified_safe"] is True
    assert verified["counterexample_hypotheses"] == []


def test_mnemorph_archaeology_bayesian_update_and_morpheus_recovery():
    mn = MnemorphArchaeologyEngine()
    # Observations matching regenerative_reconstruction: focal drop (0.32) followed by strong regrowth (0.90)
    res = mn.analyze_assays(
        assay_observations=[
            {"intervention": "focal_lesion", "observed_retention": 0.32},
            {"intervention": "regenerative_regrowth", "observed_retention": 0.90},
        ],
        recovery_trajectory=[0.32, 0.55, 0.78, 0.90],
    )
    assert res["answer_authority"] is False
    assert sum(res["posterior"].values()) == pytest.approx(1.0, abs=1e-5)
    assert res["map_mechanism"] == "regenerative_reconstruction"
    assert res["paired_brier_score"] >= 0.0
    assert res["morpheus_recovery"]["recovery_delta"] == pytest.approx(0.58)
    assert res["recommended_next_assay"] not in ("focal_lesion", "regenerative_regrowth")


def test_nexus_engine_and_rest_api_v95_endpoints():
    engine = ne.build_default_engine()
    crc = engine.calibrate_conformal_crc(
        margins=[0.1, 0.25, 0.45, 0.65, 0.85],
        losses=[1.0, 1.0, 0.0, 0.0, 0.0],
        target_risk_alpha=0.25,
    )
    assert crc["answer_authority"] is False
    assert crc["risk_certified"] is True
    assert crc["risk_coverage_profile"]["aurc"] >= 0.0
    assert crc["risk_coverage_profile"]["e_aurc"] >= 0.0

    # REST endpoints via FastAPI TestClient
    from starlette.testclient import TestClient

    svc = api.NexusApiService(engine=engine)
    app = api.create_app(svc)
    client = TestClient(app)

    r_sig = client.get("/v1/signals")
    assert r_sig.status_code == 200
    assert r_sig.json()["v95_frontier_ecosystem"]["tether_forensics_engine"] is True

    r_tether = client.post("/v1/tether/geometry", json={"expert_probs": [[0.8, 0.2], [0.7, 0.3]]})
    assert r_tether.status_code == 200
    assert r_tether.json()["answer_authority"] is False

    r_fault = client.post(
        "/v1/faultline/panel",
        json={"failure_matrix": [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], "panel_size": 3},
    )
    assert r_fault.status_code == 200
    assert r_fault.json()["selected_panel"]["worst_condition_failure_rate"] == 0.0

    r_cw = client.post("/v1/crosswire/synthesize", json={"observed_trace": [], "candidate_action": "EXEC"})
    assert r_cw.status_code == 200
    assert r_cw.json()["shield_verdict"] == "ABSTAIN_UNSAFE"

    r_mn = client.post(
        "/v1/mnemorph/assay",
        json={
            "assay_observations": [{"intervention": "synapse_scramble", "observed_retention": 0.08}],
            "recovery_trajectory": [0.2, 0.25, 0.3],
        },
    )
    assert r_mn.status_code == 200
    assert "posterior" in r_mn.json()

    r_crc = client.post(
        "/v1/conformal/crc",
        json={"margins": [0.2, 0.4, 0.6, 0.8], "losses": [1.0, 0.0, 0.0, 0.0], "target_risk_alpha": 0.25},
    )
    assert r_crc.status_code == 200
    assert r_crc.json()["risk_certified"] is True
