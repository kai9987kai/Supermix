import json
import math
import os
from pathlib import Path
import sys

import pytest


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import run_v51_chat_response_fidelity_gate as gate


class _StubEngine:
    __v51_verified_surface__ = "source"
    instances = []
    mismatch_prompt = None
    nonfinite_timing = False
    candidate_count = gate.TOP_CANDIDATE_COUNT
    duplicate_candidate = False
    applied_value = True
    adaptive_compute_overrides = {}
    runtime_default_overrides = {}
    load_status_overrides = {}
    status_overrides = {}
    available_label_count = 10
    response_suffix = ""

    def __init__(self, device, device_info, defaults):
        self.device = device
        self.device_info = device_info
        self.constructor_defaults = defaults
        self.defaults = {
            "adaptive_exit_tol": 0.001,
            "adaptive_exit_entropy": 0.2,
            "prediction_stability_patience": 2,
            "prediction_stability_tol": 0.005,
            "prediction_stability_margin": 0.0005,
            "prediction_stability_rank_depth": 3,
        }
        self.defaults.update(type(self).runtime_default_overrides)
        self.calls = []
        self.available_labels = list(range(type(self).available_label_count))
        type(self).instances.append(self)

    def load(self, weights, metadata):
        self.loaded = (weights, metadata)
        status = {
            "ok": True,
            "load_ms": 2.5,
            "model_size": "cognitive_leap_ultra_expert",
            "feature_mode": "context_mix_v4",
            "available_labels": len(self.available_labels),
        }
        status.update(type(self).load_status_overrides)
        return status

    def status(self):
        status = {
            "loaded": True,
            "weights": self.loaded[0],
            "meta": self.loaded[1],
            "runtime_compute_supported": True,
            "sessions": 0,
            "reasoning_cycles": 3,
            "adaptive_compute": False,
            "auto_compute": False,
        }
        status.update(type(self).status_overrides)
        return status

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        adaptive = kwargs["adaptive_compute"]
        prompt = kwargs["user_text"]
        mismatch = adaptive and prompt == type(self).mismatch_prompt
        candidates = [
            "candidate one",
            "candidate two",
            "candidate three",
            "candidate four",
            "candidate five",
        ][: type(self).candidate_count]
        if type(self).duplicate_candidate and len(candidates) >= 2:
            candidates[1] = candidates[0]
        if mismatch:
            candidates[0], candidates[1] = candidates[1], candidates[0]
        compute = {
            "applied": type(self).applied_value,
            "adaptive_compute": adaptive,
            "requested_reasoning_cycles": 8 if adaptive else 3,
            "cycles_used": 2 if adaptive else 3,
            "exit_reason": "prediction_stable" if adaptive else "max_cycles",
            "exit_tol": self.defaults["adaptive_exit_tol"],
            "exit_entropy_threshold": self.defaults["adaptive_exit_entropy"],
            "prediction_stability_patience": self.defaults[
                "prediction_stability_patience"
            ],
            "prediction_stability_tol": self.defaults[
                "prediction_stability_tol"
            ],
            "prediction_stability_margin": 0.0005,
            "prediction_stability_rank_depth": 3,
            "prediction_verifier_active": adaptive,
            "prediction_class_indices": list(self.available_labels),
        }
        if adaptive:
            compute.update(
                {
                    "prediction_rank_depth": 3,
                    "decision_reference_cycles": 3,
                    "prediction_decision_margin": 0.1,
                    "prediction_margin": 0.2,
                    "prediction_class_count": len(self.available_labels),
                    "prediction_class_selection_valid": True,
                }
            )
            compute.update(type(self).adaptive_compute_overrides)
        return {
            "ok": True,
            "response": (
                "adaptive mismatch" if mismatch else f"answer:{prompt}"
            )
            + type(self).response_suffix,
            "style_mode": "balanced",
            "timing_ms": {
                "infer": math.nan if type(self).nonfinite_timing else 1.0,
                "rank_pick": 0.25,
                "total": 1.25,
                "cycles_used": 2 if adaptive else 3,
            },
            "compute": compute,
            "auto_compute_plan": None,
            "top_candidates": [
                {"text": text, "score": 1.0 - index * 0.1}
                for index, text in enumerate(candidates)
            ],
        }


class _PackagedStubEngine(_StubEngine):
    __v51_verified_surface__ = "packaged"


class _PackagedBehaviorDriftEngine(_PackagedStubEngine):
    response_suffix = " [packaged drift]"


@pytest.fixture(autouse=True)
def _reset_stub():
    _StubEngine.instances = []
    _StubEngine.mismatch_prompt = None
    _StubEngine.nonfinite_timing = False
    _StubEngine.candidate_count = gate.TOP_CANDIDATE_COUNT
    _StubEngine.duplicate_candidate = False
    _StubEngine.applied_value = True
    _StubEngine.adaptive_compute_overrides = {}
    _StubEngine.runtime_default_overrides = {}
    _StubEngine.load_status_overrides = {}
    _StubEngine.status_overrides = {}
    _StubEngine.available_label_count = 10
    _StubEngine.response_suffix = ""


def _artifacts(tmp_path):
    weights = tmp_path / "checkpoint.pth"
    metadata = tmp_path / "meta.json"
    weights.write_bytes(b"checkpoint")
    metadata.write_text('{"buckets": {}}', encoding="utf-8")
    return weights, metadata


def _clock():
    ticks = iter(index / 1000 for index in range(1000))
    return lambda: next(ticks)


def _custom_prompts():
    return [
        {"id": "one", "category": "a", "prompt": "first prompt"},
        {"id": "two", "category": "b", "prompt": "second prompt"},
    ]


def _run(tmp_path, **overrides):
    kwargs = {
        "weights": gate.DEFAULT_WEIGHTS,
        "metadata": gate.DEFAULT_META,
        "device": "cpu-stub",
        "device_info": {"requested": "stub", "resolved": "cpu-stub"},
        "engine_factory": _StubEngine,
        "packaged_engine_factory": _PackagedStubEngine,
        "clock": _clock(),
        "created_at": "2026-07-22T00:00:00+00:00",
        "provenance": {"test": True},
    }
    kwargs.update(overrides)
    return gate.run_gate(**kwargs)


def test_cli_and_builtin_matrix_are_pinned_to_release_protocol():
    args = gate.build_parser().parse_args([])
    prompts, source = gate.load_prompt_matrix(None)

    assert Path(args.weights) == gate.DEFAULT_WEIGHTS
    assert Path(args.metadata) == gate.DEFAULT_META
    assert gate.FIXED_CYCLES == 3
    assert gate.ADAPTIVE_MAX_CYCLES == 8
    assert gate.PREDICTION_STABILITY_MARGIN == 5e-4
    assert gate.REQUIRED_RELEASE_PREDICTION_STABILITY_MARGIN == 5e-4
    assert gate.REQUIRED_RELEASE_DECISION_REFERENCE_CYCLES == 3
    assert gate.AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH == 3
    assert gate.AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS == {
        "adaptive_exit_tol": 0.001,
        "adaptive_exit_entropy": 0.2,
        "prediction_stability_patience": 2.0,
        "prediction_stability_tol": 0.005,
    }
    assert len(prompts) == gate.CANONICAL_RELEASE_PROMPT_MATRIX_COUNT
    assert gate.prompt_matrix_sha256(prompts) == gate.CANONICAL_RELEASE_PROMPT_MATRIX_SHA256
    assert {
        category: sum(row["category"] == category for row in prompts)
        for category in {row["category"] for row in prompts}
    } == gate.CANONICAL_RELEASE_PROMPT_MATRIX_CATEGORIES
    assert source["origin"] == "builtin"


def test_custom_prompt_json_is_normalized_hashed_and_validated(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"prompts": ["alpha", {"id": "beta", "text": "beta text"}]}),
        encoding="utf-8",
    )

    prompts, source = gate.load_prompt_matrix(path)

    assert prompts == [
        {"id": "custom-001", "category": "custom", "prompt": "alpha"},
        {"id": "beta", "category": "custom", "prompt": "beta text"},
    ]
    assert source["origin"] == "json_file"
    assert len(source["source_file_sha256"]) == 64
    with pytest.raises(ValueError, match="Duplicate prompt"):
        gate.normalize_prompt_matrix(["same", "same"])


def test_custom_prompt_matrix_is_diagnostic_and_release_ineligible(tmp_path):
    payload = _run(
        tmp_path,
        prompts=_custom_prompts(),
        prompt_source={
            "origin": "json_file",
            "path": "custom.json",
            "source_file_sha256": "0" * 64,
        },
    )

    assert payload["summary"]["any_fidelity_mismatch_count"] == 0
    assert payload["prompt_matrix"]["diagnostic_only"] is True
    assert payload["prompt_matrix"]["release_eligible"] is False
    assert payload["claim_scope"]["matrix_kind"] == "diagnostic_prompt_matrix"
    assert payload["gates"]["passed"] is False
    check = payload["gates"]["checks"]["canonical_builtin_release_prompt_matrix"]
    assert check["passed"] is False
    assert check["origin"] == "json_file"


@pytest.mark.parametrize("mutation", ["changed", "truncated"])
def test_builtin_matrix_mutation_or_truncation_is_not_release_eligible(
    tmp_path, mutation
):
    prompts = [dict(row) for row in gate.FROZEN_RELEASE_PROMPT_MATRIX]
    if mutation == "changed":
        prompts[0]["prompt"] += " changed"
    else:
        prompts.pop()

    payload = _run(
        tmp_path,
        prompts=prompts,
        prompt_source={
            "origin": "builtin",
            "path": None,
            "source_file_sha256": None,
        },
    )

    assert payload["gates"]["passed"] is False
    assert payload["prompt_matrix"]["release_eligible"] is False
    canonical = payload["prompt_matrix"]["canonical_release_contract"]
    assert canonical["sha256"] == gate.CANONICAL_RELEASE_PROMPT_MATRIX_SHA256
    assert canonical["count"] == gate.CANONICAL_RELEASE_PROMPT_MATRIX_COUNT


def test_gate_uses_isolated_sessions_release_defaults_and_exact_comparisons(tmp_path):
    payload = _run(tmp_path)
    engine = _StubEngine.instances[-1]

    assert payload["gates"]["passed"] is True
    assert payload["summary"]["response_text_mismatch_count"] == 0
    assert payload["summary"]["top_candidate_text_order_mismatch_count"] == 0
    assert payload["claim_scope"] == {
        "matrix_kind": "frozen_release_prompt_matrix",
        "artifact_kind": "canonical_default_v51_artifacts",
        "statement": (
            "Deterministic regression evidence for the exact canonical v51 "
            "prompt matrix, checkpoint, metadata, and runtime identity only."
        ),
        "held_out_claim": False,
        "universal_chat_fidelity_claim": False,
        "release_eligible": True,
    }
    assert len(payload["checkpoint"]["sha256"]) == 64
    assert len(payload["metadata"]["sha256"]) == 64
    assert len(payload["prompt_matrix"]["sha256"]) == 64
    assert payload["settings"]["prediction_stability_rank_depth"] == {
        "selection": "runtime_release_default",
        "authoritative": 3,
        "resolved": 3,
        "resolved_by_surface": {"source": 3.0, "packaged": 3.0},
        "explicitly_overridden_by_gate": False,
    }
    assert payload["settings"]["adaptive_runtime_defaults"] == {
        "adaptive_exit_tol": 0.001,
        "adaptive_exit_entropy": 0.2,
        "prediction_stability_patience": 2.0,
        "prediction_stability_tol": 0.005,
        "prediction_stability_margin": 0.0005,
        "prediction_stability_rank_depth": 3.0,
    }
    assert payload["settings"]["authoritative_adaptive_runtime_defaults"] == {
        "adaptive_exit_tol": 0.001,
        "adaptive_exit_entropy": 0.2,
        "prediction_stability_patience": 2.0,
        "prediction_stability_tol": 0.005,
    }

    assert [call["adaptive_compute"] for call in engine.calls[:4]] == [False, True, True, False]
    assert [call["reasoning_cycles"] for call in engine.calls[:4]] == [3, 8, 8, 3]
    assert len(engine.calls) == gate.CANONICAL_RELEASE_PROMPT_MATRIX_COUNT * 2
    assert len({call["session_id"] for call in engine.calls}) == len(engine.calls)
    for call in engine.calls:
        assert call["response_temperature"] == 0.0
        assert call["auto_compute"] is False
        assert call["prediction_stability_margin"] == 5e-4
        assert call["show_top_responses"] == 5
        assert "prediction_stability_rank_depth" not in call
    assert payload["source_package_parity"]["passed"] is True
    assert payload["surface_specific_runtime_hashes"]["required_for_release"] is False
    assert payload["gates"]["checks"][
        "source_package_engine_exact_behavior_parity"
    ]["passed"] is True
    assert payload["gates"]["checks"][
        "isolated_source_packaged_module_provenance"
    ]["passed"] is True
    assert payload["gates"]["checks"][
        "canonical_default_checkpoint_and_metadata_identity"
    ]["passed"] is True
    assert payload["gates"]["checks"]["canonical_model_runtime_identity"][
        "passed"
    ] is True
    assert payload["gates"]["checks"]["canonical_builtin_release_prompt_matrix"]["passed"] is True
    assert payload["gates"]["checks"][
        "authoritative_chat_app_adaptive_runtime_defaults"
    ]["passed"] is True


def test_release_adaptive_defaults_are_literal_and_immutable():
    with pytest.raises(TypeError):
        gate.AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS["adaptive_exit_tol"] = 999.0


def test_coordinated_chat_app_default_drift_cannot_redefine_release(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate.chat_app, "DEFAULT_ADAPTIVE_EXIT_TOL", 999.0)

    payload = _run(tmp_path)

    check = payload["gates"]["checks"][
        "canonical_chat_app_adaptive_defaults_unchanged"
    ]
    assert check["passed"] is False
    assert check["mismatches"]["adaptive_exit_tol"] == {
        "actual": 999.0,
        "required": 0.001,
    }
    assert payload["gates"]["passed"] is False


def test_custom_checkpoint_and_metadata_are_diagnostic_only(tmp_path):
    weights, metadata = _artifacts(tmp_path)

    payload = _run(tmp_path, weights=weights, metadata=metadata)

    identity = payload["gates"]["checks"][
        "canonical_default_checkpoint_and_metadata_identity"
    ]
    assert identity["passed"] is False
    assert identity["diagnostic_only"] is True
    assert payload["claim_scope"]["artifact_kind"] == "diagnostic_custom_artifacts"
    assert payload["claim_scope"]["release_eligible"] is False
    assert payload["gates"]["passed"] is False


@pytest.mark.parametrize(
    ("load_overrides", "label_count"),
    [
        ({"model_size": "base"}, 10),
        ({"feature_mode": "legacy"}, 10),
        ({}, 9),
    ],
)
def test_noncanonical_model_identity_blocks_release(
    tmp_path, load_overrides, label_count
):
    _StubEngine.load_status_overrides = load_overrides
    _StubEngine.available_label_count = label_count

    payload = _run(tmp_path)

    identity = payload["gates"]["checks"]["canonical_model_runtime_identity"]
    assert identity["passed"] is False
    assert payload["claim_scope"]["release_eligible"] is False
    assert payload["gates"]["passed"] is False


@pytest.mark.parametrize(
    ("status_key", "mutated_value"),
    [
        ("loaded", False),
        ("runtime_compute_supported", False),
        ("sessions", 1),
        ("reasoning_cycles", 8),
    ],
)
def test_noncanonical_loaded_runtime_status_blocks_release(
    tmp_path, status_key, mutated_value
):
    _StubEngine.status_overrides = {status_key: mutated_value}

    payload = _run(tmp_path)

    identity = payload["gates"]["checks"]["canonical_model_runtime_identity"]
    assert identity["passed"] is False
    assert payload["gates"]["passed"] is False


def test_packaged_behavior_drift_blocks_release_even_when_each_mode_is_stable(
    tmp_path,
):
    payload = _run(
        tmp_path,
        packaged_engine_factory=_PackagedBehaviorDriftEngine,
    )

    assert payload["summary"]["any_fidelity_mismatch_count"] == 0
    parity = payload["gates"]["checks"][
        "source_package_engine_exact_behavior_parity"
    ]
    assert parity["passed"] is False
    assert parity["violation_count"] == gate.CANONICAL_RELEASE_PROMPT_MATRIX_COUNT * 2
    assert all(
        row["mismatched_fields"] == ["response"]
        for row in parity["violations"]
    )
    assert payload["gates"]["passed"] is False


@pytest.mark.parametrize(
    ("runtime_key", "mutated_value"),
    [
        ("adaptive_exit_tol", 999.0),
        ("adaptive_exit_entropy", 999.0),
        ("prediction_stability_patience", 1),
        ("prediction_stability_tol", 1.0),
    ],
)
def test_noncanonical_engine_adaptive_defaults_block_release(
    tmp_path, runtime_key, mutated_value
):
    _StubEngine.runtime_default_overrides = {runtime_key: mutated_value}

    payload = _run(tmp_path)

    check = payload["gates"]["checks"][
        "authoritative_chat_app_adaptive_runtime_defaults"
    ]
    assert check["passed"] is False
    assert check["mismatches_by_surface"]["source"][runtime_key][
        "actual"
    ] == float(mutated_value)
    assert check["mismatches_by_surface"]["packaged"][runtime_key][
        "actual"
    ] == float(mutated_value)
    assert payload["gates"]["passed"] is False


@pytest.mark.parametrize(
    ("telemetry_key", "mutated_value", "violation"),
    [
        ("exit_tol", 999.0, "configured_exit_tol_mismatch"),
        (
            "exit_entropy_threshold",
            999.0,
            "configured_exit_entropy_threshold_mismatch",
        ),
        (
            "prediction_stability_patience",
            1.0,
            "configured_prediction_stability_patience_mismatch",
        ),
        (
            "prediction_stability_tol",
            1.0,
            "configured_prediction_stability_tol_mismatch",
        ),
    ],
)
def test_noncanonical_observed_adaptive_controls_block_release(
    tmp_path, telemetry_key, mutated_value, violation
):
    _StubEngine.adaptive_compute_overrides = {telemetry_key: mutated_value}

    payload = _run(tmp_path)

    assert payload["gates"]["checks"][
        "authoritative_chat_app_adaptive_runtime_defaults"
    ]["passed"] is True
    contract = payload["gates"]["checks"][
        "runtime_release_verifier_contract_observed"
    ]
    assert contract["passed"] is False
    adaptive_violations = [
        row for row in contract["violations"] if row["mode"] == "adaptive"
    ]
    assert len(adaptive_violations) == gate.CANONICAL_RELEASE_PROMPT_MATRIX_COUNT * 2
    assert all(violation in row["violations"] for row in adaptive_violations)
    assert payload["gates"]["passed"] is False


def test_gate_reports_response_and_candidate_order_mismatches(tmp_path):
    _StubEngine.mismatch_prompt = gate.FROZEN_RELEASE_PROMPT_MATRIX[1]["prompt"]

    payload = _run(tmp_path)

    assert payload["gates"]["passed"] is False
    assert payload["summary"]["response_text_mismatch_count"] == 1
    assert payload["summary"]["top_candidate_text_order_mismatch_count"] == 1
    assert payload["summary"]["any_fidelity_mismatch_count"] == 1
    comparison = payload["prompt_results"][1]["comparison"]
    assert comparison["mismatch_kinds"] == [
        "response_text",
        "top_candidate_text_order",
    ]


def test_gate_rejects_nonfinite_runtime_evidence(tmp_path):
    _StubEngine.nonfinite_timing = True

    with pytest.raises(ValueError, match="must be finite"):
        _run(tmp_path)


@pytest.mark.parametrize(
    ("candidate_count", "duplicate", "message"),
    [
        (4, False, "exactly 5"),
        (5, True, "must be unique"),
    ],
)
def test_gate_rejects_truncated_or_duplicate_candidate_evidence(
    tmp_path, candidate_count, duplicate, message
):
    _StubEngine.candidate_count = candidate_count
    _StubEngine.duplicate_candidate = duplicate

    with pytest.raises(ValueError, match=message):
        _run(tmp_path)


def test_gate_rejects_string_boolean_runtime_evidence(tmp_path):
    _StubEngine.applied_value = "false"

    with pytest.raises(ValueError, match="must be a boolean"):
        _run(tmp_path)


def test_prediction_stable_exit_requires_decision_margin_floor(tmp_path):
    _StubEngine.adaptive_compute_overrides = {
        "prediction_decision_margin": gate.PREDICTION_STABILITY_MARGIN / 2,
    }

    payload = _run(tmp_path)

    assert payload["gates"]["passed"] is False
    violations = payload["gates"]["checks"][
        "runtime_release_verifier_contract_observed"
    ]["violations"]
    assert all(row["mode"] == "adaptive" for row in violations)
    assert all(
        "prediction_stable_decision_margin_below_floor" in row["violations"]
        for row in violations
    )


def test_decision_reference_budget_exit_is_allowed_with_exact_cycle_evidence(
    tmp_path,
):
    _StubEngine.adaptive_compute_overrides = {
        "exit_reason": "decision_reference_budget",
        "cycles_used": 3,
    }

    payload = _run(tmp_path)

    contract = payload["gates"]["checks"][
        "runtime_release_verifier_contract_observed"
    ]
    assert contract["passed"] is True
    assert contract["required_decision_reference_cycles"] == 3
    assert payload["summary"]["adaptive_exit_reasons"] == {
        "decision_reference_budget": gate.CANONICAL_RELEASE_PROMPT_MATRIX_COUNT
    }


def test_decision_reference_budget_exit_rejects_wrong_cycle(tmp_path):
    _StubEngine.adaptive_compute_overrides = {
        "exit_reason": "decision_reference_budget",
        "cycles_used": 2,
    }

    payload = _run(tmp_path)

    violations = payload["gates"]["checks"][
        "runtime_release_verifier_contract_observed"
    ]["violations"]
    assert all(row["mode"] == "adaptive" for row in violations)
    assert all(
        "decision_reference_budget_cycle_mismatch" in row["violations"]
        for row in violations
    )


def test_source_package_byte_mismatch_blocks_release_pass(tmp_path):
    source = tmp_path / "source_runtime.py"
    package = tmp_path / "package_runtime.py"
    source.write_bytes(b"same contract\n")
    package.write_bytes(b"mutated contract\n")

    payload = _run(
        tmp_path,
        source_package_parity_pairs=[("runtime", source, package)],
    )

    assert payload["summary"]["any_fidelity_mismatch_count"] == 0
    assert payload["source_package_parity"]["passed"] is False
    assert payload["source_package_parity"]["pairs"][0]["exact_bytes"] is False
    assert payload["gates"]["checks"]["source_package_runtime_exact_parity"]["passed"] is False
    assert payload["gates"]["passed"] is False


def test_strict_json_rejects_nonfinite_payloads():
    with pytest.raises(ValueError):
        gate._strict_json({"bad": float("nan")})


def test_main_writes_artifact_and_enforces_failed_gate(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    weights, metadata = _artifacts(tmp_path)
    prompts_path = tmp_path / "prompts.json"
    prompts_path.write_text('["one"]', encoding="utf-8")

    monkeypatch.setattr(gate, "resolve_device", lambda *_args, **_kwargs: ("cpu", {"resolved": "cpu"}))
    monkeypatch.setattr(gate, "configure_torch_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda **_kwargs: {"gates": {"passed": False}, "finite": 1.0},
    )

    code = gate.main(
        [
            "--weights",
            str(weights),
            "--meta",
            str(metadata),
            "--prompts-json",
            str(prompts_path),
            "--output",
            str(output),
            "--enforce-gates",
        ]
    )

    assert code == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "finite": 1.0,
        "gates": {"passed": False},
    }
