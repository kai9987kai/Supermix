import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source" / "grounding_runtime.py"
RUNTIME_PATH = ROOT / "runtime_python" / "grounding_runtime.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load("source_grounding_prompt_profile", SOURCE_PATH)
RUNTIME = _load("runtime_grounding_prompt_profile", RUNTIME_PATH)


def _profile():
    return {
        "schema_version": "supermix-prompt-understanding-v1",
        "knowledge": {
            "factual": True,
            "freshness_required": True,
            "evidence_requested": True,
            "citations_requested": True,
            "strict_evidence_only": False,
        },
        "safety": {
            "personal_crisis_signal": False,
            "urgent_health_signal": False,
        },
    }


def test_grounding_reads_current_interaction_epistemic_risk_key(monkeypatch):
    monkeypatch.setattr(
        SOURCE,
        "_load_prompt_understanding_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("supplied prompt profile was re-analyzed")
        ),
    )
    plan = SOURCE.plan_grounding(
        "Summarize the release.",
        interaction_plan={
            "risk": {"epistemic_score": 0.74},
            "deliberation": {"epistemic_risk": 0.61},
        },
        prompt_profile=_profile(),
    )

    assert plan["epistemic_risk"] == 0.74
    assert plan["freshness_required"] is True
    assert plan["citation_requested"] is True
    assert plan["evidence_recommended"] is True


def test_science_and_prediction_facets_request_evidence_without_claiming_tool_authority():
    profile = _profile()
    profile["knowledge"] = {
        key: False for key in profile["knowledge"]
    }
    profile["reasoning"] = {
        "mathematical": False,
        "scientific": True,
        "predictive": True,
        "causal": True,
    }

    plan = SOURCE.plan_grounding(
        "Predict the outcome of this scientific experiment.",
        prompt_profile=profile,
    )

    assert plan["evidence_recommended"] is True
    assert plan["reasoning_domains"] == ["science", "prediction", "causal"]
    assert {
        "scientific_reasoning",
        "prediction_requires_calibration",
        "causal_reasoning",
    } <= set(plan["reasons"])
    assert plan["authority"]["controls_routes"] is False


def test_forbidden_experiment_does_not_reactivate_science_grounding():
    prompt_module_path = ROOT / "source" / "prompt_understanding.py"
    prompt_module = _load("source_prompt_for_grounding_negation", prompt_module_path)
    query = "Do not design an experiment; just rewrite the paragraph."
    profile = prompt_module.analyze_prompt(query)

    plan = SOURCE.plan_grounding(query, prompt_profile=profile)

    assert profile["reasoning"]["scientific"] is False
    assert "evidence_or_calibration" not in profile["response_contract"]["required_capabilities"]
    assert plan["evidence_recommended"] is False
    assert "scientific_reasoning" not in plan["reasons"]


def test_evidence_bundle_uses_supplied_plan_without_replanning(monkeypatch):
    plan = SOURCE.plan_grounding(
        "What is the latest release?",
        prompt_profile=_profile(),
    )

    def unexpected_replan(*_args, **_kwargs):
        raise AssertionError("evidence bundle re-planned the prompt")

    monkeypatch.setattr(SOURCE, "plan_grounding", unexpected_replan)
    bundle = SOURCE.build_evidence_bundle(
        "What is the latest release?",
        [{"title": "Release", "text": "Version 52 is current."}],
        prompt_profile=_profile(),
        grounding_plan=plan,
    )

    assert bundle["plan"] == plan
    assert bundle["evidence"]


def test_finalizer_uses_supplied_plan_and_bundle_without_replanning(monkeypatch):
    prompt = "What is the latest release?"
    plan = SOURCE.plan_grounding(prompt, prompt_profile=_profile())
    bundle = SOURCE.build_evidence_bundle(
        prompt,
        [{"title": "Release", "text": "Version 52 is current."}],
        prompt_profile=_profile(),
        grounding_plan=plan,
    )

    def unexpected_replan(*_args, **_kwargs):
        raise AssertionError("grounding finalizer re-planned the prompt")

    monkeypatch.setattr(SOURCE, "plan_grounding", unexpected_replan)
    result = SOURCE.finalize_grounded_response(
        "Version 52 is current.",
        prompt,
        grounding_plan=plan,
        evidence_bundle=bundle,
        prompt_profile=_profile(),
    )

    assert result["changed"] is False
    assert result["reason"] == "audit_only"


def test_raw_prompt_retains_strict_evidence_override_authority(monkeypatch):
    ordinary_prompt = "Tell me a short story about a lighthouse."
    invented_strict_plan = {
        "strict_evidence_only": True,
        "max_evidence_items": 6,
    }
    monkeypatch.setattr(
        SOURCE,
        "plan_grounding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("supplied grounding plan was ignored")
        ),
    )
    ordinary = SOURCE.finalize_grounded_response(
        "A lighthouse shone across the bay.",
        ordinary_prompt,
        grounding_plan=invented_strict_plan,
        evidence_bundle=[],
        prompt_profile=_profile(),
    )
    raw_strict = SOURCE.finalize_grounded_response(
        "Paris.",
        "Use only the supplied evidence: what is the capital of France?",
        grounding_plan={
            "strict_evidence_only": False,
            "max_evidence_items": 6,
        },
        evidence_bundle=[],
        prompt_profile=_profile(),
    )

    assert ordinary["changed"] is False
    assert ordinary["reason"] == "audit_only"
    assert raw_strict["changed"] is True
    assert raw_strict["reason"] == "strict_evidence_no_evidence"


def test_source_runtime_grounding_profile_contracts_are_exact_mirrors():
    assert SOURCE_PATH.read_bytes() == RUNTIME_PATH.read_bytes()
