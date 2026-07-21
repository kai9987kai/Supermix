from source.omni_collective_v42_model import _budget_hint_v42, _prompt_variants_v42


def test_budget_hint_v42_routes_hard_and_simple_prompts():
    assert _budget_hint_v42("Debug this failing pytest traceback.") == "deep"
    assert _budget_hint_v42("Summarize this paragraph.", response_confidence=0.8, domain_confidence=0.8) == "medium"
    assert _budget_hint_v42("Say hello.", response_confidence=0.8, domain_confidence=0.8) == "short"


def test_prompt_variants_v42_adds_budget_verifier_and_agentic_prompts():
    variants = _prompt_variants_v42("Patch this Python test failure and explain the fix.")

    assert any("reasoning budget" in item.lower() for item in variants)
    assert any("verifier pass" in item.lower() for item in variants)
    assert any("agentic coding request" in item.lower() for item in variants)
