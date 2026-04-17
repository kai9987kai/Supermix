from source.omni_collective_v41_model import _prompt_variants_v41


def test_prompt_variants_v41_adds_hidden_plan_and_reflection():
    variants = _prompt_variants_v41("Explain why regression tests matter after a refactor.")

    assert any("hidden plan" in item.lower() for item in variants)
    assert any("reflection pass" in item.lower() for item in variants)
    assert any("human reader" in item.lower() for item in variants)


def test_prompt_variants_v41_adds_code_repair_prompt_for_debugging_requests():
    variants = _prompt_variants_v41("The stack trace says Access is denied during checkpoint save. What should I do?")

    assert any("code-review and repair pass" in item.lower() for item in variants)
