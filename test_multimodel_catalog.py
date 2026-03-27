from pathlib import Path

from source.multimodel_catalog import ModelRecord, choose_auto_model


def _record(key: str, kind: str, capabilities: tuple[str, ...], score: float | None = None) -> ModelRecord:
    return ModelRecord(
        key=key,
        label=key,
        family="test",
        kind=kind,
        capabilities=capabilities,
        zip_path=Path(f"{key}.zip"),
        common_row_key=key,
        common_overall_exact=score,
    )


def test_auto_prefers_image_model_for_visual_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("v36_native", "native_image", ("image",), 0.15),
        _record("v38_native_xlite_fp16", "native_image", ("image",), 0.01),
    ]
    chosen, reason = choose_auto_model(records, "Generate a cinematic poster of a lighthouse at night.")
    assert chosen is not None
    assert chosen.key == "v36_native"
    assert "image" in reason.lower()


def test_auto_prefers_fast_model_for_short_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("v30_lite", "champion_chat", ("chat",), 0.01),
    ]
    chosen, _reason = choose_auto_model(records, "Quick answer please.")
    assert chosen is not None
    assert chosen.key == "v30_lite"


def test_auto_prefers_best_reasoning_model_for_code_prompt() -> None:
    records = [
        _record("qwen_v28", "qwen_adapter", ("chat",), 0.02),
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("v35_final", "champion_chat", ("chat",), 0.15),
    ]
    chosen, _reason = choose_auto_model(records, "Debug this Python stack trace and explain the root cause.")
    assert chosen is not None
    assert chosen.key == "v33_final"


def test_auto_prefers_math_specialist_for_equation_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("math_equation_micro_v1", "math_equation", ("chat",), None),
    ]
    chosen, reason = choose_auto_model(records, "Solve 3*x^2 - 12 = 0.")
    assert chosen is not None
    assert chosen.key == "math_equation_micro_v1"
    assert "math" in reason.lower() or "equation" in reason.lower()
