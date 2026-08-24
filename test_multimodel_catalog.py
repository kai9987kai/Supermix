from pathlib import Path

from source.multimodel_catalog import (
    MODEL_SPECS,
    ModelRecord,
    choose_auto_model,
    describe_model_artifact_name,
    discover_model_records,
)


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


def test_auto_prefers_dcgan_model_for_gan_prompt() -> None:
    records = [
        _record("v36_native", "native_image", ("image",), 0.15),
        _record("dcgan_mnist_model", "dcgan_image", ("image",), None),
        _record("dcgan_v2_in_progress", "dcgan_image", ("image",), None),
    ]
    chosen, reason = choose_auto_model(records, "Generate a DCGAN CIFAR retro sample grid.")
    assert chosen is not None
    assert chosen.key == "dcgan_v2_in_progress"
    assert "dcgan" in reason.lower() or "gan" in reason.lower()


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


def test_auto_prefers_latest_stable_model_for_latest_prompt() -> None:
    records = [
        _record("omni_collective_v48", "omni_collective_v48", ("chat", "vision"), 0.7557),
        _record("omni_collective_v47", "omni_collective_v47", ("chat", "vision"), 0.7110),
        _record("omni_collective_v46", "omni_collective_v46", ("chat", "vision"), 0.71),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.60),
    ]
    chosen, reason = choose_auto_model(records, "Use the latest local frontier model for this task.")
    assert chosen is not None
    assert chosen.key == "omni_collective_v47"
    assert "latest" in reason.lower()


def test_auto_prefers_v46_for_general_reasoning_when_v48_is_present() -> None:
    records = [
        _record("omni_collective_v48", "omni_collective_v48", ("chat", "vision"), 0.7557),
        _record("omni_collective_v47", "omni_collective_v47", ("chat", "vision"), 0.7110),
        _record("omni_collective_v46", "omni_collective_v46", ("chat", "vision"), 0.7477),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.60),
    ]
    chosen, reason = choose_auto_model(records, "Analyze this algorithm tradeoff and suggest the best architecture.")
    assert chosen is not None
    assert chosen.key == "omni_collective_v46"
    assert "reasoning" in reason.lower() or "coding" in reason.lower()


def test_auto_prefers_v40_for_benchmark_reasoning_even_when_v48_exists() -> None:
    records = [
        _record("omni_collective_v48", "omni_collective_v48", ("chat", "vision"), 0.7557),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.60),
    ]
    chosen, reason = choose_auto_model(records, "Analyze this benchmark tradeoff and recommend the best local reasoning model.")
    assert chosen is not None
    assert chosen.key == "v40_benchmax"
    assert "benchmark" in reason.lower()


def test_auto_prefers_math_specialist_for_equation_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("math_equation_micro_v1", "math_equation", ("chat",), None),
    ]
    chosen, reason = choose_auto_model(records, "Solve 3*x^2 - 12 = 0.")
    assert chosen is not None
    assert chosen.key == "math_equation_micro_v1"
    assert "math" in reason.lower() or "equation" in reason.lower()


def test_auto_prefers_protein_specialist_for_protein_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("protein_folding_micro_v1", "protein_folding", ("chat",), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(records, "Why does pLDDT matter in protein structure prediction?")
    assert chosen is not None
    assert chosen.key == "protein_folding_micro_v1"
    assert "protein" in reason.lower() or "fold" in reason.lower()


def test_auto_prefers_mattergen_specialist_for_materials_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("mattergen_micro_v1", "mattergen_generation", ("chat",), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(records, "Generate a CIF-style perovskite absorber candidate with a 1.8 eV band gap.")
    assert chosen is not None
    assert chosen.key == "mattergen_micro_v1"
    assert "material" in reason.lower() or "crystal" in reason.lower()


def test_auto_prefers_3d_specialist_for_openscad_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("three_d_generation_micro_v1", "three_d_generation", ("chat",), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(records, "Write a small OpenSCAD model for a phone stand with a centered hole.")
    assert chosen is not None
    assert chosen.key == "three_d_generation_micro_v1"
    assert "3d" in reason.lower() or "openscad" in reason.lower() or "cad" in reason.lower()


def test_auto_prefers_uploaded_image_specialist() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
        _record("science_vision_micro_v1", "image_recognition", ("chat", "vision"), None),
        _record("omni_collective_v1", "omni_collective", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(
        records,
        "What does this uploaded image show?",
        action_mode="auto",
        uploaded_image_path=r"C:\temp\sample.png",
    )
    assert chosen is not None
    assert chosen.key == "science_vision_micro_v1"
    assert "image" in reason.lower() or "visual" in reason.lower()


def test_auto_prefers_newer_omni_collective_for_model_choice_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v8_preview", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v7", "omni_collective_v7", ("chat", "vision"), 0.1067),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
        _record("omni_collective_v1", "omni_collective", ("chat", "vision"), None),
        _record("omni_collective_v2", "omni_collective", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(records, "Which model should I use for this mixed coding and image-analysis task?")
    assert chosen is not None
    assert chosen.key == "omni_collective_v8"
    assert "model" in reason.lower() or "fused" in reason.lower()


def test_auto_prefers_v40_for_reasoning_prompt() -> None:
    records = [
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v8_preview", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v7", "omni_collective_v7", ("chat", "vision"), 0.1067),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v2", "omni_collective", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(records, "Analyze this algorithm tradeoff and suggest the best architecture.")
    assert chosen is not None
    assert chosen.key == "v40_benchmax"
    assert "reasoning" in reason.lower() or "strongest" in reason.lower()


def test_auto_does_not_promote_v8_preview_over_default_reasoning_route() -> None:
    records = [
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v8_preview", "omni_collective_v8", ("chat", "vision"), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
    ]
    chosen, reason = choose_auto_model(records, "Analyze this benchmark tradeoff and recommend the best local reasoning model.")
    assert chosen is not None
    assert chosen.key == "v40_benchmax"
    assert "reasoning" in reason.lower() or "strongest" in reason.lower()


def test_describe_model_artifact_name_identifies_known_and_unknown_artifacts() -> None:
    finished = describe_model_artifact_name("supermix_omni_collective_v8_frontier_20260408.zip")
    assert finished["known"] is True
    assert finished["key"] == "omni_collective_v8"
    assert "vision" in finished["capabilities"]

    known = describe_model_artifact_name("supermix_omni_collective_v8_preview_20260407_001155.zip")
    assert known["known"] is True
    assert known["key"] == "omni_collective_v8_preview"
    assert "chat" in known["capabilities"]

    unknown = describe_model_artifact_name("totally_custom_external_model.zip")
    assert unknown["known"] is False
    assert unknown["label"] == "totally_custom_external_model"


def test_cognitive_leap_packages_have_distinct_bounded_champion_contracts(
    tmp_path: Path,
) -> None:
    expected = {
        "supermix_cognitive_leap_v50_chat_20260812.zip": (
            "cognitive_leap_v50",
            "champion_model_chat_v50_cognitive_leap.pth",
            "chat_model_meta_v50_cognitive_leap.json",
        ),
        "supermix_cognitive_leap_ultra_v51_demo_20260812.zip": (
            "cognitive_leap_ultra_v51_demo",
            "cognitive_leap_ultra_v51_trained.pth",
            "chat_demo_meta_v51.json",
        ),
        "supermix_cognitive_leap_ultra_v51_1_balanced_blend30_20260812.zip": (
            "cognitive_leap_ultra_v51_1_balanced_blend30",
            "cognitive_leap_ultra_v51_1_balanced_blend30.pth",
            "chat_demo_meta_v51_1_balanced_blend30.json",
        ),
    }
    for package_name in expected:
        (tmp_path / package_name).write_bytes(b"catalog-discovery-does-not-load-archives")

    records = {
        record.key: record
        for record in discover_model_records(
            models_dir=tmp_path,
            common_summary_path=tmp_path / "missing-summary.json",
        )
    }

    assert set(expected) == {
        package_name for package_name in expected if describe_model_artifact_name(package_name)["known"]
    }
    for package_name, (model_key, weights_name, meta_name) in expected.items():
        description = describe_model_artifact_name(package_name)
        assert description["key"] == model_key
        record = records[model_key]
        assert record.kind == "champion_chat"
        assert record.preferred_weights == (weights_name,)
        assert record.preferred_meta == (meta_name,)
        assert record.common_overall_exact is None

    specs = {spec.key: spec for spec in MODEL_SPECS}
    for model_key in (
        "cognitive_leap_v50",
        "cognitive_leap_ultra_v51_demo",
        "cognitive_leap_ultra_v51_1_balanced_blend30",
    ):
        assert specs[model_key].manual_only is True

    for model_key in (
        "cognitive_leap_ultra_v51_demo",
        "cognitive_leap_ultra_v51_1_balanced_blend30",
    ):
        note = specs[model_key].note.lower()
        assert "synthetic" in note
        assert "not evidence of broad assistant superiority" in note

    balanced_key = "cognitive_leap_ultra_v51_1_balanced_blend30"
    balanced = records[balanced_key]
    assert balanced.label == "Cognitive Leap Ultra v51.1 Balanced Blend (Unpromoted)"
    assert balanced.manual_only is True
    assert "failed the 15/20 seed criterion" in balanced.note
    assert "aggregate synthetic mod-10 improvement" in balanced.note
    assert balanced.to_dict()["manual_only"] is True

    chosen, reason = choose_auto_model(
        list(records.values()),
        "Analyze this synthetic modular arithmetic benchmark.",
    )
    assert chosen is None
    assert "fell back" in reason.lower()
