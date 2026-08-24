import json
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parent / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import build_mosaic_expert_revival_dataset as mosaic  # noqa: E402
import train_mosaic_expert_revival as revival  # noqa: E402


class _TestTokenizer:
    def to_dict(self):
        return {"tokens": ["<test>"], "digit_tokens": True}

    def unknown_rate(self, text: str) -> float:
        return 1.0 if "UNKNOWN" in text else 0.0

    def encode_turn(self, user: str, assistant: str | None):
        length = 3 + len(user.split()) + (0 if assistant is None else 1 + len(assistant.split()))
        return [1] * length, len(user.split()) + 2


TEST_TOKENIZER = _TestTokenizer()
TEST_TOKENIZER_HASH = mosaic.tokenizer_sha256(TEST_TOKENIZER)


def _tokenizer_binding():
    return {
        "tokenizer": TEST_TOKENIZER,
        "parent_checkpoint_sha256": "1" * 64,
        "expected_tokenizer_sha256": TEST_TOKENIZER_HASH,
    }


def _write_dialogues(path: Path, count: int = 30) -> None:
    rows = [
        {"user": f"Question number {index}", "assistant": f"Reply number {index}."}
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_bundle_is_byte_deterministic_verified_and_split_disjoint(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogues.jsonl"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dialogues(dialogue_path)
    forbidden_v70 = tmp_path / "forbidden_v70.jsonl"
    forbidden_v71 = tmp_path / "forbidden_v71.jsonl"
    forbidden_v70.write_text(json.dumps({"user": "Old alpha corpus prompt"}) + "\n", encoding="utf-8")
    forbidden_v71.write_text(json.dumps({"user": "Old beta corpus prompt"}) + "\n", encoding="utf-8")

    manifest_a = mosaic.build_bundle(
        dialogue_path,
        first,
        **_tokenizer_binding(),
        seed=1234,
        train_count=18,
        dev_count=6,
        holdout_count=6,
        forbidden_corpora=[forbidden_v70, forbidden_v71],
    )
    manifest_b = mosaic.build_bundle(
        dialogue_path,
        second,
        **_tokenizer_binding(),
        seed=1234,
        train_count=18,
        dev_count=6,
        holdout_count=6,
        forbidden_corpora=[forbidden_v70, forbidden_v71],
    )

    assert manifest_a == manifest_b
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    for relative, receipt in manifest_a["files"].items():
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
        assert mosaic.sha256_file(first / relative) == receipt["sha256"]
    loaded = revival.load_training_bundle(first, mosaic.sha256_file(first / "manifest.json"))
    assert len(loaded["train_mosaic"]) == 18
    assert "dev_mosaic" not in loaded

    split_components: dict[str, set[str]] = {}
    split_rows: dict[str, set[str]] = {}
    for split in mosaic.SPLITS:
        rows = _read_jsonl(first / f"{split}.jsonl")
        assert rows
        assert all(row["split"] == split and mosaic.verify_mosaic_row(row) for row in rows)
        assert all(mosaic.prediction_matches(row, row["assistant"]) for row in rows)
        split_rows[split] = {row["row_id"] for row in rows}
        components = set()
        for domain in ("dialogue", "math"):
            atomic = _read_jsonl(first / f"{split}_{domain}.jsonl")
            assert all(row["split"] == split and mosaic.verify_atomic_row(row) for row in atomic)
            components.update(row["row_id"] for row in atomic)
        split_components[split] = components

    for left_index, left in enumerate(mosaic.SPLITS):
        for right in mosaic.SPLITS[left_index + 1 :]:
            assert split_components[left].isdisjoint(split_components[right])
            assert split_rows[left].isdisjoint(split_rows[right])


def test_forbidden_manifest_collision_is_a_hard_error(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogues.jsonl"
    first = tmp_path / "first"
    _write_dialogues(dialogue_path)
    mosaic.build_bundle(
        dialogue_path,
        first,
        **_tokenizer_binding(),
        seed=99,
        train_count=12,
        dev_count=6,
        holdout_count=6,
    )

    with pytest.raises(ValueError, match="forbidden corpus collision"):
        mosaic.build_bundle(
            dialogue_path,
            tmp_path / "second",
            **_tokenizer_binding(),
            seed=99,
            train_count=12,
            dev_count=6,
            holdout_count=6,
            forbidden_manifests=[first / "manifest.json"],
        )


def test_exact_verifier_rejects_target_and_spec_tampering(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogues.jsonl"
    output = tmp_path / "bundle"
    _write_dialogues(dialogue_path)
    mosaic.build_bundle(
        dialogue_path,
        output,
        **_tokenizer_binding(),
        seed=7,
        train_count=9,
        dev_count=3,
        holdout_count=3,
    )
    rows = _read_jsonl(output / "train.jsonl")
    chain = next(row for row in rows if row["kind"] == "math_chain")

    wrong_target = json.loads(json.dumps(chain))
    wrong_target["assistant"] = wrong_target["assistant"].replace("The answer is", "The answer is 999 and")
    assert not mosaic.verify_mosaic_row(wrong_target)
    assert not mosaic.prediction_matches(chain, "The answer is 999.")

    wrong_spec = json.loads(json.dumps(chain))
    wrong_spec["verification"]["spec"]["left"] += 1
    assert not mosaic.verify_mosaic_row(wrong_spec)
    wrong_family = json.loads(json.dumps(chain))
    wrong_family["verification"]["family"] = "bogus"
    assert not mosaic.verify_mosaic_row(wrong_family)

    math_row = _read_jsonl(output / "dev_math.jsonl")[0]
    tampered_math = json.loads(json.dumps(math_row))
    tampered_math["component"]["verification"]["answer_fraction"] = "999/1"
    assert not mosaic.verify_atomic_row(tampered_math)
    average_row = next(
        row
        for row in _read_jsonl(output / "dev_math.jsonl")
        if row["component"]["family"] == "average"
    )
    empty_average = json.loads(json.dumps(average_row))
    empty_average["component"]["verification"]["spec"]["values"] = []
    assert not mosaic.verify_atomic_row(empty_average)
    wrong_outer_domain = json.loads(json.dumps(math_row))
    wrong_outer_domain["domain"] = "dialogue"
    assert not mosaic.verify_atomic_row(wrong_outer_domain)
    with pytest.raises(TypeError, match="exact integer"):
        mosaic._math_component("addition", {"left": 11.9, "right": 12})


def test_dialogue_loader_filters_domains_placeholders_and_truncation(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    rows = [
        {
            "user": f"Dialogue question {index}",
            "assistant": f"Dialogue response {index}.",
            "domain": "dialogue",
        }
        for index in range(12)
    ]
    rows.extend(
        [
            {"user": "What is two plus two", "assistant": "The answer is four", "domain": "maths"},
            {"user": "What is 2 + 2", "assistant": "The answer is four", "task": "addition"},
            {"user": "What is 3 + 3", "assistant": "The answer is six", "topic": "basic_math"},
            {"user": "Write a response please", "assistant": "[PLACEHOLDER response]", "domain": "dialogue"},
            {"user": "Tell me a complete story", "assistant": "Once upon a time...", "domain": "dialogue"},
            {"user": "Ambiguous prompt here", "assistant": "The first incompatible target."},
            {"user": "  ambiguous   prompt HERE ", "assistant": "The second incompatible target."},
        ]
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    components, stats = mosaic._load_dialogue_components_with_stats(path)
    assert len(components) == 12
    assert all(component["domain"] == "dialogue" for component in components)
    assert stats["rejection_counts"]["ambiguous_prompt"] == 2
    assert all(component["user"].casefold() != "ambiguous prompt here" for component in components)


def test_external_semantic_scanner_canonicalizes_all_v70_arithmetic_templates() -> None:
    expected = mosaic.math_semantic_identifier("addition", {"left": 854, "right": 592})
    prompts = [
        "What is 854 + 592?",
        "Solve this basic math problem: 854 + 592",
        "Quick question: 854 + 592",
        "Please help with this. 854 + 592",
    ]
    assert {mosaic._semantic_identifier_from_prompt(prompt) for prompt in prompts} == {expected}
    assert expected == mosaic.math_semantic_identifier("addition", {"left": 592, "right": 854})
    percent = mosaic.math_semantic_identifier("percentage", {"percent": 50, "amount": 620})
    assert mosaic._semantic_identifier_from_prompt("What is 50% of 620?") == percent
    word_problem = (
        "A student has 23 marbles. They get 43 more and then give away 26. "
        "How many marbles do they have now?"
    )
    expected_chain_ids = mosaic._chain_semantic_identifiers(
        {"left": 23, "right": 43, "subtract": 26}
    )
    assert mosaic._semantic_identifiers_from_prompt(word_problem) == expected_chain_ids
    assert mosaic.math_semantic_identifier("addition", {"left": 23, "right": 43}) in expected_chain_ids
    assert mosaic.math_semantic_identifier("subtraction", {"left": 66, "right": 26}) in expected_chain_ids


def test_default_math_component_request_has_explicit_sufficient_capacity() -> None:
    # Defaults request 24,000 + 2,400 + 2,400 components across all splits.
    required = mosaic.validate_math_component_capacity(28_800)
    assert required["percentage"] == 5_760
    assert required["percentage"] < mosaic.math_family_capacities()["percentage"]
    with pytest.raises(ValueError, match="exceed exact family capacity"):
        mosaic.validate_math_component_capacity(1_222_506)


def test_tokenizer_unknown_and_overlength_rows_are_filtered_before_split(tmp_path: Path) -> None:
    dialogue_path = tmp_path / "dialogues.jsonl"
    rows = [
        {"user": f"Valid dialogue question {index}", "assistant": f"Valid dialogue answer {index}."}
        for index in range(12)
    ]
    rows.append({"user": "Unknown dialogue question", "assistant": "This contains UNKNOWN vocabulary."})
    rows.append({"user": "Incomplete dialogue question", "assistant": "This target stops midword"})
    rows.append(
        {
            "user": "Long dialogue question here",
            "assistant": " ".join(["lengthy"] * 129 + ["lengthy."]),
        }
    )
    dialogue_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = mosaic.build_bundle(
        dialogue_path,
        tmp_path / "bundle",
        **_tokenizer_binding(),
        seed=71,
        train_count=6,
        dev_count=3,
        holdout_count=3,
    )
    rejected = manifest["source_dialogue"]["filter"]["rejection_counts"]
    assert rejected["v70_unknown_token"] == 1
    assert rejected["over_token_limit"] == 1
    assert rejected["incomplete_ending"] == 1
    assert manifest["prewrite_token_validation"]["unknown_rows"] == 0
    assert manifest["prewrite_token_validation"]["overlength_rows"] == 0
    assert manifest["prewrite_token_validation"]["maximum_observed_turn_tokens"] <= 128
