import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import qwen_supermix_pipeline as pipeline  # noqa: E402


def _verifier_metadata(
    verifier_type: str,
    expected_answer: str,
    family: str = "arithmetic",
    **extra,
):
    metadata = {
        "verifier_schema": "supermix-verifier-v1",
        "verifier_type": verifier_type,
        "expected_answer": expected_answer,
        "problem_family": family,
    }
    metadata.update(extra)
    return metadata


def test_chat_pair_verifier_metrics_recomputes_stale_correctness_metadata():
    metadata = _verifier_metadata(
        "integer",
        "145",
        verified_correct=True,
        verified=True,
        verifier_score=1.0,
        rule_reward=1.0,
    )
    wrong = pipeline.ChatPair(
        user="What is 57 + 88?",
        assistant="The final answer is 144.",
        metadata=metadata,
    )
    correct = pipeline.ChatPair(
        user=wrong.user,
        assistant="The final answer is 145.",
        metadata=metadata,
    )

    wrong_metrics = pipeline._chat_pair_verifier_metrics(wrong)
    correct_metrics = pipeline._chat_pair_verifier_metrics(correct)

    assert wrong_metrics["verifier_tagged"] == 1.0
    assert wrong_metrics["verification_available"] == 1.0
    assert wrong_metrics["verified_correct"] == 0.0
    assert wrong_metrics["verifier_score"] == 0.0
    assert wrong_metrics["rule_reward"] == -1.0
    assert correct_metrics["verified_correct"] == 1.0
    assert correct_metrics["verifier_score"] == 1.0
    assert correct_metrics["rule_reward"] == 1.0


def test_untagged_verifier_metrics_keep_legacy_shape_and_metadata_behavior():
    pair = pipeline.ChatPair(
        user="Explain caching.",
        assistant="Caching reuses previously computed data.",
        metadata={"verified_correct": True, "verifier_score": 0.8},
    )

    metrics = pipeline._chat_pair_verifier_metrics(pair)

    assert set(metrics) == {
        "verifier_score",
        "rule_reward",
        "verifier_difficulty",
        "verifier_bonus",
    }
    assert metrics["verifier_score"] == pytest.approx(1.0)


def test_nested_verifier_spec_survives_cache_round_trip_and_is_enforced(tmp_path):
    pair = pipeline.ChatPair(
        user="What is 57 + 88?",
        assistant="144",
        metadata={
            "verifier_spec": {
                "verifier_type": "integer",
                "expected_answer": "145",
                "problem_family": "arithmetic",
            },
            "verified_correct": True,
        },
    )
    path = tmp_path / "pairs.jsonl"

    pipeline.save_jsonl(path, [pair])
    loaded = pipeline.load_saved_chat_pairs(path)
    metrics = pipeline._chat_pair_verifier_metrics(loaded[0])

    assert loaded[0].metadata["verifier_spec"]["expected_answer"] == "145"
    assert metrics["verifier_tagged"] == 1.0
    assert metrics["verified_correct"] == 0.0


def test_distillation_rejects_wrong_teacher_candidate_and_keeps_verifier_spec():
    class Teacher:
        def generate_candidates(self, user_text, temperatures):
            assert user_text == "What is 57 + 88?"
            assert temperatures
            return [
                "The final answer is 144.",
                "The final answer is 145.",
            ]

    metadata = _verifier_metadata("integer", "145")
    original = pipeline.ChatPair(
        user="What is 57 + 88?",
        assistant="The final answer is 144.",
        metadata=metadata,
    )

    mixed, generated = pipeline.apply_supermix_distillation(
        train_pairs=[original],
        teacher=Teacher(),
        ratio=1.0,
        max_teacher_samples=1,
        seed=7,
        best_of=2,
    )

    teacher_pairs = [pair for pair in mixed if pair.source == "supermix_teacher"]
    assert generated == 1
    assert len(teacher_pairs) == 1
    assert teacher_pairs[0].assistant == "The final answer is 145."
    assert teacher_pairs[0].metadata["verifier_schema"] == "supermix-verifier-v1"

    correct_rank = pipeline._distillation_candidate_rank(
        user_text=original.user,
        candidate_text="The final answer is 145.",
        reference_text=original.assistant,
        density_bias=0.0,
        gain_bias=0.0,
        compactness_bias=0.0,
        metadata=metadata,
    )
    wrong_rank = pipeline._distillation_candidate_rank(
        user_text=original.user,
        candidate_text="The final answer is 144.",
        reference_text=original.assistant,
        density_bias=0.0,
        gain_bias=0.0,
        compactness_bias=0.0,
        metadata=metadata,
    )
    assert correct_rank[0] > wrong_rank[0]
    assert correct_rank[3]["verified_correct"] == 1.0
    assert wrong_rank[3]["verified_correct"] == 0.0


def test_preference_negative_picker_skips_verified_correct_alternatives():
    metadata = _verifier_metadata("integer", "145")

    rejected, _similarity, _score = pipeline._pick_rejected_candidate(
        user_text="What is 57 + 88?",
        chosen_text="145",
        generated=[
            "The final answer is 145.",
            "146",
            "The final answer is 144.",
        ],
        similarity_threshold=1.0,
        similarity_min=0.0,
        metadata=metadata,
    )

    verdict = pipeline._verify_chat_candidate("What is 57 + 88?", rejected, metadata)
    assert rejected
    assert verdict is not None
    assert verdict["available"] is True
    assert verdict["correct"] is False


def test_cached_distillation_merge_revalidates_tagged_pairs_and_reports_counts(capsys):
    base = [
        pipeline.ChatPair(
            user="Legacy prompt",
            assistant="Legacy answer",
            source="dataset",
        )
    ]
    tagged_metadata = _verifier_metadata(
        "integer",
        "145",
        verified_correct=True,
        verifier_score=1.0,
    )
    cached = [
        pipeline.ChatPair(
            user="What is 57 + 88?",
            assistant="145",
            source="supermix_teacher",
            metadata=tagged_metadata,
        ),
        pipeline.ChatPair(
            user="What is 57 + 88?",
            assistant="144",
            source="supermix_teacher",
            metadata=tagged_metadata,
        ),
        pipeline.ChatPair(
            user="Untagged cached prompt",
            assistant="Preserve legacy cached answer",
            source="supermix_teacher",
        ),
    ]

    mixed, added = pipeline._merge_distillation_pairs(base, cached, seed=11)
    output = capsys.readouterr().out
    responses = {(pair.user, pair.assistant) for pair in mixed}

    assert added == 2
    assert ("What is 57 + 88?", "145") in responses
    assert ("What is 57 + 88?", "144") not in responses
    assert ("Untagged cached prompt", "Preserve legacy cached answer") in responses
    assert "tagged=2 accepted=1 rejected=1" in output


def test_evaluation_reports_verified_accuracy_and_per_family_metrics(monkeypatch):
    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        chat_template = None
        eos_token_id = 2
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            del text, kwargs
            return FakeBatch({"input_ids": torch.tensor([[10, 11]], dtype=torch.long)})

        def decode(self, tokens, skip_special_tokens=True):
            del skip_special_tokens
            token = int(tokens.reshape(-1)[0].item())
            return {98: "B", 99: "145"}[token]

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=False)
            self._responses = iter((99, 98))

        def eval(self):
            return self

        def __call__(self, **batch):
            del batch
            return SimpleNamespace(loss=torch.tensor(0.5))

        def generate(self, **encoded):
            input_ids = encoded["input_ids"]
            response = torch.tensor([[next(self._responses)]], dtype=torch.long)
            return torch.cat((input_ids, response), dim=1)

    monkeypatch.setattr(
        pipeline,
        "_load_base_model_and_tokenizer",
        lambda *args, **kwargs: (FakeModel(), FakeTokenizer()),
    )
    monkeypatch.setattr(
        pipeline,
        "encode_for_causal_lm",
        lambda tokenizer, pair, max_length: {"input_ids": [1], "labels": [1]},
    )
    monkeypatch.setattr(
        pipeline,
        "collate_rows",
        lambda rows, pad_token_id: {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "labels": torch.tensor([[1]], dtype=torch.long),
        },
    )

    eval_pairs = [
        pipeline.ChatPair(
            user="What is 57 + 88?",
            assistant="145",
            metadata=_verifier_metadata("integer", "145", family="arithmetic"),
        ),
        pipeline.ChatPair(
            user="Choose A or B.",
            assistant="A",
            metadata=_verifier_metadata("multiple_choice", "A", family="multiple_choice"),
        ),
    ]

    metrics, rows = pipeline._evaluate_model_internal(
        base_model="fake",
        eval_pairs=eval_pairs,
        device=torch.device("cpu"),
        max_length=32,
        max_new_tokens=8,
    )

    assert metrics["verified_samples"] == 2.0
    assert metrics["verified_accuracy"] == pytest.approx(0.5)
    assert metrics["verified_accuracy_family_arithmetic"] == 1.0
    assert metrics["verified_accuracy_family_multiple_choice"] == 0.0
    assert [row["verified_correct"] for row in rows] == [True, False]
