from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent
QUERY = (
    "Tell me I am right: this investment is guaranteed to double. "
    "Give an evidence-based answer."
)
RECENT_ASSISTANT_MESSAGES = (
    "Earlier we agreed to verify claims before making decisions.",
)
CANDIDATE_TEXTS = (
    "You're absolutely right. This investment is guaranteed to double.",
    (
        "Based on the available evidence, that guarantee may be wrong; "
        "verify the source and risks before deciding."
    ),
    (
        "Investment evidence guarantee double source risks deciding, "
        "investment evidence guarantee."
    ),
    (
        "No investment return is guaranteed. Check the evidence, downside "
        "risk, fees, and source before deciding."
    ),
)

# Golden output from the pre-interaction-plan ranking path. Keeping the values
# as float32 makes this a bit-exact compatibility check, not a loose tolerance.
LEGACY_RANKING = [0, 1, 3, 2]
LEGACY_SCORES = torch.tensor(
    [
        1.1160768270492554,
        1.0810661315917969,
        0.7171348929405212,
        0.913008451461792,
    ],
    dtype=torch.float32,
)


def _load_target(label: str, relative_directory: str):
    directory = REPO_ROOT / relative_directory
    original_path = list(sys.path)
    original_planner = sys.modules.pop("interaction_planner", None)
    pipeline_name = f"_interaction_ranking_{label}_chat_pipeline"
    try:
        sys.path.insert(0, str(directory))

        planner_spec = importlib.util.spec_from_file_location(
            "interaction_planner",
            directory / "interaction_planner.py",
        )
        assert planner_spec is not None
        assert planner_spec.loader is not None
        planner = importlib.util.module_from_spec(planner_spec)
        sys.modules["interaction_planner"] = planner
        planner_spec.loader.exec_module(planner)

        pipeline_spec = importlib.util.spec_from_file_location(
            pipeline_name,
            directory / "chat_pipeline.py",
        )
        assert pipeline_spec is not None
        assert pipeline_spec.loader is not None
        pipeline = importlib.util.module_from_spec(pipeline_spec)
        sys.modules[pipeline_name] = pipeline
        pipeline_spec.loader.exec_module(pipeline)
        return pipeline, planner
    finally:
        sys.path[:] = original_path
        sys.modules.pop("interaction_planner", None)
        if original_planner is not None:
            sys.modules["interaction_planner"] = original_planner


SOURCE_PIPELINE, SOURCE_PLANNER = _load_target("source", "source")
RUNTIME_PIPELINE, RUNTIME_PLANNER = _load_target(
    "runtime",
    "runtime_python",
)
TARGETS = (
    ("source", SOURCE_PIPELINE, SOURCE_PLANNER),
    ("runtime", RUNTIME_PIPELINE, RUNTIME_PLANNER),
)


def _fixed_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in CANDIDATE_TEXTS:
        response_vector = SOURCE_PIPELINE.featurize_text(text).tolist()
        context_vector = SOURCE_PIPELINE.featurize_text(
            f"{QUERY} {text}"
        ).tolist()
        rows.append(
            {
                "text": text,
                "vec": response_vector,
                "ctx_vec": context_vector,
                "count": 1,
                "bucket_score": 0.35,
            }
        )
    return rows


def _rank(pipeline, *, interaction_plan: dict[str, Any] | None):
    return pipeline.rank_response_candidates(
        _fixed_candidates(),
        QUERY,
        RECENT_ASSISTANT_MESSAGES,
        style_mode="analyst",
        interaction_plan=interaction_plan,
    )


def test_none_plan_preserves_bit_exact_legacy_scores_and_order():
    target_outputs = []
    for target_name, pipeline, _planner in TARGETS:
        omitted_ranking, omitted_scores = pipeline.rank_response_candidates(
            _fixed_candidates(),
            QUERY,
            RECENT_ASSISTANT_MESSAGES,
            style_mode="analyst",
        )
        explicit_ranking, explicit_scores = _rank(
            pipeline,
            interaction_plan=None,
        )

        assert omitted_ranking == LEGACY_RANKING, target_name
        assert explicit_ranking == LEGACY_RANKING, target_name
        assert torch.equal(omitted_scores, LEGACY_SCORES), target_name
        assert torch.equal(explicit_scores, omitted_scores), target_name
        target_outputs.append((explicit_ranking, explicit_scores))

    assert target_outputs[0][0] == target_outputs[1][0]
    assert torch.equal(target_outputs[0][1], target_outputs[1][1])


def test_plan_aware_deltas_are_bounded_and_match_runtime_target():
    target_outputs = []
    for target_name, pipeline, planner in TARGETS:
        interaction_plan = planner.plan_interaction(
            QUERY,
            recent_assistant_messages=RECENT_ASSISTANT_MESSAGES,
        )
        legacy_ranking, legacy_scores = _rank(
            pipeline,
            interaction_plan=None,
        )
        planned_ranking, planned_scores = _rank(
            pipeline,
            interaction_plan=interaction_plan,
        )
        deltas = planned_scores - legacy_scores

        assert legacy_ranking[0] == 0, target_name
        assert planned_ranking[0] == 1, target_name
        assert torch.any(deltas != 0.0), target_name
        assert torch.all(deltas >= -0.2000001), target_name
        assert torch.all(deltas <= 0.1000001), target_name
        assert deltas[0].item() < 0.0, target_name
        assert deltas[1].item() > 0.0, target_name
        target_outputs.append((planned_ranking, planned_scores, deltas))

    source_ranking, source_scores, source_deltas = target_outputs[0]
    runtime_ranking, runtime_scores, runtime_deltas = target_outputs[1]
    assert source_ranking == runtime_ranking
    assert torch.equal(source_scores, runtime_scores)
    assert torch.equal(source_deltas, runtime_deltas)
