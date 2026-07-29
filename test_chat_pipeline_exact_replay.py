import ast
import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import chat_pipeline


QUERY = (
    "A lab starts with 71 sample records and receives 8 more. It then makes "
    "3 copies of every record in the combined set and discards 2 damaged "
    "copies. How many usable copies remain?"
)
CORRECT = (
    "Combine the first two amounts: 71 + 8 = 79. Apply the multiplier: "
    "79 * 3 = 237. Remove 2: 237 - 2 = 235. Final answer: 235."
)
DISTRACTOR = (
    "Combine the first two amounts: 38 + 8 = 46. Apply the multiplier: "
    "46 * 3 = 138. Remove 9: 138 - 9 = 129. Final answer: 129."
)


def _candidate(text: str, *, exact_user_match: bool = False):
    response_vec = chat_pipeline.featurize_text(text).tolist()
    context_vec = chat_pipeline.featurize_text(
        QUERY if exact_user_match else text
    ).tolist()
    return {
        "text": text,
        "vec": response_vec,
        "ctx_vec": context_vec,
        "count": 1,
        "bucket_score": 0.8,
        "exact_user_match": exact_user_match,
    }


def test_exact_user_match_can_repeat_previous_grounded_answer():
    response = chat_pipeline.pick_response(
        candidates=[
            _candidate(CORRECT, exact_user_match=True),
            _candidate(DISTRACTOR),
        ],
        query_text=QUERY,
        recent_assistant_messages=[CORRECT],
        response_temperature=0.0,
        style_mode="balanced",
        creativity=0.0,
    )

    assert response == CORRECT


def test_unmarked_recent_answer_remains_blocked():
    for invalid_marker in (None, False, "true", 1):
        recent_candidate = _candidate(CORRECT)
        if invalid_marker is None:
            recent_candidate.pop("exact_user_match")
        else:
            recent_candidate["exact_user_match"] = invalid_marker
        response = chat_pipeline.pick_response(
            candidates=[
                recent_candidate,
                _candidate(DISTRACTOR),
            ],
            query_text=QUERY,
            recent_assistant_messages=[CORRECT],
            response_temperature=0.0,
            style_mode="balanced",
            creativity=0.0,
        )

        assert response == DISTRACTOR


def test_source_and_packaged_pick_response_are_ast_equivalent():
    root = Path(__file__).resolve().parent

    def function_dump(path: Path) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "pick_response"
        )
        return ast.dump(function, include_attributes=False)

    assert function_dump(root / "source" / "chat_pipeline.py") == function_dump(
        root / "runtime_python" / "chat_pipeline.py"
    )
