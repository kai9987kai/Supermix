import ast
import importlib.util
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
RUNTIME_DIR = ROOT / "runtime_python"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPTS = _load(
    "source_prompt_understanding_retrieval_api",
    SOURCE_DIR / "prompt_understanding.py",
)
CHAT_APP = _load(
    "source_prompt_understanding_retrieval_chat_app",
    SOURCE_DIR / "chat_app.py",
)


def _turns():
    return [
        {
            "id": "turn_1",
            "user": "Explain the database migration plan.",
            "assistant": "First back up the database, then apply migration 42.",
        }
    ]


def test_self_contained_pronoun_prompt_does_not_expand_history():
    prompt = (
        "A lab starts with 71 records and receives 8 more. It then makes "
        "3 copies of every record. How many copies are there?"
    )
    turns = _turns()
    profile = PROMPTS.analyze_prompt(prompt, recent_turns=turns)

    query = CHAT_APP._build_db_query(
        user=prompt,
        history=[(turns[0]["user"], turns[0]["assistant"])],
        memory_rows=[],
        prompt_profile=profile,
        recent_turns=turns,
    )

    assert query == prompt
    assert "migration 42" not in query


def test_true_followup_uses_only_bounded_recent_context():
    prompt = "Continue with the rollback steps."
    turns = _turns()
    profile = PROMPTS.analyze_prompt(prompt, recent_turns=turns)

    query = CHAT_APP._build_db_query(
        user=prompt,
        history=[(turns[0]["user"], turns[0]["assistant"])],
        memory_rows=[],
        max_turns=1,
        prompt_profile=profile,
        recent_turns=turns,
    )

    assert prompt in query
    assert (
        "migration plan" in query.lower()
        or "migration 42" in query.lower()
    )


def test_supplied_profile_prevents_reanalysis(monkeypatch):
    prompt = "Continue with the rollback steps."
    turns = _turns()
    profile = PROMPTS.analyze_prompt(prompt, recent_turns=turns)

    def unexpected_analysis(*_args, **_kwargs):
        raise AssertionError("retrieval re-analyzed an already profiled prompt")

    monkeypatch.setattr(CHAT_APP, "analyze_prompt", unexpected_analysis)
    query = CHAT_APP._build_db_query(
        user=prompt,
        history=[],
        memory_rows=[],
        prompt_profile=profile,
        recent_turns=turns,
    )

    assert prompt in query


def test_zero_context_budget_never_appends_prior_turns():
    prompt = "Continue with the rollback steps."
    turns = _turns()
    profile = PROMPTS.analyze_prompt(prompt, recent_turns=turns)

    query = CHAT_APP._build_db_query(
        user=prompt,
        history=[(turns[0]["user"], turns[0]["assistant"])],
        memory_rows=[],
        max_turns=0,
        prompt_profile=profile,
        recent_turns=turns,
    )

    assert query == prompt


def test_retrieval_signature_is_backward_compatible_and_profile_aware():
    parameters = inspect.signature(CHAT_APP._build_db_query).parameters

    assert list(parameters)[:4] == [
        "user",
        "history",
        "memory_rows",
        "max_turns",
    ]
    assert "prompt_profile" in parameters
    assert "recent_turns" in parameters


def test_terminal_database_lookup_preserves_exact_raw_user_authority():
    for path in (
        SOURCE_DIR / "chat_app.py",
        RUNTIME_DIR / "chat_app.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        matching_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "submit"
            ):
                continue
            keywords = {item.arg: item.value for item in node.keywords}
            if "exact_user_text" in keywords:
                matching_calls.append(keywords["exact_user_text"])

        assert matching_calls
        assert any(
            isinstance(value, ast.Name) and value.id == "user"
            for value in matching_calls
        )


def test_source_runtime_retrieval_modules_remain_exact_mirrors():
    assert (SOURCE_DIR / "chat_app.py").read_bytes() == (
        RUNTIME_DIR / "chat_app.py"
    ).read_bytes()
