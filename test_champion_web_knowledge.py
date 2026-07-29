import ast
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import chat_app
from chat_web_app import Engine


def _candidate(text: str, score: float = 1.0):
    vec = chat_app.text_to_model_input(text, feature_mode="legacy")[0, 0].tolist()
    return {
        "text": text,
        "vec": vec,
        "ctx_vec": vec,
        "bucket_score": score,
        "count": 1,
    }


class _Model(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)
        logits[..., 2] = 4.0
        return logits


class _KnowledgeDB:
    def __init__(self):
        self.queries = []

    def query(self, query_text, top_k=120, *, exact_user_text=None):
        self.queries.append((query_text, top_k, exact_user_text))
        return [
            {
                **_candidate(
                    "The Supermix verifier schema is supermix-verifier-v1.",
                    score=8.0,
                ),
                "source_uri": "dataset:verified_knowledge.jsonl",
                "source_title": "verified knowledge",
                "source_type": "local_dataset",
                "content_hash": "a" * 64,
            }
        ]


class _MemoryDB:
    def __init__(self):
        self.writes = []

    def query(self, *_args, **_kwargs):
        return []

    def add_turn(self, user_text, assistant_text):
        self.writes.append((user_text, assistant_text))


def test_web_engine_uses_terminal_equivalent_knowledge_and_memory_paths():
    engine = Engine(
        torch.device("cpu"),
        {"resolved": "cpu"},
        {"pool_mode": "all", "db_top_k": 7},
    )
    knowledge_db = _KnowledgeDB()
    memory_db = _MemoryDB()
    engine.llm_db = knowledge_db
    engine.memory_db = memory_db
    engine.model = _Model()
    engine.buckets = {2: [_candidate("An unrelated generic response.", score=0.01)]}
    engine.available_labels = [2]

    result = engine.chat(
        session_id="knowledge-parity",
        user_text="What is the Supermix verifier schema?",
        response_temperature=0.0,
        show_top_responses=2,
    )

    assert result["knowledge"] == {
        "llm_db_enabled": True,
        "memory_enabled": True,
        "llm_db_hits": 1,
        "memory_hits": 0,
    }
    assert knowledge_db.queries and knowledge_db.queries[0][1] == 7
    assert knowledge_db.queries[0][2] == "What is the Supermix verifier schema?"
    assert result["top_candidates"][0]["text"].endswith("supermix-verifier-v1.")
    assert result["response"].endswith("supermix-verifier-v1.")
    assert memory_db.writes == [
        ("What is the Supermix verifier schema?", result["response"])
    ]
    assert result["compute"]["applied"] is False


def test_exact_current_prompt_match_ignores_previous_turn_number_anchor():
    lab_prompt = (
        "A lab starts with 71 sample records and receives 8 more. "
        "It then makes 3 copies of every record in the combined set and "
        "discards 2 damaged copies. How many usable copies remain?"
    )
    exact_response = (
        "Combine the first two amounts: 71 + 8 = 79. Apply the multiplier: "
        "79 × 3 = 237. Remove 2: 237 - 2 = 235. Final answer: 235."
    )
    wrong_response = (
        "Combine the first two amounts: 58 + 10 = 68. Apply the multiplier: "
        "68 × 3 = 204. Remove 2: 204 - 2 = 202. Final answer: 202."
    )

    class _ExactKnowledgeDB:
        def __init__(self):
            self.queries = []

        def query(self, query_text, top_k=120, *, exact_user_text=None):
            self.queries.append((query_text, top_k, exact_user_text))
            if exact_user_text != lab_prompt:
                return [{**_candidate(wrong_response, score=9.0)}]
            return [
                {
                    **_candidate(wrong_response, score=9.0),
                    "exact_user_match": False,
                },
                {
                    **_candidate(exact_response, score=0.01),
                    "exact_user_match": True,
                },
            ]

    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    knowledge_db = _ExactKnowledgeDB()
    engine.llm_db = knowledge_db
    engine.model = _Model()
    engine.buckets = {2: [_candidate("An unrelated generic response.", score=0.01)]}
    engine.available_labels = [2]

    first = engine.chat(
        session_id="exact-current-prompt",
        user_text="Calculate (7 * 9) + 5.",
        response_temperature=0.0,
    )
    second = engine.chat(
        session_id="exact-current-prompt",
        user_text=lab_prompt,
        response_temperature=0.0,
        show_top_responses=3,
    )

    assert first["response"] == "The exact result is 68."
    assert second["response"] == exact_response
    assert [row["text"] for row in second["top_candidates"]] == [exact_response]
    assert knowledge_db.queries[-1][0] == lab_prompt
    assert "Calculate (7 * 9) + 5." not in knowledge_db.queries[-1][0]
    assert knowledge_db.queries[-1][2] == lab_prompt


def test_followup_context_expansion_keeps_current_prompt_separate_for_exact_matching():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    knowledge_db = _KnowledgeDB()
    engine.llm_db = knowledge_db
    engine.model = _Model()
    engine.buckets = {2: [_candidate("An unrelated generic response.", score=0.01)]}
    engine.available_labels = [2]
    engine.sessions["context-retrieval"] = [
        ("Explain the verifier schema.", "It validates supported answer contracts.")
    ]
    engine.recent["context-retrieval"] = [
        "It validates supported answer contracts."
    ]

    engine.chat(
        session_id="context-retrieval",
        user_text="What about that?",
        response_temperature=0.0,
    )

    expanded_query, _top_k, exact_user_text = knowledge_db.queries[-1]
    assert expanded_query.startswith(
        "What about that? Relevant resolved context: "
        "Prior user: Explain the verifier schema."
    )
    assert "It validates supported answer contracts." in expanded_query
    assert exact_user_text == "What about that?"


def test_source_and_packaged_web_knowledge_methods_are_ast_equivalent():
    root = Path(__file__).resolve().parent

    def method_dump(path: Path, method_name: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        engine_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Engine"
        )
        method = next(
            node
            for node in engine_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method_name
        )
        return ast.dump(method, include_attributes=False)

    source_path = root / "source" / "chat_web_app.py"
    packaged_path = root / "runtime_python" / "chat_web_app.py"
    for method_name in ("__init__", "status", "chat"):
        assert method_dump(source_path, method_name) == method_dump(
            packaged_path,
            method_name,
        )
