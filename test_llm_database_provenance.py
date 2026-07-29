import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "source"))

from llm_database import LLMDatabase, build_llm_database


def test_built_database_returns_stable_local_provenance(tmp_path: Path):
    data_path = tmp_path / "verified_knowledge.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "user": "What is the Supermix verifier schema?",
                "assistant": "The schema is supermix-verifier-v1.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "knowledge.sqlite3"

    stats = build_llm_database(str(data_path), str(db_path))
    database = LLMDatabase(str(db_path))
    try:
        rows = database.query("Supermix verifier schema", top_k=1)
    finally:
        database.close()

    assert stats == {"input_examples": 1, "unique_entries": 1}
    assert len(rows) == 1
    assert rows[0]["source_uri"] == "dataset:verified_knowledge.jsonl"
    assert rows[0]["source_title"] == "verified knowledge"
    assert rows[0]["source_type"] == "local_dataset"
    assert len(str(rows[0]["content_hash"])) == 64
    assert str(tmp_path) not in str(rows[0])


def test_query_marks_only_normalized_exact_current_prompt_matches(tmp_path: Path):
    exact_prompt = "A lab starts with 71 sample records and receives 8 more."
    data_path = tmp_path / "exact_match.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "user": exact_prompt,
                        "assistant": "Exact response.",
                    }
                ),
                json.dumps(
                    {
                        "user": "A lab starts with 58 sample records and receives 10 more.",
                        "assistant": "Similar but wrong response.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "exact_match.sqlite3"
    build_llm_database(str(data_path), str(db_path))
    database = LLMDatabase(str(db_path))
    try:
        rows = database.query(
            "follow-up context | previous answer 68",
            top_k=2,
            exact_user_text="  a LAB starts with 71 sample records and receives 8 more.  ",
        )
    finally:
        database.close()

    exact_rows = [row for row in rows if row["exact_user_match"]]
    assert len(exact_rows) == 1
    assert exact_rows[0]["text"] == "Exact response."
    assert rows[0] == exact_rows[0]
    assert not any(
        row["exact_user_match"]
        for row in rows
        if row["text"] == "Similar but wrong response."
    )


def test_opening_legacy_database_migrates_provenance_columns(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE llm_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text TEXT NOT NULL,
            context_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            ctx_vec TEXT NOT NULL,
            resp_vec TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE llm_entries_fts
        USING fts5(user_text, context_text, response_text, content='llm_entries', content_rowid='id');
        """
    )
    conn.commit()
    conn.close()

    database = LLMDatabase(str(db_path))
    try:
        columns = {
            str(row[1])
            for row in database.conn.execute("PRAGMA table_info(llm_entries);").fetchall()
        }
    finally:
        database.close()

    assert {"source_uri", "source_title", "source_type", "content_hash"} <= columns
