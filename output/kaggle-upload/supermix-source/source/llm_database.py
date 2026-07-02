import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from chat_pipeline import featurize_text, load_conversation_examples


def _extract_user_from_context(context: str) -> str:
    for line in reversed(context.splitlines()):
        if line.strip().lower().startswith("user:"):
            return line.split(":", 1)[1].strip()
    return context.strip()


def _to_json_vec(vec: torch.Tensor) -> str:
    return json.dumps(vec.tolist(), separators=(",", ":"))


def _from_json_vec(payload: str) -> torch.Tensor:
    return torch.tensor(json.loads(payload), dtype=torch.float32)


def _tokenize_for_fts(text: str) -> str:
    tokens = [tok for tok in text.lower().replace("\n", " ").split() if tok.strip()]
    return " ".join(tokens)


def _safe_match_query(text: str, max_terms: int = 10) -> str:
    terms: List[str] = []
    for tok in text.lower().replace("\n", " ").split():
        # Keep only bareword tokens for FTS MATCH safety (avoid operators like '-').
        parts = re.findall(r"[a-z0-9_]+", tok)
        for t in parts:
            if t:
                terms.append(t)
    if not terms:
        return ""
    dedup: List[str] = []
    seen = set()
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
        if len(dedup) >= max_terms:
            break
    return " OR ".join(f"\"{t}\"" for t in dedup)


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_text TEXT NOT NULL,
            context_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 1,
            ctx_vec TEXT NOT NULL,
            resp_vec TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS llm_entries_fts
        USING fts5(user_text, context_text, response_text, content='llm_entries', content_rowid='id');
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_entries_count ON llm_entries(count DESC);")
    conn.commit()


def reset_database(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS llm_entries_fts;")
    cur.execute("DROP TABLE IF EXISTS llm_entries;")
    conn.commit()
    create_schema(conn)


def build_llm_database(
    data_path: str,
    db_path: str,
    max_rows: Optional[int] = None,
    reset: bool = True,
) -> Dict[str, int]:
    examples = load_conversation_examples(data_path)
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for ex in examples:
        user = _extract_user_from_context(ex.context)
        resp = ex.response.strip()
        if not user or not resp:
            continue
        key = (user, resp)
        row = grouped.get(key)
        if row is None:
            row = {
                "user": user,
                "context": ex.context,
                "response": resp,
                "count": 0,
                "ctx_sum": torch.zeros(128, dtype=torch.float32),
                "resp_sum": torch.zeros(128, dtype=torch.float32),
            }
            grouped[key] = row
        row["count"] = int(row["count"]) + 1
        row["ctx_sum"] = row["ctx_sum"] + featurize_text(ex.context)
        row["resp_sum"] = row["resp_sum"] + featurize_text(resp)

    items = list(grouped.values())
    items.sort(key=lambda r: (-int(r["count"]), str(r["user"]), str(r["response"])))
    if max_rows is not None and max_rows > 0:
        items = items[: max_rows]

    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -200000;")

    if reset:
        reset_database(conn)
    else:
        create_schema(conn)

    cur = conn.cursor()
    insert_rows: List[Tuple[str, str, str, int, str, str]] = []
    for row in items:
        cnt = max(1, int(row["count"]))
        ctx_vec = row["ctx_sum"] / cnt
        resp_vec = row["resp_sum"] / cnt
        insert_rows.append(
            (
                str(row["user"]),
                str(row["context"]),
                str(row["response"]),
                cnt,
                _to_json_vec(ctx_vec),
                _to_json_vec(resp_vec),
            )
        )

    cur.executemany(
        """
        INSERT INTO llm_entries (user_text, context_text, response_text, count, ctx_vec, resp_vec)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        insert_rows,
    )
    conn.commit()

    # Rebuild FTS index from content table.
    cur.execute("INSERT INTO llm_entries_fts(llm_entries_fts) VALUES('rebuild');")
    conn.commit()

    stats = {
        "input_examples": len(examples),
        "unique_entries": len(items),
    }
    conn.close()
    return stats


@dataclass
class LLMDBRow:
    response_text: str
    count: int
    ctx_vec: torch.Tensor
    resp_vec: torch.Tensor
    bm25_rank: float


class LLMDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def _fetch_rows(self, query_text: str, pool_size: int = 300) -> List[LLMDBRow]:
        cur = self.conn.cursor()
        match = _safe_match_query(query_text)
        rows: List[sqlite3.Row] = []
        if match:
            cur.execute(
                """
                SELECT e.response_text, e.count, e.ctx_vec, e.resp_vec, bm25(llm_entries_fts) AS b
                FROM llm_entries_fts
                JOIN llm_entries e ON e.id = llm_entries_fts.rowid
                WHERE llm_entries_fts MATCH ?
                ORDER BY b
                LIMIT ?;
                """,
                (match, int(pool_size)),
            )
            rows = cur.fetchall()

        if not rows:
            cur.execute(
                """
                SELECT response_text, count, ctx_vec, resp_vec, 0.0 AS b
                FROM llm_entries
                ORDER BY count DESC
                LIMIT ?;
                """,
                (int(pool_size),),
            )
            rows = cur.fetchall()

        out: List[LLMDBRow] = []
        for r in rows:
            out.append(
                LLMDBRow(
                    response_text=str(r["response_text"]),
                    count=int(r["count"]),
                    ctx_vec=_from_json_vec(str(r["ctx_vec"])),
                    resp_vec=_from_json_vec(str(r["resp_vec"])),
                    bm25_rank=float(r["b"]),
                )
            )
        return out

    def query(self, query_text: str, top_k: int = 80, pool_size: int = 300) -> List[Dict[str, object]]:
        rows = self._fetch_rows(query_text=query_text, pool_size=pool_size)
        if not rows:
            return []

        q = featurize_text(query_text)
        ctx = torch.stack([row.ctx_vec for row in rows], dim=0)
        resp = torch.stack([row.resp_vec for row in rows], dim=0)
        sim_ctx = torch.mv(ctx, q)
        sim_resp = torch.mv(resp, q)

        count_bonus = torch.tensor([math.log1p(max(1, row.count)) for row in rows], dtype=torch.float32)
        bm25_penalty = torch.tensor([max(0.0, row.bm25_rank) for row in rows], dtype=torch.float32)
        if bm25_penalty.numel() > 0 and float(torch.max(bm25_penalty)) > 0:
            bm25_penalty = bm25_penalty / (float(torch.max(bm25_penalty)) + 1e-6)

        scores = 0.72 * sim_ctx + 0.28 * sim_resp + 0.04 * count_bonus - 0.03 * bm25_penalty
        keep = min(max(1, top_k), len(rows))
        top_idx = torch.topk(scores, k=keep).indices.tolist()

        out: List[Dict[str, object]] = []
        for i in top_idx:
            row = rows[i]
            out.append(
                {
                    "text": row.response_text,
                    "count": int(row.count),
                    "vec": row.resp_vec.tolist(),
                    "ctx_vec": row.ctx_vec.tolist(),
                    "bucket_score": float(torch.sigmoid(scores[i]).item()),
                }
            )
        return out


def _cmd_build(args) -> None:
    stats = build_llm_database(
        data_path=args.data,
        db_path=args.db,
        max_rows=args.max_rows,
        reset=not args.append,
    )
    print(f"Built LLM DB: {args.db}")
    print(f"Input examples: {stats['input_examples']}")
    print(f"Unique entries: {stats['unique_entries']}")


def _cmd_stats(args) -> None:
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM llm_entries;")
    total = int(cur.fetchone()[0])
    cur.execute("SELECT AVG(count), MAX(count) FROM llm_entries;")
    avg_count, max_count = cur.fetchone()
    print(f"DB: {args.db}")
    print(f"Rows: {total}")
    print(f"Avg count: {float(avg_count or 0):.2f}")
    print(f"Max count: {int(max_count or 0)}")
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="Build/query a local LLM retrieval database.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build", help="Build LLM DB from JSONL conversation data.")
    ap_build.add_argument("--data", required=True, help="Conversation JSONL file.")
    ap_build.add_argument("--db", default="llm_chat.db", help="SQLite DB output file.")
    ap_build.add_argument("--max_rows", type=int, default=None, help="Optional cap for unique DB rows.")
    ap_build.add_argument("--append", action="store_true", help="Append instead of rebuilding.")
    ap_build.set_defaults(func=_cmd_build)

    ap_stats = sub.add_parser("stats", help="Show DB stats.")
    ap_stats.add_argument("--db", default="llm_chat.db")
    ap_stats.set_defaults(func=_cmd_stats)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
