import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from chat_pipeline import FEAT_DIM, featurize_text


def _vec_to_json(vec: torch.Tensor) -> str:
    return json.dumps(vec.tolist(), separators=(",", ":"))


def _vec_from_json(payload: str) -> torch.Tensor:
    return torch.tensor(json.loads(payload), dtype=torch.float32)


@dataclass
class MemoryRow:
    user_text: str
    assistant_text: str
    created_at: float
    user_vec: torch.Tensor
    assistant_vec: torch.Tensor


class ChatMemoryDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                user_vec TEXT NOT NULL,
                assistant_vec TEXT NOT NULL
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_memory_created_at ON chat_memory(created_at DESC);")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        user = (user_text or "").strip()
        assistant = (assistant_text or "").strip()
        if not user or not assistant:
            return

        user_vec = featurize_text(user)
        assistant_vec = featurize_text(assistant)
        self.conn.execute(
            """
            INSERT INTO chat_memory (created_at, user_text, assistant_text, user_vec, assistant_vec)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                float(time.time()),
                user,
                assistant,
                _vec_to_json(user_vec),
                _vec_to_json(assistant_vec),
            ),
        )
        self.conn.commit()

    def _fetch_recent_pool(self, pool_size: int) -> List[MemoryRow]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT created_at, user_text, assistant_text, user_vec, assistant_vec
            FROM chat_memory
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (max(1, int(pool_size)),),
        )
        rows = cur.fetchall()
        out: List[MemoryRow] = []
        for row in rows:
            out.append(
                MemoryRow(
                    user_text=str(row["user_text"]),
                    assistant_text=str(row["assistant_text"]),
                    created_at=float(row["created_at"]),
                    user_vec=_vec_from_json(str(row["user_vec"])),
                    assistant_vec=_vec_from_json(str(row["assistant_vec"])),
                )
            )
        return out

    def query(
        self,
        query_text: str,
        top_k: int = 4,
        pool_size: int = 400,
        recency_half_life_hours: float = 168.0,
    ) -> List[Dict[str, object]]:
        rows = self._fetch_recent_pool(pool_size=pool_size)
        if not rows:
            return []

        q = featurize_text(query_text)
        now = float(time.time())
        half_life_sec = max(3600.0, float(recency_half_life_hours) * 3600.0)

        user_stack = torch.stack([row.user_vec for row in rows], dim=0)
        assistant_stack = torch.stack([row.assistant_vec for row in rows], dim=0)
        sim_user = torch.mv(user_stack, q)
        sim_assistant = torch.mv(assistant_stack, q)

        recency_vals: List[float] = []
        for row in rows:
            age = max(0.0, now - float(row.created_at))
            recency_vals.append(math.exp(-age / half_life_sec))
        recency = torch.tensor(recency_vals, dtype=torch.float32)

        scores = 0.62 * sim_user + 0.28 * sim_assistant + 0.10 * recency
        keep = min(max(1, int(top_k)), len(rows))
        top_idx = torch.topk(scores, k=keep).indices.tolist()

        out: List[Dict[str, object]] = []
        for i in top_idx:
            row = rows[i]
            out.append(
                {
                    "user_text": row.user_text,
                    "assistant_text": row.assistant_text,
                    "user_vec": row.user_vec.tolist(),
                    "assistant_vec": row.assistant_vec.tolist(),
                    "created_at": row.created_at,
                    "score": float(scores[i].item()),
                }
            )
        return out


def render_memory_block(memories: Sequence[Dict[str, object]], max_chars: int = 1400) -> str:
    if not memories:
        return ""

    lines: List[str] = ["Relevant memory from earlier chats:"]
    for i, row in enumerate(memories, start=1):
        user = str(row.get("user_text", "")).strip()
        assistant = str(row.get("assistant_text", "")).strip()
        if not user or not assistant:
            continue
        lines.append(f"Memory {i} User: {user}")
        lines.append(f"Memory {i} Assistant: {assistant}")
        if len("\n".join(lines)) >= max_chars:
            break
    block = "\n".join(lines).strip()
    if len(block) > max_chars:
        block = block[:max_chars].rstrip()
    return block
