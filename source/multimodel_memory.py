from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")

MEMORY_PATTERNS = (
    (
        re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z '\-]{0,48})", re.IGNORECASE),
        "identity",
        lambda match: f"User name: {match.group(1).strip()}",
    ),
    (
        re.compile(r"\bcall me\s+([A-Za-z][A-Za-z '\-]{0,48})", re.IGNORECASE),
        "identity",
        lambda match: f"Preferred name: {match.group(1).strip()}",
    ),
    (
        re.compile(r"\bi prefer\s+(.{3,120})", re.IGNORECASE),
        "preference",
        lambda match: f"User preference: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bplease use\s+(.{3,120})", re.IGNORECASE),
        "preference",
        lambda match: f"Preferred approach: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bi(?:'m| am) working on\s+(.{3,120})", re.IGNORECASE),
        "project",
        lambda match: f"Current project: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bthis project is\s+(.{3,120})", re.IGNORECASE),
        "project",
        lambda match: f"Project detail: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bremember that\s+(.{3,160})", re.IGNORECASE),
        "fact",
        lambda match: f"Remembered fact: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bi like\s+(.{3,120})", re.IGNORECASE),
        "preference",
        lambda match: f"User likes: {match.group(1).strip().rstrip('.!?')}",
    ),
)


def _norm(text: str, limit: int = 260) -> str:
    cooked = MULTISPACE_RE.sub(" ", str(text or "").strip())
    return cooked[:limit]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(text or ""))}


def _safe_slug(text: str) -> str:
    cooked = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    cooked = "-".join(part for part in cooked.split("-") if part)
    return cooked[:80] or "session"


def _now_ts() -> float:
    return float(time.time())


class ConversationMemoryStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        return self.root_dir / f"{_safe_slug(session_id)}.json"

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._path_for(session_id)
        if not path.exists():
            now = _now_ts()
            return {
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
                "memories": [],
                "turns": [],
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        payload.setdefault("session_id", session_id)
        payload.setdefault("created_at", _now_ts())
        payload.setdefault("updated_at", payload["created_at"])
        payload.setdefault("memories", [])
        payload.setdefault("turns", [])
        return payload

    def save_session(self, session_id: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["session_id"] = session_id
        payload["updated_at"] = _now_ts()
        path = self._path_for(session_id)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def clear_session(self, session_id: str) -> None:
        self._path_for(session_id).unlink(missing_ok=True)

    def _extract_memories(self, user_text: str, assistant_text: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        lower_user = _norm(user_text, limit=320)
        for pattern, kind, builder in MEMORY_PATTERNS:
            match = pattern.search(lower_user)
            if not match:
                continue
            note = _norm(builder(match), limit=220)
            if not note:
                continue
            found.append(
                {
                    "kind": kind,
                    "text": note,
                    "source": "user",
                    "score": 1.0,
                    "updated_at": _now_ts(),
                }
            )
        assistant = _norm(assistant_text, limit=320)
        if assistant:
            found.append(
                {
                    "kind": "lesson",
                    "text": f"Successful answer pattern: {assistant}",
                    "source": "assistant",
                    "score": 0.35,
                    "updated_at": _now_ts(),
                }
            )
        return found

    def update(
        self,
        *,
        session_id: str,
        user_text: str,
        assistant_text: str,
        model_key: str,
        route_reason: str,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        consultants: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        turn = {
            "ts": _now_ts(),
            "user": _norm(user_text, limit=1200),
            "assistant": _norm(assistant_text, limit=1600),
            "model_key": _norm(model_key, limit=120),
            "route_reason": _norm(route_reason, limit=240),
            "tools": list(tools or []),
            "consultants": list(consultants or []),
        }
        turns = list(payload.get("turns") or [])
        turns.append(turn)
        payload["turns"] = turns[-80:]

        existing = list(payload.get("memories") or [])
        for item in self._extract_memories(user_text, assistant_text):
            note = item["text"]
            match = next((row for row in existing if str(row.get("text") or "").strip().lower() == note.lower()), None)
            if match is None:
                existing.append(item)
                continue
            match["updated_at"] = _now_ts()
            match["score"] = round(float(match.get("score") or 0.0) + 0.2, 3)
        payload["memories"] = existing[-60:]
        self.save_session(session_id, payload)
        return payload

    def build_context(self, session_id: str, prompt: str, *, max_memories: int = 5, max_examples: int = 2) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        prompt_tokens = _tokens(prompt)

        ranked_memories: List[tuple[float, Dict[str, Any]]] = []
        for item in list(payload.get("memories") or []):
            text = _norm(item.get("text") or "", limit=220)
            if not text:
                continue
            score = float(item.get("score") or 0.0)
            overlap = len(prompt_tokens & _tokens(text))
            if overlap:
                score += overlap * 0.55
            ranked_memories.append((score, dict(item)))
        ranked_memories.sort(key=lambda pair: pair[0], reverse=True)
        selected_memories = [row for _score, row in ranked_memories[:max_memories]]

        ranked_turns: List[tuple[float, Dict[str, Any]]] = []
        for turn in list(payload.get("turns") or [])[:-1]:
            score = len(prompt_tokens & _tokens(turn.get("user") or ""))
            if score <= 0:
                continue
            ranked_turns.append((float(score), dict(turn)))
        ranked_turns.sort(key=lambda pair: pair[0], reverse=True)
        selected_turns = [row for _score, row in ranked_turns[:max_examples]]

        blocks: List[str] = []
        memory_notes = [_norm(row.get("text") or "", limit=220) for row in selected_memories if _norm(row.get("text") or "", limit=220)]
        if memory_notes:
            blocks.append("Persistent conversation memory:\n" + "\n".join(f"- {note}" for note in memory_notes))
        exemplar_rows: List[str] = []
        for row in selected_turns:
            user = _norm(row.get("user") or "", limit=180)
            assistant = _norm(row.get("assistant") or "", limit=220)
            if not user or not assistant:
                continue
            exemplar_rows.append(f"- User: {user}\n  Assistant: {assistant}")
        if exemplar_rows:
            blocks.append("Relevant prior conversation examples:\n" + "\n".join(exemplar_rows))

        return {
            "memory_notes": memory_notes,
            "example_count": len(exemplar_rows),
            "turn_count": len(payload.get("turns") or []),
            "context_block": "\n\n".join(blocks).strip(),
            "raw": payload,
        }

    def session_snapshot(self, session_id: str) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        memories = [
            _norm(item.get("text") or "", limit=160)
            for item in list(payload.get("memories") or [])[-8:]
            if _norm(item.get("text") or "", limit=160)
        ]
        turns = list(payload.get("turns") or [])
        recent_turns = [
            {
                "user": _norm(turn.get("user") or "", limit=140),
                "assistant": _norm(turn.get("assistant") or "", limit=180),
                "model_key": _norm(turn.get("model_key") or "", limit=80),
            }
            for turn in turns[-4:]
        ]
        return {
            "session_id": session_id,
            "memory_count": len(payload.get("memories") or []),
            "turn_count": len(turns),
            "memories": memories,
            "recent_turns": recent_turns,
            "updated_at": payload.get("updated_at"),
        }

    def global_status(self) -> Dict[str, Any]:
        files = list(self.root_dir.glob("*.json"))
        return {
            "session_files": len(files),
            "root_dir": str(self.root_dir),
        }
