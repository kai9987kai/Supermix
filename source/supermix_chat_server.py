"""A multi-model chat server with streaming, admission control and eviction.

`mimomix_talk_web_app.py` serves exactly one checkpoint per process. With nine
checkpoints on disk that means nine ports, nine copies of torch, and no way to
compare two models without alt-tabbing. It also has, measured by grep, **zero**
streaming, queueing or cancellation primitives: every request takes a global
lock and holds it for the whole generation, so a second caller waits with no
feedback and no way out.

This server is additive -- the single-model app is untouched, and every
checkpoint it loads uses the same `load_talk_checkpoint` contract. It adds the
four things that were missing:

**Model switching.** One process, many checkpoints, chosen per request. The UI
gets a dropdown; `/api/models` lists what is available.

**Bounded residency.** Checkpoints load on first use and the least recently used
is evicted past `--max-resident`. This is not tidiness: a 15.6 GB box running
five servers is how the v64 training run met a segfault, and holding nine models
at once would repeat it.

**Streaming.** Tokens are sent as they are produced over SSE, so a 60-token
reply on a CPU box shows progress instead of a blank pane for four seconds.

**Admission control.** A semaphore bounds concurrent generations. Callers over
the limit get an immediate `503` with a `Retry-After`, rather than silently
queueing behind a lock -- a fast honest refusal beats an unbounded wait.

    python source/supermix_chat_server.py --model v70=output/v70_moe/v70_moe.partial.pt \\
        --model v68=output/v68_average_fix/v68_average_fix.pt --port 8780
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_text as text_utils  # noqa: E402
import answer_check
import prompt_normaliser  # noqa: E402
import eval_problem_solving as solving  # noqa: E402
import recall_index  # noqa: E402
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

MAX_NEW_TOKENS = 256
#: Hard cap on prompt+history tokens fed to the model. Older turns fall off the
#: front rather than growing without bound.
MAX_CONTEXT = 512
#: Sessions kept in memory, oldest evicted. Unbounded session state is a slow
#: memory leak on a long-lived server.
MAX_SESSIONS = 256

#: Prior turns replayed into the prompt. **Zero by default, deliberately.**
#:
#: Turn-aligned packing (v63) gives every training turn its own block, so these
#: models have never seen a multi-turn prompt -- replaying history feeds them a
#: shape they were not trained on. Measured on v70: asked "why is my script
#: failing" in a fresh session it answers "Check the traceback first, then we
#: can isolate the failing function"; asked the same thing after two arithmetic
#: turns it answers "8 = 100, total =10.0".
#:
#: The feature is kept because a model trained on multi-turn data would want it,
#: and because the measurement above is worth being able to reproduce. It is off
#: because for every checkpoint in this repo it makes replies worse.
DEFAULT_HISTORY_TURNS = 0
MAX_HISTORY_TURNS = 6
MAX_MESSAGE_CHARACTERS = 4000


@dataclass
class LoadedModel:
    name: str
    checkpoint: str
    model: Any
    tokenizer: Any
    extra: Dict[str, Any]
    recall: Optional[Any] = None
    #: Serialises generation for one model. Two models can generate at once;
    #: one model cannot, because a single `nn.Module` is not re-entrant.
    lock: threading.Lock = field(default_factory=threading.Lock)


class ModelRegistry:
    """Lazily loads checkpoints and evicts the least recently used."""

    def __init__(self, spec: Dict[str, str], max_resident: int = 2,
                 corpora: Optional[Dict[str, str]] = None):
        self.spec = dict(spec)
        self.corpora = dict(corpora or {})
        self.max_resident = max(1, int(max_resident))
        self._resident: "OrderedDict[str, LoadedModel]" = OrderedDict()
        self._guard = threading.Lock()

    def names(self) -> List[str]:
        return list(self.spec)

    def describe(self) -> List[Dict[str, Any]]:
        out = []
        for name, path in self.spec.items():
            entry: Dict[str, Any] = {
                "name": name,
                "checkpoint": path,
                "resident": name in self._resident,
                "recall_index": name in self.corpora,
            }
            loaded = self._resident.get(name)
            if loaded is not None:
                # Best-effort detail. This endpoint is polled by the UI after
                # every reply, so it must degrade to a shorter row rather than
                # 500 the model picker if a checkpoint exposes something
                # unexpected.
                try:
                    entry["vocab_size"] = loaded.tokenizer.vocab_size
                    entry["parameters"] = sum(p.numel() for p in loaded.model.parameters())
                    entry["dev_loss"] = loaded.extra.get("best_dev_loss")
                except Exception:  # noqa: BLE001 - detail is optional, listing is not
                    entry["detail_unavailable"] = True
            out.append(entry)
        return out

    def acquire(self, name: str) -> LoadedModel:
        if name not in self.spec:
            raise KeyError(f"unknown model {name!r}; available: {sorted(self.spec)}")
        with self._guard:
            existing = self._resident.get(name)
            if existing is not None:
                self._resident.move_to_end(name)
                return existing

        # Load outside the registry guard: a checkpoint takes seconds to read and
        # holding the guard would block every other model's requests meanwhile.
        model, tokenizer, payload = load_talk_checkpoint(self.spec[name])
        model.eval()
        index = None
        corpus = self.corpora.get(name)
        if corpus:
            index = recall_index.RecallIndex.from_jsonl(corpus)
        loaded = LoadedModel(
            name=name, checkpoint=self.spec[name], model=model, tokenizer=tokenizer,
            extra=dict(payload.get("extra") or {}), recall=index,
        )

        with self._guard:
            self._resident[name] = loaded
            self._resident.move_to_end(name)
            while len(self._resident) > self.max_resident:
                evicted, _ = self._resident.popitem(last=False)
                if evicted == name:  # never evict what was just requested
                    self._resident[name] = loaded
                    break
        return loaded


# -- generation -------------------------------------------------------------


def stream_tokens(
    loaded: LoadedModel,
    message: str,
    history: Sequence[Tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    should_stop,
) -> Iterator[str]:
    """Yield decoded text incrementally.

    Recomputes the full prefix each step rather than carrying a KV cache. That
    is O(n^2), and at these sizes and lengths it is not the bottleneck -- the
    point of streaming here is that the caller sees progress, not that decoding
    is optimal. `should_stop` is polled every token so a disconnected client
    stops the work instead of paying for a reply nobody will read.
    """

    tokenizer = loaded.tokenizer
    # Prior turns first, so the model sees the conversation rather than an
    # isolated question. The single-model app carried history and this server
    # dropped it; that was a regression, not a simplification.
    ids: List[int] = []
    for past_user, past_reply in history:
        turn, _ = tokenizer.encode_turn(past_user, past_reply)
        ids.extend(turn)
    prompt, _ = tokenizer.encode_turn(message, None)
    ids.extend(prompt)
    generated: List[int] = []
    emitted = ""

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if should_stop():
                break
            window = torch.tensor([ids + generated][0][-MAX_CONTEXT:], dtype=torch.long).unsqueeze(0)
            logits = loaded.model(window, return_mtp=False).logits[0, -1]

            if greedy:
                nxt = int(torch.argmax(logits))
            else:
                scaled = logits / max(1e-6, temperature)
                probs = torch.softmax(scaled, dim=-1)
                order = torch.argsort(probs, descending=True)
                sorted_probs = probs[order]
                keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
                keep[0] = True
                choice = torch.multinomial(sorted_probs[keep], 1)
                nxt = int(order[keep][choice])

            if nxt == text_utils.EOS:
                break
            generated.append(nxt)

            # Decode the whole reply each step and emit only the new suffix:
            # tokens carry their own leading whitespace, so decoding one at a
            # time and concatenating would be correct here, but decoding the
            # whole prefix is robust to any future tokenizer that is not.
            full = tokenizer.decode(generated)
            if len(full) > len(emitted):
                yield full[len(emitted):]
                emitted = full


# -- server -----------------------------------------------------------------


def build_app(registry: ModelRegistry, max_concurrency: int = 2,
              history_turns: int = DEFAULT_HISTORY_TURNS,
              normalise_prompts: bool = True):
    from flask import Flask, Response, request as flask_request, stream_with_context

    app = Flask(__name__)
    # Bounds *generations*, not connections. Listing models or loading the page
    # is never refused; only the expensive path is.
    admission = threading.BoundedSemaphore(max_concurrency)
    # session id -> [(user, reply), ...]. Bounded, oldest evicted: an unbounded
    # dict keyed on a client-supplied string is a slow memory leak.
    sessions: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()
    sessions_guard = threading.Lock()

    def history_for(session_id: str) -> List[Tuple[str, str]]:
        if history_turns <= 0:
            return []
        with sessions_guard:
            return list(sessions.get(session_id, []))[-history_turns:]

    def remember(session_id: str, user: str, reply: str) -> None:
        with sessions_guard:
            turns = sessions.setdefault(session_id, [])
            turns.append((user, reply))
            del turns[:-MAX_HISTORY_TURNS]
            sessions.move_to_end(session_id)
            while len(sessions) > MAX_SESSIONS:
                sessions.popitem(last=False)
    stats = {"requests": 0, "refused": 0, "tokens": 0}
    stats_guard = threading.Lock()

    def no_store(payload, status=200):
        response = app.response_class(
            json.dumps(payload, ensure_ascii=False), status=status,
            mimetype="application/json",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    def read_request() -> Tuple[Dict[str, Any], Optional[str]]:
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return {}, "request body must be a JSON object"
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            return {}, "message must be a non-empty string"
        if len(message) > MAX_MESSAGE_CHARACTERS:
            return {}, "message is too long"
        name = payload.get("model") or registry.names()[0]
        if name not in registry.spec:
            return {}, f"unknown model {name!r}"
        session_id = payload.get("session_id")
        # Map how a person writes an operation onto the token v74 was trained
        # on. Measured: "what is 47 times 6" answers 242, "What is 47 x 6?"
        # answers 282. The rewrite is reported back so the interface can show
        # what was actually asked rather than implying the model parsed the
        # original.
        rewrite = prompt_normaliser.normalise(message) if normalise_prompts \
            else prompt_normaliser.Normalised(message, None, message)
        return {
            "message": rewrite.prompt,
            "asked": message,
            "normalised_rule": rewrite.rule if rewrite.changed else None,
            "model": name,
            "session_id": session_id if isinstance(session_id, str) and session_id else "default",
            "check": bool(payload.get("check", True)),
            "max_new_tokens": max(1, min(int(payload.get("max_new_tokens", 64)), MAX_NEW_TOKENS)),
            "temperature": max(0.05, min(float(payload.get("temperature", 0.9)), 2.0)),
            "top_p": max(0.05, min(float(payload.get("top_p", 0.9)), 1.0)),
            "greedy": payload.get("mode", "greedy") == "greedy",
        }, None

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html")

    @app.get("/api/models")
    def models():
        return no_store({"models": registry.describe(), "max_resident": registry.max_resident})

    @app.get("/api/stats")
    def statistics():
        with stats_guard:
            return no_store(dict(stats, concurrency_limit=max_concurrency))

    def generate_once(loaded, message, options) -> Dict[str, Any]:
        """One complete reply, non-streamed. Shared by compare and benchmark."""

        started = time.perf_counter()
        pieces: List[str] = []
        with loaded.lock:
            for chunk in stream_tokens(
                loaded, message, [], options["max_new_tokens"],
                options["temperature"], options["top_p"], options["greedy"],
                should_stop=lambda: False,
            ):
                pieces.append(chunk)
        elapsed = time.perf_counter() - started
        reply = "".join(pieces).strip()
        result: Dict[str, Any] = {
            "model": loaded.name,
            "reply": reply,
            "latency_ms": round(elapsed * 1000, 2),
        }
        if loaded.recall is not None:
            result["recall"] = loaded.recall.score(reply).to_dict()
        verdict = answer_check.check(message, reply)
        result["check"] = verdict.to_dict() if verdict else None
        return result

    @app.post("/api/compare")
    def compare():
        """Ask several models the same question and return every answer.

        Comparing models is the thing this repo spends its time on, and until now
        it meant running one server per checkpoint and alt-tabbing. The answers
        are independently verified, so a disagreement shows which model is right
        rather than only that they differ.
        """

        options, error = read_request()
        if error:
            return no_store({"error": error}, 400)
        payload = flask_request.get_json(silent=True) or {}
        names = payload.get("models") or registry.names()
        if not isinstance(names, list) or not names:
            return no_store({"error": "models must be a non-empty list"}, 400)
        unknown = [n for n in names if n not in registry.spec]
        if unknown:
            return no_store({"error": f"unknown models: {unknown}"}, 400)

        if not admission.acquire(blocking=False):
            with stats_guard:
                stats["refused"] += 1
            response = no_store({"error": "server busy"}, 503)
            response.headers["Retry-After"] = "2"
            return response
        try:
            # Sequential, not concurrent: `--max-resident` may be smaller than
            # the number of models asked for, and racing them would thrash the
            # registry rather than answer faster.
            results = []
            for name in names:
                try:
                    results.append(generate_once(registry.acquire(name), options["message"], options))
                except Exception as exc:  # noqa: BLE001
                    results.append({"model": name, "error": str(exc)})
            with stats_guard:
                stats["requests"] += 1
            agree = {
                r["check"]["predicted"] for r in results
                if r.get("check") and r["check"].get("predicted") is not None
            }
            return no_store({
                "message": options["message"],
                "results": results,
                # Only meaningful when the question was checkable at all.
                "answers_agree": (len(agree) <= 1) if agree else None,
            })
        finally:
            admission.release()

    @app.post("/api/benchmark")
    def benchmark():
        """Score a model on freshly generated problems, from the browser.

        The same generators `eval_problem_solving` uses, so the number here and
        the number in a receipt mean the same thing. Capped small because
        generation is slow on CPU and this holds an admission slot.
        """

        payload = flask_request.get_json(silent=True) or {}
        name = payload.get("model") or registry.names()[0]
        if name not in registry.spec:
            return no_store({"error": f"unknown model {name!r}"}, 400)
        count = max(1, min(int(payload.get("problems", 20)), 60))

        if not admission.acquire(blocking=False):
            response = no_store({"error": "server busy"}, 503)
            response.headers["Retry-After"] = "5"
            return response
        try:
            loaded = registry.acquire(name)
            problems = solving.generate_novel(count, seed=int(payload.get("seed", 65)))
            per_task: Dict[str, Dict[str, int]] = {}
            for problem in problems:
                out = generate_once(
                    loaded, problem.prompt,
                    {"max_new_tokens": 64, "temperature": 0.9, "top_p": 0.9, "greedy": True},
                )
                bucket = per_task.setdefault(problem.task, {"n": 0, "correct": 0})
                bucket["n"] += 1
                bucket["correct"] += int(
                    solving.is_correct(solving.extract_answer(out["reply"]), problem.answer)
                )
            total = sum(b["n"] for b in per_task.values())
            correct = sum(b["correct"] for b in per_task.values())
            return no_store({
                "model": name,
                "problems": total,
                "correct": correct,
                "accuracy": round(correct / max(1, total), 4),
                "by_task": {
                    k: {**v, "accuracy": round(v["correct"] / max(1, v["n"]), 4)}
                    for k, v in sorted(per_task.items())
                },
                "note": (
                    "freshly generated problems, never drawn from training; a "
                    "memorised answer scores zero. Small samples are noisy -- "
                    f"n={total} carries roughly +-{int(196 / max(1, total) ** 0.5)} points"
                ),
            })
        finally:
            admission.release()

    @app.post("/api/reset")
    def reset():
        payload = flask_request.get_json(silent=True) or {}
        session_id = payload.get("session_id") or "default"
        with sessions_guard:
            sessions.pop(session_id, None)
        return no_store({"session_id": session_id, "cleared": True})

    @app.get("/health")
    def health():
        return no_store({"status": "ok", "models": registry.names()})

    @app.post("/api/chat")
    def chat():
        options, error = read_request()
        if error:
            return no_store({"error": error}, 400)

        if not admission.acquire(blocking=False):
            with stats_guard:
                stats["refused"] += 1
            response = no_store(
                {"error": "server busy", "concurrency_limit": max_concurrency}, 503
            )
            response.headers["Retry-After"] = "2"
            return response

        try:
            loaded = registry.acquire(options["model"])
        except Exception as exc:  # noqa: BLE001 - report, never 500 silently
            admission.release()
            return no_store({"error": str(exc)}, 400)

        started = time.perf_counter()
        pieces: List[str] = []

        cancelled = threading.Event()
        history = history_for(options["session_id"])

        def events() -> Iterator[str]:
            produced = 0
            try:
                with loaded.lock:
                    for chunk in stream_tokens(
                        loaded, options["message"], history,
                        options["max_new_tokens"],
                        options["temperature"], options["top_p"], options["greedy"],
                        should_stop=cancelled.is_set,
                    ):
                        produced += 1
                        pieces.append(chunk)
                        yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
                elapsed = time.perf_counter() - started
                reply = "".join(pieces).strip()
                summary: Dict[str, Any] = {
                    "model": loaded.name,
                    "reply": reply,
                    # What the model was actually given. When these differ the
                    # interface shows both, so a rewrite is never invisible.
                    "asked": options["asked"],
                    "asked_as": options["message"],
                    "normalised_rule": options["normalised_rule"],
                    "tokens": produced,
                    "latency_ms": round(elapsed * 1000, 2),
                    "tokens_per_second": round(produced / max(1e-6, elapsed), 2),
                }
                if loaded.recall is not None:
                    summary["recall"] = loaded.recall.score(reply).to_dict()
                if options["check"]:
                    # Independently re-derive the answer from the question. A
                    # `None` here means the question was not one of the shapes
                    # this can verify, and the interface must show that as
                    # "not checked" rather than as a pass.
                    verdict = answer_check.check(options["message"], reply)
                    summary["check"] = verdict.to_dict() if verdict else None
                remember(options["session_id"], options["message"], reply)
                with stats_guard:
                    stats["requests"] += 1
                    stats["tokens"] += produced
                yield f"event: done\ndata: {json.dumps(summary)}\n\n"
            except GeneratorExit:
                # The client went away mid-stream. Flask closes the generator;
                # setting the flag stops the decode loop on its next token
                # instead of finishing a reply nobody will read.
                cancelled.set()
                raise
            finally:
                admission.release()

        response = Response(stream_with_context(events()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    return app


PAGE = """<!doctype html><meta charset="utf-8">
<title>Supermix chat</title>
<style>
:root{--bg:#12141a;--panel:#191c24;--ink:#e8eaf0;--dim:#9aa2b4;--line:#2a2f3c;--accent:#7aa2f7}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600}
select,input,button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 9px;font:inherit}
button{cursor:pointer}button:hover{border-color:var(--accent)}
main{max-width:860px;margin:0 auto;padding:16px}
.msg{padding:10px 12px;border-radius:8px;margin:10px 0;white-space:pre-wrap}
.user{background:#1d2330;border:1px solid var(--line)}
.bot{background:var(--panel);border:1px solid var(--line)}
.rewrite{color:#9aa0a6;font-style:italic}
    .meta{margin-top:7px;font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.recall{margin-top:5px;padding:3px 8px;border-radius:4px;font-size:11.5px;font-weight:600;display:inline-block}
.check{margin-top:5px;padding:3px 8px;border-radius:4px;font-size:11.5px;font-weight:600;display:inline-block;margin-right:6px}
.ok{background:#14432a;color:#9ae6b4}.bad{background:#5a1d1d;color:#ffb4b4}.unknown{background:#2a2f3c;color:#9aa2b4}
.recalled{background:#5a1d1d;color:#ffb4b4}.partial{background:#5a4a1d;color:#ffe0a0}.novel{background:#1d4a2a;color:#a8e6b8}
form{display:flex;gap:8px;margin-top:14px}form input{flex:1}
.note{color:var(--dim);font-size:12px;margin:6px 0 0}
.cursor{opacity:.55}
.tog{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:5px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:10px 0}
.col{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.col h3{margin:0 0 6px;font-size:12px;color:var(--accent);font-weight:600}
.agree{margin:4px 0 0;font-size:12px;font-weight:600}
.agree.yes{color:#9ae6b4}.agree.no{color:#ffb4b4}
.bench{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.bench td{padding:2px 10px 2px 0;font-size:12px;font-variant-numeric:tabular-nums}
</style>
<header>
  <h1>Supermix chat</h1>
  <select id="model"></select>
  <select id="mode"><option value="greedy">greedy</option><option value="sample">sample</option></select>
  <input id="ntok" type="number" value="64" min="1" max="256" style="width:80px" title="max tokens">
  <label class="tog"><input type="checkbox" id="cmp"> compare all</label>
  <button id="bench" type="button">Benchmark</button>
  <button id="reset" type="button">Clear</button>
  <span class="note" id="status"></span>
</header>
<main>
  <div id="log"></div>
  <form id="form"><input id="input" placeholder="say something…" autocomplete="off"><button id="send">Send</button></form>
  <p class="note">Replies stream as they are generated. Models load on first use and the least recently used is evicted, so the first message to a model is slower. Maths answers are re-derived from your question and checked independently; other questions show NOT CHECKED.</p>
</main>
<script>
const el = id => document.getElementById(id);
const SESSION = 'web-' + Math.random().toString(36).slice(2);
async function loadModels(){
  const r = await fetch('/api/models'); const d = await r.json();
  el('model').innerHTML = d.models.map(m =>
    `<option value="${m.name}">${m.name}${m.resident?' •':''}</option>`).join('');
  el('status').textContent = `${d.models.length} models, ${d.max_resident} resident max`;
}
function add(cls, text){
  const n = document.createElement('div'); n.className = 'msg ' + cls; n.textContent = text;
  el('log').appendChild(n); window.scrollTo(0, document.body.scrollHeight); return n;
}
async function send(ev){
  ev.preventDefault();
  const text = el('input').value.trim(); if(!text) return;
  el('input').value = ''; el('send').disabled = true;
  add('user', text);
  const bot = add('bot', ''); const cur = document.createElement('span');
  cur.className = 'cursor'; cur.textContent = '▋'; bot.appendChild(cur);
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:text, model:el('model').value, mode:el('mode').value,
                            session_id:SESSION, max_new_tokens:Number(el('ntok').value)})});
    if(!r.ok){ const e = await r.json().catch(()=>({error:r.status}));
      bot.textContent = 'error: ' + (e.error || r.status); el('send').disabled = false; return; }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    for(;;){
      const {value, done} = await reader.read(); if(done) break;
      buf += dec.decode(value, {stream:true});
      let cut;
      while((cut = buf.indexOf('\\n\\n')) >= 0){
        const frame = buf.slice(0, cut); buf = buf.slice(cut+2);
        const ev = (frame.match(/^event: (.+)$/m)||[])[1];
        const data = JSON.parse((frame.match(/^data: (.+)$/m)||[])[1] || '{}');
        if(ev === 'token'){ cur.insertAdjacentText('beforebegin', data.text); }
        else if(ev === 'done'){
          cur.remove();
          const m = document.createElement('div'); m.className='meta';
          m.textContent = `${data.model} · ${data.tokens} tokens · ${data.tokens_per_second.toFixed(1)} tok/s · ${data.latency_ms.toFixed(0)} ms`;
          bot.appendChild(m);
          if(data.normalised_rule){
            // The question was rewritten into the corpus format before the
            // model saw it. Show it: answering a different question than the
            // one typed, without saying so, misrepresents the model.
            const n = document.createElement('div'); n.className='meta rewrite';
            n.textContent = 'asked as: ' + data.asked_as + '  (' + data.normalised_rule + ')';
            bot.appendChild(n);
          }
          if(data.check !== undefined){
            const c = data.check; const b = document.createElement('div');
            if(c === null){ b.className='check unknown'; b.textContent='NOT CHECKED — not a question I can verify'; }
            else if(c.correct){ b.className='check ok'; b.textContent='CORRECT — ' + c.expected; }
            else { b.className='check bad';
                   b.textContent='WRONG — answered ' + (c.predicted===null?'nothing':c.predicted) + ', should be ' + c.expected; }
            bot.appendChild(b);
          }
          if(data.recall){
            const v = data.recall.verdict;
            const b = document.createElement('div');
            b.className = 'recall ' + (v==='largely_recalled'?'recalled':v==='part_recalled'?'partial':'novel');
            b.textContent = (v==='largely_recalled'?'RECALLED':v==='part_recalled'?'PART RECALLED':
                             v==='mostly_novel'?'NOVEL':'TOO SHORT') +
              (data.recall.windows ? ` — ${(data.recall.verbatim_rate*100).toFixed(0)}% verbatim` : '');
            bot.appendChild(b);
          }
          loadModels();
        }
      }
    }
  }catch(e){ bot.textContent = 'error: ' + e.message; }
  finally{ el('send').disabled = false; el('input').focus(); }
}
function badge(kind, text){ const b=document.createElement('div'); b.className=kind; b.textContent=text; return b; }
function verdictBadge(c){
  if(c === undefined) return null;
  if(c === null) return badge('check unknown','NOT CHECKED');
  if(c.correct)   return badge('check ok','CORRECT — ' + c.expected);
  return badge('check bad','WRONG — said ' + (c.predicted===null?'nothing':c.predicted) + ', is ' + c.expected);
}
async function compareAll(ev){
  ev.preventDefault();
  const text = el('input').value.trim(); if(!text) return;
  el('input').value=''; el('send').disabled = true;
  add('user', text);
  const holder = document.createElement('div'); holder.className='cols';
  el('log').appendChild(holder);
  const wait = document.createElement('div'); wait.className='note'; wait.textContent='asking every model…';
  el('log').appendChild(wait);
  try{
    const r = await fetch('/api/compare', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:text, max_new_tokens:Number(el('ntok').value), mode:el('mode').value})});
    const d = await r.json(); wait.remove();
    if(!r.ok){ holder.textContent = 'error: ' + (d.error||r.status); return; }
    for(const res of d.results){
      const col = document.createElement('div'); col.className='col';
      const h = document.createElement('h3'); h.textContent = res.model; col.appendChild(h);
      const body = document.createElement('div'); body.textContent = res.error || res.reply || '(empty)';
      col.appendChild(body);
      const v = verdictBadge(res.check); if(v) col.appendChild(v);
      if(res.recall){ const rv=res.recall.verdict;
        col.appendChild(badge('recall ' + (rv==='largely_recalled'?'recalled':rv==='part_recalled'?'partial':'novel'),
          rv==='largely_recalled'?'RECALLED':rv==='part_recalled'?'PART RECALLED':rv==='mostly_novel'?'NOVEL':'TOO SHORT')); }
      if(res.latency_ms!==undefined){ const m=document.createElement('div'); m.className='meta';
        m.textContent = res.latency_ms.toFixed(0)+' ms'; col.appendChild(m); }
      holder.appendChild(col);
    }
    if(d.answers_agree !== null && d.answers_agree !== undefined){
      const a=document.createElement('p'); a.className='agree ' + (d.answers_agree?'yes':'no');
      a.textContent = d.answers_agree ? 'models agree on the answer' : 'models disagree — the badges show which is right';
      el('log').appendChild(a);
    }
    loadModels();
  }catch(e){ wait.remove(); holder.textContent='error: '+e.message; }
  finally{ el('send').disabled=false; el('input').focus(); window.scrollTo(0,document.body.scrollHeight); }
}
el('bench').addEventListener('click', async ()=>{
  el('bench').disabled = true;
  const box = document.createElement('div'); box.className='bench';
  box.textContent = 'benchmarking ' + el('model').value + ' on 20 fresh problems…';
  el('log').appendChild(box); window.scrollTo(0,document.body.scrollHeight);
  try{
    const r = await fetch('/api/benchmark', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model:el('model').value, problems:20})});
    const d = await r.json();
    if(!r.ok){ box.textContent='error: '+(d.error||r.status); return; }
    box.innerHTML = '<b>'+d.model+'</b> — '+d.correct+'/'+d.problems+' = '+(d.accuracy*100).toFixed(0)+'%';
    const t=document.createElement('table');
    for(const [k,v] of Object.entries(d.by_task)){
      const row=t.insertRow(); row.insertCell().textContent=k;
      row.insertCell().textContent=v.correct+'/'+v.n;
      row.insertCell().textContent=(v.accuracy*100).toFixed(0)+'%';
    }
    box.appendChild(t);
    const n=document.createElement('div'); n.className='note'; n.textContent=d.note; box.appendChild(n);
  }catch(e){ box.textContent='error: '+e.message; }
  finally{ el('bench').disabled=false; }
});
el('form').addEventListener('submit', e => el('cmp').checked ? compareAll(e) : send(e));
el('reset').addEventListener('click', async ()=>{ el('log').innerHTML='';
  await fetch('/api/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id:SESSION})}); });
loadModels();
</script>
"""


def parse_model_spec(values: Sequence[str]) -> Dict[str, str]:
    spec: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model expects name=path, got {value!r}")
        name, path = value.split("=", 1)
        if not Path(path).is_file():
            raise ValueError(f"checkpoint for {name!r} not found: {path}")
        spec[name] = path
    if not spec:
        raise ValueError("at least one --model is required")
    return spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="a checkpoint to serve; repeatable")
    parser.add_argument("--corpus", action="append", default=[], metavar="NAME=PATH",
                        help="training corpus for a model, enabling its recall meter")
    parser.add_argument(
        "--no-normalise",
        action="store_true",
        help=(
            "send prompts exactly as typed. By default a maths question is "
            "rewritten into the corpus's own format (\"47 times 6\" -> "
            "\"What is 47 x 6?\"), because the models answer the trained form "
            "and not the natural one; the rewrite is always shown in the reply"
        ),
    )
    parser.add_argument("--max-resident", type=int, default=2,
                        help="how many checkpoints may be in memory at once")
    parser.add_argument("--max-concurrency", type=int, default=2,
                        help="concurrent generations before callers get 503")
    parser.add_argument(
        "--history-turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help=(
            "prior turns to replay into the prompt. 0 (default) because "
            "turn-aligned packing means these models never saw multi-turn "
            "context, and replaying it measurably degrades replies"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--torch_threads", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    spec = parse_model_spec(args.model)
    corpora = parse_model_spec(args.corpus) if args.corpus else {}
    unknown = set(corpora) - set(spec)
    if unknown:
        raise SystemExit(f"--corpus names no such model: {sorted(unknown)}")

    registry = ModelRegistry(spec, max_resident=args.max_resident, corpora=corpora)
    app = build_app(registry, max_concurrency=args.max_concurrency,
                    history_turns=max(0, min(args.history_turns, MAX_HISTORY_TURNS)),
                    normalise_prompts=not args.no_normalise)
    print(f"[supermix chat] {len(spec)} models: {', '.join(spec)}")
    print(f"[supermix chat] http://{args.host}:{args.port}"
          f"  (resident<={args.max_resident}, concurrency<={args.max_concurrency},"
          f" history={args.history_turns})")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
