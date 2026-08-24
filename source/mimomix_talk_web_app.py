"""Chat interface for the v57 talking MiMoMix model.

Unlike `mimomix_reasoner_web_app.py`, this one really is a language model: the
text you get back is generated token by token by `MiMoMixModel`, not assembled
by a parser. Nothing here writes sentences on the model's behalf.

What it is not: knowledgeable. The corpus is 120,000 templated coding-assistant
turns with 292 distinct word types, and the shipped checkpoint's tokenizer holds
582 of them counting both spacing forms and the specials, so the model speaks
fluently inside that
register and cannot say anything outside it. A word not in the vocabulary cannot
be produced at all. The page states this rather than letting a confident tone
imply otherwise.

Two decoding paths are offered because they answer different questions:

* **greedy** runs `mimomix_decoding.speculative_generate`, which uses the MTP
  depths as a draft model and is *provably token-identical* to one-at-a-time
  greedy decoding. It is deterministic, so the same prompt always gives the same
  reply, and it reports a measured acceptance length.
* **sample** applies temperature and nucleus filtering. It is not identical to
  greedy and gives varied replies; it re-runs the prefix per token rather than
  using the KV cache, which is simpler and fast enough at this size.

Trust boundary: the prompt is data. It is tokenized and fed to the model, and
nothing in it selects a checkpoint, a decoding mode, or a budget -- those come
from typed request fields only. The service binds 127.0.0.1.

Usage::

    python source/mimomix_talk_web_app.py --checkpoint output/v57_talk/v57_talk.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_decoding as decoding  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
import recall_index  # noqa: E402
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

MAX_NEW_TOKENS = 160
MAX_PROMPT_TOKENS = 384
MAX_HISTORY_TURNS = 4


class TalkService:
    """One loaded v57 checkpoint, serving generated text."""

    def __init__(
        self,
        checkpoint_path: Path,
        receipt_path: Optional[Path] = None,
        corpus_path: Optional[Path] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.tokenizer, payload = load_talk_checkpoint(self.checkpoint_path)
        self.extra: Dict[str, Any] = dict(payload.get("extra") or {})
        self._lock = threading.Lock()
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

        # Optional, because a checkpoint can be served without its corpus. When
        # it is available, every reply is checked against the training text --
        # this model's most fluent sentence appears verbatim in 17.6% of rows,
        # and printing it unannotated presents recall as composition.
        self.recall: Optional[recall_index.RecallIndex] = None
        if corpus_path is not None:
            self.recall = recall_index.RecallIndex.from_jsonl(corpus_path)
        self.receipt: Dict[str, Any] = {}
        self.receipt_path: Optional[str] = None
        candidate = receipt_path or (self.checkpoint_path.parent / "talk_results.json")
        if Path(candidate).is_file():
            try:
                self.receipt = json.loads(Path(candidate).read_text(encoding="utf-8"))
                self.receipt_path = str(candidate)
            except (json.JSONDecodeError, OSError):
                self.receipt = {}

    # -- status ------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        config = self.model.config
        report = self.model.parameter_report()
        final = self.receipt.get("final_validation") or {}
        corpus = self.receipt.get("corpus") or {}
        return {
            "model": "v57_mimomix_talk",
            "architecture": "mimomix_core.MiMoMixModel (v53), trained on text",
            "checkpoint": str(self.checkpoint_path),
            "receipt_path": self.receipt_path,
            "recall_index": self.recall.to_dict() if self.recall else None,
            "parameters": report,
            "config": {
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "n_layers": config.n_layers,
                "attention_layout": list(self.model.layout),
                "sliding_window": config.sliding_window,
                "use_moe": config.use_moe,
                "n_routed_experts": config.n_routed_experts,
                "moe_top_k": config.moe_top_k,
                "n_mtp_layers": config.n_mtp_layers,
                "thinking_cycles": config.thinking_cycles,
                "thinking_max_cycles": config.thinking_max_cycles,
            },
            "vocabulary": {
                "size": self.tokenizer.vocab_size,
                "note": (
                    "a word outside this vocabulary can never be generated; the "
                    "vocabulary is the ceiling on what the model can say"
                ),
            },
            "training": {
                "validation_loss": final.get("loss"),
                "perplexity": final.get("perplexity"),
                "bits_per_token": final.get("bits_per_token"),
                "train_pairs": corpus.get("train_pairs"),
                "train_seconds": self.receipt.get("train_seconds"),
                "corpus_note": corpus.get("note"),
            },
        }

    # -- generation --------------------------------------------------------

    def _prompt_ids(self, session: List[Dict[str, str]], message: str) -> List[int]:
        """Build the prompt from a bounded history plus the new message.

        Training packed single turns contiguously into fixed-length blocks, so
        the model has seen a turn following another turn. A short history is
        therefore only mildly out of distribution; a long one is not, which is
        why it is bounded.
        """

        ids: List[int] = []
        for turn in session[-MAX_HISTORY_TURNS:]:
            ids += [text_utils.BOS, text_utils.USER]
            ids += self.tokenizer.encode(turn["user"])
            ids += [text_utils.ASSISTANT]
            ids += self.tokenizer.encode(turn["assistant"])
            ids += [text_utils.EOS]
        head, _ = self.tokenizer.encode_turn(message, None)
        ids += head
        if len(ids) > MAX_PROMPT_TOKENS:
            ids = ids[-MAX_PROMPT_TOKENS:]
        return ids

    @torch.no_grad()
    def _sample(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: Optional[int],
    ) -> torch.Tensor:
        """Temperature + nucleus sampling, recomputing the prefix each token.

        Deliberately cache-free: the KV-cache path in `mimomix_decoding` exists to
        support the greedy-equivalence guarantee, and reusing it here would risk
        silently breaking that guarantee for a feature that does not need it.
        """

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        ids = prompt_ids
        produced: List[int] = []
        for _ in range(max_new_tokens):
            logits = self.model(ids[:, -self.model.config.max_position_embeddings :],
                                return_mtp=False).logits[:, -1]
            logits = logits / max(1e-4, temperature)
            probabilities = logits.softmax(dim=-1)
            ordered, order = probabilities.sort(dim=-1, descending=True)
            cumulative = ordered.cumsum(dim=-1)
            keep = cumulative - ordered <= top_p
            keep[..., 0] = True
            ordered = torch.where(keep, ordered, torch.zeros_like(ordered))
            ordered = ordered / ordered.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(ordered, num_samples=1, generator=generator)
            token = order.gather(-1, choice)
            produced.append(int(token))
            if int(token) == text_utils.EOS:
                break
            ids = torch.cat([ids, token], dim=1)
        return torch.tensor([produced], dtype=torch.long)

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        message = request.get("message")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        if not message.strip():
            raise ValueError("message must not be empty")
        if len(message) > 4000:
            raise ValueError("message is too long")

        session_id = request.get("session_id")
        session_id = session_id if isinstance(session_id, str) and session_id else "default"
        mode = request.get("mode", "greedy")
        if mode not in {"greedy", "sample"}:
            raise ValueError("mode must be 'greedy' or 'sample'")
        max_new_tokens = _clamp_int(request.get("max_new_tokens", 64), 1, MAX_NEW_TOKENS)
        temperature = _clamp_float(request.get("temperature", 0.9), 0.05, 2.0)
        top_p = _clamp_float(request.get("top_p", 0.9), 0.05, 1.0)
        cycles = request.get("thinking_cycles")
        if cycles is not None:
            cycles = _clamp_int(cycles, 1, self.model.config.thinking_max_cycles)

        unknown_rate = self.tokenizer.unknown_rate(message)
        session = self._sessions.setdefault(session_id, [])
        prompt_ids = self._prompt_ids(session, message)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)

        started = time.perf_counter()
        with self._lock:
            self.model.eval()
            if mode == "greedy":
                result = decoding.speculative_generate(
                    self.model,
                    prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=text_utils.EOS,
                    thinking_cycles=cycles,
                )
                new_tokens = result.new_tokens
                stats = {
                    "acceptance_length": round(result.stats.acceptance_length, 4),
                    "acceptance_rate": round(result.stats.acceptance_rate, 4),
                    "trunk_forwards": int(result.stats.verify_forwards),
                    "identical_to_plain_greedy": True,
                }
            else:
                seed = request.get("seed")
                # The only numeric field that used to skip the clamp helpers,
                # and the only one that could raise out of this method.
                if seed is not None:
                    seed = _clamp_int(seed, 0, 2**31 - 1)
                new_tokens = self._sample(
                    prompt_tensor,
                    max_new_tokens,
                    temperature,
                    top_p,
                    seed,
                )
                stats = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "identical_to_plain_greedy": False,
                }
            telemetry = self.model.telemetry()
        elapsed = time.perf_counter() - started

        reply = self.tokenizer.decode(new_tokens[0].tolist()).strip()
        produced = int(new_tokens.shape[1])
        session.append({"user": message, "assistant": reply})
        del session[:-MAX_HISTORY_TURNS]

        thinking = telemetry.get("thinking") or {}
        return {
            "reply": reply,
            "session_id": session_id,
            "mode": mode,
            "tokens_generated": produced,
            "prompt_tokens": len(prompt_ids),
            "latency_ms": round(elapsed * 1000.0, 2),
            "tokens_per_second": round(produced / max(1e-6, elapsed), 2),
            "decoding": stats,
            "thinking": {
                "cycles_used": thinking.get("cycles_used"),
                "exit_reason": thinking.get("exit_reason"),
            },
            "prompt_unknown_word_rate": round(unknown_rate, 4),
            # Present only when a corpus was supplied. `null` means "not
            # checked", which is different from "checked and found novel" --
            # conflating the two would let an unindexed server imply originality
            # it never measured.
            "recall": self.recall.score(reply).to_dict() if self.recall else None,
            "warnings": (
                # Read the size off the loaded tokenizer rather than naming a
                # number here: this string said "340-word" while the shipped
                # checkpoint carried 582 types, because the vocabulary grew when
                # the sentence-initial token bug was fixed and the copy did not.
                [f"some words in your message are outside the model's "
                 f"{self.tokenizer.vocab_size}-word vocabulary and were "
                 f"replaced with <unk>"]
                if unknown_rate > 0
                else []
            ),
        }

    def reset(self, session_id: str) -> Dict[str, Any]:
        self._sessions.pop(session_id, None)
        return {"session_id": session_id, "cleared": True}


def _clamp_int(value: Any, low: int, high: int) -> int:
    """Coerce and clamp, converting every rejection into a 400, never a 500.

    `OverflowError` has to be caught alongside the obvious two: `int()` raises
    it -- not `ValueError` -- for infinity, and Python's JSON parser accepts a
    bare `Infinity` literal, so a request body can reach here with one. Only
    `ValueError` and `TypeError` are handled by the route, so an uncaught
    `OverflowError` surfaces as an internal error instead of a rejected field.

    NaN needs no special case: it fails every comparison, so `min`/`max` return
    the bound rather than propagating it.
    """

    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected an integer, got {value!r}") from error


def _clamp_float(value: Any, low: float, high: float) -> float:
    """Coerce and clamp. See `_clamp_int` on `OverflowError`.

    Here it arrives by a second route as well: a JSON integer literal has no
    size limit, and `float()` raises `OverflowError` on one too large to
    represent.
    """

    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected a number, got {value!r}") from error


# ---------------------------------------------------------------------------
# Flask surface
# ---------------------------------------------------------------------------


def build_app(service: TalkService):
    from flask import Flask, jsonify, request as flask_request

    app = Flask(__name__)

    def no_store(payload: Dict[str, Any], status: int = 200):
        response = jsonify(payload)
        response.status_code = status
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> str:
        return HTML_TEMPLATE

    @app.get("/api/status")
    def status():
        return no_store(service.status())

    @app.post("/api/chat")
    def chat():
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return no_store({"error": "request body must be a JSON object"}, 400)
        try:
            return no_store(service.chat(payload))
        except (ValueError, TypeError) as error:
            return no_store({"error": str(error)}, 400)

    @app.post("/api/reset")
    def reset():
        payload = flask_request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        return no_store(service.reset(session_id if isinstance(session_id, str) else "default"))

    @app.get("/health")
    def health():
        return no_store({"status": "ok", "checkpoint": str(service.checkpoint_path)})

    app.config["TALK_SERVICE"] = service
    return app


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supermix v57 &mdash; MiMoMix Talk</title>
<style>
:root{ --bg:#0b0f14; --panel:#121821; --panel-2:#0e141c; --line:#1f2937; --ink:#e5e7eb;
  --dim:#9ca3af; --ok:#34d399; --no:#f87171; --accent:#60a5fa; --warm:#fbbf24; }
@media (prefers-color-scheme: light){
  :root{ --bg:#f6f7f9; --panel:#fff; --panel-2:#f0f2f5; --line:#dfe3e8; --ink:#111827;
         --dim:#5b6472; --ok:#047857; --no:#b91c1c; --accent:#1d4ed8; --warm:#a16207; } }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;height:100vh}
header{padding:13px 20px;border-bottom:1px solid var(--line);background:var(--panel-2)}
h1{margin:0;font-size:16px}
.sub{color:var(--dim);font-size:12px;margin-top:3px}
.banner{margin:12px 20px 0;padding:10px 12px;border:1px solid var(--warm);border-radius:8px;
  background:color-mix(in srgb,var(--warm) 10%,transparent);font-size:12.5px}
#log{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:11px}
.msg{max-width:min(720px,92%);padding:10px 13px;border-radius:10px;border:1px solid var(--line);
  white-space:pre-wrap;word-break:break-word}
.me{align-self:flex-end;background:var(--accent);color:#fff;border-color:transparent}
.bot{align-self:flex-start;background:var(--panel)}
.bot.bad{border-color:var(--no)}
.meta{margin-top:7px;font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.meta b{color:var(--ink);font-weight:600}
.recall{margin-top:5px;padding:4px 8px;border-radius:4px;font-weight:600;display:inline-block}
.recall.recalled{background:#5a1d1d;color:#ffb4b4}
.recall.partial{background:#5a4a1d;color:#ffe0a0}
.recall.novel{background:#1d4a2a;color:#a8e6b8}
.warn{color:var(--warm)}
form{display:flex;gap:8px;padding:11px 20px;border-top:1px solid var(--line);background:var(--panel-2);
  flex-wrap:wrap;align-items:center}
input[type=text]{flex:1;min-width:220px;font:inherit;color:var(--ink);background:var(--panel);
  border:1px solid var(--line);border-radius:8px;padding:9px 12px}
select,input[type=number]{font:inherit;color:var(--ink);background:var(--panel);
  border:1px solid var(--line);border-radius:8px;padding:7px 9px}
button{font:inherit;font-weight:600;background:var(--accent);color:#fff;border:0;border-radius:8px;
  padding:9px 16px;cursor:pointer}
button.ghost{background:var(--panel);color:var(--ink);border:1px solid var(--line);font-weight:500}
button:disabled{opacity:.5;cursor:default}
.chips{display:flex;gap:6px;flex-wrap:wrap;padding:0 20px 10px}
.chip{font-size:12px;padding:4px 9px;border:1px solid var(--line);border-radius:999px;
  background:var(--panel);color:var(--dim);cursor:pointer}
footer{padding:10px 20px 16px;color:var(--dim);font-size:11px;border-top:1px solid var(--line)}
label{font-size:11px;color:var(--dim)}
</style>
</head>
<body>
<header>
  <h1>Supermix v57 &mdash; MiMoMix Talk</h1>
  <div class="sub" id="sub">loading&hellip;</div>
</header>

<div class="banner">
  <strong>This text is generated.</strong> Every reply below is produced token by token by the
  model &mdash; no parser, no templates, nothing written on its behalf. It is also
  <strong>not knowledgeable</strong>: it was trained on templated coding-assistant dialogue with a
  <span id="vocab">&hellip;</span>-word vocabulary, so it speaks only in that register and cannot
  produce a word it has never seen. Treat everything it says as style, not fact.
</div>

<div id="log"></div>

<div class="chips">
  <span class="chip" data-q="hello">hello</span>
  <span class="chip" data-q="can you help me with tests">can you help me with tests</span>
  <span class="chip" data-q="why is my script failing">why is my script failing</span>
  <span class="chip" data-q="what is your name">what is your name</span>
  <span class="chip" data-q="thanks">thanks</span>
</div>

<form id="form">
  <input id="input" type="text" autocomplete="off" placeholder="say something&hellip;" />
  <div><label for="mode">decoding</label><br>
    <select id="mode">
      <option value="greedy">greedy (MTP, deterministic)</option>
      <option value="sample">sample</option>
    </select></div>
  <div><label for="temp">temp</label><br>
    <input id="temp" type="number" value="0.9" min="0.05" max="2" step="0.05" style="width:76px"></div>
  <div><label for="ntok">max tokens</label><br>
    <input id="ntok" type="number" value="64" min="1" max="160" step="8" style="width:82px"></div>
  <button id="send" type="submit">Send</button>
  <button id="reset" class="ghost" type="button">Reset</button>
</form>

<footer id="foot"></footer>

<script>
(function(){
  const el = id => document.getElementById(id);
  const SESSION = 'talk-' + Math.random().toString(36).slice(2);
  const log = el('log');

  function bubble(text, cls){
    const d = document.createElement('div');
    d.className = 'msg ' + cls;
    d.textContent = text;
    log.appendChild(d); log.scrollTop = log.scrollHeight;
    return d;
  }

  async function send(text){
    if(!text.trim()) return;
    bubble(text, 'me');
    el('send').disabled = true;
    const pending = bubble('…', 'bot');
    try{
      const r = await fetch('/api/chat', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          message: text, session_id: SESSION,
          mode: el('mode').value,
          temperature: Number(el('temp').value),
          max_new_tokens: Number(el('ntok').value)
        })
      });
      const d = await r.json();
      if(!r.ok){ pending.textContent = 'error: ' + (d.error || r.status); pending.className = 'msg bot bad'; }
      else {
        pending.textContent = d.reply || '(empty reply)';
        const m = document.createElement('div'); m.className = 'meta';
        let line = d.tokens_generated + ' tokens · ' + d.tokens_per_second.toFixed(1) + ' tok/s · ' +
                   d.latency_ms.toFixed(0) + ' ms · ' + d.mode;
        if(d.decoding.acceptance_length !== undefined){
          line += ' · MTP acceptance ' + d.decoding.acceptance_length.toFixed(2) +
                  ' (' + d.decoding.trunk_forwards + ' trunk forwards)';
        }
        if(d.thinking && d.thinking.cycles_used) line += ' · ' + d.thinking.cycles_used + ' thinking cycles';
        m.textContent = line;
        pending.appendChild(m);
        if(d.recall){
          // The reply's provenance, not its statistics. A model whose most
          // fluent sentence appears in 17.6% of its training rows should say so
          // next to the sentence, not in a footnote nobody reads.
          const r2 = document.createElement('div');
          const v = d.recall.verdict;
          r2.className = 'meta recall ' + (v === 'largely_recalled' ? 'recalled'
                        : v === 'part_recalled' ? 'partial' : 'novel');
          const label = v === 'largely_recalled' ? 'RECALLED'
                      : v === 'part_recalled' ? 'PART RECALLED'
                      : v === 'mostly_novel' ? 'NOVEL' : 'TOO SHORT';
          if(v === 'too_short_to_judge'){
            r2.textContent = label + ' — reply is shorter than the 8-word window';
          } else {
            r2.textContent = label + ' — ' + (d.recall.verbatim_rate*100).toFixed(0) +
              '% of 8-word windows appear verbatim in training' +
              (d.recall.longest_verbatim_words ?
                 ', longest run ' + d.recall.longest_verbatim_words + ' words' : '');
          }
          pending.appendChild(r2);
        }
        if(d.warnings && d.warnings.length){
          const w = document.createElement('div'); w.className = 'meta warn';
          w.textContent = d.warnings.join(' ');
          pending.appendChild(w);
        }
      }
    }catch(e){ pending.textContent = 'error: ' + e.message; pending.className = 'msg bot bad'; }
    finally{ el('send').disabled = false; el('input').focus(); log.scrollTop = log.scrollHeight; }
  }

  el('form').addEventListener('submit', e => {
    e.preventDefault();
    const t = el('input').value; el('input').value = ''; send(t);
  });
  el('reset').addEventListener('click', async () => {
    await fetch('/api/reset', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({session_id: SESSION})});
    log.replaceChildren();
    bubble('History cleared.', 'bot');
  });
  document.querySelectorAll('.chip').forEach(c =>
    c.addEventListener('click', () => send(c.dataset.q)));

  fetch('/api/status').then(r => r.json()).then(s => {
    el('vocab').textContent = s.vocabulary.size;
    const t = s.training || {};
    el('sub').textContent =
      s.parameters.total.toLocaleString() + ' parameters (' +
      s.parameters.active_per_token.toLocaleString() + ' active/token) · ' +
      s.config.n_layers + ' layers ' + (s.config.attention_layout||[]).map(k=>k==='global'?'G':'L').join('') +
      ' · ' + (s.config.use_moe ? s.config.n_routed_experts + ' experts top-' + s.config.moe_top_k : 'dense') +
      ' · vocab ' + s.vocabulary.size +
      (t.perplexity ? ' · held-out perplexity ' + t.perplexity : '');
    el('foot').textContent =
      'Architecture is mimomix_core.MiMoMixModel — the v53 stack (hybrid sliding-window/global ' +
      'attention with learnable sinks, auxiliary-loss-free sparse MoE, multi-token prediction, ' +
      'recursive thinking core) trained on text for the first time. Greedy replies are produced by ' +
      'MTP self-speculative decoding, which is token-identical to plain greedy decoding and only ' +
      'changes the cost. ' + (t.corpus_note || '');
  }).catch(() => { el('sub').textContent = 'status unavailable'; });

  bubble('Say something. I only know one register: short coding-assistant replies.', 'bot');
  el('input').focus();
})();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the v57 talking MiMoMix model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--receipt", default=None)
    parser.add_argument(
        "--corpus",
        default=None,
        help=(
            "JSONL training corpus. When given, every reply is checked against "
            "it and the interface reports how much was recalled verbatim rather "
            "than composed. Costs ~30s of indexing at startup"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8157)
    parser.add_argument("--torch_threads", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"checkpoint not found: {checkpoint}")
        return 2
    service = TalkService(
        checkpoint,
        Path(args.receipt) if args.receipt else None,
        Path(args.corpus) if args.corpus else None,
    )
    status = service.status()
    app = build_app(service)
    print(f"[Supermix v57] serving {checkpoint}")
    print(f"  {status['parameters']['total']:,} parameters, vocabulary {status['vocabulary']['size']}, "
          f"perplexity {status['training'].get('perplexity')}")
    print(f"[Supermix v57] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
