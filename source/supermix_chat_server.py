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
import hashlib
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_text as text_utils  # noqa: E402
import answer_check  # noqa: E402
import prompt_normaliser  # noqa: E402
import eval_problem_solving as solving  # noqa: E402
import recall_index  # noqa: E402
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

MAX_NEW_TOKENS = 256
DEFAULT_MAX_NEW_TOKENS = solving.DEFAULT_MAX_NEW_TOKENS
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
                    cfg = getattr(loaded.model, "config", None)
                    if cfg is not None:
                        entry["architecture"] = {
                            "cns_core": bool(getattr(cfg, "use_cns_core", False)),
                            "cns_dual_soma": bool(getattr(cfg, "cns_dual_soma", False)),
                            "cns_hebbian_trace": bool(getattr(cfg, "cns_hebbian_trace", False)),
                            "cns_predictive_error": bool(getattr(cfg, "cns_predictive_error", False)),
                            "thinking_entropy_halt": bool(getattr(cfg, "thinking_entropy_halt", False)),
                            "run_name": loaded.extra.get("run_name"),
                        }
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


@dataclass
class GenerationProgress:
    tokens: int = 0
    finish_reason: str = "length"
    acceptance_length: Optional[float] = None
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    verify_forwards: int = 0


def generation_details(reply: str, progress: GenerationProgress, cap: int) -> Dict[str, Any]:
    out = {
        "tokens": progress.tokens,
        "max_new_tokens": cap,
        "finish_reason": progress.finish_reason,
        "hit_token_cap": progress.finish_reason == "length",
        "truncated": solving.is_truncated(reply, progress.tokens, cap),
    }
    if progress.acceptance_length is not None:
        out["speculative_stats"] = {
            "acceptance_length": round(float(progress.acceptance_length), 4),
            "drafted_tokens": int(progress.drafted_tokens),
            "accepted_draft_tokens": int(progress.accepted_draft_tokens),
            "verify_forwards": int(progress.verify_forwards),
        }
    return out


def _architecture_telemetry_snapshot(model: Any) -> Optional[Dict[str, Any]]:
    fn = getattr(model, "telemetry", None)
    if not callable(fn):
        return None
    try:
        tele = fn()
        summary: Dict[str, Any] = {}
        cns = tele.get("cns_core")
        if isinstance(cns, dict):
            summary["cns_core"] = {
                "alive_nodes": cns.get("alive_nodes"),
                "alive_edges": cns.get("edges_installed"),
                "active_fraction": cns.get("active_fraction"),
                "dual_soma": cns.get("dual_soma", False),
                "glial_homeostasis": cns.get("glial_homeostasis", False),
                "hebbian_trace": cns.get("hebbian_trace", False),
                "hebbian_gain_mean": cns.get("hebbian_gain_mean"),
                "hebbian_strength_mean_abs": cns.get("hebbian_strength_mean_abs"),
                "predictive_error": cns.get("predictive_error", False),
                "pred_error_norm": cns.get("pred_error_norm"),
                "pred_error_strength_mean_abs": cns.get("pred_error_strength_mean_abs"),
                "spectral_radius_est": cns.get("spectral_radius_est"),
            }
        thk = tele.get("thinking")
        if isinstance(thk, dict):
            summary["thinking"] = {
                "cycles_used": thk.get("cycles_used"),
                "exit_reason": thk.get("exit_reason"),
                "cycle_entropies": thk.get("cycle_entropies"),
            }
        return summary or None
    except Exception:
        return None


def _scf_verify_reply(asked: str, normalised_query: str, reply: str) -> Optional[Dict[str, Any]]:
    try:
        from scf.verification import SCFVerificationGate

        res = SCFVerificationGate().verify(
            query=asked,
            candidate_answer=reply,
            normalised_query=normalised_query,
        )
        if not res.get("verified"):
            return {
                "verified": False,
                "verifier_source": res.get("verifier_source", "none"),
            }
        expected = res.get("verified_answer")
        predicted_num = solving.extract_answer(reply)
        agrees: Optional[bool] = None
        if expected is not None and predicted_num is not None:
            try:
                agrees = solving.is_correct(predicted_num, float(expected))
            except Exception:
                agrees = None
        return {
            "verified": True,
            "verified_answer": expected,
            "verifier_source": res.get("verifier_source"),
            "model_agrees": agrees,
            "independent_certificate": res.get("independent_certificate"),
            "proof_capsule": res.get("proof_capsule"),
        }
    except Exception:
        return None


def _scf_runtime_trace(
    asked: str,
    reply: str,
    model_name: str,
    mode: str,
    recent_turns: Sequence[Tuple[str, str]] = (),
) -> Optional[Dict[str, Any]]:
    try:
        from scf import CognitiveFabricRuntime, SCFFeatureFlags

        flags = SCFFeatureFlags.all_enabled(speculative_decoding=(mode == "speculative"))
        runtime = CognitiveFabricRuntime()
        res = runtime.run_task(
            asked,
            flags_override=flags,
            recent_turns=list(recent_turns),
            adaptive_thinking=(mode != "speculative"),
            candidate_generators={
                "MiMoMix": lambda _q: {
                    "family": "MiMoMix",
                    "model_name": model_name,
                    "answer": reply,
                    "confidence": 0.62,
                    "verified": False,
                    "model_conditional": True,
                }
            },
            enable_neural_candidates=False,
        )
        rec = res.get("receipt") or {}
        routing = res.get("routing") or {}
        return {
            "scf_active": bool(res.get("scf_active")),
            "verified": bool(res.get("verified")),
            "final_answer": res.get("final_answer"),
            "selected_family": routing.get("selected_family"),
            "domain": (routing.get("assessment") or {}).get("domain"),
            "decoding_policy": (res.get("adaptive_budget") or {}).get("decoding_policy"),
            "experimental_components_used": rec.get("experimental_components_used") or [],
            "production_ready_claim": bool(rec.get("production_ready_claim", False)),
            "total_latency_ms": rec.get("total_latency_ms"),
            "evidence_links": rec.get("evidence_links") or [],
        }
    except Exception:
        return None


def stream_speculative_tokens(
    loaded: LoadedModel,
    message: str,
    history: Sequence[Tuple[str, str]],
    max_new_tokens: int,
    should_stop,
    progress: Optional[GenerationProgress] = None,
) -> Iterator[str]:
    """Run exact greedy MTP self-speculative decode and yield word chunks."""
    import mimomix_decoding as decoding

    progress = progress if progress is not None else GenerationProgress()
    tokenizer = loaded.tokenizer
    ids: List[int] = []
    for past_user, past_reply in history:
        turn, _ = tokenizer.encode_turn(past_user, past_reply)
        ids.extend(turn)
    prompt, _ = tokenizer.encode_turn(message, None)
    ids.extend(prompt)
    prompt_tensor = torch.tensor([ids[-MAX_CONTEXT:]], dtype=torch.long)

    with torch.no_grad():
        res = decoding.speculative_generate(
            loaded.model,
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=text_utils.EOS,
        )
    new_toks = res.new_tokens[0].tolist()
    progress.tokens = len(new_toks)
    progress.finish_reason = "eos" if (new_toks and new_toks[-1] == text_utils.EOS) else (
        "length" if len(new_toks) >= max_new_tokens else "eos"
    )
    if new_toks and new_toks[-1] == text_utils.EOS:
        new_toks = new_toks[:-1]
    progress.acceptance_length = float(getattr(res.stats, "acceptance_length", 1.0))
    progress.drafted_tokens = int(getattr(res.stats, "drafted_tokens", 0))
    progress.accepted_draft_tokens = int(getattr(res.stats, "accepted_draft_tokens", 0))
    progress.verify_forwards = int(getattr(res.stats, "verify_forwards", 0))

    emitted = ""
    for i in range(1, len(new_toks) + 1):
        if should_stop():
            progress.finish_reason = "cancelled"
            break
        full = tokenizer.decode(new_toks[:i])
        if len(full) > len(emitted):
            yield full[len(emitted):]
            emitted = full


def check_reply(message: str, reply: str):
    rule = prompt_normaliser.normalise(message).rule
    if rule in prompt_normaliser.COGNITIVE_LEAD_IN:
        return None
    verdict = answer_check.check(message, reply)
    covered = (set(prompt_normaliser.LEAD_IN) | set(prompt_normaliser.SCIENCE_LEAD_IN)
               | {"arithmetic", "average", "percent", "two_step", "sequence", "algebra_one_step"})
    # A substring checker must not undo the full-request parser's refusal.
    # Keep the separate checker families (e.g. bounded code traces) unchanged.
    if verdict is not None:
        if verdict.task in covered and rule not in covered:
            return None
    return verdict


def stream_tokens(
    loaded: LoadedModel,
    message: str,
    history: Sequence[Tuple[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    should_stop,
    progress: Optional[GenerationProgress] = None,
) -> Iterator[str]:
    """Yield decoded text incrementally.

    Recomputes the full prefix each step rather than carrying a KV cache. That
    is O(n^2), and at these sizes and lengths it is not the bottleneck -- the
    point of streaming here is that the caller sees progress, not that decoding
    is optimal. `should_stop` is polled every token so a disconnected client
    stops the work instead of paying for a reply nobody will read.
    """

    progress = progress if progress is not None else GenerationProgress()
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
                progress.finish_reason = "cancelled"
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
                progress.finish_reason = "eos"
                break
            generated.append(nxt)
            progress.tokens += 1

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

    def integer_option(payload, key, default, minimum, maximum):
        value = payload.get(key, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be an integer from {minimum} to {maximum}")
        return value

    def float_option(payload, key, default, minimum, maximum):
        value = payload.get(key, default)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not minimum <= value <= maximum):
            raise ValueError(f"{key} must be a finite number from {minimum} to {maximum}")
        return float(value)

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
        if not isinstance(name, str) or name not in registry.spec:
            return {}, f"unknown model {name!r}"
        session_id = payload.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            return {}, "session_id must be a string"
        if type(payload.get("check", True)) is not bool:
            return {}, "check must be a boolean"
        if type(payload.get("scf_verify", False)) is not bool:
            return {}, "scf_verify must be a boolean"
        if type(payload.get("scf_runtime", False)) is not bool:
            return {}, "scf_runtime must be a boolean"
        mode = payload.get("mode", "greedy")
        if mode not in ("greedy", "sample", "speculative"):
            return {}, "mode must be greedy, speculative, or sample"
        try:
            cap = integer_option(payload, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS, 1, MAX_NEW_TOKENS)
            temperature = float_option(payload, "temperature", 0.9, 0.05, 2.0)
            top_p = float_option(payload, "top_p", 0.9, 0.05, 1.0)
        except ValueError as exc:
            return {}, str(exc)
        rewrite = prompt_normaliser.normalise(message) if normalise_prompts \
            else prompt_normaliser.Normalised(message, None, message)
        return {
            "message": rewrite.prompt,
            "asked": message,
            "normalised_rule": rewrite.rule if rewrite.changed else None,
            "model": name,
            "session_id": session_id if isinstance(session_id, str) and session_id else "default",
            "check": bool(payload.get("check", True)),
            "scf_verify": bool(payload.get("scf_verify", False)),
            "scf_runtime": bool(payload.get("scf_runtime", False)),
            "mode": mode,
            "max_new_tokens": cap,
            "temperature": temperature,
            "top_p": top_p,
            "greedy": mode in ("greedy", "speculative"),
        }, None

    @app.get("/")
    def index():
        return Response(PAGE.replace("__DEFAULT_MAX_NEW_TOKENS__", str(DEFAULT_MAX_NEW_TOKENS)),
                        mimetype="text/html")

    @app.get("/api/models")
    def models():
        return no_store({"models": registry.describe(), "max_resident": registry.max_resident})

    @app.get("/api/scf_status")
    def scf_status():
        try:
            from scf import SCF_INTERNAL_NAME, SCF_VERSION, SCFFeatureFlags
            from scf.models import MiMoMixAdapter, SCFModelRegistry
            from scf.verification import SCFVerificationGate

            summary = SCFModelRegistry().summary()
            atlas_probe = SCFVerificationGate().connectome_atlas.verify_query(
                "How many Li20 neurons does the male CNS contain?"
            )
            return no_store({
                "version": SCF_VERSION,
                "name": SCF_INTERNAL_NAME,
                "default_flags": SCFFeatureFlags().to_dict(),
                "all_enabled_flags": SCFFeatureFlags.all_enabled().to_dict(),
                "connectome_atlas_ready": bool(atlas_probe and atlas_probe.get("solved")),
                "talk_checkpoints": MiMoMixAdapter().discover_talk_checkpoints(),
                "families": {
                    k: {
                        "maturity_status": v.get("maturity_status"),
                        "weights": len(v.get("available_weights") or []),
                        "experimental_isolated": bool(v.get("experimental_isolated", False)),
                    }
                    for k, v in summary.items()
                },
                "server_models": registry.describe(),
            })
        except Exception as exc:  # noqa: BLE001
            return no_store({"error": str(exc)}, 500)

    @app.get("/api/stats")
    def statistics():
        with stats_guard:
            return no_store(dict(stats, concurrency_limit=max_concurrency))

    def generate_once(loaded, message, options) -> Dict[str, Any]:
        """One complete reply, non-streamed. Shared by compare and benchmark."""

        started = time.perf_counter()
        pieces: List[str] = []
        progress = GenerationProgress()
        with loaded.lock:
            if options.get("mode") == "speculative":
                gen_iter = stream_speculative_tokens(
                    loaded, message, [], options["max_new_tokens"],
                    should_stop=lambda: False, progress=progress,
                )
            else:
                gen_iter = stream_tokens(
                    loaded, message, [], options["max_new_tokens"],
                    options["temperature"], options["top_p"], options["greedy"],
                    should_stop=lambda: False,
                    progress=progress,
                )
            for chunk in gen_iter:
                pieces.append(chunk)
        elapsed = time.perf_counter() - started
        reply = "".join(pieces).strip()
        result: Dict[str, Any] = {
            "model": loaded.name,
            "reply": reply,
            "latency_ms": round(elapsed * 1000, 2),
            **generation_details(reply, progress, options["max_new_tokens"]),
        }
        arch_tele = _architecture_telemetry_snapshot(loaded.model)
        if arch_tele is not None:
            result["architecture_telemetry"] = arch_tele
        if loaded.recall is not None:
            result["recall"] = loaded.recall.score(reply).to_dict()
        if options.get("check", True):
            verdict = check_reply(message, reply)
            result["check"] = verdict.to_dict() if verdict else None
        if options.get("scf_verify", False):
            result["scf_check"] = _scf_verify_reply(options.get("asked", message), message, reply)
        if options.get("scf_runtime", False):
            result["scf_runtime"] = _scf_runtime_trace(
                options.get("asked", message),
                reply,
                loaded.name,
                options.get("mode", "greedy"),
            )
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
        names = payload.get("models", registry.names())
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
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
                "asked": options["asked"],
                "asked_as": options["message"],
                "normalised_rule": options["normalised_rule"],
                "results": results,
                # Only meaningful when the question was checkable at all.
                "answers_agree": (len(agree) <= 1) if agree else None,
            })
        finally:
            admission.release()

    @app.post("/api/benchmark")
    def benchmark():
        """Run a small reproducible development sample with evaluator settings."""

        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return no_store({"error": "request body must be a JSON object"}, 400)
        name = payload.get("model") or registry.names()[0]
        if not isinstance(name, str) or name not in registry.spec:
            return no_store({"error": f"unknown model {name!r}"}, 400)
        try:
            count = integer_option(payload, "problems", 20, 1, 60)
            seed = integer_option(payload, "seed", 65, -(2 ** 63), 2 ** 63 - 1)
            cap = integer_option(payload, "max_new_tokens", DEFAULT_MAX_NEW_TOKENS, 1, MAX_NEW_TOKENS)
        except ValueError as exc:
            return no_store({"error": str(exc)}, 400)

        if not admission.acquire(blocking=False):
            response = no_store({"error": "server busy"}, 503)
            response.headers["Retry-After"] = "5"
            return response
        try:
            loaded = registry.acquire(name)
            use_scf = bool(payload.get("scf_verify", False))
            problems = solving.generate_novel(count, seed=seed)
            per_task: Dict[str, Dict[str, int]] = {}
            scf_hybrid_correct = 0
            for problem in problems:
                out = generate_once(
                    loaded, problem.prompt,
                    {"max_new_tokens": cap, "temperature": 0.9, "top_p": 0.9,
                     "greedy": True, "check": False},
                )
                bucket = per_task.setdefault(problem.task, {"n": 0, "correct": 0, "truncated": 0, "scf_correct": 0})
                bucket["n"] += 1
                bucket["truncated"] += int(out["truncated"])
                is_neural_ok = bool(
                    solving.is_correct(solving.extract_answer(out["reply"]), problem.answer)
                )
                bucket["correct"] += int(is_neural_ok)
                is_hyb_ok = is_neural_ok
                if use_scf:
                    sc = _scf_verify_reply(problem.prompt, prompt_normaliser.normalise(problem.prompt).prompt, out["reply"])
                    if sc and sc.get("verified") and sc.get("verified_answer") is not None:
                        try:
                            is_hyb_ok = bool(solving.is_correct(solving.extract_answer(str(sc["verified_answer"])), problem.answer))
                        except Exception:
                            pass
                bucket["scf_correct"] += int(is_hyb_ok)
                scf_hybrid_correct += int(is_hyb_ok)
            total = sum(b["n"] for b in per_task.values())
            correct = sum(b["correct"] for b in per_task.values())
            resp_payload: Dict[str, Any] = {
                "model": name,
                "problems": total,
                "correct": correct,
                "accuracy": round(correct / max(1, total), 4),
                "truncated_replies": sum(b["truncated"] for b in per_task.values()),
                "development_only": True,
                "training_overlap_checked": False,
                "selection_authorized": False,
                "settings": {
                    "seed": seed,
                    "tasks": list(solving.GENERATORS),
                    "shared_rng": False,
                    "generator_fingerprint": solving.generator_fingerprint(),
                    "problem_fingerprint": hashlib.sha256(json.dumps(
                        [(p.task, p.prompt, p.answer) for p in problems],
                        ensure_ascii=True, separators=(",", ":"), allow_nan=False,
                    ).encode("utf-8")).hexdigest(),
                    "max_new_tokens": cap,
                    "greedy": True,
                    "normalise_prompts": False,
                    "tolerance": solving.TOLERANCE,
                },
                "by_task": {
                    k: {**v, "accuracy": round(v["correct"] / max(1, v["n"]), 4)}
                    for k, v in sorted(per_task.items())
                },
                "note": (
                    "Development sample from training-related generators. Training "
                    "overlap and prompt/answer collisions have not been checked; "
                    "memorisation can contribute to this score. Small samples are "
                    "noisy and may omit task families. This is not sealed holdout "
                    "or model-promotion evidence."
                ),
            }
            if use_scf:
                resp_payload["scf_hybrid_correct"] = scf_hybrid_correct
                resp_payload["scf_hybrid_accuracy"] = round(scf_hybrid_correct / max(1, total), 4)
            return no_store(resp_payload)
        finally:
            admission.release()

    @app.post("/api/reset")
    def reset():
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return no_store({"error": "request body must be a JSON object"}, 400)
        session_id = payload.get("session_id") or "default"
        if not isinstance(session_id, str):
            return no_store({"error": "session_id must be a string"}, 400)
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
            progress = GenerationProgress()
            try:
                with loaded.lock:
                    if options.get("mode") == "speculative":
                        gen_iter = stream_speculative_tokens(
                            loaded, options["message"], history,
                            options["max_new_tokens"],
                            should_stop=cancelled.is_set, progress=progress,
                        )
                    else:
                        gen_iter = stream_tokens(
                            loaded, options["message"], history,
                            options["max_new_tokens"],
                            options["temperature"], options["top_p"], options["greedy"],
                            should_stop=cancelled.is_set,
                            progress=progress,
                        )
                    for chunk in gen_iter:
                        pieces.append(chunk)
                        yield f"event: token\ndata: {json.dumps({'text': chunk})}\n\n"
                elapsed = time.perf_counter() - started
                reply = "".join(pieces).strip()
                summary: Dict[str, Any] = {
                    "model": loaded.name,
                    "reply": reply,
                    "asked": options["asked"],
                    "asked_as": options["message"],
                    "normalised_rule": options["normalised_rule"],
                    **generation_details(reply, progress, options["max_new_tokens"]),
                    "latency_ms": round(elapsed * 1000, 2),
                    "tokens_per_second": round(progress.tokens / max(1e-6, elapsed), 2),
                }
                arch_tele = _architecture_telemetry_snapshot(loaded.model)
                if arch_tele is not None:
                    summary["architecture_telemetry"] = arch_tele
                if loaded.recall is not None:
                    summary["recall"] = loaded.recall.score(reply).to_dict()
                if options["check"]:
                    verdict = check_reply(options["message"], reply)
                    summary["check"] = verdict.to_dict() if verdict else None
                if options.get("scf_verify", False):
                    summary["scf_check"] = _scf_verify_reply(
                        options["asked"], options["message"], reply
                    )
                if options.get("scf_runtime", False):
                    summary["scf_runtime"] = _scf_runtime_trace(
                        options["asked"],
                        reply,
                        loaded.name,
                        options.get("mode", "greedy"),
                        history,
                    )
                remember(options["session_id"], options["message"], reply)
                with stats_guard:
                    stats["requests"] += 1
                    stats["tokens"] += progress.tokens
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


PAGE = r"""<!doctype html><meta charset="utf-8">
<title>Supermix Chat · v99 / SCF Cognitive Fabric</title>
<style>
:root{--bg:#0b101b;--panel:#131b2b;--card:#182236;--ink:#e9eff8;--dim:#8ea0be;--line:#26344f;--accent:#60a5fa;--accent2:#a78bfa;--ok:#34d399;--bad:#f87171}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at top,#111a2e 0%,var(--bg) 60%);color:var(--ink);font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;min-height:100vh;display:flex;flex-direction:column}
header{padding:12px 18px;border-bottom:1px solid var(--line);background:rgba(13,19,31,0.92);backdrop-filter:blur(10px);display:flex;gap:9px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:20}
h1{font-size:15px;margin:0;font-weight:700;background:linear-gradient(120deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
select,input,button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 10px;font:inherit}
button{cursor:pointer;transition:border-color .15s,background .15s}button:hover:not(:disabled){border-color:var(--accent);background:#1b263c}button:disabled{opacity:.5;cursor:not-allowed}
main{max-width:980px;width:100%;margin:0 auto;padding:18px;flex:1;display:flex;flex-direction:column}
#log{flex:1}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.chip{font-size:11.5px;padding:4px 10px;border-radius:999px;background:var(--panel);border:1px solid var(--line);color:var(--dim);cursor:pointer}
.chip:hover{color:var(--ink);border-color:var(--accent)}
.msg{padding:12px 14px;border-radius:10px;margin:10px 0;white-space:pre-wrap;position:relative}
.user{background:#18263e;border:1px solid #2b4269;margin-left:42px}
.bot{background:var(--card);border:1px solid var(--line);margin-right:24px}
.rewrite{color:#9aa8bf;font-style:italic}
.meta{margin-top:8px;font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.copy-btn{font-size:11px;padding:2px 7px;border-radius:5px;margin-left:auto}
.recall,.check{margin-top:6px;padding:3px 9px;border-radius:5px;font-size:11.5px;font-weight:600;display:inline-block;margin-right:6px}
.ok{background:#123d2a;color:#86efac;border:1px solid #1e6243}.bad{background:#4f1919;color:#fca5a5;border:1px solid #7f2929}.unknown{background:#232d42;color:#9eb0cc}
.recalled{background:#4f1919;color:#fca5a5}.partial{background:#4a3c16;color:#fde68a}.novel{background:#123d2a;color:#86efac}
.tele{margin-top:6px;padding:6px 10px;border-radius:6px;background:#0e1524;border:1px solid var(--line);font-size:11.5px;color:#b8c7e0;font-variant-numeric:tabular-nums}
details.proof{margin-top:6px;font-size:11.5px;color:var(--dim)}details.proof summary{cursor:pointer;color:var(--accent)}details.proof pre{margin:6px 0 0;padding:8px;border-radius:6px;background:#0b101b;border:1px solid var(--line);overflow:auto;color:#cfe0fa;font-size:11px}
form{display:flex;gap:8px;margin-top:14px;align-items:flex-end}form input,form textarea{flex:1}
textarea{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 10px;font:inherit;resize:vertical;min-height:36px}
#send{background:linear-gradient(120deg,var(--accent),var(--accent2));color:#07101f;font-weight:700;border:none}
#stop{display:none;border-color:#7f2929;color:#fca5a5}
.note{color:var(--dim);font-size:12px;margin:7px 0 0}
.cursor{opacity:.65}
.tog{font-size:12px;color:var(--dim);display:flex;align-items:center;gap:5px;cursor:pointer}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:10px 0}
.col{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.col h3{margin:0 0 6px;font-size:12px;color:var(--accent);font-weight:600}
.agree{margin:4px 0 0;font-size:12px;font-weight:600}.agree.yes{color:var(--ok)}.agree.no{color:var(--bad)}
.bench{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.bench td{padding:2px 10px 2px 0;font-size:12px;font-variant-numeric:tabular-nums}
</style>
<header>
  <h1>Supermix Chat</h1>
  <select id="model"></select>
  <select id="mode">
    <option value="speculative">speculative (MTP)</option>
    <option value="greedy">greedy</option>
    <option value="sample">sample</option>
  </select>
  <input id="ntok" type="number" value="__DEFAULT_MAX_NEW_TOKENS__" min="1" max="256" style="width:72px" title="max tokens">
  <input id="temp" type="number" value="0.7" min="0.05" max="2.0" step="0.05" style="width:66px" title="temperature (sample mode)">
  <label class="tog"><input type="checkbox" id="scf" checked> SCF verify</label>
  <label class="tog"><input type="checkbox" id="fabric" checked> Cognitive Fabric</label>
  <label class="tog"><input type="checkbox" id="cmp"> compare all</label>
  <button id="bench" type="button">Benchmark</button>
  <button id="scfstatus" type="button">SCF Status</button>
  <button id="exportmd" type="button">Export</button>
  <button id="reset" type="button">Clear</button>
  <span class="note" id="status"></span>
</header>
<main>
  <div class="chips">
    <button type="button" class="chip" onclick="usePrompt('nums = [10, 12, 4]\nr = nums[1] + nums[2] Final r?')">Code AST index</button>
    <button type="button" class="chip" onclick="usePrompt('x = 0\nwhile x <= 40: x = x + 11 Give the value of x.')">While loop trace</button>
    <button type="button" class="chip" onclick="usePrompt('In how many ways can 2 items be arranged in order from 5?')">Permutations P(5,2)</button>
    <button type="button" class="chip" onclick="usePrompt('What is 20% of 275, then add 46?')">Two-step 20%+46</button>
    <button type="button" class="chip" onclick="usePrompt('How many Li20 neurons does the male CNS contain?')">Connectome Li20</button>
    <button type="button" class="chip" onclick="usePrompt('In the male CNS, how many synapses does type Tm2 make onto type Dm3c?')">Connectome Tm2-&gt;Dm3c</button>
    <button type="button" class="chip" onclick="usePrompt('Find the impulse delivered by 41 N applied for 5 s.')">Physics impulse</button>
    <button type="button" class="chip" onclick="usePrompt('can you help me with tests')">Coding chat</button>
  </div>
  <div id="log"></div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Ask Supermix (Enter sends, Shift+Enter newline — try math, Python AST code, physics, or Male-CNS connectome)…" autocomplete="off"></textarea>
    <button id="stop" type="button">Stop</button>
    <button id="send">Send</button>
  </form>
  <p class="note">Replies stream live from the selected checkpoint with optional <strong>Speculative MTP</strong> decoding, live <strong>CNS / PredErr / Hebbian / Thinking</strong> telemetry, <strong>SCF verify</strong> deterministic proof checks, and full <strong>Cognitive Fabric</strong> routing.</p>
</main>
<script>
const el = id => document.getElementById(id);
const SESSION = 'web-' + ((window.crypto && crypto.randomUUID) ? crypto.randomUUID().slice(0,8) : String(Date.now()));
const STORAGE_KEY = 'supermix.chat.transcript.v2';
let activeAbort = null;
const transcript = [];
function saveTranscript(){ try{ localStorage.setItem(STORAGE_KEY, JSON.stringify(transcript.slice(-40))); }catch(_e){} }
function usePrompt(t){ el('input').value = t; el('input').focus(); }
async function loadModels(){
  const r = await fetch('/api/models'); const d = await r.json();
  el('model').innerHTML = d.models.map(m => {
    const a = m.architecture || {};
    const tag = a.cns_predictive_error ? (String(a.run_name||m.name).includes('v99') ? ' [v99 PredErr]' : ' [v98 PredErr]')
      : a.cns_hebbian_trace ? ' [v97 Hebbian]' : a.cns_core ? ' [CNS]' : '';
    return `<option value="${m.name}">${m.name}${tag}${m.resident?' •':''}</option>`;
  }).join('');
  el('status').textContent = `${d.models.length} models · max ${d.max_resident} resident`;
}
function add(cls, text){
  const n = document.createElement('div'); n.className = 'msg ' + cls; n.textContent = text;
  el('log').appendChild(n); window.scrollTo(0, document.body.scrollHeight); return n;
}
function badge(kind, text){ const b=document.createElement('div'); b.className=kind; b.textContent=text; return b; }
function renderAnnotations(bot, data){
  const m = document.createElement('div'); m.className='meta';
  const parts = [`${data.model}`, `${data.tokens} tokens`, `${data.tokens_per_second.toFixed(1)} tok/s`, `${data.latency_ms.toFixed(0)} ms`];
  if(data.speculative_stats && data.speculative_stats.acceptance_length){
    parts.push(`MTP accept ${data.speculative_stats.acceptance_length}x`);
  }
  const summarySpan = document.createElement('span');
  summarySpan.textContent = parts.join(' · ');
  m.appendChild(summarySpan);
  const cp = document.createElement('button'); cp.type='button'; cp.className='copy-btn'; cp.textContent='Copy';
  cp.onclick = ()=>{ if(navigator.clipboard) navigator.clipboard.writeText(data.reply || ''); cp.textContent='Copied'; setTimeout(()=>cp.textContent='Copy',1200); };
  m.appendChild(cp);
  bot.appendChild(m);
  if(data.hit_token_cap){ bot.appendChild(badge('meta', 'Token limit reached')); }
  if(data.normalised_rule){
    const n = document.createElement('div'); n.className='meta rewrite';
    n.textContent = 'asked as: ' + data.asked_as + ' (' + data.normalised_rule + ')';
    bot.appendChild(n);
  }
  if(data.architecture_telemetry){
    const t = data.architecture_telemetry; const bits = [];
    if(t.cns_core){
      bits.push(`CNS alive=${t.cns_core.alive_nodes}/${t.cns_core.alive_edges||'?'}`);
      if(t.cns_core.hebbian_trace) bits.push(`Hebbian gain=${Number(t.cns_core.hebbian_gain_mean||1).toFixed(3)}`);
      if(t.cns_core.predictive_error) bits.push(`PredErr norm=${Number(t.cns_core.pred_error_norm||0).toFixed(4)}`);
      if(t.cns_core.spectral_radius_est!==undefined && t.cns_core.spectral_radius_est!==null) bits.push(`ρ=${Number(t.cns_core.spectral_radius_est).toFixed(3)}`);
    }
    if(t.thinking){
      bits.push(`Think cycles=${t.thinking.cycles_used} (${t.thinking.exit_reason})`);
      if(t.thinking.cycle_entropies) bits.push(`H=[${t.thinking.cycle_entropies.map(v=>v.toFixed(2)).join(', ')}]`);
    }
    if(bits.length) bot.appendChild(badge('tele', bits.join(' · ')));
  }
  if(data.check !== undefined){
    const c = data.check;
    if(c === null) bot.appendChild(badge('check unknown','CHECKER: open-ended / outside legacy subset'));
    else if(c.correct) bot.appendChild(badge('check ok','LEGACY CHECK OK — ' + c.expected));
    else bot.appendChild(badge('check bad','LEGACY CHECK DIFF — answered ' + (c.predicted===null?'nothing':c.predicted) + ', expected ' + c.expected));
  }
  if(data.scf_check && data.scf_check.verified){
    const sc = data.scf_check;
    const tag = sc.model_agrees === true ? 'SCF VERIFIED MATCH' : (sc.model_agrees === false ? 'SCF VERIFIED OVERRIDE' : 'SCF VERIFIED');
    bot.appendChild(badge(sc.model_agrees === false ? 'check bad' : 'check ok', `${tag}: ${sc.verified_answer} (${sc.verifier_source})`));
  }
  if(data.scf_runtime || (data.scf_check && (data.scf_check.independent_certificate || data.scf_check.proof_capsule))){
    const det = document.createElement('details'); det.className='proof';
    const sum = document.createElement('summary');
    const rt = data.scf_runtime || {};
    sum.textContent = rt.scf_active ? `Cognitive Fabric: family=${rt.selected_family} · domain=${rt.domain} · verified=${rt.verified} (${rt.total_latency_ms} ms)` : 'Verification Proof / Certificate';
    det.appendChild(sum);
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify({scf_runtime: data.scf_runtime || null, certificate: data.scf_check ? data.scf_check.independent_certificate : null, proof_capsule: data.scf_check ? data.scf_check.proof_capsule : null}, null, 2);
    det.appendChild(pre);
    bot.appendChild(det);
  }
  if(data.recall){
    const v = data.recall.verdict;
    const label = (v==='largely_recalled'?'RECALLED':v==='part_recalled'?'PART RECALLED':v==='mostly_novel'?'NOVEL':'TOO SHORT') +
      (data.recall.windows ? ` — ${(data.recall.verbatim_rate*100).toFixed(0)}% verbatim` : '');
    bot.appendChild(badge('recall ' + (v==='largely_recalled'?'recalled':v==='part_recalled'?'partial':'novel'), label));
  }
  transcript.push({role:'bot', model:data.model, reply:data.reply, verified_answer:data.scf_check?data.scf_check.verified_answer:null});
  saveTranscript();
}
el('stop').addEventListener('click', ()=>{ if(activeAbort) activeAbort.abort(); });
async function send(ev){
  ev.preventDefault();
  const text = el('input').value.trim(); if(!text) return;
  el('input').value = ''; el('send').disabled = true; el('stop').style.display='inline-block';
  activeAbort = new AbortController();
  transcript.push({role:'user', text}); saveTranscript();
  add('user', text);
  const bot = add('bot', ''); const cur = document.createElement('span');
  cur.className = 'cursor'; cur.textContent = '▋'; bot.appendChild(cur);
  try{
    const r = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, signal:activeAbort.signal,
      body: JSON.stringify({message:text, model:el('model').value, mode:el('mode').value, temperature:Number(el('temp').value||0.7),
                            scf_verify:!!el('scf').checked, scf_runtime:!!el('fabric').checked, session_id:SESSION, max_new_tokens:Number(el('ntok').value)})});
    if(!r.ok){ const e = await r.json().catch(()=>({error:r.status}));
      bot.textContent = 'error: ' + (e.error || r.status); return; }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    for(;;){
      const {value, done} = await reader.read(); if(done) break;
      buf += dec.decode(value, {stream:true});
      let cut;
      while((cut = buf.indexOf('\n\n')) >= 0){
        const frame = buf.slice(0, cut); buf = buf.slice(cut+2);
        const evName = (frame.match(/^event: (.+)$/m)||[])[1];
        const data = JSON.parse((frame.match(/^data: (.+)$/m)||[])[1] || '{}');
        if(evName === 'token'){ cur.insertAdjacentText('beforebegin', data.text); }
        else if(evName === 'done'){ cur.remove(); renderAnnotations(bot, data); loadModels(); }
      }
    }
  }catch(e){ cur.remove(); if(e.name==='AbortError') bot.appendChild(badge('meta','Generation cancelled')); else bot.textContent='error: '+e.message; }
  finally{ activeAbort=null; el('stop').style.display='none'; el('send').disabled=false; el('input').focus(); }
}
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
      body: JSON.stringify({message:text, max_new_tokens:Number(el('ntok').value), mode:el('mode').value, temperature:Number(el('temp').value||0.7), scf_verify:!!el('scf').checked, scf_runtime:!!el('fabric').checked})});
    const d = await r.json(); wait.remove();
    if(!r.ok){ holder.textContent = 'error: ' + (d.error||r.status); return; }
    for(const res of d.results){
      const col = document.createElement('div'); col.className='col';
      const h = document.createElement('h3'); h.textContent = res.model; col.appendChild(h);
      const body = document.createElement('div'); body.textContent = res.error || res.reply || '(empty)';
      col.appendChild(body);
      const v = verdictBadge(res.check); if(v) col.appendChild(v);
      if(res.scf_check && res.scf_check.verified){
        col.appendChild(badge(res.scf_check.model_agrees===false?'check bad':'check ok', 'SCF: ' + res.scf_check.verified_answer));
      }
      if(res.architecture_telemetry && res.architecture_telemetry.cns_core){
        const c=res.architecture_telemetry.cns_core;
        col.appendChild(badge('tele', `CNS=${c.alive_nodes}` + (c.hebbian_trace?` Hebb=${Number(c.hebbian_gain_mean||1).toFixed(3)}`:'') + (c.predictive_error?` PE=${Number(c.pred_error_norm||0).toFixed(4)}`:'')));
      }
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
  box.textContent = 'benchmarking ' + el('model').value + ' on all 41 problem families…';
  el('log').appendChild(box); window.scrollTo(0,document.body.scrollHeight);
  try{
    const r = await fetch('/api/benchmark', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({model:el('model').value, problems:41, seed:82, scf_verify:!!el('scf').checked, max_new_tokens:Number(el('ntok').value)})});
    const d = await r.json();
    if(!r.ok){ box.textContent='error: '+(d.error||r.status); return; }
    const scfTxt = d.scf_hybrid_accuracy !== undefined ? ` | SCF Hybrid: ${d.scf_hybrid_correct}/${d.problems} = ${(d.scf_hybrid_accuracy*100).toFixed(1)}%` : '';
    box.textContent = `${d.model} — Neural: ${d.correct}/${d.problems} = ${(d.accuracy*100).toFixed(1)}%${scfTxt}`;
    box.appendChild(badge('meta', d.settings.max_new_tokens+' token cap; '+d.truncated_replies+' truncated replies'));
    const t=document.createElement('table');
    for(const [k,v] of Object.entries(d.by_task)){
      const row=t.insertRow(); row.insertCell().textContent=k;
      row.insertCell().textContent=v.correct+'/'+v.n;
      row.insertCell().textContent=(v.accuracy*100).toFixed(0)+'%';
      if(v.scf_correct!==undefined) row.insertCell().textContent='SCF '+v.scf_correct+'/'+v.n;
    }
    box.appendChild(t);
    const n=document.createElement('div'); n.className='note'; n.textContent=d.note; box.appendChild(n);
  }catch(e){ box.textContent='error: '+e.message; }
  finally{ el('bench').disabled=false; }
});
el('scfstatus').addEventListener('click', async ()=>{
  const box = document.createElement('div'); box.className='bench'; box.textContent='loading Cognitive Fabric status…';
  el('log').appendChild(box); window.scrollTo(0,document.body.scrollHeight);
  try{
    const r = await fetch('/api/scf_status'); const d = await r.json();
    box.textContent = `Supermix ${d.name} (${d.version}) · Atlas ready=${d.connectome_atlas_ready} · Discovered talk checkpoints=${(d.talk_checkpoints||[]).length}`;
    const pre = document.createElement('pre'); pre.style.fontSize='11px'; pre.style.overflow='auto';
    pre.textContent = JSON.stringify({families: d.families, server_models: d.server_models}, null, 2);
    box.appendChild(pre);
  }catch(e){ box.textContent='error: '+e.message; }
});
el('exportmd').addEventListener('click', ()=>{
  const md = transcript.map(x => x.role==='user' ? `### User\n${x.text}\n` : `### ${x.model}\n${x.reply}\n` + (x.verified_answer ? `> SCF verified: ${x.verified_answer}\n` : '')).join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([md || '# Supermix Chat Session\n'], {type:'text/markdown'}));
  a.download = 'supermix-chat-transcript.md'; a.click();
});
el('input').addEventListener('keydown', e => {
  if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); el('form').requestSubmit(); }
});
el('form').addEventListener('submit', e => el('cmp').checked ? compareAll(e) : send(e));
el('reset').addEventListener('click', async ()=>{ el('log').innerHTML=''; transcript.length=0; try{localStorage.removeItem(STORAGE_KEY);}catch(_e){}
  await fetch('/api/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({session_id:SESSION})}); });
loadModels();
</script>
"""

def discover_default_models(root: Optional[Path] = None) -> Dict[str, str]:
    """Auto-discover local MiMoMix talk checkpoints (v96/v97/v93/v89) for zero-arg startup."""
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    candidates = (
        ("v96_physio_neurogenesis", "output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt"),
        ("v98_scf_predictive", "output/v98_scf_predictive/v98_scf_predictive.selected.pt"),
        ("v99_scf_evolution", "output/v99_scf_evolution/v99_scf_evolution.selected.pt"),
        ("v97_scf_connectome", "output/v97_scf_connectome/v97_scf_connectome.selected.pt"),
        ("v93_neurogenesis", "output/v93_neurogenesis/v93_neurogenesis.selected.pt"),
        ("v91_cns_connectome", "output/v91_cns_connectome/v91_cns_connectome.selected.pt"),
        ("v89_corpus", "output/v89_corpus/v89_corpus.pt"),
        ("v80_omni", "output/v80_omni/v80_omni.pt"),
        ("v74_broad", "output/v74_broad/v74_broad.pt"),
        ("v57_talk_v2", "output/hf_v57_publish/v57_talk_v2.pt"),
    )
    return {name: str(base / rel) for name, rel in candidates if (base / rel).is_file()}


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
                        help="a checkpoint to serve; repeatable (auto-discovers local v96/v97 when omitted)")
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

    spec = parse_model_spec(args.model) if args.model else discover_default_models()
    if not spec:
        raise SystemExit("no --model given and no local checkpoints found")
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
