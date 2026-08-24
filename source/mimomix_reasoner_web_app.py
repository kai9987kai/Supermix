"""Web interface for a trained v56 latent state-machine reasoner.

What this serves that the v53 API could not: **a real checkpoint**.
`mimomix_api.py` builds its backends in-process from a seed, so `--serve` can
only ever return noise, and the v53 line has no checkpoint I/O at all. This app
loads a `supermix-v56-reasoner-checkpoint-v1` file, reports where it came from,
and refuses to start without one.

What it shows that a chat box cannot: **the reasoning itself**. The model's state
is a distribution over latent states and each operation is a row-stochastic
matrix, so the interface can display the state after every step and the learned
operator for each operation. That is the model's actual computation, read off
`ReasonerOutput.state_trace` and `ReasonerOutput.operator_log_probs` -- not a
rendering of what the architecture is supposed to do.

Trust boundary: only typed request fields steer this service. There is no prompt,
no tool call, and no free text that reaches the model -- a request is a start
digit and four (operation, operand) pairs, each range-checked. The service binds
127.0.0.1; remote exposure needs its own authentication and network controls.

Usage::

    python source/mimomix_reasoner_web_app.py --checkpoint output/v56_curriculum/v56_curriculum_160k.pt
    python source/mimomix_reasoner_web_app.py --checkpoint ... --port 8156 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import reasoner_chat as chat  # noqa: E402
import reasoner_curriculum as curriculum  # noqa: E402
from benchmark_cognitive_leap_ultra_v51 import make_chained_task  # noqa: E402
from mimomix_reasoner import load_reasoner  # noqa: E402

OP_NAMES = ("add", "multiply", "subtract")
MAX_EVAL_SIZE = 5000
#: The bars from the repo's recorded artifacts, shown next to every live number
#: so a reader never has to take the comparison on trust.
BASELINE_REFERENCE = {
    "v51_canonical_eval_default": 0.1710,
    "v51_canonical_best_any_setting": 0.1720,
    "v51_highest_recorded_any_cohort": 0.199219,
    "majority_class_floor": 0.1430,
    "bayes_all_operations_without_start": 0.4105,
}


class ReasonerService:
    """Thread-safe wrapper around one loaded checkpoint."""

    def __init__(self, checkpoint_path: Path, receipt_path: Optional[Path] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.model, payload = load_reasoner(self.checkpoint_path)
        self.payload_extra: Dict[str, Any] = dict(payload.get("extra") or {})
        self._lock = threading.Lock()
        self._last_telemetry: Dict[str, Any] = {}
        #: last answer per session, so "then times 3" can continue. This is the
        #: only state a message can leave behind, and it is a single integer.
        self._sessions: Dict[str, int] = {}
        self.receipt: Dict[str, Any] = {}

        candidate = receipt_path or (self.checkpoint_path.parent / "benchmark_results.json")
        if Path(candidate).is_file():
            try:
                self.receipt = json.loads(Path(candidate).read_text(encoding="utf-8"))
                self.receipt_path: Optional[str] = str(candidate)
            except (json.JSONDecodeError, OSError):
                self.receipt_path = None
        else:
            self.receipt_path = None

    # -- status ------------------------------------------------------------

    def decode_trace(self, trace: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """The answer the class head would give from each state along the trace.

        There is deliberately **no per-state decode map** here. The class head is
        a linear map over the whole log-probability vector, not a per-state
        lookup, so "latent state 6 means answer 4" is not a well-defined
        statement -- it depends on where the rest of the mass sits. Probing the
        head with an artificially confident one-hot gives an answer that looks
        authoritative and disagrees with what the model actually does.

        So this reports the only faithful thing: run the real head on the real
        state at each step and say what it would answer from there. That is
        directly comparable to the true running value, and it is the model's own
        computation rather than an interpretation of it.
        """

        rows: List[Dict[str, Any]] = []
        with torch.no_grad():
            for state in trace:
                probabilities = self.model.class_head(state).softmax(dim=-1)[0]
                answer = int(probabilities.argmax())
                rows.append(
                    {
                        "answer": answer,
                        "probability": round(float(probabilities[answer]), 6),
                    }
                )
        return rows

    def status(self) -> Dict[str, Any]:
        config = self.model.config
        report = self.model.parameter_report()
        receipt = self.receipt
        recorded = receipt.get("eval_default") or {}
        return {
            "model": "v56_latent_state_reasoner",
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_extra": self.payload_extra,
            "receipt_path": self.receipt_path,
            "parameters": report,
            "config": {
                "n_states": config.n_states,
                "n_classes": config.n_classes,
                "slot_layout": config.slot_layout,
                "share_block_encoder": config.share_block_encoder,
                "operator_context": config.operator_context,
                "hidden_size": config.hidden_size,
                "n_layers": config.n_layers,
                "use_moe": config.use_moe,
                "n_routed_experts": config.n_routed_experts,
                "moe_top_k": config.moe_top_k,
                "thinking_cycles": config.thinking_cycles,
                "thinking_max_cycles": config.thinking_max_cycles,
                "attention_layout": list(self.model.layout),
                "sliding_window": config.sliding_window,
            },
            "recorded_evaluation": {
                "protocol": (receipt.get("training_data") or {}).get("protocol"),
                "accuracy": recorded.get("accuracy"),
                "nll": recorded.get("nll"),
                "ece": recorded.get("ece"),
                "brier": recorded.get("brier"),
                "test_size": receipt.get("test_size"),
                "test_seed": receipt.get("test_seed"),
                "train_seconds": receipt.get("train_seconds"),
            },
            "reference": BASELINE_REFERENCE,
        }

    # -- single problem ----------------------------------------------------

    def solve(self, request: Dict[str, Any]) -> Dict[str, Any]:
        start = _require_int(request, "start", 0, curriculum.N_CLASSES - 1)
        raw_ops = request.get("operations")
        if not isinstance(raw_ops, list) or len(raw_ops) != curriculum.N_OPS:
            raise ValueError(f"operations must be a list of exactly {curriculum.N_OPS} items")
        op_types: List[int] = []
        operands: List[int] = []
        for index, entry in enumerate(raw_ops):
            if not isinstance(entry, dict):
                raise ValueError(f"operations[{index}] must be an object")
            op_types.append(_require_int(entry, "op", 0, 2, f"operations[{index}].op"))
            operands.append(
                _require_int(entry, "operand", 1, 9, f"operations[{index}].operand")
            )

        noise = float(request.get("noise", 0.01))
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be between 0.0 and 1.0")
        cycles = request.get("thinking_cycles")
        if cycles is not None:
            cycles = _clamp_int(cycles, 1, self.model.config.thinking_max_cycles)
        adaptive = bool(request.get("adaptive", False))

        start_tensor = torch.tensor([start])
        ops_tensor = torch.tensor([op_types])
        operands_tensor = torch.tensor([operands])
        generator = torch.Generator().manual_seed(int(request.get("seed", 0)))
        x, truth = curriculum.encode_chain(
            start_tensor, ops_tensor, operands_tensor, generator=generator, noise=noise
        )

        started = time.perf_counter()
        with self._lock, torch.no_grad():
            self.model.eval()
            out = self.model(x, thinking_cycles=cycles, adaptive_thinking=adaptive)
            self._last_telemetry = dict(out.telemetry)
            self._last_telemetry["forward_examples"] = 1
        latency_ms = (time.perf_counter() - started) * 1000.0

        probabilities = out.logits.softmax(dim=-1)[0]
        prediction = int(probabilities.argmax())
        trace = [row[0].exp().tolist() for row in out.state_trace]
        operators = [op[0].exp().tolist() for op in out.operator_log_probs]

        # The intermediate ground-truth states are shown for inspection only.
        # They are never supplied to the model and never used as a training
        # signal -- the model sees the input vector and nothing else.
        truth_states = [start]
        accumulator = start
        for op, operand in zip(op_types, operands):
            if op == 0:
                accumulator = (accumulator + operand) % curriculum.N_CLASSES
            elif op == 1:
                accumulator = (accumulator * operand) % curriculum.N_CLASSES
            else:
                accumulator = (accumulator - operand) % curriculum.N_CLASSES
            truth_states.append(accumulator)

        thinking = out.telemetry.get("thinking") or {}
        return {
            "problem": {
                "start": start,
                "operations": [
                    {"op": op, "op_name": OP_NAMES[op], "operand": operand}
                    for op, operand in zip(op_types, operands)
                ],
                "noise": noise,
                "expression": _expression(start, op_types, operands),
            },
            "prediction": prediction,
            "true_answer": int(truth[0]),
            "correct": prediction == int(truth[0]),
            "confidence": round(float(probabilities[prediction]), 6),
            "class_probabilities": [round(float(p), 6) for p in probabilities],
            "state_trace": [[round(float(p), 6) for p in row] for row in trace],
            "decoded_trace": self.decode_trace(out.state_trace),
            "truth_state_trace": truth_states,
            "operators": [[[round(float(p), 6) for p in row] for row in op] for op in operators],
            "verifier": {
                "quality_probability": thinking.get("quality_probability"),
                "continue_probability": thinking.get("continue_probability"),
                "temperature": thinking.get("temperature"),
            },
            "thinking": {
                "cycles_used": thinking.get("cycles_used"),
                "exit_reason": thinking.get("exit_reason"),
                "adaptive": adaptive,
                "requested_cycles": cycles,
            },
            "latency_ms": round(latency_ms, 3),
            "telemetry": out.telemetry,
        }

    # -- live evaluation ---------------------------------------------------

    def evaluate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Score a fresh held-out cohort from the untouched v51 generator."""

        size = _clamp_int(request.get("size", 1000), 1, MAX_EVAL_SIZE)
        seed = _clamp_int(request.get("seed", 52), 0, 10_000_000)
        cycles = request.get("thinking_cycles")
        if cycles is not None:
            cycles = _clamp_int(cycles, 1, self.model.config.thinking_max_cycles)
        adaptive = bool(request.get("adaptive", False))

        x, y = make_chained_task(size, seed=seed)
        x = x.squeeze(1)
        started = time.perf_counter()
        chunks: List[torch.Tensor] = []
        with self._lock, torch.no_grad():
            self.model.eval()
            for index in range(0, size, 1000):
                out = self.model(
                    x[index : index + 1000], thinking_cycles=cycles, adaptive_thinking=adaptive
                )
                chunks.append(out.logits)
                self._last_telemetry = dict(out.telemetry)
                self._last_telemetry["forward_examples"] = int(
                    x[index : index + 1000].shape[0]
                )
        latency_ms = (time.perf_counter() - started) * 1000.0

        logits = torch.cat(chunks, dim=0)
        probabilities = logits.softmax(dim=-1)
        prediction = logits.argmax(dim=-1)
        correct = (prediction == y).float()
        confidence, _ = probabilities.max(dim=-1)
        counts = torch.bincount(y, minlength=self.model.config.n_classes)
        one_hot = F.one_hot(y, num_classes=int(logits.shape[-1])).to(probabilities.dtype)

        order = torch.argsort(confidence, descending=True)
        ranked = correct[order]
        half = max(1, ranked.shape[0] // 2)

        return {
            "size": size,
            "seed": seed,
            "is_canonical_test_set": bool(seed == 52 and size == 1000),
            "accuracy": round(float(correct.mean()), 6),
            "nll": round(float(F.cross_entropy(logits, y)), 6),
            "brier": round(float(((probabilities - one_hot) ** 2).sum(dim=-1).mean()), 6),
            "mean_confidence": round(float(confidence.mean()), 6),
            "selective_accuracy_at_50pct_coverage": round(float(ranked[:half].mean()), 6),
            "majority_class_floor": round(float(counts.max()) / float(size), 6),
            "uniform_cross_entropy": round(math.log(float(self.model.config.n_classes)), 6),
            "thinking_cycles": cycles,
            "adaptive": adaptive,
            "latency_ms": round(latency_ms, 2),
            "reference": BASELINE_REFERENCE,
        }

    # -- chat --------------------------------------------------------------

    def chat(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Answer a natural-language arithmetic question.

        The parser reads the text; the model never sees it. Chains longer than
        four operations are executed as repeated model calls, each one starting
        from the model's own previous answer -- so its errors compound, and
        `model_calls` says how many chances it had to make one.
        """

        message = request.get("message")
        if not isinstance(message, str):
            raise ValueError("message must be a string")
        if len(message) > 2000:
            raise ValueError("message is too long")
        session_id = request.get("session_id")
        session_id = session_id if isinstance(session_id, str) and session_id else "default"

        previous = self._sessions.get(session_id)
        try:
            problem = chat.parse_problem(message, previous_answer=previous)
        except chat.ParseError as error:
            return {
                "understood": False,
                "reply": f"{error}\n\n{chat.describe_capabilities()}",
                "session_id": session_id,
                "model_calls": 0,
            }

        chunks = chat.chunk_operations(problem.operations)
        current = problem.start
        steps: List[Dict[str, Any]] = []
        truth = problem.start
        for index, group in enumerate(chunks):
            solved = self.solve(
                {
                    "start": int(current),
                    "operations": [{"op": op, "operand": operand} for op, operand in group],
                    "noise": float(request.get("noise", 0.01)),
                    "seed": 0,
                }
            )
            steps.append(
                {
                    "call": index + 1,
                    "start": int(current),
                    "operations": solved["problem"]["operations"],
                    "prediction": solved["prediction"],
                    "confidence": solved["confidence"],
                    "verifier_quality": solved["verifier"]["quality_probability"],
                    "state_trace": solved["state_trace"],
                    "decoded_trace": solved["decoded_trace"],
                    "truth_state_trace": solved["truth_state_trace"],
                    "true_answer_this_call": solved["true_answer"],
                }
            )
            current = solved["prediction"]
            truth = solved["true_answer"] if index == 0 else truth
        # Ground truth for the whole chain, computed independently of the model.
        truth = problem.start
        for op, operand in problem.operations:
            if op == 0:
                truth = (truth + operand) % 10
            elif op == 1:
                truth = (truth * operand) % 10
            else:
                truth = (truth - operand) % 10

        answer = int(current)
        correct = answer == truth
        confidence = min(step["confidence"] for step in steps)
        self._sessions[session_id] = answer

        reply = f"{problem.expression()} = **{answer}**"
        if len(chunks) > 1:
            reply += (
                f"\n\nThat took {len(chunks)} model calls: the input has four operator "
                f"slots, so I fed the model's own answer forward between them. Errors "
                f"compound across calls."
            )
        if not correct:
            reply += f"\n\nThat is wrong; the true answer is {truth}."
        return {
            "understood": True,
            "reply": reply,
            "session_id": session_id,
            "problem": problem.to_dict(),
            "answer": answer,
            "true_answer": truth,
            "correct": correct,
            "confidence": round(float(confidence), 6),
            "model_calls": len(chunks),
            "steps": steps,
        }

    def telemetry(self) -> Dict[str, Any]:
        return dict(self._last_telemetry)


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def _require_int(source: Dict[str, Any], key: str, low: int, high: int, label: str = "") -> int:
    if key not in source:
        raise ValueError(f"{label or key} is required")
    value = source[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label or key} must be an integer")
    number = int(value)
    if number != value or not low <= number <= high:
        raise ValueError(f"{label or key} must be an integer in {low}..{high}")
    return number


def _clamp_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected an integer, got {value!r}") from error
    return max(low, min(high, number))


def _expression(start: int, op_types: Sequence[int], operands: Sequence[int]) -> str:
    symbols = {0: "+", 1: "*", 2: "-"}
    text = str(start)
    for op, operand in zip(op_types, operands):
        text = f"({text} {symbols[int(op)]} {operand})"
    return f"{text} mod 10"


# ---------------------------------------------------------------------------
# Flask surface
# ---------------------------------------------------------------------------


def build_app(service: ReasonerService):
    """Build the Flask app around a loaded service.

    Imported lazily so `ReasonerService` and its tests never require Flask.
    """

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

    @app.get("/lab")
    def lab():
        """Serve the v53 observatory same-origin so its live panel can measure.

        Opened as a `file://` page the observatory has no server to read, so its
        live panel correctly reports that it is offline. Served here it reads
        this checkpoint's `/api/status` and `/api/telemetry`.
        """

        page = SOURCE_DIR.parent / "web_static" / "mimomix_lab.html"
        if not page.is_file():
            return no_store({"error": "web_static/mimomix_lab.html is not present"}, 404)
        response = app.response_class(
            page.read_text(encoding="utf-8"), mimetype="text/html"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/status")
    def status():
        return no_store(service.status())

    @app.post("/api/solve")
    def solve():
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return no_store({"error": "request body must be a JSON object"}, 400)
        try:
            return no_store(service.solve(payload))
        except (ValueError, TypeError) as error:
            return no_store({"error": str(error)}, 400)

    @app.post("/api/evaluate")
    def evaluate():
        payload = flask_request.get_json(silent=True)
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return no_store({"error": "request body must be a JSON object"}, 400)
        try:
            return no_store(service.evaluate(payload))
        except (ValueError, TypeError) as error:
            return no_store({"error": str(error)}, 400)

    @app.post("/api/chat")
    def chat_route():
        payload = flask_request.get_json(silent=True)
        if not isinstance(payload, dict):
            return no_store({"error": "request body must be a JSON object"}, 400)
        try:
            return no_store(service.chat(payload))
        except (ValueError, TypeError) as error:
            return no_store({"error": str(error)}, 400)

    @app.get("/chat")
    def chat_page() -> str:
        return CHAT_TEMPLATE

    @app.get("/api/telemetry")
    def telemetry():
        return no_store(service.telemetry())

    @app.get("/health")
    def health():
        return no_store({"status": "ok", "checkpoint": str(service.checkpoint_path)})

    app.config["REASONER_SERVICE"] = service
    return app


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supermix v56 &mdash; Latent State Reasoner</title>
<style>
:root{
  --bg:#0b0f14; --panel:#121821; --panel-2:#0e141c; --line:#1f2937;
  --ink:#e5e7eb; --dim:#9ca3af; --ok:#34d399; --no:#f87171; --accent:#60a5fa; --warm:#fbbf24;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#f6f7f9; --panel:#ffffff; --panel-2:#f0f2f5; --line:#dfe3e8;
         --ink:#111827; --dim:#5b6472; --ok:#047857; --no:#b91c1c; --accent:#1d4ed8; --warm:#a16207; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:18px 22px;border-bottom:1px solid var(--line);background:var(--panel-2)}
h1{margin:0;font-size:17px;letter-spacing:.2px}
.sub{color:var(--dim);font-size:12px;margin-top:4px}
main{padding:18px 22px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));max-width:1500px}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
section h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim)}
.wide{grid-column:1/-1}
label{display:block;font-size:11px;color:var(--dim);margin-bottom:3px}
select,input,button{font:inherit;color:var(--ink);background:var(--panel-2);
  border:1px solid var(--line);border-radius:6px;padding:6px 8px}
button{cursor:pointer;background:var(--accent);color:#fff;border-color:transparent;font-weight:600}
button:disabled{opacity:.55;cursor:default}
button.ghost{background:var(--panel-2);color:var(--ink);border-color:var(--line);font-weight:500}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:flex-end}
.stat{display:flex;justify-content:space-between;gap:12px;padding:3px 0;border-bottom:1px dotted var(--line);font-variant-numeric:tabular-nums}
.stat:last-child{border-bottom:0}
.stat span:first-child{color:var(--dim)}
.ok{color:var(--ok);font-weight:600}.no{color:var(--no);font-weight:600}
.big{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums}
.expr{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;color:var(--warm);word-break:break-all}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:12px}
th,td{padding:3px 5px;text-align:center;border:1px solid var(--line)}
th{color:var(--dim);font-weight:600}
td.lab,th.lab{text-align:left;color:var(--dim);white-space:nowrap}
.scroll{overflow-x:auto}
.cell{position:relative}
.cell .v{position:relative;z-index:1}
.bar{position:absolute;inset:0;background:var(--accent);opacity:.18;transform-origin:left}
.mark{outline:2px solid var(--warm);outline-offset:-2px}
.note{color:var(--dim);font-size:11px;margin-top:8px;line-height:1.55}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;background:var(--panel-2);
  border:1px solid var(--line);font-size:11px;color:var(--dim)}
footer{padding:14px 22px 28px;color:var(--dim);font-size:11px;max-width:1100px;line-height:1.6}
</style>
</head>
<body>
<header>
  <h1>Supermix v56 &mdash; Latent State Reasoner</h1>
  <div class="sub" id="subtitle">loading checkpoint&hellip;</div>
</header>

<main>
  <section>
    <h2>Problem</h2>
    <div class="row" id="builder"></div>
    <div class="row" style="margin-top:10px">
      <div><label for="cycles">thinking cycles</label><select id="cycles"></select></div>
      <div><label for="adaptive">halting</label>
        <select id="adaptive"><option value="0">fixed budget</option><option value="1">adaptive (ACT)</option></select>
      </div>
      <div><label for="noise">input noise &sigma;</label>
        <select id="noise"><option value="0.01">0.01 (task default)</option><option value="0">0 (clean)</option></select>
      </div>
      <button id="solve">Solve</button>
      <button id="random" class="ghost">Random</button>
    </div>
    <div class="expr" id="expr" style="margin-top:10px"></div>
    <div class="note">Only these typed fields reach the model. There is no prompt and no free text.</div>
  </section>

  <section>
    <h2>Answer</h2>
    <div class="row" style="align-items:baseline;gap:20px">
      <div><label>predicted</label><div class="big" id="pred">&mdash;</div></div>
      <div><label>true</label><div class="big" id="truth">&mdash;</div></div>
      <div><label>verdict</label><div class="big" id="verdict">&mdash;</div></div>
    </div>
    <div id="answerStats" style="margin-top:10px"></div>
  </section>

  <section class="wide">
    <h2>Latent state after each operation</h2>
    <div class="scroll"><table id="traceTable"></table></div>
    <div class="note">Each row is the model's own state distribution over its
      <span id="nstates">10</span> latent states, read from <code>state_trace</code>.
      <strong>Latent state indices are not answers.</strong> Nothing forces state 3 to mean 3,
      and the class head is a linear map over the whole distribution rather than a per-state
      lookup, so no "state <em>s</em> means answer <em>a</em>" table would be truthful.
      <em>Would answer</em> is the only faithful decoding: the real class head run on that
      row's real state, i.e. what the model would say if it had to answer from there.
      <em>True so far</em> is the generator's running value, shown for inspection only &mdash;
      it is never given to the model and never used as a training signal. The
      <span style="color:var(--warm)">outlined</span> cell is simply the row's peak. The
      <em>start</em> row is not graded: the class head is trained on final states, so reading
      it on the initial state is out of distribution and disagreeing there is expected.</div>
  </section>

  <section class="wide">
    <h2>Learned transition operator</h2>
    <div class="row">
      <div><label for="opSlot">operation slot</label><select id="opSlot"></select></div>
      <div class="pill" id="opLabel">&mdash;</div>
    </div>
    <div class="scroll" style="margin-top:10px"><table id="opTable"></table></div>
    <div class="note">Row <em>s</em> is the model's distribution over the next state given
      current state <em>s</em>. A row that concentrates on one column has learned a function;
      a near-uniform row has not, and row entropy quantifies which. This is shown in latent
      space and is deliberately <em>not</em> checked against the arithmetic rule: the latent
      indices are the model's own labels, so "state 3 should go to state 4" is not a claim
      this page can make. Whether the composition is right is measured by accuracy, below.</div>
  </section>

  <section>
    <h2>Live held-out evaluation</h2>
    <div class="row">
      <div><label for="evalSeed">generator seed</label><input id="evalSeed" type="number" value="52" min="0" style="width:100px"></div>
      <div><label for="evalSize">samples</label><input id="evalSize" type="number" value="1000" min="1" max="5000" style="width:90px"></div>
      <button id="runEval">Evaluate</button>
    </div>
    <div id="evalStats" style="margin-top:10px"></div>
    <div class="note" id="evalNote"></div>
  </section>

  <section>
    <h2>Runtime telemetry</h2>
    <div id="telemetryStats"></div>
    <div class="note">Sparse-MoE expert load and attention-sink mass from the last forward
      pass, measured &mdash; not simulated.</div>
  </section>

  <section class="wide">
    <h2>Checkpoint and recorded evaluation</h2>
    <div id="statusStats"></div>
  </section>
</main>

<footer id="footer"></footer>

<script>
(function(){
  const el = id => document.getElementById(id);
  const fmt = (x, d=4) => (x===null||x===undefined||Number.isNaN(x)) ? '—' : Number(x).toFixed(d);
  const OP_NAMES = ['add','multiply','subtract'];
  const SYMBOLS = ['+','*','-'];
  let STATE = null, NSTATES = 10, NOPS = 4;

  async function api(path, body){
    const options = body === undefined
      ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)};
    const response = await fetch(path, options);
    const data = await response.json().catch(() => ({}));
    if(!response.ok) throw new Error(data && data.error ? data.error : ('HTTP ' + response.status));
    return data;
  }

  function stats(target, rows){
    target.replaceChildren();
    rows.forEach(([key, value, cls]) => {
      const line = document.createElement('div'); line.className = 'stat';
      const k = document.createElement('span'); k.textContent = key;
      const v = document.createElement('span'); v.textContent = value;
      if(cls) v.className = cls;
      line.append(k, v); target.appendChild(line);
    });
  }

  function option(value, text){
    const o = document.createElement('option'); o.value = String(value); o.textContent = text; return o;
  }

  function buildProblemInputs(){
    const host = el('builder'); host.replaceChildren();
    const startWrap = document.createElement('div');
    const startLabel = document.createElement('label'); startLabel.textContent = 'start digit';
    const start = document.createElement('select'); start.id = 'start';
    for(let d=0; d<10; d++) start.appendChild(option(d, d));
    startWrap.append(startLabel, start); host.appendChild(startWrap);

    for(let k=0; k<NOPS; k++){
      const wrap = document.createElement('div');
      const label = document.createElement('label'); label.textContent = 'op ' + (k+1);
      const op = document.createElement('select'); op.id = 'op'+k;
      OP_NAMES.forEach((name, index) => op.appendChild(option(index, name)));
      const operand = document.createElement('select'); operand.id = 'operand'+k;
      for(let v=1; v<=9; v++) operand.appendChild(option(v, v));
      const inner = document.createElement('div'); inner.className = 'row';
      inner.append(op, operand);
      wrap.append(label, inner); host.appendChild(wrap);
    }
    [start, ...Array.from({length:NOPS}, (_, k) => el('op'+k))].forEach(node => {
      if(node) node.onchange = renderExpression;
    });
    for(let k=0;k<NOPS;k++){ const n = el('operand'+k); if(n) n.onchange = renderExpression; }
  }

  function readProblem(){
    const operations = [];
    for(let k=0;k<NOPS;k++){
      operations.push({op: Number(el('op'+k).value), operand: Number(el('operand'+k).value)});
    }
    return {start: Number(el('start').value), operations};
  }

  function renderExpression(){
    const p = readProblem();
    let text = String(p.start);
    p.operations.forEach(o => { text = '(' + text + ' ' + SYMBOLS[o.op] + ' ' + o.operand + ')'; });
    el('expr').textContent = text + ' mod 10';
  }

  function heatCell(value, marked){
    const td = document.createElement('td'); td.className = 'cell' + (marked ? ' mark' : '');
    const bar = document.createElement('div'); bar.className = 'bar';
    bar.style.transform = 'scaleX(' + Math.max(0, Math.min(1, value)) + ')';
    const span = document.createElement('span'); span.className = 'v';
    span.textContent = value >= 0.0005 ? value.toFixed(3) : '·';
    td.append(bar, span); return td;
  }

  function renderTrace(data){
    const table = el('traceTable'); table.replaceChildren();
    const head = document.createElement('tr');
    const corner = document.createElement('th'); corner.className = 'lab'; corner.textContent = 'after';
    head.appendChild(corner);
    for(let s=0; s<NSTATES; s++) head.appendChild(Object.assign(document.createElement('th'), {textContent: s}));
    head.appendChild(Object.assign(document.createElement('th'), {textContent:'would answer'}));
    head.appendChild(Object.assign(document.createElement('th'), {textContent:'true so far'}));
    head.appendChild(Object.assign(document.createElement('th'), {textContent:'entropy'}));
    table.appendChild(head);

    data.state_trace.forEach((row, index) => {
      const tr = document.createElement('tr');
      const name = index === 0 ? 'start' :
        ('op ' + index + ' (' + data.problem.operations[index-1].op_name + ' ' +
         data.problem.operations[index-1].operand + ')');
      const lab = document.createElement('td'); lab.className = 'lab'; lab.textContent = name;
      tr.appendChild(lab);
      const truth = data.truth_state_trace[index];
      const top = row.indexOf(Math.max.apply(null, row));
      row.forEach((value, s) => tr.appendChild(heatCell(value, s === top)));
      const decoded = (data.decoded_trace && data.decoded_trace[index]) || null;
      const arg = document.createElement('td');
      if(decoded){
        arg.textContent = decoded.answer + ' (' + decoded.probability.toFixed(2) + ')';
        // The class head is trained on final states only, so reading it on the
        // initial state is out of distribution and means nothing. Grading it
        // would paint a red mark on every correct answer.
        arg.className = index === 0 ? '' : (decoded.answer === truth ? 'ok' : 'no');
        if(index === 0) arg.title = 'not graded: the head is trained on final states';
      } else { arg.textContent = '—'; }
      tr.appendChild(arg);
      tr.appendChild(Object.assign(document.createElement('td'), {textContent: truth}));
      let entropy = 0; row.forEach(p => { if(p > 0) entropy -= p * Math.log(p); });
      tr.appendChild(Object.assign(document.createElement('td'), {textContent: entropy.toFixed(3)}));
      table.appendChild(tr);
    });
  }

  function renderOperator(){
    if(!STATE) return;
    const slot = Number(el('opSlot').value);
    const matrix = STATE.operators[slot];
    const meta = STATE.problem.operations[slot];
    el('opLabel').textContent = meta.op_name + ' by ' + meta.operand + '  →  s′ = (s ' +
      SYMBOLS[meta.op] + ' ' + meta.operand + ') mod 10';
    const table = el('opTable'); table.replaceChildren();
    const head = document.createElement('tr');
    const corner = document.createElement('th'); corner.className = 'lab'; corner.textContent = 's \\ s′';
    head.appendChild(corner);
    for(let s=0;s<NSTATES;s++) head.appendChild(Object.assign(document.createElement('th'), {textContent:s}));
    head.appendChild(Object.assign(document.createElement('th'), {textContent:'peak'}));
    head.appendChild(Object.assign(document.createElement('th'), {textContent:'row entropy'}));
    table.appendChild(head);
    matrix.forEach((row, s) => {
      const tr = document.createElement('tr');
      const lab = document.createElement('td'); lab.className='lab'; lab.textContent = s;
      tr.appendChild(lab);
      const peak = Math.max.apply(null, row);
      const top = row.indexOf(peak);
      row.forEach((value, next) => tr.appendChild(heatCell(value, next === top)));
      const peakCell = document.createElement('td');
      peakCell.textContent = peak.toFixed(3);
      peakCell.className = peak > 0.9 ? 'ok' : (peak < 0.5 ? 'no' : '');
      tr.appendChild(peakCell);
      let entropy = 0; row.forEach(p => { if(p > 0) entropy -= p * Math.log(p); });
      tr.appendChild(Object.assign(document.createElement('td'), {textContent: entropy.toFixed(3)}));
      table.appendChild(tr);
    });
  }

  function renderAnswer(data){
    el('pred').textContent = data.prediction;
    el('truth').textContent = data.true_answer;
    const verdict = el('verdict');
    verdict.textContent = data.correct ? 'correct' : 'wrong';
    verdict.className = 'big ' + (data.correct ? 'ok' : 'no');
    el('expr').textContent = data.problem.expression;
    stats(el('answerStats'), [
      ['confidence in answer', fmt(data.confidence)],
      ['verifier p(correct)', fmt(data.verifier.quality_probability)],
      ['verifier p(continue)', fmt(data.verifier.continue_probability)],
      ['verifier temperature', fmt(data.verifier.temperature, 3)],
      ['thinking cycles used', fmt(data.thinking.cycles_used, 1)],
      ['exit reason', data.thinking.exit_reason || '—'],
      ['latency', fmt(data.latency_ms, 2) + ' ms'],
    ]);
  }

  function renderTelemetry(t){
    if(!t || !t.parameters){ stats(el('telemetryStats'), [['telemetry','not yet measured']]); return; }
    const rows = [
      ['attention layout', (t.attention_layout||[]).map(k => k==='global'?'G':'L').join('')],
      ['sliding window', t.sliding_window],
      ['sequence length (slots)', t.sequence_length],
      ['mean attention-sink mass', fmt(t.mean_sink_mass, 5)],
      ['operator normalised entropy', fmt(t.operator_normalised_entropy, 4)],
      ['operator mean peak probability', fmt(t.operator_mean_max_probability, 4)],
      ['final state entropy', fmt(t.state_entropy, 4)],
    ];
    (t.router_entropy||[]).forEach((value, index) => {
      rows.push(['MoE layer ' + index + ' router entropy', fmt(value, 4)]);
    });
    const examples = t.forward_examples || 0;
    // Idle experts on a single-example forward are arithmetic, not starvation:
    // top-k over a handful of slot tokens cannot reach every expert. Only a
    // large batch says anything about balance.
    const meaningful = examples >= 64;
    if(examples) rows.push(['last forward measured over', examples + ' example' + (examples===1?'':'s')]);
    (t.expert_load||[]).forEach((load, index) => {
      const total = load.reduce((a,b)=>a+b, 0) || 1;
      const idle = load.filter(v => v === 0).length;
      rows.push(['MoE layer ' + index + ' expert share',
        load.map(v => (v/total*100).toFixed(0)+'%').join(' ') +
        (idle ? ('  (' + idle + (meaningful ? ' starved' : ' idle this forward') + ')') : ''),
        meaningful ? (idle ? 'no' : 'ok') : '']);
    });
    if(!meaningful && (t.expert_load||[]).length){
      rows.push(['expert balance', 'run the held-out evaluation to measure it']);
    }
    stats(el('telemetryStats'), rows);
  }

  function renderStatus(s){
    NSTATES = s.config.n_states; el('nstates').textContent = NSTATES;
    STATE_CLASS = s.state_class_map || null;
    el('subtitle').textContent =
      s.parameters.total.toLocaleString() + ' parameters (' +
      s.parameters.active_per_input.toLocaleString() + ' active/input) · ' +
      s.config.n_layers + ' layers ' + (s.config.attention_layout||[]).map(k=>k==='global'?'G':'L').join('') +
      ' · ' + (s.config.use_moe ? s.config.n_routed_experts + ' experts top-' + s.config.moe_top_k : 'dense') +
      ' · ' + s.config.n_states + ' latent states';
    const recorded = s.recorded_evaluation || {};
    const rows = [
      ['checkpoint', s.checkpoint],
      ['training protocol', recorded.protocol || '—'],
      ['operator context', s.config.operator_context],
      ['shared block encoder', String(s.config.share_block_encoder)],
      ['slot layout', s.config.slot_layout],
      ['recorded held-out accuracy',
        recorded.accuracy === null || recorded.accuracy === undefined ? '—' : fmt(recorded.accuracy),
        recorded.accuracy > s.reference.v51_canonical_best_any_setting ? 'ok' : 'no'],
      ['recorded NLL', fmt(recorded.nll)],
      ['recorded ECE', fmt(recorded.ece)],
      ['recorded Brier', fmt(recorded.brier)],
      ['recorded on', recorded.test_size ? (recorded.test_size + ' samples, seed ' + recorded.test_seed) : '—'],
      ['— previous best (v51, eval_default)', fmt(s.reference.v51_canonical_eval_default)],
      ['— previous best (any setting)', fmt(s.reference.v51_canonical_best_any_setting)],
      ['— highest previously recorded, any cohort', fmt(s.reference.v51_highest_recorded_any_cohort)],
      ['— majority-class floor', fmt(s.reference.majority_class_floor)],
      ['— Bayes limit without the start digit', fmt(s.reference.bayes_all_operations_without_start)],
    ];
    stats(el('statusStats'), rows);

    el('footer').textContent =
      'This page reports measurements from one checkpoint on one synthetic task: a 4-step ' +
      'chained modular-arithmetic problem with 5,314,410 distinct inputs. The state ' +
      'distributions and transition matrices are the model’s own computation, not an ' +
      'illustration of the architecture. Accuracy here is not evidence of language ability, ' +
      'reasoning in general, or transfer to any other task, and the verifier’s p(correct) ' +
      'is calibrated on this task only. Reference bars are the repo’s recorded v51 results ' +
      'on the identical held-out protocol.';
  }

  async function solve(){
    const button = el('solve'); button.disabled = true;
    try{
      const problem = readProblem();
      problem.noise = Number(el('noise').value);
      problem.adaptive = el('adaptive').value === '1';
      const cycles = el('cycles').value;
      problem.thinking_cycles = cycles === 'default' ? null : Number(cycles);
      STATE = await api('/api/solve', problem);
      renderAnswer(STATE); renderTrace(STATE); renderOperator(); renderTelemetry(STATE.telemetry);
    }catch(error){
      stats(el('answerStats'), [['error', error.message, 'no']]);
    }finally{ button.disabled = false; }
  }

  async function runEval(){
    const button = el('runEval'); button.disabled = true;
    stats(el('evalStats'), [['status','measuring…']]);
    try{
      const data = await api('/api/evaluate', {
        seed: Number(el('evalSeed').value),
        size: Number(el('evalSize').value),
        adaptive: el('adaptive').value === '1',
      });
      stats(el('evalStats'), [
        ['accuracy', fmt(data.accuracy), data.accuracy > data.reference.v51_canonical_best_any_setting ? 'ok' : 'no'],
        ['NLL', fmt(data.nll)],
        ['Brier', fmt(data.brier)],
        ['mean confidence', fmt(data.mean_confidence)],
        ['selective accuracy @50% coverage', fmt(data.selective_accuracy_at_50pct_coverage)],
        ['majority-class floor (this cohort)', fmt(data.majority_class_floor)],
        ['previous best on this task', fmt(data.reference.v51_canonical_best_any_setting)],
        ['latency', fmt(data.latency_ms, 1) + ' ms for ' + data.size + ' samples'],
      ]);
      el('evalNote').textContent = data.is_canonical_test_set
        ? 'This is the canonical v51 held-out set (seed 52, 1000 samples) — directly comparable to the recorded 0.1710.'
        : 'Fresh cohort. Note that (seed, size) jointly determine the sample: changing size changes the whole cohort, not just its length.';
      renderTelemetry(await api('/api/telemetry'));
    }catch(error){
      stats(el('evalStats'), [['error', error.message, 'no']]);
    }finally{ button.disabled = false; }
  }

  function randomise(){
    el('start').value = String(Math.floor(Math.random()*10));
    for(let k=0;k<NOPS;k++){
      el('op'+k).value = String(Math.floor(Math.random()*3));
      el('operand'+k).value = String(1 + Math.floor(Math.random()*9));
    }
    renderExpression(); solve();
  }

  async function boot(){
    buildProblemInputs();
    renderExpression();
    let status = null;
    try{
      status = await api('/api/status');
      renderStatus(status);
    }catch(error){
      el('subtitle').textContent = 'status error: ' + error.message; return;
    }
    const cycles = el('cycles');
    cycles.appendChild(option('default', 'default (' + status.config.thinking_cycles + ')'));
    for(let c=1;c<=status.config.thinking_max_cycles;c++) cycles.appendChild(option(c, c));
    const slots = el('opSlot');
    for(let k=0;k<NOPS;k++) slots.appendChild(option(k, 'operation ' + (k+1)));
    slots.onchange = renderOperator;
    el('solve').onclick = solve;
    el('random').onclick = randomise;
    el('runEval').onclick = runEval;
    solve();
  }

  boot();
})();
</script>
</body>
</html>
"""


CHAT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supermix v56 &mdash; Reasoner Chat</title>
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
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel-2)}
h1{margin:0;font-size:16px}
.sub{color:var(--dim);font-size:12px;margin-top:3px}
.banner{margin:12px 20px 0;padding:10px 12px;border:1px solid var(--warm);border-radius:8px;
  background:color-mix(in srgb,var(--warm) 10%,transparent);color:var(--ink);font-size:12.5px}
#log{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:min(760px,92%);padding:10px 13px;border-radius:10px;border:1px solid var(--line);
  white-space:pre-wrap;word-break:break-word}
.me{align-self:flex-end;background:var(--accent);color:#fff;border-color:transparent}
.bot{align-self:flex-start;background:var(--panel)}
.bot.bad{border-color:var(--no)}
.meta{margin-top:8px;font-size:11.5px;color:var(--dim);font-variant-numeric:tabular-nums}
.meta b{color:var(--ink);font-weight:600}
table{border-collapse:collapse;margin-top:8px;font-size:11.5px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:2px 6px;text-align:center}
th{color:var(--dim)}
td.lab{text-align:left;color:var(--dim);white-space:nowrap}
.ok{color:var(--ok);font-weight:600}.no{color:var(--no);font-weight:600}
form{display:flex;gap:8px;padding:12px 20px;border-top:1px solid var(--line);background:var(--panel-2)}
input{flex:1;font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);
  border-radius:8px;padding:9px 12px}
button{font:inherit;font-weight:600;background:var(--accent);color:#fff;border:0;border-radius:8px;
  padding:9px 16px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
.chips{display:flex;gap:6px;flex-wrap:wrap;padding:0 20px 10px}
.chip{font-size:12px;padding:4px 9px;border:1px solid var(--line);border-radius:999px;
  background:var(--panel);color:var(--dim);cursor:pointer}
footer{padding:10px 20px 16px;color:var(--dim);font-size:11px;border-top:1px solid var(--line)}
a{color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1>Supermix v56 &mdash; Reasoner Chat</h1>
  <div class="sub" id="sub">loading&hellip;</div>
</header>

<div class="banner">
  <strong>This model cannot talk.</strong> It has no tokenizer and no language ability.
  The parsing on this page is ordinary deterministic code, not a model; the reasoner only
  ever receives a 128-dimensional vector and returns one digit. Anything conversational you
  see here was written by the parser, not generated.
</div>

<div id="log"></div>

<div class="chips">
  <span class="chip" data-q="7 times 3 plus 8 minus 5 times 6">7 times 3 plus 8 minus 5 times 6</span>
  <span class="chip" data-q="4 + 9 * 7 - 2 * 3">4 + 9 * 7 - 2 * 3</span>
  <span class="chip" data-q="then times 3">then times 3</span>
  <span class="chip" data-q="2 plus 3 times 4 minus 1 times 7 plus 6 times 2">8 operations</span>
  <span class="chip" data-q="hello, who are you?">hello, who are you?</span>
</div>

<form id="form">
  <input id="input" autocomplete="off" placeholder="e.g. 7 times 3 plus 8 minus 5 times 6" />
  <button id="send" type="submit">Send</button>
</form>

<footer id="foot"></footer>

<script>
(function(){
  const el = id => document.getElementById(id);
  const SESSION = 'chat-' + Math.random().toString(36).slice(2);
  const log = el('log');

  function bubble(text, cls){
    const div = document.createElement('div');
    div.className = 'msg ' + cls;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  function traceTable(step){
    const table = document.createElement('table');
    const head = document.createElement('tr');
    ['after','would answer','true'].forEach(t =>
      head.appendChild(Object.assign(document.createElement('th'), {textContent: t})));
    table.appendChild(head);
    step.decoded_trace.forEach((d, i) => {
      const tr = document.createElement('tr');
      const label = i === 0 ? 'start' :
        ('op ' + i + ' (' + step.operations[i-1].op_name + ' ' + step.operations[i-1].operand + ')');
      const lab = document.createElement('td'); lab.className = 'lab'; lab.textContent = label;
      const got = document.createElement('td');
      got.textContent = d.answer + ' (' + d.probability.toFixed(2) + ')';
      // row 0 reads the head on the initial state, which it was never trained
      // on; grading it would mark every correct answer red
      got.className = i === 0 ? '' : (d.answer === step.truth_state_trace[i] ? 'ok' : 'no');
      if(i === 0) got.title = 'not graded: the head is trained on final states';
      const want = document.createElement('td');
      want.textContent = step.truth_state_trace[i];
      tr.append(lab, got, want); table.appendChild(tr);
    });
    return table;
  }

  function render(data){
    const div = bubble(data.reply.replace(/\*\*/g, ''), 'bot' + (data.understood && !data.correct ? ' bad' : ''));
    if(!data.understood) return;
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML =
      'answer <b>' + data.answer + '</b> &middot; true <b>' + data.true_answer + '</b> &middot; ' +
      '<span class="' + (data.correct ? 'ok' : 'no') + '">' + (data.correct ? 'correct' : 'wrong') + '</span>' +
      ' &middot; confidence <b>' + data.confidence.toFixed(3) + '</b>' +
      ' &middot; model calls <b>' + data.model_calls + '</b>';
    div.appendChild(meta);
    data.steps.forEach((step, i) => {
      if(data.steps.length > 1){
        const h = document.createElement('div');
        h.className = 'meta';
        h.textContent = 'call ' + step.call + ' — start ' + step.start + ', answered ' + step.prediction;
        div.appendChild(h);
      }
      div.appendChild(traceTable(step));
    });
    log.scrollTop = log.scrollHeight;
  }

  async function send(text){
    if(!text.trim()) return;
    bubble(text, 'me');
    el('send').disabled = true;
    try{
      const r = await fetch('/api/chat', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({message: text, session_id: SESSION})
      });
      const data = await r.json();
      if(!r.ok){ bubble('error: ' + (data.error || r.status), 'bot bad'); }
      else { render(data); }
    }catch(e){ bubble('error: ' + e.message, 'bot bad'); }
    finally{ el('send').disabled = false; el('input').focus(); }
  }

  el('form').addEventListener('submit', e => {
    e.preventDefault();
    const text = el('input').value; el('input').value = '';
    send(text);
  });
  document.querySelectorAll('.chip').forEach(chip =>
    chip.addEventListener('click', () => send(chip.dataset.q)));

  fetch('/api/status').then(r => r.json()).then(s => {
    const rec = s.recorded_evaluation || {};
    el('sub').textContent =
      s.parameters.total.toLocaleString() + ' parameters · ' + s.config.n_states + ' latent states · ' +
      'held-out accuracy ' + (rec.accuracy === null || rec.accuracy === undefined ? 'n/a' : rec.accuracy) +
      ' on 4-operation problems';
    el('foot').textContent =
      'Every answer above is checked against the generator’s own arithmetic, so "wrong" means ' +
      'the model was wrong — nothing here is graded by the model itself. Accuracy applies to ' +
      '4-operation chains; longer chains run the model repeatedly and compound its errors. ' +
      'This is one synthetic task and is not evidence of language ability or general reasoning. ' +
      'Full instrument: /  ·  observatory: /lab';
  }).catch(() => { el('sub').textContent = 'status unavailable'; });

  bubble('Ask me a chained modular-arithmetic problem. Start digit 0-9, then add / multiply / ' +
         'subtract with operands 1-9, all mod 10.', 'bot');
  el('input').focus();
})();
</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a trained v56 reasoner")
    parser.add_argument(
        "--checkpoint", required=True, help="path to a supermix-v56-reasoner-checkpoint-v1 file"
    )
    parser.add_argument("--receipt", default=None, help="benchmark_results.json (defaults alongside)")
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address; remote exposure needs its own auth"
    )
    parser.add_argument("--port", type=int, default=8156)
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
    service = ReasonerService(checkpoint, Path(args.receipt) if args.receipt else None)
    status = service.status()
    app = build_app(service)
    print(f"[Supermix v56] serving {checkpoint}")
    print(
        f"  {status['parameters']['total']:,} parameters, "
        f"recorded held-out accuracy {status['recorded_evaluation'].get('accuracy')}"
    )
    print(f"[Supermix v56] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
