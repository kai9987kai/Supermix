"""The MiMoMix thinking API: one endpoint, one router, two backends.

``readme task.txt`` sketched a MiMo-shaped API -- a single ``/v1/think`` that
routes a cheap "flash" thinker by default and escalates to a heavier backend for
long context, multi-tool workflows, and deep tasks. This is that design, made
real against the local MiMoMix stack rather than a remote vendor endpoint:

* routing is decided by :mod:`mimomix_controller`'s deterministic plan, not by a
  hand-written ``if`` chain, so the same rules govern the API and the runtime;
* the thinking budget is the recursive-cycle ladder, and the response reports
  which budget was accepted and why;
* generation runs through MTP self-speculative decoding, so the response can
  report a real acceptance length instead of a marketing number;
* every turn feeds the Dem-Lab observatory, and ``/v1/telemetry`` exposes it.

The service core is framework-free and fully testable on its own.
:func:`create_app` adds an optional Flask surface; nothing in
:class:`ThinkingService` imports Flask.

Safety posture matches the rest of the repo: binds ``127.0.0.1`` by default,
never evaluates request content as code, has no network egress, and treats
message content strictly as data.
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

import mimomix_controller as controller
import mimomix_decoding as decoding
from mimomix_core import MiMoMixConfig, MiMoMixModel
from mimomix_observatory import Observatory


__all__ = [
    "ByteTokenizer",
    "BackendSpec",
    "ThinkingService",
    "create_app",
    "default_backends",
    "main",
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class ByteTokenizer:
    """A dependency-free, lossless UTF-8 byte tokenizer.

    Byte-level means every string round-trips exactly and there is no vocabulary
    file to drift out of sync with a checkpoint. It is not efficient -- a real
    deployment wants BPE -- but it makes the API demonstrable end-to-end without
    shipping a tokenizer artifact.
    """

    PAD = 0
    BOS = 1
    EOS = 2
    N_SPECIAL = 3
    VOCAB_SIZE = N_SPECIAL + 256

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        ids = [self.BOS] if add_bos else []
        ids.extend(self.N_SPECIAL + b for b in text.encode("utf-8"))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        payload = bytes(
            int(i) - self.N_SPECIAL for i in ids if int(i) >= self.N_SPECIAL
        )
        return payload.decode("utf-8", errors="replace")

    def encode_messages(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        """Flatten a chat transcript. Roles are literal text, never directives.

        Message content is data. Nothing in a message can select a backend,
        change a budget, or enable a tool -- those come from the request's own
        typed fields, which the caller controls.
        """

        parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            parts.append(f"<{role}>{content}</{role}>")
        return self.encode("".join(parts))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class BackendSpec:
    """One routable local model."""

    name: str
    model: MiMoMixModel
    #: modes this backend is eligible for
    modes: Tuple[str, ...] = ("fast", "deep", "agent")
    latency_class: str = "medium"
    description: str = ""

    def summary(self) -> Dict[str, object]:
        report = self.model.parameter_report()
        return {
            "name": self.name,
            "modes": list(self.modes),
            "latency_class": self.latency_class,
            "description": self.description,
            "layers": self.model.config.n_layers,
            "attention_layout": list(self.model.layout),
            "sliding_window": self.model.config.sliding_window,
            "max_context": self.model.config.max_position_embeddings,
            "mtp_layers": self.model.config.n_mtp_layers,
            "max_thinking_cycles": self.model.config.thinking_max_cycles,
            "total_parameters": report["total"],
            "active_parameters_per_token": report["active_per_token"],
        }


def default_backends(vocab_size: int = ByteTokenizer.VOCAB_SIZE, seed: int = 0) -> List[BackendSpec]:
    """Two untrained local backends shaped like the flash/pro split.

    These exist so the API is runnable and testable out of the box. They are
    randomly initialised: they produce well-formed responses and honest
    telemetry, and their *text* is noise. Load real weights before drawing any
    conclusion from what they say.
    """

    torch.manual_seed(seed)
    flash = MiMoMixModel(
        MiMoMixConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            n_layers=6,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=256,
            sliding_window=128,
            hybrid_ratio=5,
            native_context=256,
            max_position_embeddings=1024,
            n_routed_experts=8,
            moe_top_k=2,
            moe_intermediate_size=64,
            n_mtp_layers=3,
            thinking_cycles=1,
            thinking_max_cycles=4,
        )
    )
    pro = MiMoMixModel(
        MiMoMixConfig(
            vocab_size=vocab_size,
            hidden_size=192,
            n_layers=12,
            n_heads=6,
            n_kv_heads=2,
            intermediate_size=384,
            sliding_window=128,
            hybrid_ratio=6,
            native_context=256,
            max_position_embeddings=4096,
            n_routed_experts=16,
            moe_top_k=2,
            moe_intermediate_size=96,
            n_mtp_layers=3,
            thinking_cycles=2,
            thinking_max_cycles=8,
        )
    )
    return [
        BackendSpec(
            name="mimomix-flash",
            model=flash,
            modes=("fast", "deep"),
            latency_class="low",
            description="5:1 SWA/global hybrid, 8 experts top-2, 3 MTP depths; the default thinker.",
        ),
        BackendSpec(
            name="mimomix-pro",
            model=pro,
            modes=("deep", "agent"),
            latency_class="high",
            description="6:1 hybrid, 16 experts top-2, deeper recursion; long context and tool use.",
        ),
    ]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


@dataclass
class ThinkingService:
    """Framework-free implementation of ``/v1/think``."""

    backends: List[BackendSpec] = field(default_factory=default_backends)
    tokenizer: ByteTokenizer = field(default_factory=ByteTokenizer)
    policy: controller.ThinkingPolicy = field(default_factory=controller.ThinkingPolicy)
    observatory: Observatory = field(default_factory=Observatory)
    max_output_tokens_limit: int = 512
    max_prompt_tokens: int = 4096
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.backends:
            raise ValueError("at least one backend is required")
        for spec in self.backends:
            spec.model.eval()
            if spec.model.config.vocab_size < self.tokenizer.VOCAB_SIZE:
                raise ValueError(
                    f"backend {spec.name!r} has vocab_size {spec.model.config.vocab_size}, "
                    f"below the tokenizer's {self.tokenizer.VOCAB_SIZE}"
                )

    # -- routing ---------------------------------------------------------

    def route(self, plan: controller.ThinkingPlan, prompt_tokens: int) -> BackendSpec:
        """Pick the cheapest eligible backend that can hold the context.

        Eligibility is mode plus context length. Ordering is by declared
        latency class, so a request only reaches ``pro`` when ``flash`` cannot
        serve it -- the token-efficiency posture, not a quality ranking.
        """

        order = {"low": 0, "medium": 1, "high": 2}
        eligible = [
            spec
            for spec in self.backends
            if plan.mode in spec.modes
            and spec.model.config.max_position_embeddings >= prompt_tokens
            and spec.model.config.thinking_max_cycles >= plan.floor_cycles
        ]
        if not eligible:
            # Fall back to the largest-context backend rather than failing: a
            # truncated answer beats a 500, and the response says what happened.
            return max(self.backends, key=lambda s: s.model.config.max_position_embeddings)
        return min(eligible, key=lambda s: (order.get(s.latency_class, 1), s.model.config.n_layers))

    # -- the endpoint ----------------------------------------------------

    def think(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError("request must be a dict")

        messages = request.get("messages") or []
        raw_input_ids = request.get("input_ids")
        if not messages and not raw_input_ids:
            raise ValueError("request needs 'messages' or 'input_ids'")

        mode = request.get("mode")
        if mode is not None and mode not in controller.VALID_MODES:
            raise ValueError(f"mode must be one of {controller.VALID_MODES}")

        max_output_tokens = int(request.get("max_output_tokens", 32))
        if not 1 <= max_output_tokens <= self.max_output_tokens_limit:
            raise ValueError(
                f"max_output_tokens must lie in [1, {self.max_output_tokens_limit}]"
            )

        tools = request.get("tools") or []
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")

        if raw_input_ids:
            token_ids = [int(t) for t in raw_input_ids]
        else:
            token_ids = self.tokenizer.encode_messages(messages)
        truncated = False
        if len(token_ids) > self.max_prompt_tokens:
            token_ids = token_ids[-self.max_prompt_tokens :]
            truncated = True
        if not token_ids:
            raise ValueError("prompt encoded to zero tokens")

        features = controller.RequestFeatures(
            prompt_tokens=len(token_ids),
            requested_acts=max(1, int(request.get("requested_acts", 1))),
            tool_calls_available=len(tools),
            has_conflict=bool(request.get("has_conflict", False)),
            needs_evidence=bool(request.get("needs_evidence", False)),
            safety_critical=bool(request.get("safety_critical", False)),
            max_output_tokens=max_output_tokens,
        )

        with self._lock:
            plan = controller.plan_request(
                features,
                policy=self.policy,
                mode=mode,
                max_model_cycles=max(s.model.config.thinking_max_cycles for s in self.backends),
            )
            backend = self.route(plan, len(token_ids))
            # Re-plan against the chosen backend so the ladder never proposes a
            # budget that backend cannot execute.
            plan = controller.plan_request(
                features,
                policy=self.policy,
                mode=plan.mode,
                max_model_cycles=backend.model.config.thinking_max_cycles,
            )

            input_ids = torch.tensor([token_ids], dtype=torch.long)
            _, decision = controller.decide(
                backend.model, input_ids, features=features, policy=self.policy, plan=plan
            )

            use_speculative = bool(request.get("speculative", True)) and (
                backend.model.config.n_mtp_layers > 0
            )
            generate = (
                decoding.speculative_generate if use_speculative else decoding.greedy_generate
            )
            result = generate(
                backend.model,
                input_ids,
                max_new_tokens=max_output_tokens,
                eos_token_id=self.tokenizer.EOS,
                thinking_cycles=decision.accepted_budget,
                adaptive_thinking=False,
            )

            new_tokens = result.new_tokens[0].tolist()
            self.observatory.record(result.telemetry, decision=decision, tokens=new_tokens)

        return {
            "model": backend.name,
            "mode": plan.mode,
            "output": self.tokenizer.decode(new_tokens),
            "output_token_ids": new_tokens,
            "prompt_tokens": len(token_ids),
            "prompt_truncated": truncated,
            "latency_class": backend.latency_class,
            "thinking": {
                "accepted_budget": decision.accepted_budget,
                "exit_reason": decision.exit_reason,
                "cycles_spent": decision.cycles_spent,
                "cycle_reduction": decision.cycle_reduction,
                "plan": plan.to_dict(),
                "probes": [p.to_dict() for p in decision.probes],
            },
            "decoding": result.stats.to_dict(),
            "telemetry": result.telemetry,
            "tools_declared": len(tools),
            "warnings": self._warnings(backend, plan, tools),
        }

    def _warnings(
        self, backend: BackendSpec, plan: controller.ThinkingPlan, tools: Sequence[Any]
    ) -> List[str]:
        warnings: List[str] = []
        if tools:
            warnings.append(
                "tools_declared_but_not_executed: this service plans for tool use "
                "but does not call tools; the caller owns execution"
            )
        if plan.mode not in backend.modes:
            warnings.append(f"backend_{backend.name}_not_preferred_for_mode_{plan.mode}")
        return warnings

    # -- auxiliary endpoints ---------------------------------------------

    def models(self) -> Dict[str, object]:
        return {"backends": [spec.summary() for spec in self.backends]}

    def telemetry(self) -> Dict[str, object]:
        with self._lock:
            return self.observatory.report(vocab_size=self.tokenizer.VOCAB_SIZE)

    def health(self) -> Dict[str, object]:
        return {
            "status": "ok",
            "backends": [spec.name for spec in self.backends],
            "turns_observed": len(self.observatory.history),
        }


# ---------------------------------------------------------------------------
# Optional Flask surface
# ---------------------------------------------------------------------------


def create_app(service: Optional[ThinkingService] = None):
    """Build a Flask app around a :class:`ThinkingService`.

    Imported lazily so the service and its tests never require Flask.
    """

    from flask import Flask, jsonify, request as flask_request

    service = service or ThinkingService()
    app = Flask(__name__)

    @app.post("/v1/think")
    def think_route():
        try:
            payload = flask_request.get_json(force=True, silent=False) or {}
        except Exception:
            return jsonify({"error": "request body must be JSON"}), 400
        try:
            return jsonify(service.think(payload))
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/v1/models")
    def models_route():
        return jsonify(service.models())

    @app.get("/v1/telemetry")
    def telemetry_route():
        response = jsonify(service.telemetry())
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health_route():
        return jsonify(service.health())

    app.config["MIMOMIX_SERVICE"] = service
    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _example() -> Dict[str, object]:
    """Prompt-free demonstration, matching the repo's ``--example`` convention."""

    service = ThinkingService()
    request = {
        "messages": [{"role": "user", "content": "Summarise the plan and list two risks."}],
        "max_output_tokens": 12,
        "requested_acts": 2,
    }
    response = service.think(request)
    return {
        "routed_model": response["model"],
        "mode": response["mode"],
        "accepted_budget": response["thinking"]["accepted_budget"],
        "exit_reason": response["thinking"]["exit_reason"],
        "cycle_reduction": response["thinking"]["cycle_reduction"],
        "acceptance_length": response["decoding"]["acceptance_length"],
        "attention_layout": response["telemetry"]["attention_layout"],
        "parameters": response["telemetry"]["parameters"],
        "note": "backends are randomly initialised; the generated text is noise by design",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MiMoMix thinking API")
    parser.add_argument("--example", action="store_true", help="print a prompt-free routing demo and exit")
    parser.add_argument("--serve", action="store_true", help="run the Flask server")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (remote exposure needs its own auth)")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args(argv)

    if args.example or not args.serve:
        print(json.dumps(_example(), indent=2))
        return 0

    app = create_app()
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
