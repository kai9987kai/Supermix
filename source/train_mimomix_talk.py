"""Train the v53 MiMoMix language model to actually generate text.

`docs/V53_MIMOMIX_ARCHITECTURE.md` is explicit that its own numbers prove the
mechanisms work and nothing about quality, because the backends are randomly
initialised: *"Their text is noise by design."* This script removes that caveat
for one corpus. It trains `MiMoMixModel` -- hybrid SWA/global attention with
learnable sinks, auxiliary-loss-free sparse MoE, multi-token prediction, and the
recursive thinking core -- on real dialogue, and writes a checkpoint the API can
serve.

What the result is, stated before any number: a **small domain-specific chat
model**. The corpus is 120,000 templated coding-assistant turns with 292 distinct
word types. The model learns to hold a turn in that register. It has no world
knowledge, because there is none in the data to learn, and a word outside the
vocabulary can never be generated.

Held-out loss is measured on rows the model never trained on, but the corpus is
templated and only 37,543 of its 120,000 responses are distinct, so a validation
row's response may still appear in training. That makes validation perplexity a
measure of fit to the template distribution, not of generalisation to unseen
language. The receipt records it under that name.

Usage::

    python source/train_mimomix_talk.py --steps 4000
    python source/train_mimomix_talk.py --steps 200 --pairs 4000 --run_name smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_decoding as decoding  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel, SparseMoEFeedForward  # noqa: E402

CHECKPOINT_SCHEMA = "supermix-v57-talk-checkpoint-v1"
RECEIPT_SCHEMA = "supermix-v57-talk-benchmark-v1"

#: Prompts used to watch the model learn to talk. They are held fixed across the
#: whole run so successive samples are comparable.
PROBE_PROMPTS = (
    "hello",
    "can you help me with tests",
    "why is my script failing",
    "what is your name",
    "write a unit test for login",
)


def save_talk_checkpoint(
    path: Path,
    model: MiMoMixModel,
    tokenizer: text_utils.WordTokenizer,
    extra: Optional[Dict[str, Any]] = None,
    optimiser: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """Persist config, weights and tokenizer together.

    The tokenizer travels with the weights on purpose: a checkpoint whose
    vocabulary is missing is not reloadable, and one loaded against the wrong
    vocabulary produces confident nonsense rather than an error.

    `optimiser` and `scheduler` are optional and, when given, make the checkpoint
    a true resume point rather than just a set of weights. Restoring weights
    alone restarts AdamW's moments and the learning-rate schedule, which is not
    free: continuing v62 that way sent dev loss from 0.8919 up to 1.0036 and cost
    roughly 1,500 steps re-warming before it recovered. On a multi-day run that
    is hours of wasted compute, so the state is written when available.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "tokenizer": tokenizer.to_dict(),
        "extra": extra or {},
    }
    if optimiser is not None:
        payload["optimiser_state"] = optimiser.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()

    # Write beside the target and rename, rather than over it.
    #
    # `torch.save` truncates the destination first, so a process that dies
    # partway through leaves a corrupt file where the checkpoint was. That is
    # not hypothetical here: v64 and v74 both segfaulted at an eval/checkpoint
    # boundary, and for a `.partial.pt` the file being overwritten is the only
    # thing standing between a crash and losing the whole run -- v74 was 9.2
    # hours in. `os.replace` is atomic on POSIX and on Windows, so the previous
    # checkpoint survives intact until the new one is completely written.
    staging = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, staging)
        os.replace(staging, path)
    except BaseException:
        # Includes KeyboardInterrupt: leaving a stray .tmp beside the
        # checkpoint would be read by nothing but would confuse the next
        # person to look in the directory.
        staging.unlink(missing_ok=True)
        raise


def load_talk_checkpoint(path, map_location: str = "cpu"):
    """Rebuild `(model, tokenizer, payload)` from a v57 checkpoint."""

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"not a {CHECKPOINT_SCHEMA} checkpoint")
    config = MiMoMixConfig(**payload["config"])
    model = MiMoMixModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


@torch.no_grad()
def generate_reply(
    model: MiMoMixModel,
    tokenizer: text_utils.WordTokenizer,
    prompt: str,
    max_new_tokens: int = 48,
    speculative: bool = True,
    thinking_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate one reply, optionally through MTP self-speculative decoding.

    Speculative decoding here is not a quality choice: for greedy decoding it is
    provably token-identical to one-at-a-time generation, so it only changes how
    many trunk forwards the reply costs.
    """

    model.eval()
    ids, _ = tokenizer.encode_turn(prompt, None)
    prompt_ids = torch.tensor([ids], dtype=torch.long)
    started = time.perf_counter()
    generate = decoding.speculative_generate if speculative else decoding.greedy_generate
    result = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=text_utils.EOS,
        thinking_cycles=thinking_cycles,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    reply = tokenizer.decode(result.new_tokens[0].tolist()).strip()
    return {
        "prompt": prompt,
        "reply": reply,
        "tokens": int(result.new_tokens.shape[1]),
        "latency_ms": round(elapsed, 2),
        "acceptance_length": round(result.stats.acceptance_length, 4),
        "trunk_forwards": int(result.stats.verify_forwards),
    }


@torch.no_grad()
def evaluate(
    model: MiMoMixModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Held-out loss over reply tokens only, plus perplexity and bits/token."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for index in range(0, inputs.shape[0], batch_size):
        batch_x = inputs[index : index + batch_size]
        batch_y = labels[index : index + batch_size]
        out = model(batch_x, return_mtp=False)
        shift_logits = out.logits[:, :-1]
        shift_labels = batch_y[:, 1:]
        counted = int((shift_labels != -100).sum())
        if counted == 0:
            continue
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += counted
    mean = total_loss / max(1, total_tokens)
    return {
        "loss": round(mean, 6),
        "perplexity": round(math.exp(min(20.0, mean)), 4),
        "bits_per_token": round(mean / math.log(2), 6),
        "scored_tokens": total_tokens,
    }


@torch.no_grad()
def routing_report(
    model: MiMoMixModel, inputs: torch.Tensor, batch_size: int = 8, batches: int = 8
) -> Dict[str, Any]:
    """Accumulate expert load over several batches, not one forward.

    The bias rule is a control loop and it rings, so a single batch measures the
    phase of the oscillation. Worse, reading it after a *generation* call reports
    a batch of one with a handful of tokens, where most experts are idle as
    arithmetic rather than starved.
    """

    layers = [m for m in model.modules() if isinstance(m, SparseMoEFeedForward)]
    if not layers:
        return {"moe_layers": 0}
    model.eval()
    totals = [torch.zeros(layer.n_routed) for layer in layers]
    counted = 0
    for index in range(0, min(inputs.shape[0], batch_size * batches), batch_size):
        model(inputs[index : index + batch_size], return_mtp=False)
        for position, layer in enumerate(layers):
            totals[position] += layer.last_expert_load.detach().cpu()
        counted += 1

    rows = []
    for position, load in enumerate(totals):
        mass = float(load.sum())
        share = (load / mass) if mass > 0 else load
        n = int(load.numel())
        entropy = float(-(share * share.clamp_min(1e-12).log()).sum())
        rows.append(
            {
                "layer": position,
                "normalised_entropy": round(entropy / math.log(n), 6) if n > 1 else 1.0,
                "starved_experts": int((load == 0).sum()),
            }
        )
    return {
        "moe_layers": len(rows),
        "batches_accumulated": counted,
        "mean_normalised_entropy": round(
            sum(r["normalised_entropy"] for r in rows) / len(rows), 6
        ),
        "total_starved": sum(r["starved_experts"] for r in rows),
        "per_layer": rows,
    }


def build_config(args: argparse.Namespace, vocab_size: int) -> MiMoMixConfig:
    return MiMoMixConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        intermediate_size=args.intermediate_size,
        sliding_window=args.sliding_window,
        hybrid_ratio=args.hybrid_ratio,
        native_context=args.sequence_length,
        max_position_embeddings=args.sequence_length,
        rope_scaling="none",
        use_moe=not args.no_moe,
        n_routed_experts=args.n_routed_experts,
        moe_top_k=args.moe_top_k,
        moe_intermediate_size=args.moe_intermediate_size,
        n_mtp_layers=args.n_mtp_layers,
        mtp_loss_weight=args.mtp_loss_weight,
        use_thinking_core=not args.no_thinking_core,
        thinking_cycles=args.thinking_cycles,
        thinking_max_cycles=args.thinking_max_cycles,
        thinking_residual_init=getattr(args, "thinking_residual_init", 0.0),
    )


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))
    device, device_info = resolve_device(args.device, preference=args.device_preference)

    corpus = text_utils.load_chat_pairs(
        args.database, limit=args.pairs, validation_fraction=args.validation_fraction, seed=args.seed
    )
    # Fields are counted separately, never concatenated: joining them would only
    # ever show the tokenizer a reply's first word with a leading space, and the
    # sentence-initial form would be missing from the vocabulary.
    tokenizer = text_utils.WordTokenizer.build(
        (field for pair in corpus.train for field in pair), max_vocab=args.max_vocab
    )
    sample_texts = [u for u, _ in corpus.validation[:400]] + [a for _, a in corpus.validation[:400]]
    text_utils.assert_roundtrip(tokenizer, sample_texts[:200])
    coverage = tokenizer.vocabulary_report(sample_texts)

    train_x, train_y = text_utils.build_training_tensors(
        corpus.train, tokenizer, args.sequence_length
    )
    val_x, val_y = text_utils.build_training_tensors(
        corpus.validation, tokenizer, args.sequence_length
    )

    config = build_config(args, tokenizer.vocab_size)
    model = MiMoMixModel(config).to(device)
    parameters = model.parameter_report()

    print(f"v57 MiMoMix talk | corpus {corpus.source}")
    print(f"  pairs        {len(corpus.train):,} train / {len(corpus.validation):,} validation")
    print(f"  vocabulary   {tokenizer.vocab_size} types, coverage {coverage['coverage']:.4f}")
    print(f"  sequences    {tuple(train_x.shape)} train / {tuple(val_x.shape)} validation")
    print(f"  parameters   {parameters['total']:,} total / {parameters['active_per_token']:,} active")
    print(f"  layout       {''.join('G' if k=='global' else 'L' for k in model.layout)}"
          f"  window {config.sliding_window}")
    print(f"  device       {device_info.get('resolved', device)}", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=max(1, args.steps), pct_start=args.pct_start
    )

    history: List[Dict[str, Any]] = []
    samples_over_time: List[Dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    started = time.perf_counter()
    running = 0.0
    seen = 0

    for step in range(1, args.steps + 1):
        model.train()
        pick = torch.randint(0, train_x.shape[0], (args.batch_size,), generator=generator)
        batch_x = train_x[pick].to(device)
        batch_y = train_y[pick].to(device)
        out = model(batch_x, labels=batch_y)
        optimiser.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        model.step_router_bias()  # one bias update per optimizer step
        running += float(out.lm_loss.detach())
        seen += 1

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val_x, val_y, args.eval_batch_size)
            row = {
                "step": step,
                "train_lm_loss": round(running / max(1, seen), 6),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
                **metrics,
            }
            history.append(row)
            running, seen = 0.0, 0
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            probe = generate_reply(model, tokenizer, PROBE_PROMPTS[0], args.sample_tokens)
            samples_over_time.append({"step": step, **probe})
            print(
                f"step {step:>5}/{args.steps}  train {row['train_lm_loss']:.4f}  "
                f"val {metrics['loss']:.4f}  ppl {metrics['perplexity']:.2f}  "
                f"{row['elapsed_seconds']:.0f}s   reply: {probe['reply'][:60]!r}",
                flush=True,
            )

    if best_state is not None and args.restore_best:
        model.load_state_dict(best_state)

    final = evaluate(model, val_x, val_y, args.eval_batch_size)
    conversations = [
        generate_reply(model, tokenizer, prompt, args.sample_tokens) for prompt in PROBE_PROMPTS
    ]
    # Greedy and speculative must emit identical text; only the cost differs.
    parity = generate_reply(model, tokenizer, PROBE_PROMPTS[1], args.sample_tokens, speculative=False)
    speculative = [c for c in conversations if c["prompt"] == PROBE_PROMPTS[1]][0]

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{args.run_name}.pt"
    report: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "model": "v57_mimomix_talk",
        "architecture": "mimomix_core.MiMoMixModel (v53), trained on text",
        "corpus": {
            **corpus.to_dict(),
            "note": (
                "templated coding-assistant dialogue; 37,543 distinct responses over "
                "120,000 rows, so validation perplexity measures fit to the template "
                "distribution, not generalisation to unseen language"
            ),
        },
        "tokenizer": coverage,
        "config": config.to_dict(),
        "parameters": parameters,
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "device": str(device_info.get("resolved", device)),
        "train_seconds": round(time.perf_counter() - started, 1),
        "history": history,
        "sample_progression": samples_over_time,
        "final_validation": final,
        "uniform_baseline_loss": round(math.log(tokenizer.vocab_size), 6),
        "conversations": conversations,
        "decoding_parity": {
            "prompt": PROBE_PROMPTS[1],
            "greedy_reply": parity["reply"],
            "speculative_reply": speculative["reply"],
            "identical": parity["reply"] == speculative["reply"],
            "greedy_trunk_forwards": parity["trunk_forwards"],
            "speculative_trunk_forwards": speculative["trunk_forwards"],
            "acceptance_length": speculative["acceptance_length"],
        },
        "routing": routing_report(model, val_x, args.eval_batch_size),
        "checkpoint_path": str(checkpoint_path),
    }
    checks = {
        "learned_something": final["loss"] < 0.5 * math.log(tokenizer.vocab_size),
        "produces_non_empty_replies": all(c["reply"].strip() for c in conversations),
        "speculative_matches_greedy": report["decoding_parity"]["identical"],
        "no_starved_experts": report["routing"].get("total_starved", 0) == 0,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())

    save_talk_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        extra={
            "run_name": args.run_name,
            "validation_loss": final["loss"],
            "perplexity": final["perplexity"],
            "created_at": report["created_at"],
        },
    )
    atomic_json(output_dir / "talk_results.json", report)
    return report


def print_summary(report: Dict[str, Any]) -> None:
    final = report["final_validation"]
    print()
    print("== v57 MiMoMix talk ==")
    print(f"  parameters        {report['parameters']['total']:,} "
          f"({report['parameters']['active_per_token']:,} active/token)")
    print(f"  vocabulary        {report['tokenizer']['vocab_size']} types "
          f"(coverage {report['tokenizer']['coverage']:.4f})")
    print(f"  validation loss   {final['loss']:.4f}  perplexity {final['perplexity']:.2f}  "
          f"({final['bits_per_token']:.3f} bits/token)")
    print(f"  uniform baseline  {report['uniform_baseline_loss']:.4f}")
    parity = report["decoding_parity"]
    print(f"  MTP decoding      acceptance {parity['acceptance_length']:.3f}, "
          f"{parity['speculative_trunk_forwards']} vs {parity['greedy_trunk_forwards']} forwards, "
          f"identical to greedy: {parity['identical']}")
    routing = report["routing"]
    if routing.get("moe_layers"):
        print(f"  routing entropy   {routing['mean_normalised_entropy']:.3f}, "
              f"starved {routing['total_starved']}")
    print()
    print("  conversations:")
    for row in report["conversations"]:
        print(f"    user  {row['prompt']}")
        print(f"    model {row['reply'][:150]}")
    print()
    for name, passed in report["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  checkpoint  {report['checkpoint_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MiMoMix to generate text")
    parser.add_argument("--database", default=str(SOURCE_DIR.parent / "databases" / "llm_chat.db"))
    parser.add_argument("--pairs", type=int, default=None, help="limit rows read from the database")
    parser.add_argument("--validation_fraction", type=float, default=0.02)
    parser.add_argument("--max_vocab", type=int, default=16384)
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--sample_tokens", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--restore_best", action="store_true", default=True)
    parser.add_argument("--no_restore_best", dest="restore_best", action="store_false")

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--sliding_window", type=int, default=64)
    parser.add_argument("--hybrid_ratio", type=int, default=3)
    parser.add_argument("--no_moe", action="store_true")
    parser.add_argument("--n_routed_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=128)
    parser.add_argument("--n_mtp_layers", type=int, default=2)
    parser.add_argument("--mtp_loss_weight", type=float, default=0.3)
    parser.add_argument("--no_thinking_core", action="store_true")
    parser.add_argument("--thinking_cycles", type=int, default=2)
    parser.add_argument("--thinking_max_cycles", type=int, default=4)
    parser.add_argument(
        "--thinking_residual_init",
        type=float,
        default=0.0,
        help=(
            "initial value of the scalar gate the recursive thinking core is "
            "multiplied by. 0.0 (default) reproduces v57/v58 exactly and leaves "
            "the core inert; above zero gives it a gradient path from step 0"
        ),
    )

    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--run_name", default="v57_talk")
    parser.add_argument("--output_dir", default=str(SOURCE_DIR.parent / "output" / "v57_talk"))
    parser.add_argument("--enforce_gates", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print_summary(report)
    if args.enforce_gates and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
