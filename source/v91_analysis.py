"""v91 post-training analysis: paired, per-row, and path-noise-free.

The pre-registered exact-match McNemar on 630 problems needs a ~3-point gain
for p < 0.05, which a 2,500-step warm start is unlikely to produce. This tool
measures what that test cannot:

``dev``
    Per-row dev loss for any number of checkpoints, on the *same* 11,230 dev
    rows the trainer held out (the split is rebuilt with the trainer's own
    code). With turn-aligned packing each row is one block, so every row is a
    paired item. A checkpoint can be scored in three modes:

    * ``on``   -- as trained;
    * ``off``  -- the connectome core's gate forced to zero;
    * ``mean`` -- the core's added vector replaced by its mean over reply
      tokens, so a branch that learned only a constant offset scores like
      ``on`` here but not in ``off``.

    ``on`` vs ``off`` on the *same weights* has no training-path noise, which a
    comparison between two training runs always has.

``compare``
    Paired statistics for pairs of those per-row losses: mean difference,
    percentile-bootstrap 95% interval, and the fraction of rows each side wins.

``weights``
    State-dict diagnostics with no forward pass: gate size, how far each edge
    moved from its connectome initialisation, and the signed spectral radius of
    the recurrence the core actually runs.

``gate_off_checkpoint``
    Writes a copy of a checkpoint with the gate zeroed, for
    ``eval_problem_solving.py`` (exact-match with the branch closed).

Run only when no training is in flight -- this machine cannot run two heavy
jobs at once.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SOURCE_DIR)

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from train_mimomix_generalisation import build_parser, load_corpus_pairs  # noqa: E402
from train_mimomix_talk import load_talk_checkpoint  # noqa: E402

#: The corpus and split flags every v91 arm (and v89) trained with.
V91_SPLIT_ARGS = [
    "--corpus_jsonl", "datasets/v89/v89_combined.jsonl", "--min_response_characters", "1",
    "--digit_tokens", "--sequence_length", "128", "--max_vocab", "16384", "--turn_aligned_packing",
]


def dev_rows(split_args: Sequence[str] = V91_SPLIT_ARGS) -> List[Tuple[str, str]]:
    """The trainer's dev rows, rebuilt with the trainer's own functions."""

    args = build_parser().parse_args(list(split_args))
    pairs = load_corpus_pairs(
        args.database, limit=args.pairs, corpus_jsonl=args.corpus_jsonl,
        min_response_characters=args.min_response_characters,
    )
    split = splits.build_generalisation_split(
        pairs,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
        target_row_fraction=args.tier3_row_fraction,
        max_row_fraction_per_sentence=args.max_row_fraction_per_sentence,
        seed=args.split_seed,
        source=args.corpus_jsonl,
    )
    return list(split.dev)


class CoreMode:
    """Context manager that switches a model's connectome core between modes."""

    def __init__(self, model: torch.nn.Module, mode: str, mean_delta: Optional[torch.Tensor] = None):
        self.core = getattr(model, "cns_core", None)
        self.mode = mode
        self.mean_delta = mean_delta
        self.saved_gate: Optional[torch.Tensor] = None
        self.original_forward = None

    def __enter__(self):
        if self.mode == "on":
            return self
        if self.core is None:
            raise ValueError(f"mode {self.mode!r} needs a model with a connectome core")
        if self.mode == "off":
            self.saved_gate = self.core.gate.detach().clone()
            with torch.no_grad():
                self.core.gate.zero_()
        elif self.mode == "mean":
            if self.mean_delta is None:
                raise ValueError("mean mode needs mean_delta")
            delta = self.mean_delta
            self.original_forward = self.core.forward
            self.core.forward = lambda hidden: hidden + delta.to(hidden.dtype)  # type: ignore[assignment]
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        return self

    def __exit__(self, *exc):
        if self.saved_gate is not None:
            with torch.no_grad():
                self.core.gate.copy_(self.saved_gate)
        if self.original_forward is not None:
            self.core.forward = self.original_forward  # type: ignore[assignment]
        return False


@torch.no_grad()
def per_row_losses(model, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int = 16
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Summed reply-token loss and reply-token count for every block."""

    model.eval()
    sums = np.zeros(inputs.shape[0], dtype=np.float64)
    counts = np.zeros(inputs.shape[0], dtype=np.int64)
    for start in range(0, inputs.shape[0], batch_size):
        x = inputs[start:start + batch_size].long()
        y = labels[start:start + batch_size].long()
        logits = model(x, return_mtp=False).logits[:, :-1]
        target = y[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            reduction="none", ignore_index=-100,
        ).view(target.shape)
        valid = target != -100
        sums[start:start + x.shape[0]] = (loss * valid).sum(1).double().numpy()
        counts[start:start + x.shape[0]] = valid.sum(1).numpy()
    return sums, counts


@torch.no_grad()
def mean_core_delta(model, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int = 16) -> torch.Tensor:
    """Mean of what the core adds to the residual stream, over reply tokens."""

    core = model.cns_core
    captured: Dict[str, Any] = {"sum": None, "n": 0}
    original = core.forward
    current: Dict[str, torch.Tensor] = {}

    def capture(hidden):
        out = original(hidden)
        delta = (out - hidden)[:, :-1][current["mask"]]
        captured["sum"] = delta.sum(0) if captured["sum"] is None else captured["sum"] + delta.sum(0)
        captured["n"] += delta.shape[0]
        return out

    core.forward = capture  # type: ignore[assignment]
    try:
        model.eval()
        for start in range(0, inputs.shape[0], batch_size):
            x = inputs[start:start + batch_size].long()
            y = labels[start:start + batch_size].long()
            current["mask"] = y[:, 1:] != -100
            model(x, return_mtp=False)
    finally:
        core.forward = original  # type: ignore[assignment]
    return captured["sum"] / max(1, captured["n"])


def run_dev(specs: Sequence[str], output: str, batch_size: int, limit: Optional[int]) -> Dict[str, Any]:
    """``specs`` are ``name=checkpoint:mode`` strings."""

    parsed = []
    for spec in specs:
        name, rest = spec.split("=", 1)
        path, mode = rest.rsplit(":", 1)
        parsed.append((name, path, mode))
    started = time.time()
    rows = dev_rows()
    if limit:
        rows = rows[:limit]
    print(f"dev rows {len(rows):,} ({time.time() - started:.0f}s)", flush=True)

    results: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}
    reference_tokens: Optional[List[str]] = None
    tensors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    loaded: Dict[str, Any] = {}
    for name, path, mode in parsed:
        if path not in loaded:
            loaded.clear()  # one checkpoint in memory at a time
            loaded[path] = load_talk_checkpoint(path)
        model, tokenizer, _ = loaded[path]
        if reference_tokens is None:
            reference_tokens = list(tokenizer.tokens)
            tensors = text_utils.build_training_tensors(rows, tokenizer, 128, turn_aligned=True)
        elif list(tokenizer.tokens) != reference_tokens:
            raise SystemExit(f"{path} has a different vocabulary; rows would not be comparable")
        inputs, labels = tensors
        t0 = time.time()
        mean_delta = mean_core_delta(model, inputs, labels, batch_size) if mode == "mean" else None
        with CoreMode(model, mode, mean_delta):
            sums, counts = per_row_losses(model, inputs, labels, batch_size)
        arrays[f"{name}__sum"] = sums
        arrays[f"{name}__count"] = counts
        results[name] = {
            "checkpoint": path,
            "mode": mode,
            "rows": int(len(sums)),
            "token_mean_loss": float(sums.sum() / max(1, counts.sum())),
            "row_mean_loss": float(np.mean(sums / np.maximum(counts, 1))),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name:<24} {mode:<4} token-mean {results[name]['token_mean_loss']:.6f} "
              f"({results[name]['seconds']:.0f}s)", flush=True)
    np.savez_compressed(os.path.splitext(output)[0] + ".rows.npz", **arrays)
    report = {"schema": "supermix-v91-dev-rows-v1", "rows": len(rows), "results": results}
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def paired(a_sum, a_cnt, b_sum, b_cnt, resamples: int = 10000, seed: int = 91) -> Dict[str, Any]:
    """B minus A, per row (negative = B has lower loss)."""

    a = a_sum / np.maximum(a_cnt, 1)
    b = b_sum / np.maximum(b_cnt, 1)
    diff = b - a
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.empty(resamples)
    token_boots = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        boots[i] = diff[idx].mean()
        token_boots[i] = (b_sum[idx].sum() - a_sum[idx].sum()) / max(1, a_cnt[idx].sum())
    token_diff = (b_sum.sum() - a_sum.sum()) / max(1, a_cnt.sum())
    return {
        "rows": int(n),
        "row_mean_diff": float(diff.mean()),
        "row_mean_diff_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "token_mean_diff": float(token_diff),
        "token_mean_diff_ci95": [float(np.percentile(token_boots, 2.5)), float(np.percentile(token_boots, 97.5))],
        "rows_b_better": int((diff < -1e-9).sum()),
        "rows_a_better": int((diff > 1e-9).sum()),
        "rows_tied": int((np.abs(diff) <= 1e-9).sum()),
        "ci_excludes_zero": bool(np.percentile(token_boots, 2.5) > 0 or np.percentile(token_boots, 97.5) < 0),
    }


def run_compare(rows_npz: str, pairs: Sequence[str], output: str) -> Dict[str, Any]:
    data = np.load(rows_npz)
    report: Dict[str, Any] = {"schema": "supermix-v91-paired-dev-v1", "source": rows_npz,
                              "sign": "diff = second minus first; negative means the second has lower loss",
                              "pairs": {}}
    for pair in pairs:
        first, second = pair.split(",")
        result = paired(data[f"{first}__sum"], data[f"{first}__count"],
                        data[f"{second}__sum"], data[f"{second}__count"])
        report["pairs"][f"{second} - {first}"] = result
        print(f"{second:>22} - {first:<22} token diff {result['token_mean_diff']:+.6f} "
              f"CI [{result['token_mean_diff_ci95'][0]:+.6f}, {result['token_mean_diff_ci95'][1]:+.6f}] "
              f"wins {result['rows_b_better']}/{result['rows_a_better']}", flush=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def weight_report(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    if "cns_core.gate" not in state:
        return {"checkpoint": path, "cns_core": False}
    mask = state["cns_core.mask"].double()
    sign = state["cns_core.sign"].double()
    start = state["cns_core.init_fraction"].double() * state["cns_core.spectral_scale"].double()
    now = torch.nn.functional.softplus(state["cns_core.edge_logit"].double()) * mask
    edges = mask > 0
    ratio = (now[edges] / start[edges].clamp_min(1e-12))
    signed = now * sign.unsqueeze(0)
    eig = np.linalg.eigvals(signed.numpy())
    abs_eig = np.linalg.eigvals(now.numpy())
    gate = state["cns_core.gate"].double()
    leak = torch.sigmoid(state["cns_core.leak_logit"].double())
    return {
        "checkpoint": path,
        "cns_core": True,
        "gate_mean_abs": float(gate.abs().mean()),
        "gate_max_abs": float(gate.abs().max()),
        "gate_l2": float(gate.norm()),
        "edge_strength_ratio_median": float(ratio.median()),
        "edge_strength_ratio_p05_p95": [float(torch.quantile(ratio, 0.05)), float(torch.quantile(ratio, 0.95))],
        "edges_grown_2x": int((ratio > 2).sum()),
        "edges_shrunk_half": int((ratio < 0.5).sum()),
        "abs_spectral_radius": float(np.abs(abs_eig).max()),
        "signed_spectral_radius": float(np.abs(eig).max()),
        "signed_max_real_eigenvalue": float(eig.real.max()),
        "leak_mean": float(leak.mean()),
        "leak_min_max": [float(leak.min()), float(leak.max())],
        "node_bias_mean_abs": float(state["cns_core.node_bias"].abs().mean()),
    }


def gate_off_checkpoint(path: str, output: str) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "cns_core.gate" not in payload["state_dict"]:
        raise SystemExit(f"{path} has no connectome core")
    payload["state_dict"]["cns_core.gate"] = torch.zeros_like(payload["state_dict"]["cns_core.gate"])
    extra = dict(payload.get("extra") or {})
    extra["v91_gate_forced_zero_from"] = path
    payload["extra"] = extra
    torch.save(payload, output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--spec", nargs="+", required=True, help="name=checkpoint:on|off|mean")
    d.add_argument("--output", required=True)
    d.add_argument("--batch_size", type=int, default=16)
    d.add_argument("--limit", type=int, default=None, help="first N dev rows only (smoke tests)")
    d.add_argument("--torch_threads", type=int, default=8)
    c = sub.add_parser("compare")
    c.add_argument("--rows", required=True)
    c.add_argument("--pair", nargs="+", required=True, help="first,second")
    c.add_argument("--output", required=True)
    w = sub.add_parser("weights")
    w.add_argument("--checkpoint", nargs="+", required=True)
    w.add_argument("--output", required=True)
    g = sub.add_parser("gate_off_checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "dev":
        torch.set_num_threads(args.torch_threads)
        run_dev(args.spec, args.output, args.batch_size, args.limit)
    elif args.command == "compare":
        run_compare(args.rows, args.pair, args.output)
    elif args.command == "weights":
        report = [weight_report(p) for p in args.checkpoint]
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report, indent=2))
    elif args.command == "gate_off_checkpoint":
        print(gate_off_checkpoint(args.checkpoint, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
