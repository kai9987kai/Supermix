"""What a bigger model actually costs per step on this box.

Two reasons not to guess this.

**Parameters are the wrong currency for a sparse model.** Only the experts a
token routes to run, so `n_routed_experts` buys capacity at almost no compute,
while `hidden_size` and `n_layers` buy it at full price. A config chosen on
total parameters can be three times the model and six times the wall clock, or
three times the model and barely slower, and the number alone does not say which.

**Timings on this box are not reproducible across processes.** The same
benchmark has read 2.045, 11.037 and 2.136 seconds per step on this hardware
depending on nothing that was measured, and v87 spent 3.8 hours in a block that
looked like a stall and was the CPU clock dropping to 37% while the battery
charged. So every configuration here is timed **in one process, interleaved**,
cycling through the list repeatedly rather than finishing one before starting the
next. A drift that affects all of them equally then cancels out of the ratio,
which is the quantity being used to pick a size.

    python output/v87_measurements/size_cost.py --repeats 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402

# v87's shipped configuration, and the candidates for v88.
CANDIDATES = {
    "v87 h256 L4 E48": {},
    "h256 L4 E96": {"n_routed_experts": 96},
    "h320 L4 E64": {"hidden_size": 320, "n_routed_experts": 64},
    "h320 L5 E64": {"hidden_size": 320, "n_layers": 5, "n_routed_experts": 64},
    "h320 L6 E64": {"hidden_size": 320, "n_layers": 6, "n_routed_experts": 64},
    "h384 L6 E64": {"hidden_size": 384, "n_layers": 6, "n_routed_experts": 64},
}

BASE = dict(vocab_size=8635, hidden_size=256, n_layers=4, n_heads=8,
            n_kv_heads=2, n_routed_experts=48)


def build(overrides: dict) -> MiMoMixModel:
    import inspect
    fields = inspect.signature(MiMoMixConfig.__init__).parameters
    args = dict(BASE)
    args.update(overrides)
    return MiMoMixModel(MiMoMixConfig(**{k: v for k, v in args.items()
                                         if k in fields}))


def active_parameters(model: MiMoMixModel) -> int:
    """Parameters a single token actually passes through.

    Every expert's parameters are stored, but a token is routed to `top_k` of
    them, so the rest cost memory and no arithmetic. Falls back to the total
    when the model does not expose a router, which keeps this honest rather
    than silently reporting the wrong number.
    """

    total = sum(p.numel() for p in model.parameters())
    expert_params = 0
    routed = 0
    top_k = None
    for module in model.modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        per_expert = sum(p.numel() for p in experts.parameters())
        if not per_expert:
            continue
        expert_params += per_expert
        routed += len(experts)
        top_k = top_k or getattr(module, "top_k", None) or getattr(
            getattr(module, "gate", None), "top_k", None)
    if not expert_params or not routed or not top_k:
        return total
    return int(total - expert_params + expert_params * top_k / routed * len(
        [m for m in model.modules() if getattr(m, "experts", None)]) / max(
        1, len([m for m in model.modules() if getattr(m, "experts", None)])))


def timed_step(model: MiMoMixModel, batch: torch.Tensor) -> float:
    """One forward and backward, which is what a training step costs."""

    start = time.perf_counter()
    out = model(batch, labels=batch)
    # `loss` carries the MTP term too, which is what a training step actually
    # backpropagates -- v85 learned the hard way that comparing `loss` against
    # `lm_loss` across configurations reads as a regression that is not there.
    out.loss.backward()
    model.zero_grad(set_to_none=True)
    return time.perf_counter() - start


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="cost of a larger model, measured")
    ap.add_argument("--repeats", type=int, default=3,
                    help="interleaved passes over every candidate")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--sequence_length", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--output")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    batch = torch.randint(0, BASE["vocab_size"],
                          (args.batch_size, args.sequence_length))

    models, sizes = {}, {}
    for name, over in CANDIDATES.items():
        model = build(over)
        model.train()
        models[name] = model
        sizes[name] = {
            "total": sum(p.numel() for p in model.parameters()),
            "active": active_parameters(model),
        }
        timed_step(model, batch)          # warm up, never timed
        print(f"built {name:16s} {sizes[name]['total']:>12,} params")

    print(f"\ntiming {args.repeats} interleaved passes, batch {args.batch_size} "
          f"x {args.sequence_length}, {args.threads} threads")
    samples = {name: [] for name in CANDIDATES}
    for pass_index in range(args.repeats):
        for name, model in models.items():
            samples[name].append(timed_step(model, batch))
        print(f"  pass {pass_index + 1}/{args.repeats} done", flush=True)

    baseline = min(sorted(samples["v87 h256 L4 E48"]))
    print(f"\n{'config':18s} {'total':>11s} {'active':>11s} {'s/step':>8s} "
          f"{'vs v87':>7s}  {'21.5k steps':>12s}")
    report = {}
    for name in CANDIDATES:
        best = min(samples[name])          # least contended sample
        ratio = best / baseline
        hours = best * 21500 / 3600
        report[name] = {"total_parameters": sizes[name]["total"],
                        "active_parameters": sizes[name]["active"],
                        "seconds_per_step": best, "ratio_to_v87": ratio,
                        "samples": sorted(samples[name])}
        print(f"{name:18s} {sizes[name]['total']:>11,} "
              f"{sizes[name]['active']:>11,} {best:8.3f} {ratio:6.2f}x "
              f"{hours:11.1f}h")

    print("\ns/step is the minimum of the samples, not the mean: a slow sample "
          "is contention,\nand the fastest is the closest to the true cost. "
          "Hours assume the clock stays up.")
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"schema": "supermix-v88-size-cost-v1",
             "batch_size": args.batch_size,
             "sequence_length": args.sequence_length,
             "threads": args.threads, "candidates": report}, indent=2),
            encoding="utf-8")
        print(f"\nreport -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
