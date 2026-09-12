"""Does every new v85 flag actually train, and what does it cost?

v85 added a dozen architecture flags behind default-off switches, each carrying a
citation and no Supermix measurement. Before anyone spends twenty hours on an
arm, two questions have cheap answers:

  1. Does it train at all -- forward, backward, optimiser step, no NaN, loss
     going down over a few dozen steps?
  2. What does it cost per step? An arm that is 40% slower needs to be 40%
     better just to break even on the same wall clock.

This runs a short real optimisation at the v80 shape for each flag and reports
both. It is a gate, not an experiment: 30 steps says nothing about quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
sys.path.insert(0, str(REPO / "source"))

import torch  # noqa: E402

import mimomix_core as mc  # noqa: E402

# v80's shape, so timings are comparable with the 2.045 s/step baseline.
V80 = dict(
    vocab_size=8570, hidden_size=256, n_layers=4, n_heads=8, n_kv_heads=2,
    intermediate_size=384, moe_intermediate_size=96, n_routed_experts=48,
    n_shared_experts=1, moe_top_k=2, sliding_window=64, hybrid_ratio=3,
    n_mtp_layers=2, mtp_loss_weight=0.3, use_thinking_core=True,
    thinking_cycles=2, thinking_max_cycles=4, native_context=128,
    max_position_embeddings=128, rope_scaling="none",
)

ARMS = {
    "baseline": {},
    "qk_norm": {"qk_norm": True},
    "attention_output_gate": {"attention_output_gate": True},
    "sink_swa_only": {"attention_sink_kinds": "swa"},
    "rotary_dim_half": {"rotary_dim": 16},
    "global_layers_final_only": {"global_layers": (3,)},
    "router_sigmoid": {"router_score_function": "sigmoid"},
    "balance_sequence_scope": {"moe_balance_scope": "sequence"},
    "no_shared_expert": {"n_shared_experts": 0},
    "top_k_4": {"moe_top_k": 4},
    "one_mtp_depth": {"n_mtp_layers": 1},
    "no_mtp": {"n_mtp_layers": 0},
    "warm_gate": {"thinking_residual_init": 0.1},
    "differential": {"use_differential_attention": True},
    "mla": {"use_mla": True},
    "mod": {"use_mod": True},
}


def run_arm(name, overrides, steps, batch, seq, lr, seed):
    torch.manual_seed(seed)
    cfg = mc.MiMoMixConfig(**{**V80, **overrides})
    model = mc.MiMoMixModel(cfg)
    n_total = sum(p.numel() for p in model.parameters())
    report = model.parameter_report() if hasattr(model, "parameter_report") else {}
    active = report.get("active_per_token")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed + 1)

    # A learnable toy task: predict the token that appeared `period` positions
    # back. A model that trains will drive loss down; one that is broken will
    # not, and NaN shows up immediately.
    period = 7
    losses, lm_losses, times = [], [], []
    finite = True
    for step in range(steps):
        base = torch.randint(0, cfg.vocab_size, (batch, seq + period), generator=gen)
        x = base[:, period:]
        y = base[:, :seq]
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        loss = out.loss
        if not torch.isfinite(loss):
            finite = False
            break
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            finite = False
            break
        opt.step()
        model.step_router_bias()
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
        # `loss` sums the language-model, auxiliary and MTP terms, so an arm that
        # removes a term reports a different quantity and looks flat next to the
        # baseline for a reason that has nothing to do with learning. `lm_loss`
        # is the same quantity in every arm; judge learning on it.
        lm = getattr(out, "lm_loss", None)
        lm_losses.append(float(lm.detach()) if lm is not None else float("nan"))

    times_sorted = sorted(times[3:]) if len(times) > 6 else sorted(times)
    median = times_sorted[len(times_sorted) // 2] if times_sorted else float("nan")

    def ends(series):
        if not series or any(math.isnan(v) for v in series[:3] + series[-3:]):
            return None, None
        return (sum(series[:3]) / len(series[:3]),
                sum(series[-3:]) / len(series[-3:]))

    first, last = ends(losses)
    lm_first, lm_last = ends(lm_losses)
    # Learning is judged on lm_loss, the only term every arm reports.
    judged_first = lm_first if lm_first is not None else first
    judged_last = lm_last if lm_last is not None else last
    return {
        "arm": name,
        "overrides": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in overrides.items()},
        "params_total": n_total,
        "params_active_per_token": active,
        "steps_completed": len(losses),
        "all_finite": finite,
        "loss_first3": round(first, 4) if first is not None else None,
        "loss_last3": round(last, 4) if last is not None else None,
        "lm_loss_first3": round(lm_first, 4) if lm_first is not None else None,
        "lm_loss_last3": round(lm_last, 4) if lm_last is not None else None,
        "lm_loss_drop": round(lm_first - lm_last, 4) if lm_first is not None else None,
        "judged_on": "lm_loss" if lm_first is not None else "total_loss",
        "learned": bool(judged_first is not None and judged_last < judged_first - 0.01),
        "median_s_per_step": round(median, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=85)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(REPO / "output" / "v85_measurements" / "flag_smoke.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    wanted = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARMS)

    print(f"threads {torch.get_num_threads()}  steps {args.steps}  "
          f"batch {args.batch}x{args.seq}  lr {args.lr}")
    print(f"{'arm':26s} {'params':>10s} {'active':>9s} {'s/step':>7s} "
          f"{'vs base':>8s} {'loss':>16s}  ok")
    print("-" * 92)

    results = {}
    base_time = None
    for name in wanted:
        r = run_arm(name, ARMS[name], args.steps, args.batch, args.seq, args.lr, args.seed)
        results[name] = r
        if name == "baseline":
            base_time = r["median_s_per_step"]
        rel = (f"{r['median_s_per_step'] / base_time:5.2f}x"
               if base_time else "    -")
        ok = "OK" if (r["all_finite"] and r["learned"]) else (
            "NaN" if not r["all_finite"] else "FLAT")
        active = r["params_active_per_token"]
        lo, hi = r["lm_loss_first3"], r["lm_loss_last3"]
        shown = (f"{lo:7.3f}->{hi:7.3f}" if lo is not None
                 else f"{r['loss_first3']:7.3f}->{r['loss_last3']:7.3f}")
        print(f"{name:26s} {r['params_total']:10,d} "
              f"{(f'{active:,}' if active else '-'):>9s} "
              f"{r['median_s_per_step']:7.3f} {rel:>8s} "
              f"{shown}  {ok}", flush=True)

    payload = {
        "schema": "supermix-v85-flag-smoke-v1",
        "purpose": "gate, not experiment: does each new flag train, and what does it cost",
        "settings": {"steps": args.steps, "batch": args.batch, "seq": args.seq,
                     "lr": args.lr, "seed": args.seed, "threads": args.threads,
                     "shape": "v80"},
        "arms": results,
        "non_claims": [
            f"{args.steps} steps on a synthetic copy task says nothing about quality "
            "on the real corpus. A flag that trains here can still be useless or harmful.",
            "Step times are from one machine under whatever else was running; treat "
            "ratios to baseline as the signal, not absolute seconds.",
        ],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")

    broken = [n for n, r in results.items() if not r["all_finite"]]
    flat = [n for n, r in results.items() if r["all_finite"] and not r["learned"]]
    if broken:
        print(f"\nDOES NOT TRAIN (NaN/inf): {broken}")
    if flat:
        print(f"TRAINS BUT DID NOT LEARN in {args.steps} steps: {flat}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
