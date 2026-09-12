"""Per-flag step cost, measured against drift.

A sequential sweep measured `sink_swa_only` and `rotary_dim_half` at 1.45x and
1.42x the baseline. Both have byte-identical parameter counts to the baseline
and strictly less work to do, so neither can be slower. The sweep was measuring
the machine getting slower over twenty-five minutes, not the flags.

This interleaves instead: for each arm, alternate baseline and arm blocks inside
one process and take the ratio of medians. Drift that affects both arms equally
cancels; drift that does not shows up as disagreement between rounds, which is
reported rather than averaged away.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
sys.path.insert(0, str(REPO / "source"))

import torch  # noqa: E402

import mimomix_core as mc  # noqa: E402

V80 = dict(
    vocab_size=8570, hidden_size=256, n_layers=4, n_heads=8, n_kv_heads=2,
    intermediate_size=384, moe_intermediate_size=96, n_routed_experts=48,
    n_shared_experts=1, moe_top_k=2, sliding_window=64, hybrid_ratio=3,
    n_mtp_layers=2, mtp_loss_weight=0.3, use_thinking_core=True,
    thinking_cycles=2, thinking_max_cycles=4, native_context=128,
    max_position_embeddings=128, rope_scaling="none",
)

ARMS = {
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


def build(overrides, seed=85):
    torch.manual_seed(seed)
    cfg = mc.MiMoMixConfig(**{**V80, **overrides})
    model = mc.MiMoMixModel(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    return cfg, model, opt


def time_block(model, opt, cfg, steps, gen):
    times = []
    for _ in range(steps):
        x = torch.randint(0, cfg.vocab_size, (16, 128), generator=gen)
        y = torch.randint(0, cfg.vocab_size, (16, 128), generator=gen)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(x, labels=y)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.step_router_bias()
        times.append(time.perf_counter() - t0)
    return statistics.median(times[1:] or times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(REPO / "output" / "v85_measurements" / "paired_timing.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    wanted = [a.strip() for a in args.only.split(",") if a.strip()] or list(ARMS)
    print(f"paired timing: {args.rounds} rounds x {args.steps} steps, "
          f"baseline rebuilt beside each arm\n")
    print(f"{'arm':26s} {'base s':>8s} {'arm s':>8s} {'ratio':>7s} "
          f"{'spread':>8s}  reading")
    print("-" * 82)

    results = {}
    for name in wanted:
        cfg_b, model_b, opt_b = build({})
        cfg_a, model_a, opt_a = build(ARMS[name])
        gen = torch.Generator().manual_seed(7)
        base_t, arm_t = [], []
        for _ in range(args.rounds):
            base_t.append(time_block(model_b, opt_b, cfg_b, args.steps, gen))
            arm_t.append(time_block(model_a, opt_a, cfg_a, args.steps, gen))
        ratios = [a / b for a, b in zip(arm_t, base_t)]
        ratio = statistics.median(ratios)
        spread = max(ratios) - min(ratios)
        # A spread comparable to the effect means the machine moved more than
        # the flag did, and the ratio should not be quoted.
        reading = ("cost not resolvable" if spread > abs(ratio - 1.0)
                   else "cheaper" if ratio < 0.97
                   else "same" if ratio < 1.03 else "costs more")
        results[name] = {
            "baseline_median_s": round(statistics.median(base_t), 4),
            "arm_median_s": round(statistics.median(arm_t), 4),
            "ratio": round(ratio, 3),
            "ratio_per_round": [round(r, 3) for r in ratios],
            "ratio_spread": round(spread, 3),
            "params_baseline": sum(p.numel() for p in model_b.parameters()),
            "params_arm": sum(p.numel() for p in model_a.parameters()),
            "reading": reading,
        }
        print(f"{name:26s} {results[name]['baseline_median_s']:8.3f} "
              f"{results[name]['arm_median_s']:8.3f} {ratio:7.3f} "
              f"{spread:8.3f}  {reading}", flush=True)

    payload = {
        "schema": "supermix-v85-paired-timing-v1",
        "method": ("baseline and arm interleaved in one process, ratio of medians "
                   "over rounds; a sequential sweep measured drift instead"),
        "settings": {"rounds": args.rounds, "steps": args.steps,
                     "threads": args.threads, "shape": "v80"},
        "arms": results,
        "non_claims": [
            "Ratios only. Absolute seconds on this box vary up to 5x between "
            "identical runs and are not comparable across sessions.",
            "An arm whose ratio spread exceeds its effect is reported as "
            "unresolvable rather than given a number.",
            "Cost says nothing about quality; these flags have no Supermix "
            "accuracy measurement at all.",
        ],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
