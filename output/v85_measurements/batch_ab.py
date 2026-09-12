"""Is a smaller batch free on this machine?

v85 measured that v80 saw each arithmetic task 54% as often as v74 did, because
the corpus doubled and the 18,000-step budget did not. If per-token cost is flat
in batch size, then halving the batch doubles optimiser updates per wallclock
hour at no cost, and that attacks the dilution directly without touching the
corpus or the step budget.

Measured the only way timings on this box mean anything: both arms interleaved
in one process, ratio of medians, spread reported. A standalone number here is
worthless -- the same benchmark has read 2.045, 11.037 and 2.136 s/step with
nothing changed.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
sys.path.insert(0, str(REPO / "source"))

import torch  # noqa: E402

torch.set_num_threads(8)

import mimomix_core as mc  # noqa: E402

V80 = dict(
    vocab_size=8570, hidden_size=256, n_layers=4, n_heads=8, n_kv_heads=2,
    intermediate_size=384, moe_intermediate_size=96, n_routed_experts=48,
    n_shared_experts=1, moe_top_k=2, sliding_window=64, hybrid_ratio=3,
    n_mtp_layers=2, mtp_loss_weight=0.3, use_thinking_core=True,
    thinking_cycles=2, thinking_max_cycles=4, native_context=128,
    max_position_embeddings=128, rope_scaling="none",
)
SEQ = 128
BATCHES = [4, 8, 16, 32]
ROUNDS, STEPS = 4, 6


def build():
    torch.manual_seed(85)
    cfg = mc.MiMoMixConfig(**V80)
    model = mc.MiMoMixModel(cfg)
    return cfg, model, torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


def block(model, opt, cfg, batch, gen):
    times = []
    for _ in range(STEPS):
        x = torch.randint(0, cfg.vocab_size, (batch, SEQ), generator=gen)
        y = torch.randint(0, cfg.vocab_size, (batch, SEQ), generator=gen)
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
    print(f"interleaved batch sweep, {ROUNDS} rounds x {STEPS} steps, seq {SEQ}\n")
    models = {b: build() for b in BATCHES}
    gen = torch.Generator().manual_seed(7)
    samples = {b: [] for b in BATCHES}
    for _ in range(ROUNDS):
        for b in BATCHES:               # interleaved, so drift hits every arm
            cfg, model, opt = models[b]
            samples[b].append(block(model, opt, cfg, b, gen))

    base = statistics.median(samples[16])
    print(f"{'batch':>6s} {'s/step':>8s} {'spread':>8s} {'s/sequence':>11s} "
          f"{'vs batch16':>11s} {'updates/hour':>13s}")
    print("-" * 64)
    out = {}
    for b in BATCHES:
        med = statistics.median(samples[b])
        spread = max(samples[b]) - min(samples[b])
        per_seq = med / b
        updates = 3600.0 / med
        out[b] = {"median_s_per_step": round(med, 4),
                  "spread": round(spread, 4),
                  "s_per_sequence": round(per_seq, 6),
                  "updates_per_hour": round(updates),
                  "ratio_to_batch16": round(med / base, 3)}
        print(f"{b:6d} {med:8.3f} {spread:8.3f} {per_seq:11.5f} "
              f"{med / base:11.3f} {updates:13.0f}")
    print()
    ref = out[16]["s_per_sequence"]
    print("per-sequence cost relative to batch 16 (1.00 = perfectly flat, "
          "i.e. small batches are free):")
    for b in BATCHES:
        print(f"  batch {b:3d}: {out[b]['s_per_sequence'] / ref:.3f}")
    payload = {
        "schema": "supermix-v85-batch-size-sweep-v1",
        "question": ("Is per-sequence cost flat in batch size? If so a smaller "
                     "batch buys optimiser updates for free, which is the "
                     "cheapest attack on the v85 dilution finding."),
        "method": "all batches interleaved in one process; ratio of medians",
        "settings": {"rounds": ROUNDS, "steps": STEPS, "seq": SEQ,
                     "threads": torch.get_num_threads(), "shape": "v80"},
        "batches": out,
        "non_claims": [
            "Ratios only. Absolute seconds on this box vary up to 5x between "
            "identical runs and never transfer across sessions.",
            "A smaller batch is a different optimisation trajectory, not just "
            "more updates. Flat per-sequence cost says it is affordable, not "
            "that it is better; only a paired training arm can say that.",
            "Measured under x86-64 PyTorch running via Prism emulation on ARM64 "
            "hardware, which is what this project actually runs on.",
        ],
    }
    dest = REPO / "output" / "v85_measurements" / "batch_size_sweep.json"
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
