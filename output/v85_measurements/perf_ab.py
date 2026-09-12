"""A/B the training step cost between two source trees, same shape, same seed."""
import importlib
import os
import sys
import time

import torch

SOURCE = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
torch.set_num_threads(8)

for m in [k for k in list(sys.modules) if k.startswith("mimomix")]:
    del sys.modules[m]
sys.path.insert(0, SOURCE)
mc = importlib.import_module("mimomix_core")
print("source:", SOURCE)
print("module:", mc.__file__)

V80 = dict(
    vocab_size=8570, hidden_size=256, n_layers=4, n_heads=8, n_kv_heads=2,
    intermediate_size=384, moe_intermediate_size=96, n_routed_experts=48,
    n_shared_experts=1, moe_top_k=2, sliding_window=64, hybrid_ratio=3,
    n_mtp_layers=2, mtp_loss_weight=0.3, use_thinking_core=True,
    thinking_cycles=2, thinking_max_cycles=4, native_context=128,
    max_position_embeddings=128, rope_scaling="none",
)

torch.manual_seed(0)
cfg = mc.MiMoMixConfig(**V80)
model = mc.MiMoMixModel(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
g = torch.Generator().manual_seed(1)
print("params:", sum(p.numel() for p in model.parameters()))

# Split forward / backward so a regression can be attributed.
fwd, bwd, tot = [], [], []
for i in range(STEPS):
    x = torch.randint(0, cfg.vocab_size, (16, 128), generator=g)
    y = torch.randint(0, cfg.vocab_size, (16, 128), generator=g)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(x, labels=y)
    t1 = time.perf_counter()
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    model.step_router_bias()
    t2 = time.perf_counter()
    if i >= 2:
        fwd.append(t1 - t0)
        bwd.append(t2 - t1)
        tot.append(t2 - t0)


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


print(f"RESULT  fwd {med(fwd):.3f}s  bwd+opt {med(bwd):.3f}s  total {med(tot):.3f}s/step")
