"""v92: a male-CNS connectome core with memory across tokens, tested against a
null ensemble on a frozen v89.

v91 grafted a per-token connectome branch into v89 and trained everything. The
branch opened and was then ignored: closing its gate changed dev loss by
-6e-6 nats, and the no-graft control finished ahead. Every positive result
for connectome wiring in the literature involves memory across time -- frozen
reservoirs near criticality (Suarez et al. 2021, 2024) -- and v91's core had
none: its state reset at every token.

v92 changes three things.

1. **Temporal recurrence.** The core's state ``r`` is carried along the
   sequence as a causal scan: for each position ``t``, ``inner_steps`` updates
   of ``r <- (1 - a) r + a relu(W r + u_t + b)`` starting from ``r_{t-1}``, not
   from zero. The fly wiring is now a dynamical system over the text.
2. **Frozen trunk.** v89 is frozen and only the core trains, so the network
   cannot learn to route around the branch -- if the wiring carries anything
   useful, the loss has to show it. Hidden states after block 2 are cached
   once, in full fp32, and every run replays blocks 3-4, the thinking core and
   the head from them (``forward_from``).
3. **A null ensemble, matched on the operator that runs.** The real wiring is
   trained with 3 initialisation seeds; 10 stratified rewires
   (:func:`malecns_connectome.stratified_rewire`: same degrees, self-loops,
   input fractions, E/I input mix and role blocks) are trained with seed 1.
   Every wiring is scaled to the same *signed* spectral radius -- v91 matched
   |W|, which left its null 26% more recurrent.

Controls on the same rig: the per-token v91 core (``temporal=False``), and an
"empty" core with no inter-module edges at all (a leaky memory per module).

Commands (run only when no training is in flight)::

    python source/v92_temporal_connectome.py nulls
    python source/v92_temporal_connectome.py prepare
    python source/v92_temporal_connectome.py run
    python source/v92_temporal_connectome.py report
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
import torch.nn as nn
import torch.nn.functional as F

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SOURCE_DIR)

import malecns_connectome as mc  # noqa: E402
from mimomix_core import CNS_AFFERENT_ROLES, CNS_EFFERENT_ROLES, RMSNorm  # noqa: E402

V89 = "output/v89_corpus/v89_corpus.pt"
MODULES = "datasets/v91_malecns/malecns_modules_512.npz"
NULLS = "datasets/v92_connectome/nulls_512.npz"
CACHE = "output/v92_connectome/cache"
RUNS = "output/v92_connectome/runs"
GRAFT_AFTER = 2  # the core's output joins the residual stream after block 2

#: The pre-registered run plan, interleaved so a chain cut short still holds
#: real-vs-null pairs.
PLAN: List[Dict[str, Any]] = (
    [
        {"name": "T_real_s1", "temporal": True, "wiring": "real", "seed": 1},
        {"name": "T_null_101", "temporal": True, "wiring": "null:101", "seed": 1},
        {"name": "T_real_s2", "temporal": True, "wiring": "real", "seed": 2},
        {"name": "T_null_102", "temporal": True, "wiring": "null:102", "seed": 1},
        {"name": "T_empty_s1", "temporal": True, "wiring": "empty", "seed": 1},
        {"name": "P_real_s1", "temporal": False, "wiring": "real", "seed": 1},
        {"name": "T_real_s3", "temporal": True, "wiring": "real", "seed": 3},
    ]
    + [{"name": f"T_null_{k}", "temporal": True, "wiring": f"null:{k}", "seed": 1} for k in range(103, 111)]
    + [{"name": "P_null_101", "temporal": False, "wiring": "null:101", "seed": 1}]
)


# ---------------------------------------------------------------------------
# The core
# ---------------------------------------------------------------------------


class TemporalConnectomeCore(nn.Module):
    """Dale's-law recurrent branch on the male-CNS module graph.

    ``temporal=True``: the rate vector is carried across positions (causal
    scan, ``inner_steps`` updates per token). ``temporal=False``: it restarts
    from zero at every token and runs ``inner_steps`` updates -- exactly v91's
    ConnectomeCore dynamics. Output joins the residual stream through a
    per-channel gate initialised at zero, so the untrained core is the
    identity.
    """

    def __init__(self, hidden: int, n_nodes: int = 512, inner_steps: int = 2, temporal: bool = True,
                 leak_init: float = 0.2, rms_eps: float = 1e-6):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.inner_steps = int(inner_steps)
        self.temporal = bool(temporal)
        self.norm = RMSNorm(hidden, rms_eps)
        self.read_in = nn.Linear(hidden, n_nodes, bias=False)
        self.read_out = nn.Linear(n_nodes, hidden, bias=False)
        nn.init.normal_(self.read_in.weight, std=0.02)
        nn.init.normal_(self.read_out.weight, std=0.02)
        self.edge_logit = nn.Parameter(torch.full((n_nodes, n_nodes), -10.0))
        self.node_bias = nn.Parameter(torch.zeros(n_nodes))
        leak = min(max(leak_init, 1e-4), 1 - 1e-4)
        self.leak_logit = nn.Parameter(torch.full((n_nodes,), math.log(leak / (1 - leak))))
        self.gate = nn.Parameter(torch.zeros(hidden))
        self.register_buffer("mask", torch.zeros(n_nodes, n_nodes))
        self.register_buffer("sign", torch.ones(n_nodes))
        self.register_buffer("in_mask", torch.ones(n_nodes))
        self.register_buffer("out_mask", torch.ones(n_nodes))
        self.info: Dict[str, Any] = {}

    @torch.no_grad()
    def install(self, post: np.ndarray, pre: np.ndarray, fraction: np.ndarray, sign: np.ndarray,
                roles: Sequence[str], signed_radius: float = 0.95, io: str = "afferent_efferent") -> Dict[str, Any]:
        """Install a wiring, scaled so the SIGNED matrix has ``signed_radius``."""

        n = self.n_nodes
        dense = np.zeros((n, n))
        if len(post):
            dense[post, pre] = fraction
        signed = dense * sign[None, :]
        rho = float(np.max(np.abs(np.linalg.eigvals(signed)))) if len(post) else 0.0
        scale = signed_radius / rho if rho > 0 else 1.0
        magnitude = dense * scale
        mask = (dense > 0).astype(np.float64)
        target = torch.from_numpy(magnitude).float().clamp_min(1e-6)
        inverse = torch.log(torch.expm1(target))
        self.edge_logit.copy_(torch.where(torch.from_numpy(mask).bool(), inverse, torch.full_like(inverse, -10.0)))
        self.mask.copy_(torch.from_numpy(mask))
        self.sign.copy_(torch.from_numpy(sign.astype(np.float64)))
        if io == "afferent_efferent":
            self.in_mask.copy_(torch.tensor([r in CNS_AFFERENT_ROLES for r in roles], dtype=torch.float32))
            self.out_mask.copy_(torch.tensor([r in CNS_EFFERENT_ROLES for r in roles], dtype=torch.float32))
        installed = self.weight().double().numpy()
        eig = np.linalg.eigvals(installed) if len(post) else np.zeros(1)
        self.info = {
            "edges": int(mask.sum()),
            "signed_radius_unscaled": round(rho, 6),
            "scale": round(scale, 6),
            "signed_radius": round(float(np.abs(eig).max()), 6),
            "max_real_eigenvalue": round(float(eig.real.max()), 6),
            "abs_radius": round(float(np.abs(np.linalg.eigvals(np.abs(installed))).max()), 6) if len(post) else 0.0,
            "input_nodes": int(self.in_mask.sum()),
            "output_nodes": int(self.out_mask.sum()),
        }
        return dict(self.info)

    def weight(self) -> torch.Tensor:
        return self.mask * F.softplus(self.edge_logit) * self.sign.unsqueeze(0)

    def rates(self, hidden: torch.Tensor) -> torch.Tensor:
        """Module rates ``(B, T, N)`` for hidden states ``(B, T, H)``."""

        drive = self.read_in(self.norm(hidden)) * self.in_mask + self.node_bias
        w_t = self.weight().t()
        alpha = torch.sigmoid(self.leak_logit)
        if not self.temporal:
            rate = torch.zeros_like(drive)
            for _ in range(self.inner_steps):
                rate = (1 - alpha) * rate + alpha * F.relu(rate @ w_t + drive)
            return rate
        state = drive.new_zeros(drive.shape[0], drive.shape[2])
        states = []
        for t in range(drive.shape[1]):
            u = drive[:, t]
            for _ in range(self.inner_steps):
                state = (1 - alpha) * state + alpha * F.relu(state @ w_t + u)
            states.append(state)
        return torch.stack(states, dim=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        read = self.rates(hidden) * self.out_mask
        n_out = self.out_mask.sum().clamp_min(1.0)
        read = read * torch.rsqrt(read.pow(2).sum(-1, keepdim=True) / n_out + 1e-6)
        return hidden + self.gate * self.read_out(read)


# ---------------------------------------------------------------------------
# Frozen-trunk replay
# ---------------------------------------------------------------------------


def trunk_until(model, input_ids: torch.Tensor, last_layer: int) -> torch.Tensor:
    """Hidden state after block ``last_layer``, exactly as ``MiMoMixModel.forward`` computes it."""

    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    cos, sin = model.rotary(positions)
    local_cos, local_sin = model.rotary_local(positions)
    keys = positions.new_empty((0,))
    hidden = model.embed_tokens(input_ids)
    for layer in model.layers[: last_layer + 1]:
        c, s = (local_cos, local_sin) if layer.kind == "swa" else (cos, sin)
        hidden, _ = layer(hidden, c, s, positions, keys, past_kv=None, attention_mask=None, use_cache=False)
    return hidden


def forward_from(model, hidden: torch.Tensor, first_layer: int) -> torch.Tensor:
    """Logits from blocks ``first_layer``.., the thinking core, final norm and head."""

    positions = torch.arange(hidden.shape[1], device=hidden.device)
    cos, sin = model.rotary(positions)
    local_cos, local_sin = model.rotary_local(positions)
    keys = positions.new_empty((0,))
    for layer in model.layers[first_layer:]:
        c, s = (local_cos, local_sin) if layer.kind == "swa" else (cos, sin)
        hidden, _ = layer(hidden, c, s, positions, keys, past_kv=None, attention_mask=None, use_cache=False)
    if model.thinking_core is not None:
        hidden, _ = model.thinking_core(hidden)
    return model.lm_head(model.norm(hidden))


def row_losses(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Summed reply-token loss and count per row (the trainer's shift and mask)."""

    shift = logits[:, :-1]
    target = labels[:, 1:].long()
    loss = F.cross_entropy(shift.reshape(-1, shift.shape[-1]), target.reshape(-1),
                           reduction="none", ignore_index=-100).view(target.shape)
    valid = target != -100
    return (loss * valid).sum(1), valid.sum(1)


# ---------------------------------------------------------------------------
# Null ensemble
# ---------------------------------------------------------------------------


def build_nulls(modules: str = MODULES, output: str = NULLS, seeds: Sequence[int] = range(101, 111)) -> Dict[str, Any]:
    """Stratified rewires of the real module graph, one per seed, in a NEW file.

    The v91 npz is only read: v91 arm B trains from it.
    """

    with np.load(modules, allow_pickle=True) as data:
        post, pre, fraction = data["edge_post"], data["edge_pre"], data["edge_fraction"]
        sign = data["module_sign"].astype(np.int64)
        roles = np.array([str(r) for r in data["module_role"]])
    n = len(sign)
    strata_names = sorted({(r, int(s)) for r, s in zip(roles, sign)})
    stratum_of = {k: i for i, k in enumerate(strata_names)}
    strata = np.array([stratum_of[(r, int(s))] for r, s in zip(roles, sign)])
    arrays: Dict[str, np.ndarray] = {}
    diagnostics: Dict[str, Any] = {"real": mc.graph_diagnostics(post, pre, fraction, sign.astype(float), roles, n)}
    for seed in seeds:
        r_post, r_pre, r_fraction, info = mc.stratified_rewire(post, pre, fraction, strata, n, seed=int(seed))
        arrays[f"post_{seed}"], arrays[f"pre_{seed}"], arrays[f"fraction_{seed}"] = r_post, r_pre, r_fraction
        diagnostics[str(seed)] = {
            **info,
            **mc.graph_diagnostics(r_post, r_pre, r_fraction, sign.astype(float), roles, n, reference=(post, pre)),
        }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(output, seeds=np.array(list(seeds)), **arrays)
    receipt = {"schema": "supermix-v92-null-ensemble-v1", "source": modules, "seeds": list(seeds),
               "diagnostics": diagnostics}
    with open(os.path.splitext(output)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    return receipt


def wiring_arrays(wiring: str, modules: str = MODULES, nulls: str = NULLS):
    with np.load(modules, allow_pickle=True) as data:
        sign = data["module_sign"].astype(np.float64)
        roles = [str(r) for r in data["module_role"]]
        if wiring == "real":
            return data["edge_post"], data["edge_pre"], data["edge_fraction"], sign, roles
        if wiring == "empty":
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0), sign, roles
    if wiring.startswith("null:"):
        seed = wiring.split(":", 1)[1]
        with np.load(nulls) as data:
            return data[f"post_{seed}"], data[f"pre_{seed}"], data[f"fraction_{seed}"], sign, roles
    raise ValueError(f"unknown wiring {wiring!r}")


# ---------------------------------------------------------------------------
# Cache of block-2 states
# ---------------------------------------------------------------------------


def _split_rows(split_seed_args: Optional[Sequence[str]] = None):
    import mimomix_eval_splits as splits
    from train_mimomix_generalisation import build_parser, load_corpus_pairs
    from v91_analysis import V91_SPLIT_ARGS

    args = build_parser().parse_args(list(split_seed_args or V91_SPLIT_ARGS))
    pairs = load_corpus_pairs(args.database, limit=args.pairs, corpus_jsonl=args.corpus_jsonl,
                              min_response_characters=args.min_response_characters)
    return splits.build_generalisation_split(
        pairs, dev_fraction=args.dev_fraction, test_fraction=args.test_fraction,
        target_row_fraction=args.tier3_row_fraction,
        max_row_fraction_per_sentence=args.max_row_fraction_per_sentence,
        seed=args.split_seed, source=args.corpus_jsonl,
    )


@torch.no_grad()
def prepare(cache_dir: str = CACHE, checkpoint: str = V89, train_batches: int = 1000, batch_size: int = 16,
            dev_rows: int = 2000, seed: int = 92, threads: int = 8, rows_override=None) -> Dict[str, Any]:
    """Cache fp32 hidden states after block 2 for a fixed train stream and dev subset."""

    import mimomix_text as text_utils
    from train_mimomix_talk import load_talk_checkpoint

    torch.set_num_threads(threads)
    started = time.time()
    model, tokenizer, _ = load_talk_checkpoint(checkpoint)
    model.eval()
    if rows_override is None:
        split = _split_rows()
        train_pool, dev_pool = list(split.train), list(split.dev)
    else:
        train_pool, dev_pool = rows_override
    rng = np.random.default_rng(seed)
    need = train_batches * batch_size
    picked = [train_pool[i] for i in rng.choice(len(train_pool), size=min(len(train_pool), int(need * 1.1) + 64),
                                                 replace=len(train_pool) < need)]
    train_x, train_y = text_utils.build_training_tensors(picked, tokenizer, 128, turn_aligned=True)
    if train_x.shape[0] < need:
        raise SystemExit(f"only {train_x.shape[0]} train blocks for {need} needed")
    train_x, train_y = train_x[:need], train_y[:need]
    dev_pick = [dev_pool[i] for i in sorted(rng.choice(len(dev_pool), size=min(dev_rows, len(dev_pool)), replace=False))]
    dev_x, dev_y = text_utils.build_training_tensors(dev_pick, tokenizer, 128, turn_aligned=True)
    hidden = model.config.hidden_size
    os.makedirs(cache_dir, exist_ok=True)
    train_h = np.lib.format.open_memmap(os.path.join(cache_dir, "train_h.npy"), mode="w+", dtype=np.float32,
                                        shape=(train_batches, batch_size, 128, hidden))
    for b in range(train_batches):
        x = train_x[b * batch_size:(b + 1) * batch_size].long()
        train_h[b] = trunk_until(model, x, GRAFT_AFTER).numpy()
        if (b + 1) % 100 == 0:
            print(f"  train cache {b + 1}/{train_batches} ({time.time() - started:.0f}s)", flush=True)
    train_h.flush()
    np.save(os.path.join(cache_dir, "train_y.npy"),
            train_y.view(train_batches, batch_size, 128).numpy().astype(np.int32))
    dev_h = np.lib.format.open_memmap(os.path.join(cache_dir, "dev_h.npy"), mode="w+", dtype=np.float32,
                                      shape=(dev_x.shape[0], 128, hidden))
    base_sum = np.zeros(dev_x.shape[0])
    base_cnt = np.zeros(dev_x.shape[0], dtype=np.int64)
    for start in range(0, dev_x.shape[0], batch_size):
        x = dev_x[start:start + batch_size].long()
        h = trunk_until(model, x, GRAFT_AFTER)
        dev_h[start:start + x.shape[0]] = h.numpy()
        s, c = row_losses(forward_from(model, h, GRAFT_AFTER + 1), dev_y[start:start + x.shape[0]])
        base_sum[start:start + x.shape[0]] = s.double().numpy()
        base_cnt[start:start + x.shape[0]] = c.numpy()
    dev_h.flush()
    np.save(os.path.join(cache_dir, "dev_y.npy"), dev_y.numpy().astype(np.int32))
    np.savez(os.path.join(cache_dir, "baseline_rows.npz"), sum=base_sum, count=base_cnt)
    meta = {
        "schema": "supermix-v92-cache-v1",
        "checkpoint": checkpoint,
        "graft_after_block": GRAFT_AFTER,
        "train_batches": train_batches,
        "batch_size": batch_size,
        "dev_rows": int(dev_x.shape[0]),
        "seed": seed,
        "dtype": "float32",
        "baseline_dev_token_loss": float(base_sum.sum() / max(1, base_cnt.sum())),
        "seconds": round(time.time() - started, 1),
    }
    with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def wsd_lr(step: int, total: int, peak: float, warmup: float = 0.05, decay: float = 0.2) -> float:
    """Warmup-stable-decay: linear up, flat, linear down to 0 over the last ``decay``."""

    w = max(1, int(total * warmup))
    d = max(1, int(total * decay))
    if step < w:
        return peak * (step + 1) / w
    if step >= total - d:
        return peak * max(0.0, (total - step) / d)
    return peak


def bootstrap_ci(diff_sum: np.ndarray, count: np.ndarray, resamples: int = 5000, seed: int = 92) -> List[float]:
    rng = np.random.default_rng(seed)
    n = len(diff_sum)
    boots = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        boots[i] = diff_sum[idx].sum() / max(1, count[idx].sum())
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def run_one(spec: Dict[str, Any], cache_dir: str = CACHE, out_dir: str = RUNS, checkpoint: str = V89,
            steps: int = 1000, lr: float = 3e-3, signed_radius: float = 0.95, threads: int = 8,
            model=None) -> Dict[str, Any]:
    from train_mimomix_talk import load_talk_checkpoint

    torch.set_num_threads(threads)
    started = time.time()
    if model is None:
        model, _, _ = load_talk_checkpoint(checkpoint)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(int(spec["seed"]))
    temporal = bool(spec["temporal"])
    core = TemporalConnectomeCore(
        model.config.hidden_size, 512,
        inner_steps=2 if temporal else 6,
        temporal=temporal,
        leak_init=0.2 if temporal else 0.5,
    )
    post, pre, fraction, sign, roles = wiring_arrays(spec["wiring"])
    info = core.install(post, pre, fraction, sign, roles, signed_radius=signed_radius)
    decay = [core.read_in.weight, core.read_out.weight]
    no_decay = [core.edge_logit, core.node_bias, core.leak_logit, core.gate, core.norm.weight]
    optimiser = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-12,
    )
    train_h = np.load(os.path.join(cache_dir, "train_h.npy"), mmap_mode="r")
    train_y = np.load(os.path.join(cache_dir, "train_y.npy"), mmap_mode="r")
    n_batches = train_h.shape[0]
    history = []
    running = 0.0
    for step in range(steps):
        for group in optimiser.param_groups:
            group["lr"] = wsd_lr(step, steps, lr)
        core.train()
        h = torch.from_numpy(np.array(train_h[step % n_batches]))
        y = torch.from_numpy(np.array(train_y[step % n_batches]))
        s, c = row_losses(forward_from(model, core(h), GRAFT_AFTER + 1), y)
        loss = s.sum() / c.sum().clamp_min(1)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
        optimiser.step()
        running += float(loss.detach())
        if (step + 1) % 100 == 0:
            history.append({"step": step + 1, "train_loss": round(running / 100, 6),
                            "gate_mean_abs": round(float(core.gate.detach().abs().mean()), 6),
                            "seconds": round(time.time() - started, 1)})
            print(f"  {spec['name']} step {step + 1}/{steps} train {running / 100:.5f} "
                  f"gate {history[-1]['gate_mean_abs']:.4f} {history[-1]['seconds']:.0f}s", flush=True)
            running = 0.0

    core.eval()
    dev_h = np.load(os.path.join(cache_dir, "dev_h.npy"), mmap_mode="r")
    dev_y = np.load(os.path.join(cache_dir, "dev_y.npy"), mmap_mode="r")
    base = np.load(os.path.join(cache_dir, "baseline_rows.npz"))
    sums = np.zeros(dev_h.shape[0])
    counts = np.zeros(dev_h.shape[0], dtype=np.int64)
    with torch.no_grad():
        for start in range(0, dev_h.shape[0], 16):
            h = torch.from_numpy(np.array(dev_h[start:start + 16]))
            y = torch.from_numpy(np.array(dev_y[start:start + 16]))
            s, c = row_losses(forward_from(model, core(h), GRAFT_AFTER + 1), y)
            sums[start:start + h.shape[0]] = s.double().numpy()
            counts[start:start + h.shape[0]] = c.numpy()
    if not np.array_equal(counts, base["count"]):
        raise RuntimeError("dev token counts differ from the baseline cache")
    diff = sums - base["sum"]
    delta = float(diff.sum() / max(1, counts.sum()))
    edges_now = (F.softplus(core.edge_logit.detach()) * core.mask)
    result = {
        "schema": "supermix-v92-run-v1",
        "spec": spec,
        "steps": steps,
        "lr": lr,
        "signed_radius_target": signed_radius,
        "wiring": info,
        "dev_token_loss": float(sums.sum() / max(1, counts.sum())),
        "baseline_dev_token_loss": float(base["sum"].sum() / max(1, base["count"].sum())),
        "delta_vs_v89": delta,
        "delta_ci95": bootstrap_ci(diff, counts),
        "rows_better": int((diff < -1e-9).sum()),
        "rows_worse": int((diff > 1e-9).sum()),
        "gate_mean_abs": float(core.gate.detach().abs().mean()),
        "gate_max_abs": float(core.gate.detach().abs().max()),
        "mean_leak": float(torch.sigmoid(core.leak_logit.detach()).mean()),
        "edge_mean_now": float(edges_now.sum() / core.mask.sum().clamp_min(1)),
        "history": history,
        "seconds": round(time.time() - started, 1),
    }
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{spec['name']}.rows.npz"), sum=sums, count=counts)
    torch.save({"spec": spec, "state_dict": core.state_dict(), "info": info},
               os.path.join(out_dir, f"{spec['name']}.core.pt"))
    with open(os.path.join(out_dir, f"{spec['name']}.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def run_plan(plan: Sequence[Dict[str, Any]] = PLAN, **kwargs) -> None:
    from train_mimomix_talk import load_talk_checkpoint

    model, _, _ = load_talk_checkpoint(kwargs.get("checkpoint", V89))
    out_dir = kwargs.get("out_dir", RUNS)
    for spec in plan:
        if os.path.exists(os.path.join(out_dir, f"{spec['name']}.json")):
            print(f"skip {spec['name']} (done)", flush=True)
            continue
        result = run_one(spec, model=model, **kwargs)
        print(f"== {spec['name']} delta {result['delta_vs_v89']:+.6f} CI {result['delta_ci95']} "
              f"gate {result['gate_mean_abs']:.4f} ({result['seconds']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# The pre-registered readout
# ---------------------------------------------------------------------------


def report(out_dir: str = RUNS, output: Optional[str] = None) -> Dict[str, Any]:
    results = {}
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".json"):
            with open(os.path.join(out_dir, name), encoding="utf-8") as handle:
                r = json.load(handle)
            results[r["spec"]["name"]] = r
    real = [results[k]["delta_vs_v89"] for k in results if k.startswith("T_real")]
    nulls = [results[k]["delta_vs_v89"] for k in results if k.startswith("T_null")]
    out: Dict[str, Any] = {"schema": "supermix-v92-report-v1", "runs": {
        k: {"delta": v["delta_vs_v89"], "ci95": v["delta_ci95"], "gate": v["gate_mean_abs"]}
        for k, v in results.items()}}
    if real:
        out["H1_branch_helps_frozen_v89"] = {
            "rule": "every T_real seed has delta < 0 with its 95% CI below 0",
            "holds": all(results[k]["delta_ci95"][1] < 0 for k in results if k.startswith("T_real")),
        }
    if len(real) >= 2 and len(nulls) >= 3:
        mean_real, mean_null = float(np.mean(real)), float(np.mean(nulls))
        sd_null = float(np.std(nulls, ddof=1))
        gap = mean_null - mean_real
        out["H2_fly_wiring_beats_nulls"] = {
            "rule": "(mean_null - mean_real) > 2 sd_null AND > (max_real - min_real)",
            "mean_real": mean_real, "mean_null": mean_null, "sd_null": sd_null,
            "real_spread": float(max(real) - min(real)), "gap": gap,
            "z": gap / sd_null if sd_null > 0 else None,
            "real_rank_among_nulls": [int(sum(n <= r for n in nulls)) for r in real],
            "n_real": len(real), "n_null": len(nulls),
            "holds": bool(gap > 2 * sd_null and gap > (max(real) - min(real))),
        }
    for label, a, b in (("H3_memory_vs_per_token", "T_real_s1", "P_real_s1"),
                        ("H4_wiring_vs_empty", "T_real_s1", "T_empty_s1")):
        if a in results and b in results:
            ra = np.load(os.path.join(out_dir, f"{a}.rows.npz"))
            rb = np.load(os.path.join(out_dir, f"{b}.rows.npz"))
            diff = ra["sum"] - rb["sum"]
            out[label] = {
                "diff": float(diff.sum() / max(1, ra["count"].sum())),
                "ci95": bootstrap_ci(diff, ra["count"]),
                "note": f"{a} minus {b}; negative means {a} has lower dev loss",
            }
    if output:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("nulls")
    p = sub.add_parser("prepare")
    p.add_argument("--train_batches", type=int, default=1000)
    p.add_argument("--dev_rows", type=int, default=2000)
    r = sub.add_parser("run")
    r.add_argument("--steps", type=int, default=1000)
    r.add_argument("--only", nargs="*", default=None, help="run names from PLAN")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "nulls":
        receipt = build_nulls()
        print(json.dumps({k: {m: v.get(m) for m in ("edge_overlap_with_real", "signed_spectral_radius",
                                                    "reciprocity", "modules_with_changed_inhibitory_input_count")}
                          for k, v in receipt["diagnostics"].items()}, indent=1))
    elif args.command == "prepare":
        print(json.dumps(prepare(train_batches=args.train_batches, dev_rows=args.dev_rows), indent=2))
    elif args.command == "run":
        plan = [s for s in PLAN if not args.only or s["name"] in args.only]
        run_plan(plan, steps=args.steps)
    elif args.command == "report":
        print(json.dumps(report(output="output/v92_connectome/report.json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
