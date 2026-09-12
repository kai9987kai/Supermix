"""Capture or compare reference forward-pass outputs for mimomix_core.

Usage:
    python reference_logits.py capture  <source_dir> <out.json>
    python reference_logits.py compare  <source_dir> <ref.json>

The point is an oracle I control: the agents editing mimomix_core.py report
their own regression checks, and a self-report is not evidence. This builds the
same tensors from a frozen pre-edit copy and from the live tree and compares
them, so "the default config is unchanged" becomes a number rather than a claim.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch


def build_and_run(source_dir: str):
    for mod in [m for m in list(sys.modules) if m.startswith(("mimomix", "train_mimomix"))]:
        del sys.modules[mod]
    sys.path.insert(0, source_dir)
    mc = importlib.import_module("mimomix_core")

    out = {"source": source_dir, "cases": {}}

    # Case 1: the library default config.
    # Case 2: the v80 training shape.
    cases = {
        "default": {},
        "v80_shape": dict(
            vocab_size=8570, hidden_size=256, n_layers=4, n_heads=8, n_kv_heads=2,
            intermediate_size=384, moe_intermediate_size=96, n_routed_experts=48,
            n_shared_experts=1, moe_top_k=2, sliding_window=64, hybrid_ratio=3,
            n_mtp_layers=2, mtp_loss_weight=0.3, use_thinking_core=True,
            thinking_cycles=2, thinking_max_cycles=4, native_context=128,
            max_position_embeddings=128, rope_scaling="none",
        ),
    }

    for name, overrides in cases.items():
        torch.manual_seed(1234)
        cfg = mc.MiMoMixConfig(**overrides)
        model = mc.MiMoMixModel(cfg)
        model.eval()
        g = torch.Generator().manual_seed(99)
        seq = 32 if name == "default" else 64
        x = torch.randint(0, cfg.vocab_size, (2, seq), generator=g)
        y = torch.randint(0, cfg.vocab_size, (2, seq), generator=g)
        with torch.no_grad():
            res = model(x, labels=y)
        logits = res.logits if hasattr(res, "logits") else res[0]
        params = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        out["cases"][name] = {
            "layout": list(mc.attention_layout(cfg.n_layers, cfg.hybrid_ratio, cfg.final_layer_global)),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "param_sha": hashlib.sha256(params.numpy().tobytes()).hexdigest()[:32],
            "param_sum": float(params.sum()),
            "logits_shape": list(logits.shape),
            "logits_sum": float(logits.sum()),
            "logits_absmax": float(logits.abs().max()),
            "logits_first16": [round(float(v), 6) for v in logits.reshape(-1)[:16]],
            "loss": float(res.loss) if getattr(res, "loss", None) is not None else None,
            "config": cfg.to_dict(),
        }
        print(f"  {name}: params {out['cases'][name]['n_params']:,}  "
              f"loss {out['cases'][name]['loss']}  layout {out['cases'][name]['layout']}")
    sys.path.remove(source_dir)
    return out


def main() -> int:
    mode, source_dir, path = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"{mode} from {source_dir}")
    got = build_and_run(source_dir)
    if mode == "capture":
        Path(path).write_text(json.dumps(got, indent=1), encoding="utf-8")
        print("wrote", path)
        return 0

    ref = json.loads(Path(path).read_text(encoding="utf-8"))
    ok = True
    for name, r in ref["cases"].items():
        g = got["cases"].get(name)
        if g is None:
            print(f"  {name}: MISSING in current tree")
            ok = False
            continue
        same_params = g["param_sha"] == r["param_sha"]
        dl = abs(g["logits_sum"] - r["logits_sum"])
        dloss = (abs((g["loss"] or 0) - (r["loss"] or 0)))
        first = max((abs(a - b) for a, b in zip(g["logits_first16"], r["logits_first16"])), default=0.0)
        verdict = "IDENTICAL" if (same_params and first < 1e-6 and dloss < 1e-6) else "CHANGED"
        if verdict == "CHANGED":
            ok = False
        print(f"  {name}: {verdict}")
        print(f"      params {r['n_params']:,} -> {g['n_params']:,}   init identical: {same_params}")
        print(f"      loss   {r['loss']} -> {g['loss']}   (delta {dloss:.3e})")
        print(f"      logits sum delta {dl:.3e}   max |delta| over first 16: {first:.3e}")
        if r["layout"] != g["layout"]:
            print(f"      LAYOUT CHANGED {r['layout']} -> {g['layout']}")
        newk = set(g["config"]) - set(r["config"])
        goner = set(r["config"]) - set(g["config"])
        if newk:
            print(f"      new config fields: {sorted(newk)}")
        if goner:
            print(f"      REMOVED config fields: {sorted(goner)}")
        changed = {k: (r["config"][k], g["config"][k]) for k in set(r["config"]) & set(g["config"])
                   if r["config"][k] != g["config"][k]}
        if changed:
            print(f"      CHANGED DEFAULTS: {changed}")
    print("\nRESULT:", "default behaviour preserved" if ok else "DEFAULT BEHAVIOUR CHANGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
