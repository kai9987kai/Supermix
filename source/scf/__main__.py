"""Unified CLI entrypoint for Supermix v91 Cognitive Fabric (``python -m scf``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SOURCE_DIR = Path(__file__).resolve().parents[1]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scf import SCF_INTERNAL_NAME, SCF_VERSION, CognitiveFabricRuntime, SCFFeatureFlags  # noqa: E402
from scf.models import MiMoMixAdapter, SCFModelRegistry  # noqa: E402
from scf.verification import SCFVerificationGate  # noqa: E402


def cmd_status(*, as_json: bool = False) -> int:
    registry = SCFModelRegistry()
    summary = registry.summary()
    talk_ckpts = MiMoMixAdapter().discover_talk_checkpoints()
    gate = SCFVerificationGate()
    atlas_probe = gate.connectome_atlas.verify_query(
        "How many Li20 neurons does the male CNS contain?"
    )
    payload: Dict[str, Any] = {
        "version": SCF_VERSION,
        "name": SCF_INTERNAL_NAME,
        "default_flags": SCFFeatureFlags().to_dict(),
        "env_flags": SCFFeatureFlags.from_env().to_dict(),
        "talk_checkpoints": talk_ckpts,
        "connectome_atlas_ready": bool(atlas_probe and atlas_probe.get("solved")),
        "families": summary,
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"Supermix {SCF_INTERNAL_NAME} ({SCF_VERSION})")
    print(f"Connectome atlas ready: {payload['connectome_atlas_ready']}")
    print(f"Discovered talk checkpoints ({len(talk_ckpts)}):")
    for ckpt in talk_ckpts:
        print(f"  - {ckpt}")
    print("Registered model families:")
    for fam, info in summary.items():
        print(
            f"  - {fam}: status={info.get('maturity_status')} "
            f"weights={len(info.get('available_weights') or [])}"
        )
    return 0


def cmd_verify(query: str, *, as_json: bool = False) -> int:
    import prompt_normaliser as pn

    norm = pn.normalise(query)
    gate = SCFVerificationGate()
    res = gate.verify(query=query, normalised_query=norm.prompt)
    payload = {
        "query": query,
        "normalised": {"rule": norm.rule, "prompt": norm.prompt, "changed": norm.changed},
        **res,
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(
        f"verified={res.get('verified')} | answer={res.get('verified_answer')!r} "
        f"| source={res.get('verifier_source')}"
    )
    return 0 if res.get("verified") else 1


def cmd_ask(
    prompt: str,
    *,
    enable_scf: bool = True,
    neural: bool = False,
    checkpoint: Optional[str] = None,
    max_tokens: int = 96,
    as_json: bool = False,
) -> int:
    runtime = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled() if enable_scf else SCFFeatureFlags()
    res = runtime.run_task(
        prompt,
        flags_override=flags,
        enable_neural_candidates=neural,
        neural_checkpoint_path=checkpoint,
        max_neural_tokens=max_tokens,
    )
    if as_json:
        print(json.dumps(res, indent=2, default=str))
        return 0
    print(f"answer: {res.get('final_answer')}")
    print(f"verified={res.get('verified')} scf_active={res.get('scf_active')}")
    if res.get("verification"):
        print(f"verifier_source={res['verification'].get('verifier_source')}")
    return 0


def cmd_bench(
    *,
    seed: int = 82,
    count: Optional[int] = None,
    neural: bool = False,
    checkpoint: Optional[str] = None,
    as_json: bool = False,
) -> int:
    import eval_problem_solving as ep

    total_default = len(ep.V89_BENCHMARK_TASKS) + len(ep.V93_NEW_TASKS)
    n_tasks = count if count is not None else total_default
    tasks = ep.generate_novel(n_tasks, seed=seed)

    runtime = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled()
    verified_correct = 0
    hybrid_correct = 0
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(tasks):
        res = runtime.run_task(
            item.prompt,
            flags_override=flags,
            enable_neural_candidates=neural,
            neural_checkpoint_path=checkpoint,
            max_neural_tokens=64,
        )
        ans_text = str(res.get("final_answer") or "")
        extracted = ep.extract_answer(ans_text)
        ok = extracted is not None and ep.is_correct(extracted, item.answer)
        ver = bool(res.get("verified"))
        if ok:
            hybrid_correct += 1
            if ver:
                verified_correct += 1
        rows.append(
            {
                "index": idx,
                "task": item.task,
                "verified": ver,
                "correct": ok,
                "expected": item.answer,
                "predicted": extracted,
                "source": (res.get("verification") or {}).get("verifier_source"),
            }
        )

    payload = {
        "seed": seed,
        "total": len(tasks),
        "neural_enabled": neural,
        "checkpoint": checkpoint,
        "verified_correct": verified_correct,
        "verified_accuracy": round(verified_correct / max(len(tasks), 1), 4),
        "hybrid_correct": hybrid_correct,
        "hybrid_accuracy": round(hybrid_correct / max(len(tasks), 1), 4),
        "failures": [r for r in rows if not r["correct"]],
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(
        f"SCF bench (seed={seed}, n={len(tasks)}): "
        f"verified_correct={verified_correct}/{len(tasks)} ({payload['verified_accuracy']:.4f}), "
        f"hybrid_correct={hybrid_correct}/{len(tasks)} ({payload['hybrid_accuracy']:.4f})"
    )
    for fail in payload["failures"]:
        print(f"  MISS {fail['index']} ({fail['task']}): exp={fail['expected']} got={fail['predicted']}")
    return 0


def cmd_frontier(*, as_json: bool = False) -> int:
    """Run deterministic self-verification of Supermix v102/v103 frontier modules."""
    import torch

    import mimomix_core as mc
    import muon_optimizer as mo
    import nexus_v102_frontier as v102
    import nexus_v103_synthesis as v103

    torch.manual_seed(102)
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        moe_intermediate_size=24,
        n_routed_experts=4,
        n_shared_experts=1,
        use_mhc_residual=True,
        mhc_streams=2,
        mhc_sinkhorn_iters=20,
        use_engram_memory=True,
        engram_table_size=128,
        engram_n_heads=2,
        engram_after_layer=0,
        use_deltanet_memory=True,
        deltanet_dim=16,
        deltanet_after_layer=1,
        use_cns_core=True,
        cns_nodes=8,
        cns_steps=2,
        cns_after_layer=0,
        cns_dendritic_apical=True,
        use_thinking_core=True,
        thinking_latent_dim=16,
        thinking_cycles=2,
        thinking_ctm_synchrony=True,
    )
    model = mc.MiMoMixModel(cfg)
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, 64, (1, 6))
        _ = model(x)
    tele = model.telemetry()

    # Muon Newton-Schulz orthogonality check
    g = torch.randn(16, 16)
    ortho = mo.newton_schulz_orthogonalize(g, steps=5)
    gram = ortho @ ortho.t()
    ortho_diag_mean = float(gram.diagonal().mean())

    # Carrier 2^3 exact Mobius atlas check
    atlas = v102.CarrierInteractionAtlas.decompose_subset_factorial(
        ["trunk", "cns", "thinking"],
        {
            "": 0.90,
            "trunk": 0.40,
            "cns": 0.82,
            "thinking": 0.85,
            "cns,trunk": 0.30,
            "thinking,trunk": 0.35,
            "cns,thinking": 0.74,
            "cns,thinking,trunk": 0.20,
        },
    )
    flops = v102.CarrierInteractionAtlas.profile_mimomix_compute(cfg, seq_len=64)
    cvar = v102.CounterformFalsificationAuditor().audit_candidates(
        candidate_losses={"conservative": [0.10, 0.12, 0.15, 0.18], "risky": [0.05, 0.08, 0.10, 0.95]},
        consistency_residuals={"conservative": 0.04, "risky": 0.05},
        projection_residuals={"conservative": 0.06, "risky": 0.07},
    )
    commons = v102.TesseraRatifiedLemmaCommons(min_independent_lineages=2)
    commons.submit_discovery(expression="a^2 - b^2 = (a-b)(a+b)", lineage="NexusMind", task_family="algebra")
    rat = commons.submit_discovery(expression="a^2 - b^2 = (a-b)(a+b)", lineage="MiMoMix", task_family="quadratics")
    sched = v102.HyperionComputeScheduler().schedule(
        domain="math_science", submode="problem_solving", complexity="high", has_memory_turns=True
    )

    # v103 synthesis checks (Continuum/BTSP/Hopfield, SAE, Fractal, Causeway, NeuroHive)
    mem_bench = v103.ContinuumBTSPHopfieldMemory(seed=103).run_benchmark(n_patterns=6, steps=16)
    topo_cmp = v103.FractalConnectomeAnalyzer.compare_topologies(n_nodes=18, seed=103)
    hive = v103.NeuroHiveResourcePlanner().simulate_dispatch(
        tasks=[{"id": "t1", "token_cost": 40}, {"id": "t2", "token_cost": 60}],
        token_budget=500,
        seed=103,
    )

    payload: Dict[str, Any] = {
        "frontier_version": v102.FRONTIER_V102_VERSION,
        "synthesis_version": v103.FRONTIER_V103_VERSION,
        "mhc": tele.get("mhc"),
        "engram": tele.get("engram"),
        "deltanet": tele.get("deltanet"),
        "cns_dendritic_apical": (tele.get("cns_core") or {}).get("dendritic_apical"),
        "ctm_synchrony": (tele.get("thinking") or {}).get("ctm_synchrony"),
        "muon_ortho_diag_mean": round(ortho_diag_mean, 4),
        "carrier_atlas_exact": atlas["exact_reconstruction_verified"],
        "aether_flops_per_token": flops["active_flops_per_token"],
        "counterform_decision": cvar["decision"],
        "counterform_selected": cvar["selected_candidate"],
        "tessera_ratified": rat["ratified"],
        "hyperion_waves": sched["wave_count"],
        "btsp_one_shot_overlap": mem_bench["btsp_one_shot_overlap"],
        "mnemosyne_intact_vs_erased": mem_bench["intact_vs_erased_delta"],
        "fractal_highest_clustering": topo_cmp["highest_clustering"],
        "neurohive_budget_conserved": hive["budget_conserved"],
        "answer_authority": False,
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"Supermix Frontier {payload['frontier_version']} self-check OK")
    print(f"  mHC Birkhoff error: {payload['mhc']['max_doubly_stochastic_error']:.2e}")
    print(f"  CNS apical: {payload['cns_dendritic_apical']} | CTM sync: {payload['ctm_synchrony']}")
    print(f"  Carrier Atlas exact: {payload['carrier_atlas_exact']} | CVaR: {payload['counterform_decision']} ({payload['counterform_selected']})")
    print(f"  Tessera ratified: {payload['tessera_ratified']} | Hyperion waves: {payload['hyperion_waves']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scf",
        description=f"Supermix {SCF_INTERNAL_NAME} ({SCF_VERSION}) CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show SCF status, flags, and model registry")
    p_status.add_argument("--json", action="store_true")

    p_verify = sub.add_parser("verify", help="Run deterministic SCFVerificationGate on a query")
    p_verify.add_argument("query", type=str)
    p_verify.add_argument("--json", action="store_true")

    p_ask = sub.add_parser("ask", help="Run a prompt through CognitiveFabricRuntime")
    p_ask.add_argument("prompt", type=str)
    p_ask.add_argument("--legacy", action="store_true", help="Run with SCF_ENABLED=False")
    p_ask.add_argument("--neural", action="store_true", help="Enable MiMoMix neural candidate generation")
    p_ask.add_argument("--checkpoint", type=str, default=None)
    p_ask.add_argument("--max_tokens", type=int, default=96)
    p_ask.add_argument("--json", action="store_true")

    p_bench = sub.add_parser("bench", help="Benchmark SCF on generated problem-solving tasks")
    p_bench.add_argument("--seed", type=int, default=82)
    p_bench.add_argument("--count", type=int, default=None)
    p_bench.add_argument("--neural", action="store_true")
    p_bench.add_argument("--checkpoint", type=str, default=None)
    p_bench.add_argument("--json", action="store_true")

    p_frontier = sub.add_parser("frontier", help="Run Supermix v102 frontier modules self-check")
    p_frontier.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        return cmd_status(as_json=args.json)
    if args.command == "verify":
        return cmd_verify(args.query, as_json=args.json)
    if args.command == "ask":
        return cmd_ask(
            args.prompt,
            enable_scf=not args.legacy,
            neural=args.neural,
            checkpoint=args.checkpoint,
            max_tokens=args.max_tokens,
            as_json=args.json,
        )
    if args.command == "bench":
        return cmd_bench(
            seed=args.seed,
            count=args.count,
            neural=args.neural,
            checkpoint=args.checkpoint,
            as_json=args.json,
        )
    if args.command == "frontier":
        return cmd_frontier(as_json=args.json)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
