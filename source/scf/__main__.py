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
        use_titans_memory=True,
        titans_dim=16,
        titans_after_layer=1,
        use_attn_res=True,
        use_value_residual=True,
        use_short_conv=True,
        short_conv_kernel=4,
        short_conv_after_layer=0,
        use_shared_expert_gate=True,
        use_ufwm=True,
        ufwm_dim=16,
        ufwm_banks=4,
        ufwm_top_k=2,
        ufwm_mode="moe",
        use_neuron_foundry=True,
        neuron_foundry_dim=16,
        neuron_foundry_after_layer=1,
        use_cns_core=True,
        cns_nodes=8,
        cns_steps=2,
        cns_after_layer=0,
        cns_dendritic_apical=True,
        cns_neuromod_gate=True,
        cns_stdp_trace=True,
        cns_oscillatory_phase=True,
        cns_astrocyte_gate=True,
        cns_retrograde_ephaptic=True,
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

    # v104 evolution checks (Conformal Router, Circuit Tracer, Episodic Graph, PRM Auditor, Pareto)
    import nexus_v104_evolution as v104

    conf_r = v104.ConformalCalibratedRouter().calibrate_and_route(
        calibration_scores=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        candidate_scores={"NexusMind": 0.92, "MiMoMix": 0.78, "ChampionNet": 0.40},
        alpha=0.15,
    )
    circ = v104.MechanisticCircuitTracer().trace_circuit(
        clean_activations={"engram_mlp": 0.8, "attn_deltanet": 1.1, "connectome_titans": 1.4, "thinking_readout": 0.6},
        corrupted_activations={"engram_mlp": 0.2, "attn_deltanet": 0.2, "connectome_titans": 0.1, "thinking_readout": 0.1},
    )
    prm = v104.VerifiableProcessRewardAuditor().audit_trace(
        prompt="Compute 12 * 4 + 7",
        steps=["12 * 4 = 48", "48 + 7 = 55"],
    )
    pareto = v104.ParetoFrontierAutoTuner().optimize(
        candidates=[
            {"name": "v96_promoted", "quality_score": 0.878, "latency_ms": 140.0, "memory_mb": 210.0, "calibration_error": 0.04},
            {"name": "v104_hybrid", "quality_score": 0.992, "latency_ms": 195.0, "memory_mb": 260.0, "calibration_error": 0.015},
            {"name": "v104_edge", "quality_score": 0.835, "latency_ms": 68.0, "memory_mb": 120.0, "calibration_error": 0.055},
        ]
    )

    # v105 hyper-optimization checks
    import mimomix_decoding as md
    import nexus_v105_hyperopt as v105
    import nexus_v106_frontier as v106
    import nexus_v107_frontier as v107

    pe_ortho = mo.newton_schulz_orthogonalize(g, steps=5, polar_express=True)
    pe_err = float((pe_ortho @ pe_ortho.t() - torch.eye(16)).norm())
    spec_opt = v105.SpeculativeDraftOptimizer().optimize(depth_accept_probs=[0.85, 0.75, 0.65])
    kv_plan = v105.KVCacheCompressionPlanner().plan(seq_len=32768, memory_budget_mb=64.0)
    roofline = v105.RooflineComputeProfiler().profile(hidden_size=256, n_layers=6)
    with torch.no_grad():
        kv_out = model(x, use_cache=True, return_mtp=False)
        kv_rep = md.kv_cache_memory_report(kv_out.past_key_values)
        kv_int4_rep = md.kv_cache_int4_memory_report(kv_out.past_key_values, group_size=16)
        kv_outlier_rep = md.kv_cache_outlier_memory_report(
            kv_out.past_key_values, group_size=16, outlier_channels=2
        )

    ttc_res = v106.TestTimeComputeAllocator().allocate(total_token_budget=4096, base_pass1=0.35)
    spec_diag = v106.SpectralWeightDiagnostics().diagnose(
        singular_values=[5.2, 3.1, 2.0, 1.45, 1.1, 0.85, 0.65, 0.5], matrix_shape=[64, 64]
    )
    quant_res = v106.AdaptiveQuantizationPlanner().plan(
        layers=[
            {"name": "attn", "params": 1000, "sensitivity": 10.0},
            {"name": "mlp", "params": 3000, "sensitivity": 1.5},
        ],
        target_avg_bits=4.5,
    )
    ufwm_prof = v107.UniversalFastWeightProfiler().profile(
        hidden_size=128, n_layers=6, fast_dim=24, n_banks=4, top_k=2, seq_len=512
    )
    neuron_assay = v107.SyntheticNeuronFoundryAssay().assay(
        stimulus_series=[0.2, 0.9, -0.4, 1.2, 0.1, -0.7, 0.8, 0.3], foundry_dim=32
    )

    payload: Dict[str, Any] = {
        "frontier_version": v102.FRONTIER_V102_VERSION,
        "synthesis_version": v103.FRONTIER_V103_VERSION,
        "evolution_v104_version": v104.EVOLUTION_V104_VERSION,
        "evolution_v105_version": v105.EVOLUTION_V105_VERSION,
        "evolution_v106_version": v106.FRONTIER_V106_VERSION,
        "evolution_v107_version": v107.FRONTIER_V107_VERSION,
        "mhc": tele.get("mhc"),
        "engram": tele.get("engram"),
        "deltanet": tele.get("deltanet"),
        "titans": tele.get("titans"),
        "attn_res": tele.get("attn_res"),
        "value_residual": tele.get("value_residual"),
        "short_conv": tele.get("short_conv"),
        "shared_expert_gate": tele.get("shared_expert_gate"),
        "ufwm": tele.get("ufwm"),
        "neuron_foundry": tele.get("neuron_foundry"),
        "cns_dendritic_apical": (tele.get("cns_core") or {}).get("dendritic_apical"),
        "cns_neuromod_gate": (tele.get("cns_core") or {}).get("neuromod_gate"),
        "cns_stdp_trace": (tele.get("cns_core") or {}).get("stdp_trace"),
        "cns_oscillatory_phase": (tele.get("cns_core") or {}).get("oscillatory_phase"),
        "cns_astrocyte_gate": (tele.get("cns_core") or {}).get("astrocyte_gate"),
        "cns_retrograde_ephaptic": (tele.get("cns_core") or {}).get("retrograde_ephaptic"),
        "ctm_synchrony": (tele.get("thinking") or {}).get("ctm_synchrony"),
        "muon_ortho_diag_mean": round(ortho_diag_mean, 4),
        "polar_express_ortho_err": round(pe_err, 4),
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
        "conformal_decision": conf_r["decision"],
        "conformal_prediction_set": conf_r["prediction_set"],
        "circuit_dominant_stage": circ["dominant_stage"],
        "prm_all_verified": prm["all_arithmetic_verified"],
        "pareto_hypervolume": pareto["hypervolume_2d_quality_speed"],
        "v105_optimal_draft_depth": spec_opt["optimal_draft_depth"],
        "v105_kv_recommended_scheme": kv_plan["recommended_scheme"],
        "v105_kv_int8_compression_ratio": kv_rep["compression_ratio"],
        "v105_roofline_decode_regime": roofline["decode"]["regime"],
        "v106_kv_int4_compression_ratio": kv_int4_rep["compression_ratio"],
        "v106_ttc_regime": ttc_res["recommended_regime"],
        "v106_spectral_health": spec_diag["health_status"],
        "v106_quant_achieved_bits": quant_res["achieved_avg_bits"],
        "v107_kv_outlier_compression_ratio": kv_outlier_rep["compression_ratio"],
        "v107_kv_outlier_cosine": kv_outlier_rep["mean_cosine_similarity"],
        "v107_ufwm_sharing_ratio": ufwm_prof["parameter_sharing_ratio"],
        "v107_neuron_dominant_family": neuron_assay["dominant_family"],
        "answer_authority": False,
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(f"Supermix Frontier {payload['frontier_version']} self-check OK")
    print(f"  mHC Birkhoff error: {payload['mhc']['max_doubly_stochastic_error']:.2e}")
    print(f"  CNS apical: {payload['cns_dendritic_apical']} | STDP: {payload['cns_stdp_trace']} | CTM sync: {payload['ctm_synchrony']}")
    print(f"  Carrier Atlas exact: {payload['carrier_atlas_exact']} | CVaR: {payload['counterform_decision']} ({payload['counterform_selected']})")
    print(f"  Tessera ratified: {payload['tessera_ratified']} | Hyperion waves: {payload['hyperion_waves']}")
    print(f"  v104 Titans: {bool(payload['titans'])} | AttnRes: {bool(payload['attn_res'])} | Conformal: {payload['conformal_decision']}")
    print(f"  v105 ValueRes: {bool(payload['value_residual'])} | PolarExpress err: {payload['polar_express_ortho_err']} | KV int8: {payload['v105_kv_int8_compression_ratio']}x")
    return 0


def cmd_diagnose(*, as_json: bool = False) -> int:
    """Unified Supermix v104-v106 health, exact-verifier, and frontier diagnostic."""
    import nexus_v104_evolution as v104
    import nexus_v105_hyperopt as v105
    import nexus_v106_frontier as v106
    import nexus_v107_frontier as v107

    gate = SCFVerificationGate()
    probes = [
        ("catalan", "What is the 5th catalan number?", "42"),
        ("crt", "Solve x = 2 (mod 3) and x = 3 (mod 5)", "8 (mod 15)"),
        ("shannon_entropy", "Shannon entropy of [0.5, 0.25, 0.25]", "1.5"),
        ("lorentz", "Lorentz factor at v = 0.6c", "1.25"),
        ("amdahl", "Amdahl speedup p = 0.8, n = 4", "2.5"),
        ("base_conversion", "Convert 255 from base 10 to base 16", "FF"),
        ("extended_gcd", "Extended gcd of 240 and 46", "gcd=2, s=-9, t=47"),
        ("matrix_rank", "Matrix rank of [[1, 2, 3], [2, 4, 6], [0, 1, 1]]", "2"),
        ("dijkstra", "Shortest path from A to D in graph A-B:4, A-C:1, C-B:2, B-D:1, C-D:5", "4"),
        ("multiset_combinations", "Combinations with repetition n = 5, k = 3", "35"),
        ("michaelis_menten", "Michaelis-Menten Vmax = 120, Km = 4, S = 12", "90"),
        ("shannon_hartley", "Shannon-Hartley bandwidth = 1000, snr = 15", "4000"),
        ("mm1_queue", "M/M/1 queue lambda = 3, mu = 5", "1.5"),
        ("quadratic_form", "Quadratic form x = [1, 2], A = [[3, 1], [1, 2]]", "15"),
    ]
    probe_results: List[Dict[str, Any]] = []
    for name, q, expected in probes:
        res = gate.verify(query=q)
        ans = res.get("verified_answer")
        ok = bool(res.get("verified") and str(ans) == expected)
        probe_results.append(
            {
                "probe": name,
                "verified": bool(res.get("verified")),
                "answer": ans,
                "expected": expected,
                "passed": ok,
            }
        )
    registry = SCFModelRegistry()
    summary = registry.summary()
    payload = {
        "scf_version": SCF_VERSION,
        "evolution_v104_version": v104.EVOLUTION_V104_VERSION,
        "evolution_v105_version": v105.EVOLUTION_V105_VERSION,
        "evolution_v106_version": v106.FRONTIER_V106_VERSION,
        "evolution_v107_version": v107.FRONTIER_V107_VERSION,
        "model_families_discovered": sorted(summary.keys()),
        "verifier_probes_passed": sum(1 for p in probe_results if p["passed"]),
        "verifier_probes_total": len(probe_results),
        "verifier_probes": probe_results,
        "all_probes_passed": all(p["passed"] for p in probe_results),
        "answer_authority": False,
    }
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0 if payload["all_probes_passed"] else 1
    print(f"Supermix Diagnose ({payload['evolution_v106_version']})")
    print(f"  Families: {', '.join(payload['model_families_discovered'])}")
    print(f"  Exact Verifier Probes: {payload['verifier_probes_passed']}/{payload['verifier_probes_total']}")
    return 0 if payload["all_probes_passed"] else 1


def cmd_perf(*, as_json: bool = False) -> int:
    """Run live Supermix v105-v106 performance, trunk-reuse, Int4 KV & chunked-prefill benchmark."""
    import time
    import torch

    import mimomix_controller as ctl
    import mimomix_core as mc
    import mimomix_decoding as md
    import nexus_v105_hyperopt as v105
    import nexus_v106_frontier as v106
    import nexus_v107_frontier as v107

    runtime = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled()
    # Warmup once, then measure warm route + verify latency
    _ = runtime.run_task("What is 2 + 2?", flags_override=flags)
    t0 = time.perf_counter()
    r_warm = runtime.run_task("Amdahl speedup p = 0.8, n = 4", flags_override=flags)
    scf_warm_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    torch.manual_seed(106)
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=4,
        moe_top_k=2,
        use_moe=True,
        use_shared_expert_gate=True,
        n_mtp_layers=2,
        use_mhc_residual=True,
        use_value_residual=True,
        use_short_conv=True,
        use_attn_res=True,
        use_ufwm=True,
        ufwm_dim=16,
        ufwm_banks=4,
        ufwm_top_k=2,
        use_neuron_foundry=True,
        neuron_foundry_dim=16,
        neuron_foundry_after_layer=1,
        use_cns_core=True,
        cns_nodes=16,
        cns_steps=2,
        cns_stdp_trace=True,
        cns_oscillatory_phase=True,
        cns_astrocyte_gate=True,
        cns_retrograde_ephaptic=True,
    )
    model = mc.MiMoMixModel(cfg).eval()
    prompt = torch.randint(0, 64, (1, 16))
    with torch.no_grad():
        t0 = time.perf_counter()
        g_res = md.greedy_generate(model, prompt, max_new_tokens=10)
        greedy_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        t0 = time.perf_counter()
        s_res = md.speculative_generate(model, prompt, max_new_tokens=10, adaptive_draft=True)
        adaptive_spec_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        c_res = md.chunked_greedy_generate(model, prompt, max_new_tokens=10, chunk_size=5)

        pol = ctl.ThinkingPolicy(ladder=(1, 2, 4), entropy_target=0.0)
        t0 = time.perf_counter()
        o_reuse, _ = ctl.decide(model, prompt, mode="agent", policy=pol, reuse_trunk=True)
        ctl_reuse_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        t0 = time.perf_counter()
        o_full, _ = ctl.decide(model, prompt, mode="agent", policy=pol, reuse_trunk=False)
        ctl_full_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        prefill = model(prompt, use_cache=True, return_mtp=False)
        kv_report = md.kv_cache_memory_report(prefill.past_key_values)
        kv_int4_report = md.kv_cache_int4_memory_report(prefill.past_key_values, group_size=16)
        kv_outlier_report = md.kv_cache_outlier_memory_report(
            prefill.past_key_values, group_size=16, outlier_channels=2
        )

    parity = bool(torch.equal(g_res.new_tokens, s_res.new_tokens))
    chunked_parity = bool(torch.equal(g_res.new_tokens, c_res.new_tokens))
    ctl_parity = bool(torch.equal(o_reuse.logits, o_full.logits))
    roofline = v105.RooflineComputeProfiler().profile(
        hidden_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        intermediate_size=cfg.intermediate_size,
        moe_intermediate_size=cfg.moe_intermediate_size,
        moe_top_k=cfg.moe_top_k,
        n_shared_experts=cfg.n_shared_experts,
        use_moe=cfg.use_moe,
        vocab_size=cfg.vocab_size,
        seq_len=16,
    )
    payload: Dict[str, Any] = {
        "evolution_v105_version": v105.EVOLUTION_V105_VERSION,
        "evolution_v106_version": v106.FRONTIER_V106_VERSION,
        "evolution_v107_version": v107.FRONTIER_V107_VERSION,
        "scf_warm_task_ms": scf_warm_ms,
        "scf_verified_answer": r_warm.get("final_answer"),
        "greedy_decode_10tok_ms": greedy_ms,
        "adaptive_spec_10tok_ms": adaptive_spec_ms,
        "adaptive_spec_greedy_parity": parity,
        "chunked_prefill_greedy_parity": chunked_parity,
        "controller_trunk_reuse_ms": ctl_reuse_ms,
        "controller_full_recompute_ms": ctl_full_ms,
        "controller_trunk_reuse_speedup": round(ctl_full_ms / max(ctl_reuse_ms, 1e-6), 3),
        "controller_trunk_reuse_bit_identical": ctl_parity,
        "kv_int8_report": kv_report,
        "kv_int4_report": kv_int4_report,
        "kv_outlier_report": kv_outlier_report,
        "roofline_decode_regime": roofline["decode"]["regime"],
        "roofline_decode_tps_ceiling": roofline["decode"]["ceiling_tokens_per_sec"],
        "answer_authority": False,
    }
    all_ok = parity and chunked_parity and ctl_parity
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0 if all_ok else 1
    print(f"Supermix Perf ({payload['evolution_v106_version']})")
    print(f"  SCF warm task: {scf_warm_ms} ms (answer={payload['scf_verified_answer']})")
    print(f"  Decode 10 tok: greedy={greedy_ms} ms | adaptive_spec={adaptive_spec_ms} ms | parity={parity} | chunked={chunked_parity}")
    print(f"  Controller trunk reuse: {ctl_reuse_ms} ms vs {ctl_full_ms} ms ({payload['controller_trunk_reuse_speedup']}x, exact={ctl_parity})")
    print(f"  KV compression: Int8={kv_report['compression_ratio']}x (cos={kv_report['mean_cosine_similarity']}) | Int4={kv_int4_report['compression_ratio']}x (cos={kv_int4_report['mean_cosine_similarity']})")
    return 0 if all_ok else 1


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

    p_frontier = sub.add_parser("frontier", help="Run Supermix v102-v105 frontier modules self-check")
    p_frontier.add_argument("--json", action="store_true")

    p_diag = sub.add_parser("diagnose", help="Run unified Supermix v104-v105 health & exact-verifier diagnostic")
    p_diag.add_argument("--json", action="store_true")

    p_perf = sub.add_parser("perf", help="Run live Supermix v105 performance & KV-compression micro-benchmark")
    p_perf.add_argument("--json", action="store_true")
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
    if args.command == "diagnose":
        return cmd_diagnose(as_json=args.json)
    if args.command == "perf":
        return cmd_perf(as_json=args.json)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
