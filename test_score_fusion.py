"""Contracts for calibrated score fusion.

The headline contract is negative: `legacy` must stay bit-exact. The fusion
modes exist so a future weight re-tune can be measured, not because any of them
has been shown to rank better. `test_benchmark_reports_honest_significance`
pins that honesty.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source" / "score_fusion.py"
RUNTIME_PATH = ROOT / "runtime_python" / "score_fusion.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = _load("source_score_fusion_tests", SOURCE_PATH)
runtime = _load("runtime_score_fusion_tests", RUNTIME_PATH)


@pytest.fixture(scope="module")
def pipeline():
    source_dir = ROOT / "source"
    sys.path.insert(0, str(source_dir))
    try:
        yield _load("score_fusion_chat_pipeline", source_dir / "chat_pipeline.py")
    finally:
        sys.path.remove(str(source_dir))


def _pool(pipeline, size: int = 12):
    texts = [
        "Use CREATE INDEX on the column you filter by most often.",
        "Partial indexes add a WHERE clause so only matching rows are stored.",
        "A B-tree keeps keys sorted and supports range scans.",
        "A closure captures variables from its enclosing lexical scope.",
        "Python's GIL prevents true parallel execution of bytecode.",
        "Photosynthesis converts light energy into chemical energy.",
        "The mitochondrion produces ATP in the cell.",
        "HTTP 429 means too many requests; back off and retry.",
        "Docker containers share the host kernel, unlike virtual machines.",
        "A CDN caches static assets closer to the user.",
        "TLS encrypts traffic between client and server.",
        "A vector database stores embeddings for similarity search.",
    ][:size]
    rows = []
    for index, text in enumerate(texts):
        vector = pipeline.featurize_text(text)
        rows.append({
            "text": text,
            "vec": vector.tolist(),
            "ctx_vec": vector.tolist(),
            "count": 1 + (index % 4),
            "bucket_score": 0.25 + 0.03 * (index % 6),
        })
    return rows


# --------------------------------------------------------------------------
# percentile rank
# --------------------------------------------------------------------------

def test_percentile_rank_is_uniform_and_handles_ties_by_average() -> None:
    ranks = source.percentile_rank(torch.tensor([10.0, 20.0, 30.0, 40.0]))
    assert ranks.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], abs=1e-6)

    tied = source.percentile_rank(torch.tensor([5.0, 5.0, 5.0, 9.0]))
    assert tied[0] == tied[1] == tied[2]
    assert tied[3] > tied[0]


def test_percentile_rank_is_invariant_to_candidate_order() -> None:
    """Ties must not depend on which order candidates arrived in."""

    values = torch.tensor([3.0, 1.0, 3.0, 2.0])
    permutation = torch.tensor([2, 0, 3, 1])

    direct = source.percentile_rank(values).index_select(0, permutation)
    permuted = source.percentile_rank(values.index_select(0, permutation))
    assert torch.allclose(direct, permuted)


def test_a_constant_signal_contributes_nothing() -> None:
    ranks = source.percentile_rank(torch.tensor([7.0, 7.0, 7.0]))
    assert ranks.tolist() == [0.0, 0.0, 0.0]


def test_degenerate_sizes_are_safe() -> None:
    assert source.percentile_rank(torch.zeros(0)).tolist() == []
    assert source.percentile_rank(torch.tensor([3.0])).tolist() == [0.0]
    assert source.zmuv(torch.zeros(0)).tolist() == []
    assert source.zmuv(torch.tensor([4.0, 4.0])).tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------
# the dispersion gate is what makes rank calibration safe
# --------------------------------------------------------------------------

def test_dispersion_gate_blocks_noise_from_being_amplified() -> None:
    """Rank calibration is scale-free, so noise would otherwise get full spread."""

    torch.manual_seed(0)
    noise = torch.full((16,), 0.5) + torch.randn(16) * 1e-7
    informative = torch.randn(16)

    noise_out, noise_gate = source.calibrate_signal(noise)
    real_out, real_gate = source.calibrate_signal(informative)

    assert noise_gate == 0.0
    assert float(noise_out.max() - noise_out.min()) == 0.0
    assert real_gate == 1.0
    assert float(real_out.max() - real_out.min()) == pytest.approx(1.0)


def test_gated_mode_keeps_raw_scales() -> None:
    values = torch.tensor([0.1, 0.5, 0.9, 0.3])
    gated, gate = source.calibrate_signal(values, rank_transform=False)
    assert gate == 1.0
    assert torch.allclose(gated, values)


# --------------------------------------------------------------------------
# fusion contracts
# --------------------------------------------------------------------------

def test_unknown_modes_fall_back_to_legacy() -> None:
    for bad in (None, "", "nonsense", 17, object()):
        assert source.resolve_fusion_mode(bad) == "legacy"
    for good in source.RUNTIME_FUSION_MODES:
        assert source.resolve_fusion_mode(good) == good


def test_runtime_callers_cannot_select_a_mode_measured_worse_than_legacy() -> None:
    """`calibrated` and `consensus` measured 51 and 58 points of top-1 below
    legacy on 200 real corpus probes. A config typo or a stale setting must not
    be able to reach them."""

    for harmful in source.EXPERIMENTAL_FUSION_MODES:
        assert harmful in source.FUSION_MODES
        assert harmful not in source.RUNTIME_FUSION_MODES
        assert source.resolve_fusion_mode(harmful) == "legacy"
        # Reachable only by asking for it explicitly, which the benchmark does.
        assert source.resolve_fusion_mode(harmful, allow_experimental=True) == harmful


def test_the_ranker_ignores_experimental_modes_unless_opted_in(pipeline) -> None:
    candidates = _pool(pipeline)
    query = "how do I make a slow postgres query faster?"

    legacy = pipeline.rank_response_candidates(candidates, query, [])
    for harmful in source.EXPERIMENTAL_FUSION_MODES:
        guarded = pipeline.rank_response_candidates(
            candidates, query, [], fusion_mode=harmful
        )
        assert guarded[0] == legacy[0]
        assert torch.equal(guarded[1], legacy[1])

        opted_in = pipeline.rank_response_candidates(
            candidates, query, [], fusion_mode=harmful, allow_experimental_fusion=True
        )
        assert torch.isfinite(opted_in[1]).all()


def test_calibration_makes_weights_match_their_influence() -> None:
    """The defect this module exists for, stated as a test."""

    torch.manual_seed(1)
    signals = {
        "sim_ctx": torch.randn(16) * 0.10,
        "lex_sim": torch.rand(16) * 0.09,
        "bucket_bonus": torch.rand(16) * 0.35 + 0.2,
        "freq_penalty": torch.log1p(torch.randint(1, 6, (16,)).float()),
    }
    weights = {"sim_ctx": 0.60, "lex_sim": 0.10, "bucket_bonus": 0.18, "freq_penalty": -0.03}
    total_weight = sum(abs(w) for w in weights.values())

    def gap(mode: str) -> float:
        _, diagnostics = source.fuse_signals(signals, weights, mode=mode)
        worst = 0.0
        for row in diagnostics["signals"]:
            nominal = abs(row["weight"]) / total_weight
            worst = max(worst, abs(row["influence_share"] - nominal))
        return worst

    # Legacy: lex_sim carries three times the weight of freq_penalty and a
    # fraction of its influence.
    assert gap("legacy") > 0.05
    # Calibrated: influence tracks the weight that was written down.
    assert gap("calibrated") < 0.02


def test_consensus_multiplier_is_bounded_and_rewards_agreement() -> None:
    agreed = [torch.tensor([1.0, 0.0, 0.0, 0.0]) for _ in range(4)]
    multiplier = source.consensus_multiplier(agreed)
    assert float(multiplier.max()) == pytest.approx(source.MAX_CONSENSUS_MULTIPLIER)
    assert float(multiplier.min()) >= 1.0
    assert multiplier[0] > multiplier[1]

    # A constant signal supports nobody and must not create phantom consensus.
    flat = [torch.full((4,), 0.5) for _ in range(3)]
    assert torch.allclose(source.consensus_multiplier(flat), torch.ones(4))
    assert source.consensus_multiplier([]).numel() == 0


def test_diagnostics_are_json_safe_and_carry_no_text() -> None:
    torch.manual_seed(2)
    signals = {"a": torch.randn(8), "b": torch.randn(8)}
    _, diagnostics = source.fuse_signals(signals, {"a": 0.5, "b": 0.5}, mode="calibrated")
    assert json.loads(json.dumps(diagnostics, sort_keys=True)) == diagnostics

    compact = source.fusion_diagnostics(diagnostics)
    assert json.loads(json.dumps(compact, sort_keys=True)) == compact
    assert compact["signal_count"] == 2
    assert source.fusion_diagnostics(None)["signal_count"] == 0


# --------------------------------------------------------------------------
# the contract that matters most
# --------------------------------------------------------------------------

def test_legacy_ranking_is_bit_exact_however_it_is_requested(pipeline) -> None:
    candidates = _pool(pipeline)
    query = "how do I make a slow postgres query faster?"

    baseline = pipeline.rank_response_candidates(candidates, query, [])
    for spelling in (None, "legacy", "", "unknown-mode"):
        order, scores = pipeline.rank_response_candidates(
            candidates, query, [], fusion_mode=spelling
        )
        assert order == baseline[0]
        assert torch.equal(scores, baseline[1])


def test_gated_mode_changes_nothing_when_no_signal_is_noise(pipeline) -> None:
    """Documents the measured result: the gate never fires on real signals here."""

    candidates = _pool(pipeline)
    query = "what is a closure in programming"

    legacy = pipeline.rank_response_candidates(candidates, query, [], fusion_mode="legacy")
    gated = pipeline.rank_response_candidates(candidates, query, [], fusion_mode="gated")
    assert gated[0] == legacy[0]


def test_every_mode_is_deterministic_finite_and_total(pipeline) -> None:
    candidates = _pool(pipeline)
    query = "how do I reduce latency for static files"

    for mode in source.FUSION_MODES:
        first = pipeline.rank_response_candidates(candidates, query, [], fusion_mode=mode)
        second = pipeline.rank_response_candidates(candidates, query, [], fusion_mode=mode)
        assert first[0] == second[0]
        assert torch.equal(first[1], second[1])
        assert torch.isfinite(first[1]).all()
        assert sorted(first[0]) == list(range(len(candidates)))


def test_degenerate_pools_are_safe_in_every_mode(pipeline) -> None:
    candidates = _pool(pipeline)
    for pool in ([], candidates[:1]):
        for mode in source.FUSION_MODES:
            order, scores = pipeline.rank_response_candidates(
                pool, "anything", [], fusion_mode=mode
            )
            assert len(order) == len(pool)
            assert scores.numel() == len(pool)


def test_source_and_runtime_are_exact_mirrors() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    runtime_bytes = RUNTIME_PATH.read_bytes()
    assert source_bytes == runtime_bytes
    assert hashlib.sha256(source_bytes).hexdigest() == hashlib.sha256(runtime_bytes).hexdigest()

    values = torch.tensor([0.4, 0.1, 0.9, 0.1])
    assert torch.equal(source.percentile_rank(values), runtime.percentile_rank(values))
    assert source.resolve_fusion_mode("consensus") == runtime.resolve_fusion_mode("consensus")


def test_benchmark_reports_honest_significance() -> None:
    """The harness must not present sampling noise as an improvement."""

    source_dir = ROOT / "source"
    sys.path.insert(0, str(source_dir))
    try:
        benchmark = _load("ranking_quality_bench", source_dir / "benchmark_ranking_quality.py")
    finally:
        sys.path.remove(str(source_dir))

    report = benchmark.run(bootstrap=400)
    assert json.loads(json.dumps(report, sort_keys=True)) == report
    assert report["probe_count"] >= 20

    legacy = report["modes"]["legacy"]
    assert legacy["deltas_vs_legacy"]["mrr"]["delta"] == 0.0

    # No mode may be flagged significant unless its interval excludes zero.
    for mode, entry in report["modes"].items():
        for metric, row in entry["deltas_vs_legacy"].items():
            low, high = row["ci95"]
            assert low <= high
            expected = low > 0.0 or high < 0.0
            assert row["significant"] is expected, (mode, metric)
