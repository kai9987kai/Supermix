"""Supermix v103 Frontier Synthesis Engines (Memory, Mechanistic SAE, Fractal Topology, Causeway, NeuroHive).

Self-contained, deterministic standard-library + NumPy engines synthesizing five
additional frontier research systems from 2025-2026 literature and the
``kai9987kai`` GitHub ecosystem (all strictly ``answer_authority=False`` with
canonical SHA-256 receipts):

1. **ContinuumBTSPHopfieldMemory** (Google Nested Learning 2025/2026 +
   ``OMNIS-COGNITIVE/frontier/memory_systems.py`` + ``Supermix-Poseidon/poseidon/mnemosyne.py``):
   Combines multi-timescale frequency-separated Continuum Memory (fast/med/slow),
   one-shot BTSP binary plateau imprinting, Modern Hopfield softmax energy
   retrieval, and Poseidon MNEMOSYNE's 5-mode matched lesion assay
   (``intact``, ``erased``, ``shifted``, ``random_retrieval``, ``memory_free``).

2. **TopKSparseAutoencoderProbe** (Gao et al. 2024/2025 +
   ``OMNIS-COGNITIVE/frontier/analysis.py``):
   Overcomplete Top-K Sparse Autoencoder (SAE) with unit-norm decoder columns,
   reconstruction R^2, dead-feature accounting, and causal feature ablation
   specificity scoring.

3. **FractalConnectomeAnalyzer** (``kai9987kai/SONAPSE-FRACTAL`` + ``NEUROGRAFT``):
   Reproducible finite-approximation Sierpinski, Hilbert-chain, Watts-Strogatz
   small-world, and Erdos-Renyi control graph constructors with exact clustering,
   characteristic path length, modularity proxy, degree entropy, and spectral radius.

4. **CausewaySensitivityWorkbench** (``kai9987kai/Causeway``):
   Common-random-numbers paired policy comparison across explicit assumptions
   with one-change sensitivity lenses detecting winner-reversing assumptions.

5. **NeuroHiveResourcePlanner** (``kai9987kai/NEUROHIVE``):
   Resource-constrained multi-agent cognitive dispatch with station thermal/load
   tracking, deterministic fault injection, autonomous maintenance recovery,
   and strict budget conservation verification.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FRONTIER_V103_VERSION",
    "MNEMOSYNE_LESION_MODES",
    "ContinuumBTSPHopfieldMemory",
    "TopKSparseAutoencoderProbe",
    "FractalConnectomeAnalyzer",
    "CausewaySensitivityWorkbench",
    "NeuroHiveResourcePlanner",
]

FRONTIER_V103_VERSION = "103.0.0-synthesis"
MNEMOSYNE_LESION_MODES: Tuple[str, ...] = (
    "intact",
    "erased",
    "shifted",
    "random_retrieval",
    "memory_free",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ============================================================================
# 1. ContinuumBTSPHopfieldMemory + Poseidon MNEMOSYNE 5-Mode Lesion Assay
# ============================================================================


class ContinuumBTSPHopfieldMemory:
    """Tri-paradigm neural memory (Continuum + One-Shot BTSP + Modern Hopfield) with lesion assays."""

    def __init__(
        self,
        *,
        dim: int = 12,
        out_dim: int = 4,
        btsp_bits: int = 32,
        hopfield_beta: float = 6.0,
        seed: int = 103,
    ) -> None:
        self.dim = max(2, int(dim))
        self.out_dim = max(1, int(out_dim))
        self.btsp_bits = max(8, int(btsp_bits))
        self.hopfield_beta = float(hopfield_beta)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Nested Learning multi-timescale Continuum Memory banks
        self.w_fast = self.rng.normal(0.0, 0.04, size=(self.out_dim, self.dim))
        self.w_med = self.rng.normal(0.0, 0.04, size=(self.out_dim, self.dim))
        self.w_slow = self.rng.normal(0.0, 0.04, size=(self.out_dim, self.dim))
        self.step_idx = 0

        # One-shot BTSP binary synaptic matrix
        self.btsp_weights = np.zeros((self.btsp_bits, self.btsp_bits), dtype=np.int8)

        # Modern Hopfield continuous pattern bank
        self.hopfield_patterns: List[np.ndarray] = []

    def continuum_predict(self, x: Sequence[float]) -> np.ndarray:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)[: self.dim]
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        return (0.45 * self.w_fast + 0.30 * self.w_med + 0.25 * self.w_slow) @ vec

    def continuum_update(self, x: Sequence[float], target: Sequence[float]) -> float:
        vec = np.asarray(x, dtype=np.float64).reshape(-1)[: self.dim]
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        tgt = np.asarray(target, dtype=np.float64).reshape(-1)[: self.out_dim]
        if tgt.size < self.out_dim:
            tgt = np.pad(tgt, (0, self.out_dim - tgt.size))
        self.step_idx += 1
        pred = self.continuum_predict(vec)
        err = tgt - pred
        grad = np.outer(err, vec)
        self.w_fast = 0.95 * self.w_fast + 0.22 * grad
        if self.step_idx % 3 == 0:
            self.w_med = 0.99 * self.w_med + 0.09 * grad + 0.04 * (self.w_fast - self.w_med)
        if self.step_idx % 8 == 0:
            self.w_slow = 0.997 * self.w_slow + 0.04 * grad + 0.05 * (self.w_med - self.w_slow)
        return float(np.mean(err**2))

    def btsp_imprint(self, cue: Sequence[float], target: Sequence[float], *, plateau_prob: float = 0.9) -> None:
        c = (np.asarray(cue, dtype=np.float64).reshape(-1)[: self.btsp_bits] > 0.5).astype(np.int8)
        t = (np.asarray(target, dtype=np.float64).reshape(-1)[: self.btsp_bits] > 0.5).astype(np.int8)
        if c.size < self.btsp_bits:
            c = np.pad(c, (0, self.btsp_bits - c.size))
        if t.size < self.btsp_bits:
            t = np.pad(t, (0, self.btsp_bits - t.size))
        coactive = np.outer(t, c)
        gate = (self.rng.random(coactive.shape) < float(plateau_prob)).astype(np.int8)
        self.btsp_weights = np.clip(self.btsp_weights + coactive * gate, 0, 1).astype(np.int8)

    def btsp_recall(self, cue: Sequence[float], *, k_active: int = 6) -> np.ndarray:
        c = (np.asarray(cue, dtype=np.float64).reshape(-1)[: self.btsp_bits] > 0.5).astype(np.float64)
        if c.size < self.btsp_bits:
            c = np.pad(c, (0, self.btsp_bits - c.size))
        drive = self.btsp_weights.astype(np.float64) @ c
        k = max(1, min(self.btsp_bits, int(k_active)))
        top_idx = np.argsort(drive)[-k:]
        out = np.zeros((self.btsp_bits,), dtype=np.int8)
        out[top_idx] = 1
        return out

    def hopfield_store(self, pattern: Sequence[float]) -> None:
        vec = np.asarray(pattern, dtype=np.float64).reshape(-1)[: self.dim]
        if vec.size < self.dim:
            vec = np.pad(vec, (0, self.dim - vec.size))
        nrm = float(np.linalg.norm(vec))
        if nrm > 1e-12:
            vec = vec / nrm
        self.hopfield_patterns.append(vec)
        if len(self.hopfield_patterns) > 64:
            self.hopfield_patterns.pop(0)

    def hopfield_retrieve(self, query: Sequence[float], *, mode: str = "intact") -> Tuple[np.ndarray, float]:
        if mode not in MNEMOSYNE_LESION_MODES:
            raise ValueError(f"unknown lesion mode: {mode!r}")
        q = np.asarray(query, dtype=np.float64).reshape(-1)[: self.dim]
        if q.size < self.dim:
            q = np.pad(q, (0, self.dim - q.size))
        qn = float(np.linalg.norm(q))
        if qn > 1e-12:
            q = q / qn
        if mode in {"erased", "memory_free"} or not self.hopfield_patterns:
            return np.zeros((self.dim,), dtype=np.float64), 0.0

        mat = np.stack(self.hopfield_patterns, axis=0)  # (M, D)
        if mode == "shifted" and mat.shape[0] > 1:
            keys = np.roll(mat, shift=1, axis=0)
        else:
            keys = mat

        if mode == "random_retrieval":
            pick = int(self.rng.integers(0, mat.shape[0]))
            rec = mat[pick]
            return rec, float(np.dot(q, rec))

        logits = self.hopfield_beta * (keys @ q)
        logits = logits - float(np.max(logits))
        weights = np.exp(logits)
        weights /= max(1e-12, float(np.sum(weights)))
        retrieved = weights @ mat
        rn = float(np.linalg.norm(retrieved))
        if rn > 1e-12:
            retrieved = retrieved / rn
        return retrieved, float(np.dot(q, retrieved))

    def run_benchmark(self, *, n_patterns: int = 8, steps: int = 24) -> Dict[str, Any]:
        """Run deterministic Continuum + One-Shot BTSP + Hopfield + 5-Mode Lesion Assay."""
        local_rng = np.random.default_rng(self.seed + 7)
        true_w = local_rng.normal(0.0, 0.5, size=(self.out_dim, self.dim))
        mse_hist: List[float] = []
        for _ in range(max(4, int(steps))):
            x = local_rng.normal(0.0, 1.0, size=(self.dim,))
            x /= max(1e-8, float(np.linalg.norm(x)))
            mse_hist.append(self.continuum_update(x, true_w @ x))

        # One-shot BTSP sparse recall test
        cue = np.zeros((self.btsp_bits,), dtype=np.int8)
        tgt = np.zeros((self.btsp_bits,), dtype=np.int8)
        cue[local_rng.choice(self.btsp_bits, size=6, replace=False)] = 1
        tgt[local_rng.choice(self.btsp_bits, size=6, replace=False)] = 1
        self.btsp_imprint(cue, tgt, plateau_prob=1.0)
        recalled = self.btsp_recall(cue, k_active=6)
        btsp_overlap = float(np.sum((recalled == 1) & (tgt == 1))) / 6.0

        # Modern Hopfield + Poseidon MNEMOSYNE 5-mode matched lesion assay
        self.hopfield_patterns.clear()
        stored: List[np.ndarray] = []
        for _ in range(max(3, int(n_patterns))):
            p = local_rng.normal(0.0, 1.0, size=(self.dim,))
            p /= max(1e-8, float(np.linalg.norm(p)))
            stored.append(p)
            self.hopfield_store(p)

        lesion_scores: Dict[str, float] = {}
        for mode in MNEMOSYNE_LESION_MODES:
            cos_vals: List[float] = []
            for p in stored:
                noisy = p + local_rng.normal(0.0, 0.12, size=(self.dim,))
                rec, _ = self.hopfield_retrieve(noisy, mode=mode)
                cos_vals.append(float(np.dot(p, rec)))
            lesion_scores[mode] = round(float(np.mean(cos_vals)), 6)

        payload: Dict[str, Any] = {
            "continuum_initial_mse": round(mse_hist[0], 6),
            "continuum_final_mse": round(mse_hist[-1], 6),
            "continuum_mse_reduction": round(mse_hist[0] - mse_hist[-1], 6),
            "btsp_one_shot_overlap": round(btsp_overlap, 4),
            "mnemosyne_lesion_cosine": lesion_scores,
            "intact_vs_erased_delta": round(lesion_scores["intact"] - lesion_scores["erased"], 6),
            "answer_authority": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


# ============================================================================
# 2. TopKSparseAutoencoderProbe (Monosemantic SAE + Causal Feature Ablation)
# ============================================================================


class TopKSparseAutoencoderProbe:
    """Overcomplete Top-K Sparse Autoencoder with causal feature ablation specificity."""

    def __init__(
        self,
        *,
        input_dim: int = 8,
        dict_size: int = 24,
        k_active: int = 3,
        seed: int = 103,
    ) -> None:
        self.input_dim = max(2, int(input_dim))
        self.dict_size = max(self.input_dim, int(dict_size))
        self.k_active = max(1, min(self.dict_size, int(k_active)))
        rng = np.random.default_rng(int(seed))
        w = rng.normal(0.0, 0.35, size=(self.dict_size, self.input_dim))
        w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-8
        self.w_enc = w
        self.w_dec = w.T.copy()  # (D, M)
        self.b_pre = np.zeros((self.input_dim,), dtype=np.float64)

    def encode_topk(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        pre = np.maximum(0.0, (arr - self.b_pre) @ self.w_enc.T)
        z = np.zeros_like(pre)
        if self.k_active >= self.dict_size:
            return pre
        top_idx = np.argpartition(pre, -self.k_active, axis=1)[:, -self.k_active :]
        rows = np.arange(arr.shape[0])[:, None]
        z[rows, top_idx] = pre[rows, top_idx]
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.w_dec.T + self.b_pre

    def fit_and_probe(
        self,
        samples: Sequence[Sequence[float]],
        *,
        epochs: int = 35,
        lr: float = 0.08,
        probe_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        data = np.asarray(samples, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] != self.input_dim:
            raise ValueError(f"samples must be shape (N>=4, {self.input_dim}), got {data.shape}")
        if not np.all(np.isfinite(data)):
            raise ValueError("samples contain non-finite values")

        self.b_pre = np.mean(data, axis=0)
        init_recon = self.decode(self.encode_topk(data))
        init_mse = float(np.mean((init_recon - data) ** 2))

        for _ in range(max(1, int(epochs))):
            z = self.encode_topk(data)
            recon = self.decode(z)
            err = recon - data  # (N, D)
            grad_dec = (err.T @ z) / float(data.shape[0])
            self.w_dec -= float(lr) * grad_dec
            self.w_dec /= np.linalg.norm(self.w_dec, axis=0, keepdims=True) + 1e-8
            self.w_enc = self.w_dec.T.copy()

        z_final = self.encode_topk(data)
        recon_final = self.decode(z_final)
        final_mse = float(np.mean((recon_final - data) ** 2))
        var_total = float(np.mean((data - self.b_pre) ** 2))
        r2 = float(max(0.0, 1.0 - final_mse / max(1e-12, var_total)))

        active_counts = np.sum(z_final > 0.0, axis=0)
        dead_frac = float(np.mean(active_counts == 0))
        l0_mean = float(np.mean(np.sum(z_final > 0.0, axis=1)))

        if probe_weights is None:
            p_vec = np.ones((self.input_dim,), dtype=np.float64) / math.sqrt(self.input_dim)
        else:
            p_vec = np.asarray(probe_weights, dtype=np.float64).reshape(-1)
            if p_vec.size != self.input_dim:
                raise ValueError(f"probe_weights must have length {self.input_dim}")

        logits_full = recon_final @ p_vec
        mean_act = np.mean(z_final, axis=0)
        top_feat = int(np.argmax(mean_act))
        z_top = z_final.copy()
        z_top[:, top_feat] = 0.0
        causal_shift = float(np.mean(np.abs(logits_full - (self.decode(z_top) @ p_vec))))

        low_feat = int(np.argmin(mean_act))
        z_low = z_final.copy()
        z_low[:, low_feat] = 0.0
        control_shift = float(np.mean(np.abs(logits_full - (self.decode(z_low) @ p_vec))))

        payload: Dict[str, Any] = {
            "input_dim": self.input_dim,
            "dict_size": self.dict_size,
            "k_active": self.k_active,
            "initial_mse": round(init_mse, 6),
            "final_mse": round(final_mse, 6),
            "reconstruction_r2": round(r2, 6),
            "mean_l0_active": round(l0_mean, 4),
            "dead_feature_fraction": round(dead_frac, 4),
            "top_feature_index": top_feat,
            "causal_ablation_shift": round(causal_shift, 6),
            "control_ablation_shift": round(control_shift, 6),
            "causal_specificity_ratio": round(causal_shift / max(1e-5, control_shift), 4),
            "answer_authority": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


# ============================================================================
# 3. FractalConnectomeAnalyzer (SONAPSE-FRACTAL Topology Metrics)
# ============================================================================


class FractalConnectomeAnalyzer:
    """Finite-approximation fractal & small-world graph constructor and topology analyzer."""

    @staticmethod
    def build_topology(kind: str, *, n_nodes: int = 27, seed: int = 103) -> np.ndarray:
        """Build undirected 0/1 adjacency matrix for Sierpinski, Hilbert, small_world, or random_control."""
        n = max(6, min(128, int(n_nodes)))
        rng = np.random.default_rng(int(seed))
        adj = np.zeros((n, n), dtype=np.float64)

        if kind == "sierpinski":
            # Hierarchical 3-clique self-similar Sierpinski gasket approximation
            block = 3
            for i in range(0, n, block):
                end = min(n, i + block)
                for u in range(i, end):
                    for v in range(u + 1, end):
                        adj[u, v] = adj[v, u] = 1.0
            # Bridge triangles across hierarchical scales (3, 9, 27, ...)
            stride = block
            while stride < n:
                for base in range(0, n, stride * 3):
                    g0 = base + stride - 1
                    g1 = base + stride
                    g2 = base + 2 * stride - 1
                    g3 = base + 2 * stride
                    if g0 < n and g1 < n:
                        adj[g0, g1] = adj[g1, g0] = 1.0
                    if g2 < n and g3 < n:
                        adj[g2, g3] = adj[g3, g2] = 1.0
                    if base < n and min(n - 1, base + 3 * stride - 1) > base:
                        hi = min(n - 1, base + 3 * stride - 1)
                        adj[base, hi] = adj[hi, base] = 1.0
                stride *= 3
        elif kind == "hilbert":
            # Space-filling locality chain + multiscale power-of-2 fold bridges
            for i in range(n - 1):
                adj[i, i + 1] = adj[i + 1, i] = 1.0
            step = 4
            while step < n:
                for i in range(0, n - step, step):
                    adj[i, i + step - 1] = adj[i + step - 1, i] = 1.0
                step *= 2
        elif kind == "small_world":
            # Watts-Strogatz ring (k=4) with p=0.15 deterministic rewiring
            for i in range(n):
                for offset in (1, 2):
                    j = (i + offset) % n
                    adj[i, j] = adj[j, i] = 1.0
            for i in range(n):
                if rng.random() < 0.15:
                    nbrs = np.where(adj[i] > 0)[0]
                    non_nbrs = [u for u in range(n) if u != i and adj[i, u] == 0]
                    if len(nbrs) > 0 and non_nbrs:
                        drop = int(nbrs[int(rng.integers(0, len(nbrs)))])
                        add = int(non_nbrs[int(rng.integers(0, len(non_nbrs)))])
                        adj[i, drop] = adj[drop, i] = 0.0
                        adj[i, add] = adj[add, i] = 1.0
        elif kind == "random_control":
            # Degree-comparable connected Erdos-Renyi control
            p_edge = min(0.5, 3.2 / float(max(2, n - 1)))
            for i in range(n - 1):
                adj[i, i + 1] = adj[i + 1, i] = 1.0
            for i in range(n):
                for j in range(i + 2, n):
                    if rng.random() < p_edge:
                        adj[i, j] = adj[j, i] = 1.0
        else:
            raise ValueError(f"unknown topology kind: {kind!r}")

        np.fill_diagonal(adj, 0.0)
        return adj

    @classmethod
    def analyze_adjacency(cls, adj: np.ndarray, *, label: str = "graph") -> Dict[str, Any]:
        mat = np.asarray(adj, dtype=np.float64)
        n = int(mat.shape[0])
        sym = ((mat + mat.T) > 0).astype(np.float64)
        np.fill_diagonal(sym, 0.0)
        degrees = sym.sum(axis=1)
        edges = int(degrees.sum() // 2)

        # Exact mean local clustering coefficient
        clust_vals: List[float] = []
        for i in range(n):
            nbrs = np.where(sym[i] > 0)[0]
            k = len(nbrs)
            if k < 2:
                clust_vals.append(0.0)
            else:
                sub = sym[np.ix_(nbrs, nbrs)]
                possible = k * (k - 1)
                clust_vals.append(float(sub.sum()) / float(possible))
        clustering = float(np.mean(clust_vals)) if clust_vals else 0.0

        # Exact BFS characteristic path length over reachable pairs
        total_dist = 0
        pairs = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q: deque[int] = deque([src])
            while q:
                u = q.popleft()
                for v in np.where(sym[u] > 0)[0]:
                    vi = int(v)
                    if dist[vi] == -1:
                        dist[vi] = dist[u] + 1
                        q.append(vi)
            for dst in range(src + 1, n):
                if dist[dst] > 0:
                    total_dist += dist[dst]
                    pairs += 1
        avg_path = float(total_dist) / float(max(1, pairs))

        # 3-partition modularity proxy (Newman Q over equal contiguous communities)
        m2 = max(1.0, float(degrees.sum()))
        comm = np.array([min(2, (i * 3) // max(1, n)) for i in range(n)])
        q_mod = 0.0
        for i in range(n):
            for j in range(n):
                if comm[i] == comm[j]:
                    q_mod += sym[i, j] - (degrees[i] * degrees[j]) / m2
        modularity = q_mod / m2

        # Normalized degree entropy + spectral radius
        deg_sum = float(degrees.sum())
        if deg_sum > 0:
            p_deg = degrees[degrees > 0] / deg_sum
            deg_ent = float(-np.sum(p_deg * np.log(p_deg)) / math.log(max(2, n)))
        else:
            deg_ent = 0.0
        eigvals = np.linalg.eigvalsh(sym)
        spec_rad = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0

        return {
            "label": str(label),
            "nodes": n,
            "edges": edges,
            "density": round(float(2 * edges) / float(max(1, n * (n - 1))), 6),
            "clustering_coefficient": round(clustering, 6),
            "characteristic_path_length": round(avg_path, 6),
            "modularity_q": round(modularity, 6),
            "degree_entropy": round(deg_ent, 6),
            "spectral_radius": round(spec_rad, 6),
        }

    @classmethod
    def compare_topologies(
        cls,
        *,
        kinds: Sequence[str] = ("sierpinski", "hilbert", "small_world", "random_control"),
        n_nodes: int = 27,
        seed: int = 103,
    ) -> Dict[str, Any]:
        reports = [
            cls.analyze_adjacency(cls.build_topology(k, n_nodes=n_nodes, seed=seed), label=k)
            for k in kinds
        ]
        by_label = {r["label"]: r for r in reports}
        payload: Dict[str, Any] = {
            "n_nodes": int(n_nodes),
            "seed": int(seed),
            "topologies": by_label,
            "highest_clustering": max(reports, key=lambda r: r["clustering_coefficient"])["label"],
            "highest_modularity": max(reports, key=lambda r: r["modularity_q"])["label"],
            "answer_authority": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


# ============================================================================
# 4. CausewaySensitivityWorkbench (Paired-Seed Common Random Numbers + Lenses)
# ============================================================================


class CausewaySensitivityWorkbench:
    """One-change assumption sensitivity lenses over paired common-random-numbers policies."""

    @staticmethod
    def evaluate_policies(
        *,
        policies: Mapping[str, Mapping[str, float]],
        assumptions: Optional[Mapping[str, float]] = None,
        sensitivity_flips: Optional[Mapping[str, float]] = None,
        seeds: Sequence[int] = (11, 29, 47, 73),
    ) -> Dict[str, Any]:
        """Compare policies under shared stochastic stress draws and one-change assumption lenses.

        Each policy specifies bounded knobs ``accuracy_weight``, ``cost_penalty``,
        ``safety_margin`` (in ``[0, 1]``). Assumptions govern stress amplitude and
        risk aversion; one-change lenses flip each assumption individually to check
        whether the winning policy reverses.
        """
        if not policies:
            raise ValueError("policies must be non-empty")
        base_assump: Dict[str, float] = {
            "stress_amplitude": 0.35,
            "risk_aversion": 0.60,
            "cost_sensitivity": 0.25,
            **{str(k): float(v) for k, v in (assumptions or {}).items()},
        }
        flips: Dict[str, float] = {
            "stress_amplitude": 0.85,
            "risk_aversion": 1.40,
            "cost_sensitivity": 0.75,
            **{str(k): float(v) for k, v in (sensitivity_flips or {}).items()},
        }

        def _score_all(assump: Dict[str, float]) -> Tuple[Dict[str, float], str]:
            scores: Dict[str, List[float]] = {name: [] for name in policies}
            for s in seeds:
                rng = np.random.default_rng(int(s))
                shock = float(rng.uniform(0.0, 1.0)) * assump["stress_amplitude"]
                load = float(rng.uniform(0.2, 1.0))
                for pname, cfg in policies.items():
                    acc = float(cfg.get("accuracy_weight", 0.7))
                    cost = float(cfg.get("cost_penalty", 0.3))
                    safety = float(cfg.get("safety_margin", 0.5))
                    # Common random shock hits low-safety policies harder
                    failure_risk = max(0.0, shock - safety * 0.55)
                    utility = acc - assump["risk_aversion"] * failure_risk - assump["cost_sensitivity"] * cost * load
                    scores[pname].append(utility)
            means = {name: round(float(np.mean(vals)), 6) for name, vals in scores.items()}
            winner = max(sorted(means.keys()), key=lambda k: means[k])
            return means, winner

        baseline_scores, baseline_winner = _score_all(base_assump)
        lenses: List[Dict[str, Any]] = []
        reversing: List[str] = []
        for key in sorted(base_assump.keys()):
            if key not in flips:
                continue
            alt_assump = dict(base_assump)
            alt_assump[key] = float(flips[key])
            alt_scores, alt_winner = _score_all(alt_assump)
            reversed_flag = alt_winner != baseline_winner
            if reversed_flag:
                reversing.append(key)
            lenses.append(
                {
                    "assumption": key,
                    "baseline_value": base_assump[key],
                    "flipped_value": float(flips[key]),
                    "scores": alt_scores,
                    "winner": alt_winner,
                    "winner_reversed": reversed_flag,
                }
            )

        payload: Dict[str, Any] = {
            "seeds": [int(s) for s in seeds],
            "baseline_assumptions": base_assump,
            "baseline_scores": baseline_scores,
            "baseline_winner": baseline_winner,
            "sensitivity_lenses": lenses,
            "winner_reversing_assumptions": reversing,
            "decision_robust_across_lenses": len(reversing) == 0,
            "answer_authority": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


# ============================================================================
# 5. NeuroHiveResourcePlanner (Thermal/Fault-Resilient Cognitive Dispatch)
# ============================================================================


class NeuroHiveResourcePlanner:
    """Resource-constrained, fault-resilient cognitive dispatch inspired by ``NEUROHIVE``."""

    def simulate_dispatch(
        self,
        *,
        tasks: Sequence[Mapping[str, Any]],
        token_budget: int = 1000,
        thermal_limit: float = 85.0,
        fault_rate: float = 0.15,
        enable_mnt_repair: bool = True,
        seed: int = 103,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        remaining_tokens = max(0, int(token_budget))
        spent_tokens = 0
        station_temp = 38.0
        completed: List[str] = []
        deferred: List[str] = []
        faults_injected = 0
        faults_repaired = 0
        thermal_throttles = 0

        for idx, task in enumerate(tasks):
            tid = str(task.get("id", f"task_{idx}"))
            cost = max(1, int(task.get("token_cost", 64)))
            heat = float(task.get("thermal_delta", 9.0))

            if cost > remaining_tokens:
                deferred.append(tid)
                continue

            # Thermal homeostasis cool-down throttle before exceeding limit
            if station_temp + heat > float(thermal_limit):
                station_temp = max(38.0, station_temp - 18.0)
                thermal_throttles += 1

            # Deterministic fault injection + autonomous MNT-01 repair check
            if rng.random() < float(fault_rate):
                faults_injected += 1
                if enable_mnt_repair and remaining_tokens >= cost + 12:
                    remaining_tokens -= 12
                    spent_tokens += 12
                    faults_repaired += 1
                else:
                    deferred.append(tid)
                    continue

            remaining_tokens -= cost
            spent_tokens += cost
            station_temp = max(38.0, station_temp + heat - 3.5)
            completed.append(tid)

        conservation_ok = (spent_tokens + remaining_tokens) == max(0, int(token_budget))
        sii = round(
            (len(completed) / float(max(1, len(tasks))))
            * (1.0 - 0.15 * (faults_injected - faults_repaired) / float(max(1, len(tasks)))),
            4,
        )
        payload: Dict[str, Any] = {
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "deferred_tasks": deferred,
            "spent_tokens": spent_tokens,
            "remaining_tokens": remaining_tokens,
            "budget_conserved": conservation_ok,
            "final_station_temp_c": round(station_temp, 2),
            "thermal_throttles": thermal_throttles,
            "faults_injected": faults_injected,
            "faults_repaired": faults_repaired,
            "swarm_intelligence_index": max(0.0, min(1.0, sii)),
            "answer_authority": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload
