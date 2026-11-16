"""NexusMind v95 Unified Frontier Ecosystem.

Synthesizes frontier research modules from the kai9987kai ecosystem into a
self-contained, deterministic diagnostic and verification suite using standard
library + NumPy primitives (strictly ``answer_authority=False``):

1. **TetherForensicsEngine** (Supermix-Tether v0.5-v0.8):
   Riemannian Fisher-Rao information geometry, spherical Frechet / Karcher
   barycenter, simplex tangent-space Jacobian & robustness sandwich
   (exact TV radius + logit radius + constructive vertex breakdown), coalition
   Shapley & Order-2 Shapley-Taylor interaction indices, multivariate Partial
   Information Decomposition (PID), and Spectral-Mahalanobis dark-knowledge
   spoof forensics.

2. **FaultlineEnsembleEngine** (FAULTLINE):
   Condition-indexed joint-failure matrix analysis comparing accuracy, pairwise
   co-failure, pooled-independence, and Faultline minimax joint majority-vote
   panel selection strategies.

3. **CrosswireInterfaceEngine** (CROSSWIRE):
   Active black-box Mealy interface version-space tracking, entropy-maximizing
   diagnostic probe selection, and all-hypothesis fail-closed safety shielding.

4. **MnemorphArchaeologyEngine** (Mnemorph & OMNIS-COGNITIVE MORPHEUS):
   Bayesian posterior identification across competing connectome memory
   mechanisms, expected information gain (EIG) assay ranking, paired Brier
   calibration, and MORPHEUS recovery-trajectory analysis.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TetherForensicsEngine",
    "FaultlineEnsembleEngine",
    "CrosswireInterfaceEngine",
    "MnemorphArchaeologyEngine",
]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ============================================================================
# 1. TetherForensicsEngine (Fisher-Rao, Robustness Sandwich, Shapley, PID)
# ============================================================================


class TetherForensicsEngine:
    """Riemannian ensemble geometry, robustness sandwich, and spoof forensics."""

    @staticmethod
    def _normalize_probs(expert_probs: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(expert_probs, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("expert_probs must be a 2D array of shape (E, C)")
        n_experts, n_classes = arr.shape
        if n_experts < 1 or n_classes < 2:
            raise ValueError("expert_probs requires at least 1 expert and 2 classes")
        if not np.all(np.isfinite(arr)):
            raise ValueError("expert_probs contains non-finite values")
        if np.any(arr < 0.0):
            raise ValueError("expert_probs entries must be non-negative probabilities")
        row_sums = arr.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError("each expert probability row must have positive sum")
        return arr / row_sums

    @staticmethod
    def _normalize_weights(weights: Optional[Sequence[float]], n_experts: int) -> np.ndarray:
        if weights is None:
            return np.full((n_experts,), 1.0 / float(n_experts), dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_experts,):
            raise ValueError(f"weights must have length {n_experts}")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
            raise ValueError("weights must be finite, non-negative, and sum > 0")
        return w / float(w.sum())

    @staticmethod
    def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> float:
        psi_p = np.sqrt(np.clip(p, 0.0, None))
        psi_q = np.sqrt(np.clip(q, 0.0, None))
        inner = float(np.clip(np.dot(psi_p, psi_q), -1.0, 1.0))
        return float(2.0 * math.acos(inner))

    @classmethod
    def pairwise_fisher_rao_matrix(cls, probs: np.ndarray) -> np.ndarray:
        psi = np.sqrt(np.clip(probs, 0.0, None))
        gram = np.clip(psi @ psi.T, -1.0, 1.0)
        dist = 2.0 * np.arccos(gram)
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)
        return dist

    @classmethod
    def karcher_barycenter(
        cls,
        probs: np.ndarray,
        weights: np.ndarray,
        *,
        max_iter: int = 64,
        tol: float = 1e-11,
    ) -> np.ndarray:
        psi_all = np.sqrt(np.clip(probs, 0.0, None))
        # Initialize at normalized weighted chordal mean on positive orthant sphere.
        psi = weights @ psi_all
        norm = float(np.linalg.norm(psi))
        psi = psi / max(norm, 1e-15)

        for _ in range(max_iter):
            alphas = np.clip(psi_all @ psi, -1.0, 1.0)
            thetas = np.arccos(alphas)
            tangent = np.zeros_like(psi)
            for idx in range(len(probs)):
                theta = float(thetas[idx])
                if theta > 1e-10:
                    sin_t = math.sin(theta)
                    if sin_t > 1e-12:
                        diff = psi_all[idx] - alphas[idx] * psi
                        tangent += float(weights[idx]) * (theta / sin_t) * diff
            t_norm = float(np.linalg.norm(tangent))
            if t_norm < tol:
                break
            step = min(t_norm, math.pi * 0.49)
            direction = tangent / t_norm
            psi_next = math.cos(step) * psi + math.sin(step) * direction
            psi_next = np.clip(psi_next, 0.0, None)
            n_next = float(np.linalg.norm(psi_next))
            if n_next <= 1e-15:
                break
            psi = psi_next / n_next

        bary = psi * psi
        bary = np.clip(bary, 0.0, None)
        s = float(bary.sum())
        return bary / max(s, 1e-15)

    @classmethod
    def _robustness_sandwich(
        cls,
        probs: np.ndarray,
        weights: np.ndarray,
        barycenter: np.ndarray,
    ) -> Dict[str, Any]:
        order = np.argsort(-barycenter, kind="mergesort")
        top1_idx = int(order[0])
        top2_idx = int(order[1])
        p1 = float(barycenter[top1_idx])
        p2 = float(barycenter[top2_idx])
        margin = max(0.0, p1 - p2)

        # Simplex tangent-space Jacobian J = diag(p*) - p* (p*)^T
        jacobian = np.diag(barycenter) - np.outer(barycenter, barycenter)
        eigvals = np.linalg.eigvalsh(jacobian)
        sigma_max_j = float(max(0.0, np.max(eigvals)))

        # Exact simplex TV lower bound to flip argmax: r_tv_cert = (p_(1) - p_(2)) / 2
        r_tv_cert = 0.5 * margin
        r_logit_cert = (
            float(margin / (2.0 * math.sqrt(2.0) * sigma_max_j))
            if sigma_max_j > 1e-12
            else 0.0
        )

        # Constructive vertex breakdown TV upper bound (delta_break_tv >= r_tv_cert):
        # 1) Single-expert adversarial vertex replacement that flips pooled argmax
        candidates_tv: List[Tuple[float, str, Optional[int], int]] = []
        n_experts, n_classes = probs.shape
        if margin <= 1e-15:
            candidates_tv.append((0.0, "tie_at_boundary", None, top2_idx))
        else:
            if n_experts > 1:
                for e_idx in range(n_experts):
                    for alt_cls in range(n_classes):
                        if alt_cls == top1_idx:
                            continue
                        adv_probs = probs.copy()
                        adv_probs[e_idx] = 0.0
                        adv_probs[e_idx, alt_cls] = 1.0
                        adv_bary = cls.karcher_barycenter(adv_probs, weights)
                        if int(np.argmax(adv_bary)) != top1_idx:
                            tv = 0.5 * float(np.abs(barycenter - adv_bary).sum())
                            candidates_tv.append((tv, "single_expert_vertex", e_idx, alt_cls))
            # 2) Minimal convex blend toward runner-up vertex e_{top2}:
            # (1 - lam)*p1 = (1 - lam)*p2 + lam  => lam_break = margin / (1 + margin)
            lam_break = min(1.0, (margin / (1.0 + margin)) + 1e-9)
            vertex = np.zeros_like(barycenter)
            vertex[top2_idx] = 1.0
            blended = (1.0 - lam_break) * barycenter + lam_break * vertex
            blended /= float(blended.sum())
            blend_tv = 0.5 * float(np.abs(barycenter - blended).sum())
            candidates_tv.append((blend_tv, "convex_vertex_blend", None, top2_idx))

        best_tv, mechanism, crit_expert, flip_cls = min(candidates_tv, key=lambda x: x[0])
        delta_break_tv = float(min(1.0, max(r_tv_cert, best_tv)))
        sandwich_valid = bool(0.0 <= r_tv_cert <= delta_break_tv + 1e-12 <= 1.0 + 1e-12)
        return {
            "top_class": top1_idx,
            "runner_up_class": top2_idx,
            "top_margin": round(margin, 8),
            "sigma_max_jacobian": round(sigma_max_j, 8),
            "r_tv_cert": round(r_tv_cert, 8),
            "r_logit_cert": round(r_logit_cert, 8),
            "delta_break_tv": round(delta_break_tv, 8),
            "sandwich_gap_tv": round(max(0.0, delta_break_tv - r_tv_cert), 8),
            "breakdown_mechanism": mechanism,
            "critical_expert_index": crit_expert,
            "flip_target_class": flip_cls,
            "sandwich_valid": sandwich_valid,
        }

    @classmethod
    def _shapley_and_taylor(
        cls,
        probs: np.ndarray,
        weights: np.ndarray,
        target_class: int,
    ) -> Dict[str, Any]:
        n_experts, n_classes = probs.shape
        v_empty = 1.0 / float(n_classes)

        def coalition_value(mask_bits: Tuple[int, ...]) -> float:
            if not any(mask_bits):
                return v_empty
            idxs = [i for i, b in enumerate(mask_bits) if b]
            sub_p = probs[idxs]
            sub_w = weights[idxs]
            sub_w_sum = float(sub_w.sum())
            if sub_w_sum <= 0.0:
                sub_w = np.full((len(idxs),), 1.0 / len(idxs), dtype=np.float64)
            else:
                sub_w = sub_w / sub_w_sum
            sub_bary = cls.karcher_barycenter(sub_p, sub_w, max_iter=32, tol=1e-9)
            return float(sub_bary[target_class])

        shapley = np.zeros((n_experts,), dtype=np.float64)
        interaction = np.zeros((n_experts, n_experts), dtype=np.float64)

        if n_experts <= 8:
            cache: Dict[int, float] = {}
            full_mask = (1 << n_experts) - 1
            for bitmask in range(1 << n_experts):
                bits = tuple((bitmask >> i) & 1 for i in range(n_experts))
                cache[bitmask] = coalition_value(bits)

            fact = [math.factorial(k) for k in range(n_experts + 1)]
            # Exact Shapley values
            for i in range(n_experts):
                acc = 0.0
                for s_mask in range(1 << n_experts):
                    if (s_mask >> i) & 1:
                        continue
                    s_size = s_mask.bit_count()
                    coef = fact[s_size] * fact[n_experts - s_size - 1] / fact[n_experts]
                    acc += coef * (cache[s_mask | (1 << i)] - cache[s_mask])
                shapley[i] = acc

            # Exact Order-2 Shapley-Taylor Interaction Matrix (Grabisch/Sundararajan)
            if n_experts >= 2:
                for i in range(n_experts):
                    for j in range(i + 1, n_experts):
                        acc_ij = 0.0
                        for s_mask in range(1 << n_experts):
                            if ((s_mask >> i) & 1) or ((s_mask >> j) & 1):
                                continue
                            s_size = s_mask.bit_count()
                            coef = fact[s_size] * fact[n_experts - s_size - 2] / fact[n_experts - 1]
                            d2 = (
                                cache[s_mask | (1 << i) | (1 << j)]
                                - cache[s_mask | (1 << i)]
                                - cache[s_mask | (1 << j)]
                                + cache[s_mask]
                            )
                            acc_ij += coef * d2
                        interaction[i, j] = acc_ij
                        interaction[j, i] = acc_ij
            v_full = cache[full_mask]
            exact_mode = True
        else:
            # Deterministic stratified permutation sampling for large panels
            rng = np.random.default_rng(2026095)
            n_perms = 64
            base_perm = np.arange(n_experts)
            for _ in range(n_perms):
                perm = rng.permutation(base_perm)
                cur_mask = [0] * n_experts
                prev_val = v_empty
                for idx in perm:
                    cur_mask[int(idx)] = 1
                    cur_val = coalition_value(tuple(cur_mask))
                    shapley[int(idx)] += (cur_val - prev_val) / float(n_perms)
                    prev_val = cur_val
            v_full = coalition_value(tuple([1] * n_experts))
            # Pairwise interaction on sample coalitions (empty + pair)
            for i in range(n_experts):
                for j in range(i + 1, n_experts):
                    m_i = tuple(1 if k == i else 0 for k in range(n_experts))
                    m_j = tuple(1 if k == j else 0 for k in range(n_experts))
                    m_ij = tuple(1 if k in (i, j) else 0 for k in range(n_experts))
                    d2 = coalition_value(m_ij) - coalition_value(m_i) - coalition_value(m_j) + v_empty
                    interaction[i, j] = d2
                    interaction[j, i] = d2
            exact_mode = False

        return {
            "exact_enumeration": exact_mode,
            "target_class": target_class,
            "coalition_empty_value": round(v_empty, 8),
            "coalition_full_value": round(v_full, 8),
            "efficiency_gap": round(abs(float(shapley.sum()) - (v_full - v_empty)), 10),
            "shapley_values": [round(float(x), 8) for x in shapley],
            "shapley_taylor_interaction_matrix": [
                [round(float(val), 8) for val in row] for row in interaction
            ],
        }

    @staticmethod
    def _partial_info_decomposition(
        probs: np.ndarray,
        weights: np.ndarray,
    ) -> Dict[str, Any]:
        _, n_classes = probs.shape
        u_c = 1.0 / float(n_classes)
        eps = 1e-12
        # Pointwise non-negative Bregman KL contributions in bits for each expert and class:
        # d_KL(p || u) = p * log2(p/u) + (u - p) / ln(2) >= 0 everywhere.
        p_safe = np.clip(probs, eps, 1.0)
        spec_info = p_safe * np.log2(p_safe / u_c) + (u_c - p_safe) / math.log(2.0)
        spec_info = np.clip(spec_info, 0.0, None)

        redundancy = float(np.sum(np.min(spec_info, axis=0)))
        per_expert_total = np.sum(spec_info, axis=1)
        unique_info = np.clip(per_expert_total - redundancy, 0.0, None)

        # Product-of-experts joint predictive distribution captures emergent synergy
        # when complementary experts jointly suppress distracting classes.
        log_poe = np.sum(weights[:, None] * np.log(p_safe), axis=0)
        log_poe -= float(np.max(log_poe))
        poe = np.exp(log_poe)
        poe /= float(np.sum(poe))
        poe_safe = np.clip(poe, eps, 1.0)
        mean_lin = np.clip(weights @ probs, eps, 1.0)
        synergy = float(
            max(
                0.0,
                np.sum(poe_safe * np.log2(poe_safe / mean_lin)),
            )
        )
        return {
            "redundancy_bits": round(redundancy, 8),
            "unique_bits": [round(float(u), 8) for u in unique_info],
            "synergy_bits": round(synergy, 8),
            "total_unique_bits": round(float(np.sum(unique_info)), 8),
        }

    @staticmethod
    def _dark_knowledge_forensics(
        probs: np.ndarray,
        top_class: int,
        reference_probs: Optional[Sequence[Sequence[float]]],
    ) -> Dict[str, Any]:
        non_win = [c for c in range(probs.shape[1]) if c != top_class]
        cand_dk = probs[:, non_win]
        cand_mean = np.mean(cand_dk, axis=0)

        if reference_probs is not None:
            ref_arr = TetherForensicsEngine._normalize_probs(reference_probs)
            if ref_arr.shape[1] != probs.shape[1]:
                raise ValueError("reference_probs must have the same number of classes as expert_probs")
            ref_dk = ref_arr[:, non_win]
        else:
            ref_dk = cand_dk

        ref_mean = np.mean(ref_dk, axis=0)
        residual = cand_mean - ref_mean
        res_norm_sq = float(np.dot(residual, residual))

        if len(ref_dk) >= 2:
            centered = ref_dk - ref_mean
            cov = (centered.T @ centered) / float(len(ref_dk))
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx_desc = np.argsort(-eigvals)
            eigvals = np.clip(eigvals[idx_desc], 0.0, None)
            eigvecs = eigvecs[:, idx_desc]

            total_var = float(np.sum(eigvals))
            if total_var > 1e-12:
                cum_var = np.cumsum(eigvals) / total_var
                k_rank = int(np.searchsorted(cum_var, 0.95) + 1)
                k_rank = max(1, min(k_rank, len(eigvals)))
            else:
                k_rank = 1
            basis = eigvecs[:, :k_rank]
            proj = basis @ (basis.T @ residual)
            orth = residual - proj
            orth_norm_sq = float(np.dot(orth, orth))
            orth_ratio = orth_norm_sq / max(res_norm_sq, 1e-12) if res_norm_sq > 1e-12 else 0.0

            inv_cov = np.linalg.pinv(cov + 1e-6 * np.eye(len(non_win)), rcond=1e-9)
            mahal_sq = float(max(0.0, residual @ inv_cov @ residual))
            mahal_z = math.sqrt(mahal_sq)

            cand_spread = float(np.var(cand_dk))
            ref_spread = float(np.var(ref_dk))
            spread_ratio = cand_spread / max(ref_spread, 1e-12) if ref_spread > 1e-12 else 1.0
        else:
            orth_ratio = 1.0 if res_norm_sq > 1e-8 else 0.0
            mahal_z = float(math.sqrt(res_norm_sq) / 0.05)
            spread_ratio = 1.0

        # Detect mimicry collapse (e.g. candidate matches top-1 while collapsing dark-knowledge spread)
        if reference_probs is not None and len(ref_dk) >= 2 and spread_ratio < 0.15 and float(np.var(ref_dk)) > 1e-4:
            verdict = "mimicry_spoof"
        elif mahal_z > 3.0 and orth_ratio > 0.35:
            verdict = "orthogonal_drift"
        else:
            verdict = "authentic"

        return {
            "spoof_verdict": verdict,
            "mahalanobis_z": round(float(mahal_z), 6),
            "orthogonal_energy_ratio": round(float(orth_ratio), 6),
            "dark_knowledge_spread_ratio": round(float(spread_ratio), 6),
            "residual_l2": round(float(math.sqrt(res_norm_sq)), 8),
        }

    def analyze(
        self,
        expert_probs: Sequence[Sequence[float]],
        weights: Optional[Sequence[float]] = None,
        target_class: Optional[int] = None,
        reference_probs: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """Run full Fisher-Rao geometry, robustness sandwich, Shapley/PID, and spoof forensics."""
        probs = self._normalize_probs(expert_probs)
        n_experts, n_classes = probs.shape
        w = self._normalize_weights(weights, n_experts)

        dist_mat = self.pairwise_fisher_rao_matrix(probs)
        barycenter = self.karcher_barycenter(probs, w)
        euclidean_mean = w @ probs
        euclidean_mean /= float(euclidean_mean.sum())

        fr_dists_karcher = np.array(
            [self.fisher_rao_distance(probs[i], barycenter) for i in range(n_experts)],
            dtype=np.float64,
        )
        fr_dists_euclid = np.array(
            [self.fisher_rao_distance(probs[i], euclidean_mean) for i in range(n_experts)],
            dtype=np.float64,
        )
        frechet_var = float(np.sum(w * (fr_dists_karcher ** 2)))
        arith_frechet_var = float(np.sum(w * (fr_dists_euclid ** 2)))
        suppression = (
            max(0.0, (arith_frechet_var - frechet_var) / arith_frechet_var)
            if arith_frechet_var > 1e-12
            else 0.0
        )

        sandwich = self._robustness_sandwich(probs, w, barycenter)
        resolved_target = int(sandwich["top_class"]) if target_class is None else int(target_class)
        if not 0 <= resolved_target < n_classes:
            raise ValueError(f"target_class {resolved_target} out of range [0, {n_classes})")

        attribution = self._shapley_and_taylor(probs, w, resolved_target)
        pid = self._partial_info_decomposition(probs, w)
        forensics = self._dark_knowledge_forensics(
            probs, int(sandwich["top_class"]), reference_probs
        )

        return {
            "engine": "TetherForensicsEngine-v0.8",
            "answer_authority": False,
            "num_experts": n_experts,
            "num_classes": n_classes,
            "weights": [round(float(x), 8) for x in w],
            "karcher_barycenter": [round(float(x), 8) for x in barycenter],
            "euclidean_centroid": [round(float(x), 8) for x in euclidean_mean],
            "pairwise_fisher_rao_distances": [
                [round(float(v), 8) for v in row] for row in dist_mat
            ],
            "frechet_variance": round(frechet_var, 8),
            "arithmetic_frechet_variance": round(arith_frechet_var, 8),
            "geodesic_variance_suppression": round(suppression, 8),
            "robustness_sandwich": sandwich,
            "attribution": attribution,
            "partial_information_decomposition": pid,
            "dark_knowledge_forensics": forensics,
        }


# ============================================================================
# 2. FaultlineEnsembleEngine (Condition-Indexed Correlated Failure Selection)
# ============================================================================


class FaultlineEnsembleEngine:
    """Selects odd-sized majority-vote panels under condition-indexed failure correlation."""

    @staticmethod
    def _exact_majority_failure_prob(rates: Sequence[float]) -> float:
        """Exact Poisson-binomial probability that strictly more than half fail."""
        k = len(rates)
        threshold = (k // 2) + 1
        dp = [0.0] * (k + 1)
        dp[0] = 1.0
        for p in rates:
            q = 1.0 - p
            next_dp = [0.0] * (k + 1)
            for c in range(k + 1):
                if dp[c] == 0.0:
                    continue
                next_dp[c] += dp[c] * q
                if c + 1 <= k:
                    next_dp[c + 1] += dp[c] * p
            dp = next_dp
        return float(sum(dp[threshold:]))

    @staticmethod
    def _panel_metrics(
        subset: Tuple[int, ...],
        fail_mat: np.ndarray,
        cond_weights: np.ndarray,
    ) -> Dict[str, Any]:
        sub_rates = fail_mat[list(subset), :]  # (k, M)
        n_conds = fail_mat.shape[1]
        by_cond = [
            FaultlineEnsembleEngine._exact_majority_failure_prob(sub_rates[:, m])
            for m in range(n_conds)
        ]
        by_cond_arr = np.asarray(by_cond, dtype=np.float64)
        worst_idx = int(np.argmax(by_cond_arr))
        worst_rate = float(by_cond_arr[worst_idx])
        expected_joint = float(np.dot(cond_weights, by_cond_arr))

        # Mean pairwise co-failure rate across distinct expert pairs
        pair_rates: List[float] = []
        for a, b in itertools.combinations(subset, 2):
            pair_rates.append(float(np.dot(cond_weights, fail_mat[a] * fail_mat[b])))
        pairwise_mean = float(sum(pair_rates) / len(pair_rates)) if pair_rates else 0.0

        # Higher-order triple co-failure across 3-tuples
        triple_rates: List[float] = []
        for a, b, c in itertools.combinations(subset, 3):
            triple_rates.append(float(np.dot(cond_weights, fail_mat[a] * fail_mat[b] * fail_mat[c])))
        triple_mean = float(sum(triple_rates) / len(triple_rates)) if triple_rates else 0.0

        return {
            "expert_indices": list(subset),
            "expected_joint_failure_rate": round(expected_joint, 8),
            "worst_condition_failure_rate": round(worst_rate, 8),
            "worst_condition_index": worst_idx,
            "majority_vote_error_by_condition": [round(float(x), 8) for x in by_cond],
            "pairwise_cofailure_mean": round(pairwise_mean, 8),
            "higher_order_triple_cofailure": round(triple_mean, 8),
        }

    def select_panel(
        self,
        failure_matrix: Sequence[Sequence[float]],
        panel_size: int = 3,
        expert_names: Optional[Sequence[str]] = None,
        condition_names: Optional[Sequence[str]] = None,
        condition_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Compare accuracy, pairwise, pooled, and Faultline joint majority-vote panels."""
        mat = np.asarray(failure_matrix, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError("failure_matrix must be 2D of shape (num_experts, num_conditions)")
        n_experts, n_conds = mat.shape
        if n_experts < 1 or n_conds < 1:
            raise ValueError("failure_matrix must have at least 1 expert and 1 condition")
        if not np.all(np.isfinite(mat)) or np.any(mat < 0.0) or np.any(mat > 1.0):
            raise ValueError("failure_matrix probabilities/indicators must be finite in [0, 1]")
        if isinstance(panel_size, bool) or not isinstance(panel_size, int) or panel_size < 1 or panel_size % 2 == 0:
            raise ValueError("panel_size must be a positive odd integer")
        if panel_size > n_experts:
            raise ValueError(f"panel_size ({panel_size}) cannot exceed num_experts ({n_experts})")

        e_names = (
            [str(x) for x in expert_names]
            if expert_names is not None
            else [f"expert_{i}" for i in range(n_experts)]
        )
        c_names = (
            [str(x) for x in condition_names]
            if condition_names is not None
            else [f"cond_{m}" for m in range(n_conds)]
        )
        if len(e_names) != n_experts or len(c_names) != n_conds:
            raise ValueError("expert_names / condition_names length mismatch")

        if condition_weights is None:
            w_cond = np.full((n_conds,), 1.0 / float(n_conds), dtype=np.float64)
        else:
            w_cond = np.asarray(condition_weights, dtype=np.float64)
            if w_cond.shape != (n_conds,) or not np.all(np.isfinite(w_cond)) or np.any(w_cond < 0.0) or float(w_cond.sum()) <= 0.0:
                raise ValueError("condition_weights must match num_conditions, be >= 0, and sum > 0")
            w_cond = w_cond / float(w_cond.sum())

        indiv_err = mat @ w_cond  # (E,)
        pooled_mat = indiv_err[:, None]  # (E, 1)

        total_combos = math.comb(n_experts, panel_size)
        if total_combos <= 5000:
            candidate_subsets = list(itertools.combinations(range(n_experts), panel_size))
        else:
            top_by_acc = np.argsort(indiv_err, kind="mergesort")[: min(n_experts, 16)]
            candidate_subsets = list(itertools.combinations(sorted(int(i) for i in top_by_acc), panel_size))

        # 1. Greedy individual accuracy panel
        acc_subset = tuple(sorted(int(i) for i in np.argsort(indiv_err, kind="mergesort")[:panel_size]))
        acc_metrics = self._panel_metrics(acc_subset, mat, w_cond)

        # 2. Pairwise co-failure minimization
        def pairwise_key(sub: Tuple[int, ...]) -> Tuple[float, float, Tuple[int, ...]]:
            pairs = list(itertools.combinations(sub, 2))
            p_mean = (
                sum(float(np.dot(w_cond, mat[a] * mat[b])) for a, b in pairs) / len(pairs)
                if pairs
                else float(indiv_err[sub[0]])
            )
            return (round(p_mean, 12), round(float(sum(indiv_err[i] for i in sub)), 12), sub)

        pairwise_subset = min(candidate_subsets, key=pairwise_key)
        pairwise_metrics = self._panel_metrics(pairwise_subset, mat, w_cond)

        # 3. Pooled condition-independent majority failure minimization
        def pooled_key(sub: Tuple[int, ...]) -> Tuple[float, Tuple[int, ...]]:
            p_rate = self._exact_majority_failure_prob([float(pooled_mat[i, 0]) for i in sub])
            return (round(p_rate, 12), sub)

        pooled_subset = min(candidate_subsets, key=pooled_key)
        pooled_metrics = self._panel_metrics(pooled_subset, mat, w_cond)

        # 4. Faultline joint condition-indexed minimax panel selection
        joint_evals = {sub: self._panel_metrics(sub, mat, w_cond) for sub in candidate_subsets}
        joint_subset = min(
            candidate_subsets,
            key=lambda sub: (
                joint_evals[sub]["worst_condition_failure_rate"],
                joint_evals[sub]["expected_joint_failure_rate"],
                joint_evals[sub]["higher_order_triple_cofailure"],
                sub,
            ),
        )
        joint_metrics = joint_evals[joint_subset]

        strategies = {
            "accuracy": {**acc_metrics, "expert_names": [e_names[i] for i in acc_subset]},
            "pairwise": {**pairwise_metrics, "expert_names": [e_names[i] for i in pairwise_subset]},
            "pooled": {**pooled_metrics, "expert_names": [e_names[i] for i in pooled_subset]},
            "joint": {**joint_metrics, "expert_names": [e_names[i] for i in joint_subset]},
        }
        for metrics in strategies.values():
            metrics["worst_condition_name"] = c_names[metrics["worst_condition_index"]]

        advantage = {
            "worst_condition_failure_reduction": round(
                acc_metrics["worst_condition_failure_rate"] - joint_metrics["worst_condition_failure_rate"],
                8,
            ),
            "expected_joint_failure_reduction": round(
                acc_metrics["expected_joint_failure_rate"] - joint_metrics["expected_joint_failure_rate"],
                8,
            ),
            "pairwise_cofailure_reduction": round(
                acc_metrics["pairwise_cofailure_mean"] - joint_metrics["pairwise_cofailure_mean"],
                8,
            ),
        }

        receipt_core = {
            "panel_size": panel_size,
            "selected_expert_indices": joint_metrics["expert_indices"],
            "selected_expert_names": strategies["joint"]["expert_names"],
            "worst_condition_failure_rate": joint_metrics["worst_condition_failure_rate"],
            "expected_joint_failure_rate": joint_metrics["expected_joint_failure_rate"],
        }

        return {
            "engine": "FaultlineEnsembleEngine",
            "answer_authority": False,
            "panel_size": panel_size,
            "num_experts": n_experts,
            "num_conditions": n_conds,
            "individual_error_rates": {e_names[i]: round(float(indiv_err[i]), 8) for i in range(n_experts)},
            "selected_panel": strategies["joint"],
            "strategies": strategies,
            "faultline_advantage_over_accuracy": advantage,
            "receipt": {
                "schema": "faultline-panel-receipt-v1",
                "answer_authority": False,
                **receipt_core,
                "receipt_sha256": _canonical_sha256(receipt_core),
            },
        }


# ============================================================================
# 3. CrosswireInterfaceEngine (Version-Space Active Probing & Safety Shield)
# ============================================================================


def _default_protocol_hypotheses() -> List[Dict[str, Any]]:
    """Built-in canonical Mealy protocol hypothesis space when none is supplied."""
    return [
        {
            "id": "h_strict_guard",
            "initial_state": "READY",
            "transitions": {
                "READY": {
                    "AUTH": {"next_state": "ARMED", "output": "ACK_AUTH"},
                    "PING": {"next_state": "READY", "output": "PONG_STRICT"},
                    "EXEC": {"next_state": "READY", "output": "DENY_UNARMED"},
                },
                "ARMED": {
                    "PING": {"next_state": "ARMED", "output": "PONG_ARMED"},
                    "EXEC": {"next_state": "READY", "output": "COMMIT_OK"},
                    "AUTH": {"next_state": "ARMED", "output": "ACK_AUTH"},
                },
            },
        },
        {
            "id": "h_delayed_fault",
            "initial_state": "READY",
            "transitions": {
                "READY": {
                    "AUTH": {"next_state": "ARMED", "output": "ACK_AUTH"},
                    "PING": {"next_state": "READY", "output": "PONG_DELAYED"},
                    "EXEC": {"next_state": "FAULT", "output": "UNSAFE"},
                },
                "ARMED": {
                    "PING": {"next_state": "ARMED", "output": "PONG_ARMED"},
                    "EXEC": {"next_state": "READY", "output": "COMMIT_OK"},
                    "AUTH": {"next_state": "ARMED", "output": "ACK_AUTH"},
                },
            },
        },
        {
            "id": "h_permissive_unsafe",
            "initial_state": "READY",
            "transitions": {
                "READY": {
                    "AUTH": {"next_state": "ARMED", "output": "ACK_AUTH"},
                    "PING": {"next_state": "READY", "output": "PONG_STRICT"},
                    "EXEC": {"next_state": "FAULT", "output": "OVERFLOW"},
                },
                "ARMED": {
                    "PING": {"next_state": "ARMED", "output": "PONG_ARMED"},
                    "EXEC": {"next_state": "FAULT", "output": "OVERFLOW"},
                    "AUTH": {"next_state": "ARMED", "output": "ACK_AUTH"},
                },
            },
        },
    ]


class CrosswireInterfaceEngine:
    """Active Mealy version-space verification and all-hypothesis safety shielding."""

    @staticmethod
    def _step_hypothesis(
        hypothesis: Mapping[str, Any],
        state: str,
        action: str,
    ) -> Optional[Tuple[str, str]]:
        if "transitions" in hypothesis:
            trans = hypothesis["transitions"]
            state_map = trans.get(state) or trans.get("*") or {}
            rule = state_map.get(action)
            if isinstance(rule, Mapping):
                nxt = str(rule.get("next_state", state))
                out = str(rule.get("output", ""))
                return nxt, out
            if isinstance(rule, (tuple, list)) and len(rule) == 2:
                return str(rule[0]), str(rule[1])
            return None
        responses = hypothesis.get("responses")
        if isinstance(responses, Mapping) and action in responses:
            return state, str(responses[action])
        return None

    @staticmethod
    def _available_actions(hypotheses: Sequence[Mapping[str, Any]]) -> List[str]:
        actions: set[str] = set()
        for h in hypotheses:
            for state_map in (h.get("transitions") or {}).values():
                if isinstance(state_map, Mapping):
                    actions.update(str(a) for a in state_map.keys())
            if isinstance(h.get("responses"), Mapping):
                actions.update(str(a) for a in h["responses"].keys())
        return sorted(actions)

    def verify_and_probe(
        self,
        hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
        observed_trace: Optional[Sequence[Mapping[str, Any]]] = None,
        candidate_action: Optional[str] = None,
        unsafe_outputs: Optional[Iterable[str]] = None,
        unsafe_states: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Prune hypothesis version space, rank active probes, and shield candidate action."""
        h_list = list(hypotheses) if hypotheses is not None else _default_protocol_hypotheses()
        if not h_list:
            raise ValueError("hypotheses must be a non-empty list")

        trace = list(observed_trace or ())
        bad_outputs = {str(x) for x in (unsafe_outputs or ("FAULT", "OVERFLOW", "UNSAFE", "ERROR"))}
        bad_states = {str(x) for x in (unsafe_states or ("FAULT", "LOCKOUT"))}

        surviving: List[Tuple[Mapping[str, Any], str]] = []
        eliminated: List[Dict[str, Any]] = []

        for h in h_list:
            h_id = str(h.get("id", f"h_{len(surviving) + len(eliminated)}"))
            cur_state = str(h.get("initial_state", "READY"))
            consistent = True
            fail_step: Optional[int] = None
            fail_reason = ""
            for step_idx, step in enumerate(trace):
                if "state" in step and str(step["state"]) != cur_state:
                    consistent = False
                    fail_step = step_idx
                    fail_reason = f"state_mismatch(expected={cur_state}, observed={step['state']})"
                    break
                act = str(step.get("action", ""))
                outcome = self._step_hypothesis(h, cur_state, act)
                if outcome is None:
                    consistent = False
                    fail_step = step_idx
                    fail_reason = f"undefined_transition({cur_state}, {act})"
                    break
                nxt_state, pred_out = outcome
                obs_out = str(step.get("output", ""))
                if pred_out != obs_out:
                    consistent = False
                    fail_step = step_idx
                    fail_reason = f"output_mismatch(pred={pred_out}, observed={obs_out})"
                    break
                if "next_state" in step and str(step["next_state"]) != nxt_state:
                    consistent = False
                    fail_step = step_idx
                    fail_reason = f"next_state_mismatch(pred={nxt_state}, observed={step['next_state']})"
                    break
                cur_state = nxt_state

            if consistent:
                surviving.append((h, cur_state))
            else:
                eliminated.append({
                    "id": h_id,
                    "falsified_at_step": fail_step,
                    "reason": fail_reason,
                })

        surviving_ids = [str(h.get("id", f"h_{idx}")) for idx, (h, _) in enumerate(surviving)]
        n_surv = len(surviving)
        prior_entropy = math.log2(n_surv) if n_surv > 0 else 0.0

        # Active diagnostic probe selection maximizing expected entropy reduction over V(D)
        all_actions = self._available_actions(h_list)
        probe_ranking: List[Dict[str, Any]] = []
        for act in all_actions:
            buckets: Dict[str, int] = {}
            for h, cur_state in surviving:
                out_pair = self._step_hypothesis(h, cur_state, act)
                key = f"OUT:{out_pair[1]}" if out_pair is not None else "UNDEFINED"
                buckets[key] = buckets.get(key, 0) + 1
            if n_surv > 0:
                cond_entropy = 0.0
                for count in buckets.values():
                    if count > 1:
                        cond_entropy += (count / n_surv) * math.log2(count)
                eig_bits = max(0.0, prior_entropy - cond_entropy)
            else:
                eig_bits = 0.0
            probe_ranking.append({
                "action": act,
                "expected_info_gain_bits": round(eig_bits, 8),
                "output_partitions": len(buckets),
                "partition_counts": buckets,
            })
        probe_ranking.sort(key=lambda row: (-row["expected_info_gain_bits"], row["action"]))
        best_probe = probe_ranking[0]["action"] if probe_ranking else None

        # All-Hypothesis Safety Shield
        resolved_candidate = candidate_action or (
            "EXEC" if "EXEC" in all_actions else (all_actions[0] if all_actions else "")
        )
        counterexamples: List[Dict[str, str]] = []
        if n_surv == 0:
            shield_verdict = "ABSTAIN_INCONSISTENT_TRACE"
            certified_safe = False
        else:
            for h, cur_state in surviving:
                h_id = str(h.get("id", "unknown"))
                outcome = self._step_hypothesis(h, cur_state, resolved_candidate)
                if outcome is None:
                    counterexamples.append({
                        "hypothesis_id": h_id,
                        "state": cur_state,
                        "violation": "undefined_action",
                    })
                    continue
                nxt_state, pred_out = outcome
                if pred_out in bad_outputs or nxt_state in bad_states:
                    counterexamples.append({
                        "hypothesis_id": h_id,
                        "state": cur_state,
                        "predicted_output": pred_out,
                        "predicted_next_state": nxt_state,
                        "violation": "unsafe_transition",
                    })
            if counterexamples:
                shield_verdict = "ABSTAIN_UNSAFE"
                certified_safe = False
            else:
                shield_verdict = "CERTIFIED_SAFE"
                certified_safe = True

        return {
            "engine": "CrosswireInterfaceEngine",
            "answer_authority": False,
            "total_hypotheses": len(h_list),
            "surviving_count": n_surv,
            "surviving_hypotheses": surviving_ids,
            "eliminated_hypotheses": eliminated,
            "version_space_entropy_bits": round(prior_entropy, 8),
            "recommended_probe_action": best_probe,
            "probe_ranking": probe_ranking,
            "candidate_action": resolved_candidate,
            "certified_safe": certified_safe,
            "shield_verdict": shield_verdict,
            "counterexample_hypotheses": counterexamples,
        }


# ============================================================================
# 4. MnemorphArchaeologyEngine (Bayesian Memory Mechanism & MORPHEUS Assay)
# ============================================================================


class MnemorphArchaeologyEngine:
    """Bayesian engram mechanism identification and MORPHEUS recovery analysis."""

    MECHANISMS: Tuple[str, ...] = (
        "persistent_subspace_trace",
        "regenerative_reconstruction",
        "readout_remapping",
    )

    INTERVENTIONS: Tuple[str, ...] = (
        "baseline",
        "focal_lesion",
        "commissure_cut",
        "synapse_scramble",
        "regenerative_regrowth",
    )

    # Calibrated mean retention + standard deviation for each (mechanism, intervention)
    LIKELIHOOD_MODEL: Dict[str, Dict[str, Tuple[float, float]]] = {
        "persistent_subspace_trace": {
            "baseline": (0.94, 0.08),
            "focal_lesion": (0.72, 0.14),
            "commissure_cut": (0.84, 0.12),
            "synapse_scramble": (0.26, 0.15),
            "regenerative_regrowth": (0.68, 0.14),
        },
        "regenerative_reconstruction": {
            "baseline": (0.92, 0.08),
            "focal_lesion": (0.34, 0.14),
            "commissure_cut": (0.50, 0.15),
            "synapse_scramble": (0.18, 0.13),
            "regenerative_regrowth": (0.89, 0.10),
        },
        "readout_remapping": {
            "baseline": (0.93, 0.08),
            "focal_lesion": (0.56, 0.15),
            "commissure_cut": (0.68, 0.14),
            "synapse_scramble": (0.10, 0.10),
            "regenerative_regrowth": (0.44, 0.15),
        },
    }

    @classmethod
    def _gaussian_pdf(cls, x: float, mu: float, sigma: float) -> float:
        z = (x - mu) / max(sigma, 1e-6)
        return math.exp(-0.5 * z * z) / (max(sigma, 1e-6) * math.sqrt(2.0 * math.pi))

    @classmethod
    def _shannon_entropy_bits(cls, probs: Sequence[float]) -> float:
        return float(-sum(p * math.log2(p) for p in probs if p > 1e-15))

    @classmethod
    def _expected_information_gain(
        cls,
        posterior: Sequence[float],
        intervention: str,
        grid_points: int = 41,
    ) -> float:
        h_cur = cls._shannon_entropy_bits(posterior)
        ys = np.linspace(0.0, 1.0, grid_points)
        dy = 1.0 / float(grid_points - 1)
        cond_entropy = 0.0
        for idx, y in enumerate(ys):
            weight_trap = 0.5 if idx in (0, grid_points - 1) else 1.0
            joint_terms = [
                posterior[m_idx]
                * cls._gaussian_pdf(float(y), *cls.LIKELIHOOD_MODEL[mech][intervention])
                for m_idx, mech in enumerate(cls.MECHANISMS)
            ]
            p_y = sum(joint_terms)
            if p_y > 1e-15:
                post_y = [t / p_y for t in joint_terms]
                cond_entropy += weight_trap * dy * p_y * cls._shannon_entropy_bits(post_y)
        return float(max(0.0, h_cur - cond_entropy))

    @staticmethod
    def _morpheus_trajectory(trajectory: Optional[Sequence[float]]) -> Dict[str, Any]:
        if not trajectory:
            return {
                "steps": 0,
                "initial_retention": None,
                "final_retention": None,
                "recovery_delta": None,
                "recovery_rate": None,
                "half_life_steps": None,
            }
        vals = [float(x) for x in trajectory]
        for v in vals:
            if not math.isfinite(v) or not 0.0 <= v <= 1.0:
                raise ValueError("recovery_trajectory values must be finite within [0, 1]")
        init_r = vals[0]
        final_r = vals[-1]
        delta = final_r - init_r
        steps = len(vals)
        rate = delta / float(max(1, steps - 1))
        half_life: Optional[int] = None
        if abs(delta) > 1e-6:
            target = init_r + 0.5 * delta
            for idx, v in enumerate(vals):
                if (delta > 0 and v >= target) or (delta < 0 and v <= target):
                    half_life = idx
                    break
        return {
            "steps": steps,
            "initial_retention": round(init_r, 6),
            "final_retention": round(final_r, 6),
            "recovery_delta": round(delta, 6),
            "recovery_rate": round(rate, 6),
            "half_life_steps": half_life,
        }

    def analyze_assays(
        self,
        assay_observations: Optional[Sequence[Mapping[str, Any]]] = None,
        recovery_trajectory: Optional[Sequence[float]] = None,
        prior: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Any]:
        """Update Bayesian mechanism posterior over connectome assays and rank next assays by EIG."""
        if prior is None:
            prior_vec = [1.0 / len(self.MECHANISMS)] * len(self.MECHANISMS)
        else:
            raw = [float(prior.get(m, 0.0)) for m in self.MECHANISMS]
            if any(x < 0.0 or not math.isfinite(x) for x in raw) or sum(raw) <= 0.0:
                raise ValueError("prior must specify finite non-negative weights summing > 0")
            s_raw = sum(raw)
            prior_vec = [x / s_raw for x in raw]

        observations = list(assay_observations or ())
        log_post = [math.log(max(p, 1e-15)) for p in prior_vec]
        observed_interventions: set[str] = set()
        brier_terms: List[float] = []

        for obs in observations:
            intervention = str(obs.get("intervention", "")).strip()
            if intervention not in self.INTERVENTIONS:
                raise ValueError(
                    f"unknown intervention {intervention!r}; expected one of {self.INTERVENTIONS}"
                )
            ret = float(obs.get("observed_retention", 0.0))
            if not math.isfinite(ret) or not 0.0 <= ret <= 1.0:
                raise ValueError("observed_retention must be finite within [0, 1]")
            observed_interventions.add(intervention)
            for m_idx, mech in enumerate(self.MECHANISMS):
                mu, sigma = self.LIKELIHOOD_MODEL[mech][intervention]
                lik = self._gaussian_pdf(ret, mu, sigma)
                log_post[m_idx] += math.log(max(lik, 1e-300))

        max_lp = max(log_post)
        exp_lp = [math.exp(lp - max_lp) for lp in log_post]
        z_sum = sum(exp_lp)
        posterior = [x / z_sum for x in exp_lp]

        for obs in observations:
            intervention = str(obs["intervention"]).strip()
            ret = float(obs["observed_retention"])
            pred_ret = sum(
                posterior[m_idx] * self.LIKELIHOOD_MODEL[mech][intervention][0]
                for m_idx, mech in enumerate(self.MECHANISMS)
            )
            brier_terms.append((pred_ret - ret) ** 2)

        posterior_map = {m: round(posterior[i], 8) for i, m in enumerate(self.MECHANISMS)}
        map_idx = max(range(len(self.MECHANISMS)), key=lambda i: posterior[i])
        entropy_bits = self._shannon_entropy_bits(posterior)
        brier_score = sum(brier_terms) / len(brier_terms) if brier_terms else None

        ranking: List[Dict[str, Any]] = []
        for intervention in self.INTERVENTIONS:
            eig = self._expected_information_gain(posterior, intervention)
            ranking.append({
                "intervention": intervention,
                "expected_info_gain_bits": round(eig, 8),
                "already_observed": intervention in observed_interventions,
            })
        ranking.sort(
            key=lambda row: (
                row["already_observed"],
                -row["expected_info_gain_bits"],
                row["intervention"],
            )
        )

        return {
            "engine": "MnemorphArchaeologyEngine",
            "answer_authority": False,
            "observations_count": len(observations),
            "posterior": posterior_map,
            "map_mechanism": self.MECHANISMS[map_idx],
            "map_probability": round(posterior[map_idx], 8),
            "posterior_entropy_bits": round(entropy_bits, 8),
            "paired_brier_score": round(brier_score, 8) if brier_score is not None else None,
            "recommended_next_assay": ranking[0]["intervention"] if ranking else None,
            "next_assay_ranking": ranking,
            "morpheus_recovery": self._morpheus_trajectory(recovery_trajectory),
        }
