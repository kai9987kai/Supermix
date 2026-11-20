"""Supermix v102 Frontier Research Engines (cross-ecosystem synthesis).

Self-contained, deterministic standard-library + NumPy engines synthesizing six
previously untapped ``kai9987kai`` GitHub research platforms alongside 2025-2026
frontier mechanisms into Supermix's Cognitive Fabric (all strictly
``answer_authority=False`` with SHA-256 receipts):

1. **NexusSearchHybridRRF** (``kai9987kai/NexusSearch`` + ``OMNIS-COGNITIVE/mnemorph``
   + Titans surprise-gated memory):
   Exact BM25 + hashed sparse cosine vector retrieval fused via Reciprocal Rank
   Fusion (RRF, ``k=60``), 7-class memory taxonomy, and running-centroid
   surprise scoring.

2. **TesseraRatifiedLemmaCommons** (``kai9987kai/TESSERA``):
   Bounded reusable reasoning-lemma / opcode commons enforcing independent
   rediscovery across ``>= 2`` distinct lineages/task families + deterministic
   verification before ratification, with epoch-based expiry of unused entries.

3. **CounterformFalsificationAuditor** (``kai9987kai/COUNTERFORM`` v0.2):
   Decision-sensitive CVaR (worst-tail) + mean risk scoring over sensitivity
   particles plus dual consistency/projection audit emitting ``RECOMMEND`` or
   fail-closed ``ABSTAIN_UNSUPPORTED``.

4. **CarrierInteractionAtlas** (``kai9987kai/MEMORY-CARRIER-OBSERVATORY`` +
   ``GhostInTheMachine`` v7 + ``Supermix-Aether``):
   - Exact ``2^k`` subset-factorial Mobius interaction decomposition (main,
     pairwise, 3-way, 4-way terms with exact additive reconstruction check).
   - State-predicted error stability index (``R^2`` of comparator error predicted
     by internal state, GhostInTheMachine v7).
   - Analytical compute & KV-cache resource-parity profiler for ``MiMoMixConfig``.

5. **SonapseDiscoveryMonitor** (``kai9987kai/sonapse-evo`` 2.0):
   Dual-mode Repair (Mode A) vs Discovery Opportunity (Mode B) monitor with
   Pareto multi-objective acceptance gating for structural neurogenesis.

6. **HyperionComputeScheduler** (``kai9987kai/HYPERION-20``):
   Heterogeneous cognitive service-class dependency DAG scheduler mapping task
   pipelines across Supermix's deterministic verifiers, connectome atlas,
   neural models, adaptive conformal gates, and forensic engines.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FRONTIER_V102_VERSION",
    "MEMORY_CLASSES_7",
    "NexusSearchHybridRRF",
    "TesseraRatifiedLemmaCommons",
    "CounterformFalsificationAuditor",
    "CarrierInteractionAtlas",
    "SonapseDiscoveryMonitor",
    "HyperionComputeScheduler",
]

FRONTIER_V102_VERSION = "102.0.0-frontier"

MEMORY_CLASSES_7: Tuple[str, ...] = (
    "sensory",
    "working",
    "episodic",
    "semantic",
    "procedural",
    "long_term",
    "experimental",
)

_TOKEN_RE = re.compile(r"[a-z0-9_+-]{2,}")


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
# 1. NexusSearchHybridRRF (BM25 + Sparse Vector Cosine + RRF + Titans Surprise)
# ============================================================================


class NexusSearchHybridRRF:
    """Deterministic hybrid BM25 + hashed-vector cosine + Reciprocal Rank Fusion."""

    def __init__(
        self,
        *,
        vector_dim: int = 128,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self.vector_dim = max(16, int(vector_dim))
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)
        self.rrf_k = max(1, int(rrf_k))

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(str(text or "").lower())

    def hash_embed(self, text: str) -> np.ndarray:
        """Deterministic signed feature-hashing embedding normalized to unit L2."""
        vec = np.zeros((self.vector_dim,), dtype=np.float64)
        tokens = self.tokenize(text)
        if not tokens:
            return vec
        # Unigrams + adjacent bigrams capture phrase alignment without external models
        feats = list(tokens) + [
            f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)
        ]
        for feat in feats:
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            idx = val % self.vector_dim
            sign = 1.0 if ((val >> 17) & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec

    @staticmethod
    def classify_memory(text: str, *, origin: str = "turn") -> str:
        """Map text into the 7-class OMNIS/Mnemorph cognitive memory taxonomy."""
        low = str(text or "").lower()
        if any(k in low for k in ("prefer", "always ", "never ", "keep answers", "style", "constraint", "step by step")):
            return "procedural"
        if any(k in low for k in ("remember ", "we use ", "project ", "architecture", "repo", "policy")):
            return "long_term"
        if any(k in low for k in ("define", "formula", "theorem", "male cns", "synapse", "law of")):
            return "semantic"
        if any(k in low for k in ("experiment", "ablation", "seed", "hypothesis", "benchmark")):
            return "experimental"
        if origin == "working":
            return "working"
        if origin == "sensory":
            return "sensory"
        return "episodic"

    def rank_documents(
        self,
        query: str,
        documents: Sequence[Mapping[str, Any]],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Rank documents using exact BM25 + cosine vector + RRF + Titans surprise."""
        docs = list(documents)
        if not docs:
            return {
                "hits": [],
                "query_surprise": 0.0,
                "memory_classes_active": [],
                "rrf_k": self.rrf_k,
                "answer_authority": False,
            }

        q_tokens = self.tokenize(query)
        q_vec = self.hash_embed(query)
        doc_tokens = [self.tokenize(str(d.get("text") or "")) for d in docs]
        doc_vecs = [self.hash_embed(str(d.get("text") or "")) for d in docs]

        # Running session centroid -> Titans surprise score = 1 - max(0, cos(q, centroid))
        centroid = np.mean(np.stack(doc_vecs, axis=0), axis=0)
        c_norm = float(np.linalg.norm(centroid))
        if c_norm > 1e-12 and float(np.linalg.norm(q_vec)) > 1e-12:
            centroid = centroid / c_norm
            query_surprise = float(max(0.0, min(1.0, 1.0 - max(0.0, float(np.dot(q_vec, centroid))))))
        else:
            query_surprise = 1.0 if q_tokens else 0.0

        # Exact Okapi BM25 scores
        n_docs = len(docs)
        avgdl = sum(len(toks) for toks in doc_tokens) / float(max(1, n_docs))
        df: Dict[str, int] = {}
        for toks in doc_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        bm25_scores: List[float] = []
        cos_scores: List[float] = []
        for idx, toks in enumerate(doc_tokens):
            dl = len(toks)
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for qt in q_tokens:
                freq = tf.get(qt, 0)
                if freq <= 0:
                    continue
                n_qi = df.get(qt, 0)
                idf = math.log(1.0 + (n_docs - n_qi + 0.5) / (n_qi + 0.5))
                denom = freq + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * (dl / max(avgdl, 1e-9)))
                score += idf * (freq * (self.bm25_k1 + 1.0)) / max(denom, 1e-9)
            bm25_scores.append(score)
            cos_scores.append(float(np.dot(q_vec, doc_vecs[idx])))

        # Reciprocal Rank Fusion over BM25 rank, Vector rank, and Recency rank
        bm25_order = sorted(range(n_docs), key=lambda i: (-bm25_scores[i], -i))
        cos_order = sorted(range(n_docs), key=lambda i: (-cos_scores[i], -i))
        recency_order = list(reversed(range(n_docs)))

        bm25_rank = {doc_i: r + 1 for r, doc_i in enumerate(bm25_order)}
        cos_rank = {doc_i: r + 1 for r, doc_i in enumerate(cos_order)}
        rec_rank = {doc_i: r + 1 for r, doc_i in enumerate(recency_order)}

        scored: List[Dict[str, Any]] = []
        classes_seen: set[str] = set()
        for idx, doc in enumerate(docs):
            text = str(doc.get("text") or "")
            mclass = str(doc.get("memory_class") or self.classify_memory(text))
            if mclass not in MEMORY_CLASSES_7:
                mclass = "episodic"
            classes_seen.add(mclass)
            rrf = (
                (1.0 / (self.rrf_k + bm25_rank[idx]))
                + (1.0 / (self.rrf_k + cos_rank[idx]))
                + (0.5 / (self.rrf_k + rec_rank[idx]))
            )
            # Small deterministic surprise boost for salient procedural/long-term rules
            pref_boost = 0.0025 if mclass in {"procedural", "long_term"} else 0.0
            scored.append(
                {
                    "id": str(doc.get("id", f"doc_{idx}")),
                    "memory_class": mclass,
                    "text": text[:200],
                    "bm25": round(bm25_scores[idx], 6),
                    "cosine": round(cos_scores[idx], 6),
                    "rrf_score": round(rrf + pref_boost, 6),
                    "bm25_rank": bm25_rank[idx],
                    "cosine_rank": cos_rank[idx],
                    "claims_can_authorize_tools": False,
                }
            )
        scored.sort(key=lambda item: (-item["rrf_score"], -item["bm25"], -item["cosine"]))
        return {
            "hits": scored[: max(1, int(top_k))],
            "query_surprise": round(query_surprise, 6),
            "memory_classes_active": sorted(classes_seen),
            "rrf_k": self.rrf_k,
            "answer_authority": False,
        }


# ============================================================================
# 2. TesseraRatifiedLemmaCommons (Independent Rediscovery + Expiry)
# ============================================================================


@dataclass
class _LemmaEntry:
    lemma_id: str
    signature: str
    expression: str
    lineages: set[str] = field(default_factory=set)
    task_families: set[str] = field(default_factory=set)
    verified_passed: bool = False
    ratified: bool = False
    ratified_epoch: Optional[int] = None
    last_used_epoch: int = 0
    use_count: int = 0


class TesseraRatifiedLemmaCommons:
    """Bounded shared reasoning-lemma library with independent-lineage ratification.

    Inspired by ``kai9987kai/TESSERA``:
    - A candidate lemma/motif is NOT published on first discovery.
    - It is ratified into the shared commons ONLY after ``>= min_independent_lineages``
      distinct lineages/models independently rediscover the exact canonical signature
      AND it passes deterministic verification.
    - Unused ratified lemmas expire after ``expiry_epochs`` idle epochs so the
      library stays bounded and avoids bloat or premature collapse.
    """

    def __init__(
        self,
        *,
        min_independent_lineages: int = 2,
        expiry_epochs: int = 2,
        max_active_lemmas: int = 64,
    ) -> None:
        self.min_independent_lineages = max(2, int(min_independent_lineages))
        self.expiry_epochs = max(1, int(expiry_epochs))
        self.max_active_lemmas = max(4, int(max_active_lemmas))
        self.epoch: int = 0
        self._entries: Dict[str, _LemmaEntry] = {}
        self.expired_ids: List[str] = []

    @staticmethod
    def canonical_signature(expression: str) -> str:
        norm = " ".join(str(expression or "").strip().lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    def submit_discovery(
        self,
        *,
        expression: str,
        lineage: str,
        task_family: str,
        verified_passed: bool = True,
    ) -> Dict[str, Any]:
        """Record a discovery by one lineage; ratify only upon independent rediscovery."""
        sig = self.canonical_signature(expression)
        entry = self._entries.get(sig)
        if entry is None:
            entry = _LemmaEntry(
                lemma_id=f"lem_{sig[:8]}",
                signature=sig,
                expression=str(expression).strip(),
                last_used_epoch=self.epoch,
            )
            self._entries[sig] = entry
        entry.lineages.add(str(lineage).strip() or "unknown")
        entry.task_families.add(str(task_family).strip() or "general")
        if verified_passed:
            entry.verified_passed = True
        entry.last_used_epoch = self.epoch

        newly_ratified = False
        if (
            not entry.ratified
            and entry.verified_passed
            and len(entry.lineages) >= self.min_independent_lineages
        ):
            entry.ratified = True
            entry.ratified_epoch = self.epoch
            newly_ratified = True
            self._enforce_capacity()

        return {
            "lemma_id": entry.lemma_id,
            "signature": entry.signature,
            "ratified": entry.ratified,
            "newly_ratified": newly_ratified,
            "independent_lineages": sorted(entry.lineages),
            "task_families": sorted(entry.task_families),
            "verified_passed": entry.verified_passed,
            "epoch": self.epoch,
            "answer_authority": False,
        }

    def use_lemma(self, lemma_id: str) -> bool:
        for entry in self._entries.values():
            if entry.lemma_id == lemma_id and entry.ratified:
                entry.use_count += 1
                entry.last_used_epoch = self.epoch
                return True
        return False

    def advance_epoch(self, steps: int = 1) -> Dict[str, Any]:
        """Advance task epoch and expire ratified lemmas idle past ``expiry_epochs``."""
        self.epoch += max(1, int(steps))
        to_expire: List[str] = []
        for sig, entry in list(self._entries.items()):
            idle = self.epoch - entry.last_used_epoch
            if entry.ratified and idle > self.expiry_epochs:
                to_expire.append(sig)
        expired_now: List[str] = []
        for sig in to_expire:
            removed = self._entries.pop(sig)
            self.expired_ids.append(removed.lemma_id)
            expired_now.append(removed.lemma_id)
        return {
            "epoch": self.epoch,
            "expired_now": expired_now,
            "active_ratified_count": len(self.active_lemmas()),
        }

    def _enforce_capacity(self) -> None:
        ratified = [e for e in self._entries.values() if e.ratified]
        if len(ratified) <= self.max_active_lemmas:
            return
        ratified.sort(key=lambda e: (e.use_count, e.last_used_epoch, e.lemma_id))
        while len(ratified) > self.max_active_lemmas:
            victim = ratified.pop(0)
            self._entries.pop(victim.signature, None)
            self.expired_ids.append(victim.lemma_id)

    def active_lemmas(self) -> List[Dict[str, Any]]:
        out = [
            {
                "lemma_id": e.lemma_id,
                "signature": e.signature,
                "expression": e.expression,
                "lineages": sorted(e.lineages),
                "task_families": sorted(e.task_families),
                "use_count": e.use_count,
                "last_used_epoch": e.last_used_epoch,
            }
            for e in self._entries.values()
            if e.ratified
        ]
        out.sort(key=lambda item: (-item["use_count"], item["lemma_id"]))
        return out

    def snapshot(self) -> Dict[str, Any]:
        payload = {
            "epoch": self.epoch,
            "min_independent_lineages": self.min_independent_lineages,
            "expiry_epochs": self.expiry_epochs,
            "active_ratified": self.active_lemmas(),
            "pending_candidates": len([e for e in self._entries.values() if not e.ratified]),
            "expired_count": len(self.expired_ids),
            "answer_authority": False,
        }
        payload["receipt_sha256"] = _canonical_sha256(payload)
        return payload


# ============================================================================
# 3. CounterformFalsificationAuditor (CVaR Tail Risk + Dual Audit Abstention)
# ============================================================================


class CounterformFalsificationAuditor:
    """Decision-sensitive CVaR sensitivity-particle auditor with explicit abstention.

    Inspired by ``kai9987kai/COUNTERFORM`` v0.2:
    Evaluates candidate plans/models across weighted sensitivity particles using a
    blended Mean + Conditional Value-at-Risk (CVaR_alpha) objective, and enforces
    a dual audit (consistency residual + forward projection residual). If no
    candidate clears both audits or best risk exceeds ``max_cvar_risk``, the
    auditor fails closed with ``decision="ABSTAIN_UNSUPPORTED"``.
    """

    def audit_candidates(
        self,
        *,
        candidate_losses: Mapping[str, Sequence[float]],
        particle_weights: Optional[Sequence[float]] = None,
        consistency_residuals: Optional[Mapping[str, float]] = None,
        projection_residuals: Optional[Mapping[str, float]] = None,
        cvar_alpha: float = 0.25,
        cvar_weight: float = 0.5,
        max_consistency_residual: float = 0.15,
        max_projection_residual: float = 0.20,
        max_cvar_risk: float = 0.65,
    ) -> Dict[str, Any]:
        if not candidate_losses:
            raise ValueError("candidate_losses must contain at least one candidate")
        if not (0.0 < float(cvar_alpha) <= 1.0):
            raise ValueError("cvar_alpha must be in (0, 1]")
        if not (0.0 <= float(cvar_weight) <= 1.0):
            raise ValueError("cvar_weight must be in [0, 1]")

        n_particles: Optional[int] = None
        parsed: Dict[str, np.ndarray] = {}
        for name, losses in candidate_losses.items():
            arr = np.asarray(losses, dtype=np.float64)
            if arr.ndim != 1 or arr.size < 1 or not np.all(np.isfinite(arr)):
                raise ValueError(f"invalid loss vector for candidate {name!r}")
            if n_particles is None:
                n_particles = int(arr.size)
            elif int(arr.size) != n_particles:
                raise ValueError("all candidates must share the same sensitivity particle count")
            parsed[str(name)] = arr

        assert n_particles is not None
        if particle_weights is None:
            weights = np.full((n_particles,), 1.0 / float(n_particles), dtype=np.float64)
        else:
            w = np.asarray(particle_weights, dtype=np.float64)
            if w.shape != (n_particles,) or not np.all(np.isfinite(w)) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
                raise ValueError("particle_weights must match particle length, be non-negative, and sum > 0")
            weights = w / float(w.sum())

        cons_map = dict(consistency_residuals or {})
        proj_map = dict(projection_residuals or {})

        ranked: List[Dict[str, Any]] = []
        for name, losses in parsed.items():
            mean_risk = float(np.dot(weights, losses))
            # Exact weighted upper-tail CVaR_alpha (worst alpha probability mass)
            order = np.argsort(-losses, kind="mergesort")
            rem = float(cvar_alpha)
            tail_sum = 0.0
            for idx in order:
                if rem <= 1e-15:
                    break
                take = min(rem, float(weights[idx]))
                tail_sum += take * float(losses[idx])
                rem -= take
            cvar_risk = tail_sum / float(cvar_alpha)
            blended = (1.0 - float(cvar_weight)) * mean_risk + float(cvar_weight) * cvar_risk

            cons_res = float(cons_map.get(name, 0.0))
            proj_res = float(proj_map.get(name, 0.0))
            cons_ok = cons_res <= float(max_consistency_residual)
            proj_ok = proj_res <= float(max_projection_residual)
            risk_ok = cvar_risk <= float(max_cvar_risk)
            dual_passed = bool(cons_ok and proj_ok and risk_ok)

            ranked.append(
                {
                    "candidate": name,
                    "mean_risk": round(mean_risk, 6),
                    "cvar_risk": round(cvar_risk, 6),
                    "blended_objective": round(blended, 6),
                    "worst_particle_loss": round(float(np.max(losses)), 6),
                    "consistency_residual": round(cons_res, 6),
                    "projection_residual": round(proj_res, 6),
                    "consistency_audit_passed": cons_ok,
                    "projection_audit_passed": proj_ok,
                    "risk_bound_passed": risk_ok,
                    "dual_audit_passed": dual_passed,
                }
            )

        ranked.sort(key=lambda r: (not r["dual_audit_passed"], r["blended_objective"], r["cvar_risk"], r["candidate"]))
        passing = [r for r in ranked if r["dual_audit_passed"]]
        if passing:
            decision = "RECOMMEND"
            selected = passing[0]["candidate"]
            abstention_reason = None
        else:
            decision = "ABSTAIN_UNSUPPORTED"
            selected = None
            abstention_reason = "no_candidate_cleared_dual_consistency_projection_and_cvar_bounds"

        result: Dict[str, Any] = {
            "decision": decision,
            "selected_candidate": selected,
            "abstention_reason": abstention_reason,
            "cvar_alpha": float(cvar_alpha),
            "cvar_weight": float(cvar_weight),
            "particle_count": n_particles,
            "candidates": ranked,
            "answer_authority": False,
        }
        result["receipt_sha256"] = _canonical_sha256(result)
        return result


# ============================================================================
# 4. CarrierInteractionAtlas (2^k Mobius Atlas + Ghost R^2 + Aether FLOPs)
# ============================================================================


class CarrierInteractionAtlas:
    """Subset-factorial carrier ablation atlas, authorship R^2, and FLOP profiler."""

    @staticmethod
    def decompose_subset_factorial(
        carriers: Sequence[str],
        subset_outcomes: Mapping[Tuple[str, ...], float] | Mapping[str, float],
    ) -> Dict[str, Any]:
        """Exact 2^k Mobius inversion decomposing combined carrier removals into orders 1..k.

        Matches ``MEMORY-CARRIER-OBSERVATORY``'s Interaction Atlas:
        For any subset ``S`` of removed carriers with outcome ``f(S)``, the Mobius
        interaction coefficient is:
            ``I(S) = sum_{T subset S} (-1)^{|S| - |T|} * f(T)``
        which satisfies the exact reconstruction identity ``f(S) = sum_{T subset S} I(T)``.
        """
        names = [str(c) for c in carriers]
        k = len(names)
        if k < 1 or k > 6:
            raise ValueError("carriers count must be in [1, 6]")

        # Accept either tuple keys or comma-joined / empty-string keys (JSON friendly)
        f_map: Dict[Tuple[str, ...], float] = {}
        for raw_key, raw_val in subset_outcomes.items():
            if isinstance(raw_key, str):
                parts = tuple(sorted(p.strip() for p in raw_key.split(",") if p.strip()))
            else:
                parts = tuple(sorted(str(p).strip() for p in raw_key if str(p).strip()))
            f_map[parts] = float(raw_val)

        all_subsets: List[Tuple[str, ...]] = []
        for r in range(k + 1):
            for comb in itertools.combinations(sorted(names), r):
                all_subsets.append(comb)
                if comb not in f_map:
                    raise ValueError(f"missing subset outcome for {comb!r}")

        baseline = f_map[()]
        coeffs: Dict[Tuple[str, ...], float] = {}
        for s in all_subsets:
            acc = 0.0
            s_len = len(s)
            for r in range(s_len + 1):
                sign = 1.0 if ((s_len - r) % 2 == 0) else -1.0
                for sub in itertools.combinations(s, r):
                    acc += sign * f_map[sub]
            coeffs[s] = acc

        # Exact reconstruction verification across all 2^k subsets
        max_recon_err = 0.0
        for s in all_subsets:
            recon = 0.0
            for r in range(len(s) + 1):
                for sub in itertools.combinations(s, r):
                    recon += coeffs[sub]
            max_recon_err = max(max_recon_err, abs(recon - f_map[s]))

        by_order: Dict[str, List[Dict[str, Any]]] = {}
        for order in range(1, k + 1):
            items = [
                {
                    "subset": list(s),
                    "key": ",".join(s),
                    "effect": round(coeffs[s], 8),
                }
                for s in all_subsets
                if len(s) == order
            ]
            items.sort(key=lambda item: -abs(item["effect"]))
            by_order[f"order_{order}"] = items

        result: Dict[str, Any] = {
            "carriers": sorted(names),
            "subset_count": len(all_subsets),
            "baseline_outcome": round(baseline, 8),
            "full_removal_outcome": round(f_map[tuple(sorted(names))], 8),
            "interactions_by_order": by_order,
            "max_reconstruction_error": float(max_recon_err),
            "exact_reconstruction_verified": bool(max_recon_err < 1e-9),
            "answer_authority": False,
        }
        result["receipt_sha256"] = _canonical_sha256(result)
        return result

    @staticmethod
    def authorship_error_predictability(
        internal_states: Sequence[Sequence[float]],
        comparator_errors: Sequence[float],
        *,
        ridge: float = 1e-4,
    ) -> Dict[str, Any]:
        """GhostInTheMachine v7 diagnostic: fraction of error variance predicted by internal state."""
        x = np.asarray(internal_states, dtype=np.float64)
        y = np.asarray(comparator_errors, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
            raise ValueError("internal_states (T, D) and comparator_errors (T,) need T >= 3")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("inputs must be finite")

        # Ridge-regularized linear predictor y_hat = X w + b
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
        reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(ridge)
        reg[-1, -1] = 0.0
        w = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
        y_hat = x_aug @ w
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        r2_clamped = float(max(0.0, min(1.0, r2)))
        regime = (
            "self_authored_stabilized"
            if r2_clamped >= 0.50
            else ("partially_coupled" if r2_clamped >= 0.25 else "exogenous_unpredicted")
        )
        return {
            "state_predicted_error_r2": round(r2_clamped, 6),
            "error_rms": round(float(np.sqrt(np.mean(y**2))), 6),
            "residual_rms": round(float(np.sqrt(np.mean((y - y_hat) ** 2))), 6),
            "stabilization_regime": regime,
            "samples": int(x.shape[0]),
            "answer_authority": False,
        }

    @staticmethod
    def profile_mimomix_compute(config: Any, *, seq_len: int = 128, dtype_bytes: int = 4) -> Dict[str, Any]:
        """Supermix-Aether analytical per-token FLOP and KV-cache profiler for MiMoMixConfig."""
        d = int(getattr(config, "hidden_size", 128))
        n_layers = int(getattr(config, "n_layers", 6))
        n_heads = int(getattr(config, "n_heads", 4))
        n_kv = int(getattr(config, "n_kv_heads", 2))
        head_dim = int(getattr(config, "head_dim", None) or (d // max(1, n_heads)))
        inter = int(getattr(config, "intermediate_size", 256))
        use_moe = bool(getattr(config, "use_moe", False))
        n_dense = int(getattr(config, "n_dense_layers", 1))
        moe_inter = int(getattr(config, "moe_intermediate_size", 64))
        moe_k = int(getattr(config, "moe_top_k", 2))
        moe_shared = int(getattr(config, "n_shared_experts", 1))
        n_routed = int(getattr(config, "n_routed_experts", 8))
        swa_win = int(getattr(config, "sliding_window", 128))
        hybrid_ratio = int(getattr(config, "hybrid_ratio", 5))
        use_mla = bool(getattr(config, "use_mla", False))
        mla_dim = int(getattr(config, "mla_latent_dim", 32))
        mla_pe = int(getattr(config, "mla_pe_dim", 16))
        use_cns = bool(getattr(config, "use_cns_core", False))
        cns_nodes = int(getattr(config, "cns_nodes", 512))
        cns_steps = int(getattr(config, "cns_steps", 6))
        use_think = bool(getattr(config, "use_thinking_core", False))
        think_dim = int(getattr(config, "thinking_latent_dim", 64))
        think_cycles = int(getattr(config, "thinking_cycles", 3))
        use_mhc = bool(getattr(config, "use_mhc_residual", False))
        mhc_streams = int(getattr(config, "mhc_streams", 2)) if use_mhc else 1

        # Count global vs SWA layers
        n_global = 0
        for idx in range(n_layers):
            if hybrid_ratio <= 0 or ((idx + 1) % (hybrid_ratio + 1)) == 0 or idx == n_layers - 1:
                n_global += 1
        n_swa = n_layers - n_global

        # Attention FLOPs per token (QKV + O projections + contextual SDPA)
        qkv_o_flops = 2 * d * (n_heads * head_dim + 2 * n_kv * head_dim + n_heads * head_dim)
        global_attn_flops = n_global * (qkv_o_flops + 4 * n_heads * head_dim * int(seq_len))
        swa_span = min(int(seq_len), swa_win)
        swa_attn_flops = n_swa * (qkv_o_flops + 4 * n_heads * head_dim * swa_span)

        # FFN / Sparse MoE FLOPs per token (SwiGLU = 3 linear projections = 6 * d * inter)
        n_moe_layers = max(0, n_layers - n_dense) if use_moe else 0
        n_dense_eff = n_layers - n_moe_layers
        dense_ffn_flops = n_dense_eff * (6 * d * inter)
        moe_active_ffn_flops = n_moe_layers * ((moe_k + moe_shared) * 6 * d * moe_inter + 2 * d * n_routed)
        moe_dense_equiv_flops = n_moe_layers * ((n_routed + moe_shared) * 6 * d * moe_inter)

        # Optional ConnectomeCore, RecursiveThinkingCore, and mHC residual mixing FLOPs
        cns_flops = (2 * d * cns_nodes * 2 + cns_steps * 2 * cns_nodes * cns_nodes) if use_cns else 0
        think_flops = (2 * d * think_dim * 2 + think_cycles * 6 * think_dim * think_dim) if use_think else 0
        mhc_flops = (n_layers * 2 * (mhc_streams**2) * d) if use_mhc else 0

        active_flops_per_tok = (
            global_attn_flops + swa_attn_flops + dense_ffn_flops + moe_active_ffn_flops + cns_flops + think_flops + mhc_flops
        )
        dense_equiv_flops_per_tok = (
            global_attn_flops + swa_attn_flops + dense_ffn_flops + moe_dense_equiv_flops + cns_flops + think_flops + mhc_flops
        )

        # KV cache bytes at seq_len
        if use_mla:
            global_kv_bytes = n_global * int(seq_len) * (mla_dim + mla_pe) * dtype_bytes
        else:
            global_kv_bytes = n_global * int(seq_len) * 2 * n_kv * head_dim * dtype_bytes
        swa_kv_bytes = n_swa * swa_span * 2 * n_kv * head_dim * dtype_bytes
        full_dense_kv_bytes = n_layers * int(seq_len) * 2 * n_heads * head_dim * dtype_bytes
        total_kv_bytes = global_kv_bytes + swa_kv_bytes

        return {
            "seq_len": int(seq_len),
            "n_layers": n_layers,
            "global_layers": n_global,
            "swa_layers": n_swa,
            "active_flops_per_token": int(active_flops_per_tok),
            "dense_equivalent_flops_per_token": int(dense_equiv_flops_per_tok),
            "moe_compute_sparsity_ratio": round(
                float(active_flops_per_tok) / float(max(1, dense_equiv_flops_per_tok)), 4
            ),
            "kv_cache_bytes": int(total_kv_bytes),
            "full_mha_kv_cache_bytes": int(full_dense_kv_bytes),
            "kv_cache_compression_ratio": round(
                float(full_dense_kv_bytes) / float(max(1, total_kv_bytes)), 4
            ),
            "mhc_streams": mhc_streams,
            "cns_flops_per_token": int(cns_flops),
            "thinking_flops_per_token": int(think_flops),
            "answer_authority": False,
        }


# ============================================================================
# 5. SonapseDiscoveryMonitor (Mode A Repair vs Mode B Discovery + Pareto Gate)
# ============================================================================


class SonapseDiscoveryMonitor:
    """Dual-mode structural evolution monitor inspired by ``kai9987kai/sonapse-evo`` 2.0."""

    def evaluate_step(
        self,
        *,
        train_loss: float,
        dev_loss: float,
        prev_dev_loss: Optional[float] = None,
        active_module_fraction: float = 0.60,
        spectral_radius: float = 0.90,
        target_spectral_radius: float = 0.90,
        lesion_detected: bool = False,
        spare_slots_free: int = 0,
    ) -> Dict[str, Any]:
        gen_gap = max(0.0, float(dev_loss) - float(train_loss))
        capacity_pressure = max(0.0, min(1.0, float(active_module_fraction) - 0.55) / 0.45)
        spectral_drift = abs(float(spectral_radius) - float(target_spectral_radius))
        loss_stagnation = (
            max(0.0, float(dev_loss) - float(prev_dev_loss))
            if prev_dev_loss is not None
            else 0.0
        )

        if lesion_detected or spectral_drift > 0.25:
            mode = "MODE_A_REPAIR"
            recommended_op = (
                "project_spectral_radius"
                if spectral_drift > 0.25
                else "grow_edges_from_scores"
            )
            opportunity_score = round(min(1.0, 0.65 + spectral_drift), 4)
        elif spare_slots_free > 0 and (capacity_pressure >= 0.45 or (gen_gap > 0.04 and loss_stagnation >= 0.0)):
            mode = "MODE_B_DISCOVERY"
            recommended_op = "split_module_with_witness_rollback"
            opportunity_score = round(min(1.0, 0.5 * capacity_pressure + 2.5 * gen_gap + 0.15), 4)
        else:
            mode = "STEADY_HOMEOSTASIS"
            recommended_op = "maintain_topology"
            opportunity_score = round(min(1.0, 0.3 * capacity_pressure + gen_gap), 4)

        return {
            "mode": mode,
            "recommended_operator": recommended_op,
            "opportunity_score": opportunity_score,
            "capacity_pressure": round(capacity_pressure, 4),
            "generalization_gap": round(gen_gap, 6),
            "spectral_drift": round(spectral_drift, 6),
            "pareto_ready": bool(spare_slots_free > 0 or recommended_op != "split_module_with_witness_rollback"),
            "answer_authority": False,
        }

    @staticmethod
    def pareto_accept(
        *,
        baseline_loss: float,
        candidate_loss: float,
        param_delta_fraction: float,
        spectral_radius_after: float,
        max_param_growth: float = 0.10,
        max_spectral_radius: float = 1.15,
        witness_tolerance: float = 0.005,
    ) -> Dict[str, Any]:
        """Multi-objective Pareto acceptance gate (loss vs param budget vs criticality)."""
        loss_gain = float(baseline_loss) - float(candidate_loss)
        loss_ok = loss_gain >= -float(witness_tolerance)
        budget_ok = float(param_delta_fraction) <= float(max_param_growth)
        spectral_ok = float(spectral_radius_after) <= float(max_spectral_radius)
        accepted = bool(loss_ok and budget_ok and spectral_ok)
        return {
            "accepted": accepted,
            "loss_gain": round(loss_gain, 6),
            "param_delta_fraction": round(float(param_delta_fraction), 6),
            "spectral_radius_after": round(float(spectral_radius_after), 6),
            "checks": {
                "witness_loss": loss_ok,
                "param_budget": budget_ok,
                "spectral_criticality": spectral_ok,
            },
            "answer_authority": False,
        }


# ============================================================================
# 6. HyperionComputeScheduler (Heterogeneous Cognitive DAG Scheduler)
# ============================================================================


class HyperionComputeScheduler:
    """Dependency-aware heterogeneous compute scheduler inspired by ``HYPERION-20``."""

    _SERVICE_CATALOG: Dict[str, Dict[str, Any]] = {
        "DeterministicASTVerifier": {
            "service_class": "symbolic_exact",
            "latency_class": "ultra_low",
            "answer_authority": True,
            "cost_units": 1,
        },
        "ConnectomeAtlas": {
            "service_class": "biological_graph_lookup",
            "latency_class": "ultra_low",
            "answer_authority": True,
            "cost_units": 1,
        },
        "NexusSearchRRFMemory": {
            "service_class": "hybrid_retrieval",
            "latency_class": "low",
            "answer_authority": False,
            "cost_units": 2,
        },
        "MiMoMixNeural": {
            "service_class": "hybrid_moe_connectome_lm",
            "latency_class": "medium",
            "answer_authority": False,
            "cost_units": 6,
        },
        "RecursiveThinking": {
            "service_class": "latent_iterative_refiner",
            "latency_class": "medium",
            "answer_authority": False,
            "cost_units": 4,
        },
        "ConformalRiskGate": {
            "service_class": "statistical_calibration",
            "latency_class": "ultra_low",
            "answer_authority": False,
            "cost_units": 1,
        },
        "RiemannianCollective": {
            "service_class": "manifold_consensus_forensics",
            "latency_class": "low",
            "answer_authority": False,
            "cost_units": 3,
        },
        "MAPElitesIdeation": {
            "service_class": "quality_diversity_evolution",
            "latency_class": "low",
            "answer_authority": False,
            "cost_units": 3,
        },
        "CounterformAuditor": {
            "service_class": "cvar_falsification_audit",
            "latency_class": "low",
            "answer_authority": False,
            "cost_units": 2,
        },
    }

    def schedule(
        self,
        *,
        domain: str,
        submode: str,
        complexity: str = "medium",
        speculative_allowed: bool = True,
        has_memory_turns: bool = False,
    ) -> Dict[str, Any]:
        """Build a topologically ordered parallel-wave execution plan across Supermix engines."""
        waves: List[List[Dict[str, Any]]] = []

        # Wave 0: cheap grounding / memory / fast deterministic specialists
        wave0: List[str] = ["ConformalRiskGate"]
        if has_memory_turns:
            wave0.append("NexusSearchRRFMemory")
        if domain == "connectome":
            wave0.append("ConnectomeAtlas")
        if domain in {"math_science", "code"} or submode in {"math_science", "code"}:
            wave0.append("DeterministicASTVerifier")
        waves.append([{"engine": name, **self._SERVICE_CATALOG[name]} for name in wave0])

        # Wave 1: heavy cognitive / generative specialists
        wave1: List[str] = ["MiMoMixNeural"]
        if complexity == "high" or submode == "problem_solving" or domain == "causal_epistemic":
            wave1.append("RecursiveThinking")
            wave1.append("CounterformAuditor")
        if submode == "ideation":
            wave1.append("MAPElitesIdeation")
        waves.append([{"engine": name, **self._SERVICE_CATALOG[name]} for name in wave1])

        # Wave 2: synthesis & verification gate
        wave2: List[str] = ["RiemannianCollective"]
        if "DeterministicASTVerifier" not in wave0:
            wave2.append("DeterministicASTVerifier")
        waves.append([{"engine": name, **self._SERVICE_CATALOG[name]} for name in wave2])

        total_cost = sum(node["cost_units"] for w in waves for node in w)
        critical_path_cost = sum(max(node["cost_units"] for node in w) for w in waves if w)
        payload: Dict[str, Any] = {
            "domain": str(domain),
            "submode": str(submode),
            "complexity": str(complexity),
            "speculative_allowed": bool(speculative_allowed),
            "wave_count": len(waves),
            "waves": waves,
            "total_cost_units": int(total_cost),
            "critical_path_cost_units": int(critical_path_cost),
            "parallelism_speedup_est": round(float(total_cost) / float(max(1, critical_path_cost)), 3),
            "answer_authority": False,
        }
        payload["plan_sha256"] = _canonical_sha256(payload)
        return payload
