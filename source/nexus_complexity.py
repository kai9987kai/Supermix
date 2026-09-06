"""NexusMind v88 Algorithmic Information & Complexity Analyzer.

Synthesizes foundational complexity and entropy metrics from AI-Dem-Lab
and modern mechanistic LLM evaluation:
1. **Shannon Entropy Profile**:
   - Computes empirical Shannon entropy $H(X) = -\\sum p_i \\log_2 p_i$ across
     tokens, characters, or internal hidden representation projections.
   - Computes sliding-window local entropy to capture cognitive burstiness.
2. **Algorithmic Compressibility (Kolmogorov Proxy)**:
   - Uses zlib/Lempel-Ziv compression ratio:
     $$K(x) \\approx \\frac{|C(x)|}{|x|}$$
   - Measures informative content vs redundant hallucination loops.
3. **Normalized Compression Distance (NCD)**:
   - Evaluates mutual information distance between two reasoning trajectories $x$ and $y$:
     $$NCD(x, y) = \\frac{C(xy) - \\min(C(x), C(y))}{\\max(C(x), C(y))}$$
4. **Loop & Degeneration Detector**:
   - Identifies periodic loops, degenerate repetitions, and sudden entropy collapse.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ComplexityProfileResult:
    """Detailed complexity and entropy measurements for a sequence."""
    total_tokens: int
    unique_tokens: int
    type_token_ratio: float
    shannon_entropy_bits: float
    max_possible_entropy: float
    normalized_entropy: float  # [0.0, 1.0]
    compression_ratio: float  # compressed_bytes / raw_bytes
    sliding_entropy_profile: List[float] = field(default_factory=list)
    repetitive_loop_detected: bool = False
    loop_period: Optional[int] = None
    entropy_collapse_detected: bool = False
    regime: str = "balanced_information"  # "collapsed_repetition" | "high_entropy_noise" | "balanced_information"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NCDResult:
    """Normalized Compression Distance comparison between two reasoning sequences."""
    sequence_a_len: int
    sequence_b_len: int
    compressed_a_bytes: int
    compressed_b_bytes: int
    compressed_joint_bytes: int
    ncd_score: float  # [0.0, 1.0+] (0 = identical, 1 = maximum mutual dissimilarity)
    semantic_divergence_class: str  # "near_duplicate" | "closely_aligned" | "distinct_approaches" | "orthogonal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlgorithmicComplexityAnalyzer:
    """Analyzer for information-theoretic complexity and degradation."""

    def __init__(self, window_size: int = 8):
        self.window_size = max(2, window_size)

    def analyze_sequence(self, text: str) -> ComplexityProfileResult:
        """Compute full Shannon, Lempel-Ziv, and degradation metrics."""
        tokens = text.strip().split()
        if not tokens:
            tokens = [char for char in text.strip()] or ["<empty>"]

        n_total = len(tokens)
        counts = Counter(tokens)
        n_unique = len(counts)
        ttr = n_unique / max(1, n_total)

        # Shannon Entropy
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / n_total
            if p > 0:
                entropy -= p * math.log2(p)

        max_ent = math.log2(n_total) if n_total > 1 else 1.0
        norm_ent = min(1.0, entropy / max_ent) if max_ent > 0 else 0.0

        # Algorithmic Compressibility (zlib proxy)
        raw_bytes = text.encode("utf-8")
        if len(raw_bytes) > 0:
            comp_bytes = zlib.compress(raw_bytes, level=6)
            comp_ratio = len(comp_bytes) / len(raw_bytes)
        else:
            comp_ratio = 1.0

        # Sliding Window Entropy
        sliding: List[float] = []
        step = max(1, len(tokens) // 16)
        w = min(len(tokens), self.window_size)
        for i in range(0, max(1, len(tokens) - w + 1), step):
            sub = tokens[i : i + w]
            sub_counts = Counter(sub)
            sub_ent = 0.0
            for sc in sub_counts.values():
                sp = sc / len(sub)
                if sp > 0:
                    sub_ent -= sp * math.log2(sp)
            sliding.append(round(sub_ent, 3))

        # Repetitive Loop Detection
        loop_found = False
        loop_p: Optional[int] = None
        for p in range(1, min(10, len(tokens) // 3 + 1)):
            matches = 0
            checks = 0
            for i in range(len(tokens) - p):
                checks += 1
                if tokens[i] == tokens[i + p]:
                    matches += 1
            if checks > 4 and (matches / checks) >= 0.75:
                loop_found = True
                loop_p = p
                break

        # Entropy Collapse Detection
        collapse = False
        if len(sliding) >= 3:
            recent_ent = sum(sliding[-2:]) / 2.0
            initial_ent = sum(sliding[:2]) / 2.0
            if recent_ent < 0.4 * initial_ent and recent_ent < 1.0:
                collapse = True

        if loop_found or collapse or norm_ent < 0.25:
            regime = "collapsed_repetition"
        elif norm_ent > 0.95 and comp_ratio > 0.9:
            regime = "high_entropy_noise"
        else:
            regime = "balanced_information"

        return ComplexityProfileResult(
            total_tokens=n_total,
            unique_tokens=n_unique,
            type_token_ratio=round(ttr, 4),
            shannon_entropy_bits=round(entropy, 4),
            max_possible_entropy=round(max_ent, 4),
            normalized_entropy=round(norm_ent, 4),
            compression_ratio=round(comp_ratio, 4),
            sliding_entropy_profile=sliding,
            repetitive_loop_detected=loop_found,
            loop_period=loop_p,
            entropy_collapse_detected=collapse,
            regime=regime,
        )

    def compute_ncd(self, text_a: str, text_b: str) -> NCDResult:
        """Compute Normalized Compression Distance between two reasoning texts."""
        b_a = text_a.encode("utf-8")
        b_b = text_b.encode("utf-8")
        b_ab = b_a + b" " + b_b

        c_a = len(zlib.compress(b_a, level=6)) if b_a else 0
        c_b = len(zlib.compress(b_b, level=6)) if b_b else 0
        c_ab = len(zlib.compress(b_ab, level=6)) if b_ab else 0

        max_c = max(c_a, c_b)
        min_c = min(c_a, c_b)

        if max_c == 0:
            ncd = 0.0
        else:
            ncd = max(0.0, (c_ab - min_c) / max_c)

        if ncd < 0.2:
            div_class = "near_duplicate"
        elif ncd < 0.5:
            div_class = "closely_aligned"
        elif ncd < 0.85:
            div_class = "distinct_approaches"
        else:
            div_class = "orthogonal"

        return NCDResult(
            sequence_a_len=len(text_a),
            sequence_b_len=len(text_b),
            compressed_a_bytes=c_a,
            compressed_b_bytes=c_b,
            compressed_joint_bytes=c_ab,
            ncd_score=round(ncd, 4),
            semantic_divergence_class=div_class,
        )
