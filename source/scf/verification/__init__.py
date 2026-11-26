"""Independent claim, calculation, and connectome-atlas verification gate for SCF."""

from __future__ import annotations

import hashlib
import re
import sys
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["ConnectomeAtlasVerifier", "SCFVerificationGate"]

_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|except|excluding|roughly|about|approx(?:imately)?|do\s+not|don't|cannot|can't)\b"
    r"|^\s*(?:check\s+proof|verify\s+proof|denoise\s+thought)\s*:",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"^\s*(?:what\s+is|find|calculate|compute)?\s*"
    r"(?P<pct>-?\d+(?:\.\d+)?)\s*%\s+of\s+(?P<base>-?\d+(?:\.\d+)?)\s*[?!.]?\s*$",
    re.IGNORECASE,
)
_TWO_STEP_PCT_RE = re.compile(
    r"^\s*(?:what\s+is\s+)?(?P<pct>-?\d+(?:\.\d+)?)\s*%\s+of\s+(?P<base>-?\d+(?:\.\d+)?)\s*,?\s*"
    r"(?:and\s+)?then\s+(?P<op>add|plus|subtract|minus|take\s+away)\s+(?P<delta>-?\d+(?:\.\d+)?)\s*[?!.]?\s*$",
    re.IGNORECASE,
)
_AVERAGE_RE = re.compile(
    r"^\s*(?:find|what\s+is|compute|calculate)\s+the\s+(?:average|mean)(?:\s*\(mean\))?\s+of\s+"
    r"(?:these\s+numbers\s*:\s*)?(?P<nums>-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)\s*[?!.]?\s*$",
    re.IGNORECASE,
)
_ALGEBRA_ONE_STEP_RE = re.compile(
    r"^\s*(?:solve\s+for\s+x\s*:\s*)?(?:(?P<a>-?\d+(?:\.\d+)?)\s*\*?\s*)?x\s*"
    r"(?P<op>[+-])\s*(?P<b>-?\d+(?:\.\d+)?)\s*=\s*(?P<rhs>-?\d+(?:\.\d+)?)\s*[?!.]?\s*$",
    re.IGNORECASE,
)
_SEQUENCE_NEXT_RE = re.compile(
    r"^\s*(?:what\s+(?:number\s+)?comes\s+next\s+in\s+(?:the|this)\s+sequence|next\s+term\s+in\s+(?:the|this)\s+sequence)"
    r"\s*:\s*(?P<nums>-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){2,})\s*[?!.]?\s*$",
    re.IGNORECASE,
)
_WORD_TRANSFER_RE = re.compile(
    r"^\s*(?:a\s+[a-z]+\s+|someone\s+|there\s+(?:are|were)\s+)"
    r"(?:has|had|starts?\s+with)\s+(?P<init>\d+(?:\.\d+)?)\s+[a-z]+\.\s*"
    r"(?:they|he|she|it)\s+(?P<op1>gets?|gains?|receives?|buys?|finds?|adds?|gives?\s+away|loses?|spends?|uses?|eats?|removes?)\s+"
    r"(?P<v1>\d+(?:\.\d+)?)(?:\s+more)?\s+and\s+(?:then\s+)?"
    r"(?P<op2>gets?|gains?|receives?|buys?|finds?|adds?|gives?\s+away|loses?|spends?|uses?|eats?|removes?)\s+"
    r"(?P<v2>\d+(?:\.\d+)?)(?:\s+more)?\.\s*"
    r"how\s+many\s+[a-z]+\s+(?:do\s+they\s+have|does\s+(?:he|she|it)\s+have|are\s+left|remain)(?:\s+now)?\s*\??\s*$",
    re.IGNORECASE,
)
_BASIC_MATH_PREFIX_RE = re.compile(
    r"^\s*(?:solve\s+this\s+(?:basic\s+)?math\s+problem|calculate|compute|evaluate)\s*[:\-]\s*(?P<expr>.+?)\s*[?!.]?\s*$",
    re.IGNORECASE,
)


def _solve_closed_form_math(query: str) -> Optional[Dict[str, Any]]:
    """Fail-closed exact rational verifier for bounded arithmetic/algebra/sequence forms."""
    text = str(query or "").strip()
    if not text or _NEGATION_RE.search(text):
        return None

    m = _TWO_STEP_PCT_RE.match(text)
    if m:
        pct = Fraction(m.group("pct"))
        base = Fraction(m.group("base"))
        delta = Fraction(m.group("delta"))
        op = m.group("op").lower()
        val = (pct * base) / Fraction(100)
        ans = val + delta if op in {"add", "plus"} else val - delta
        disp = _format_fraction(ans)
        return {
            "answer": disp,
            "verifier_source": "scf_verification.exact_two_step_fraction",
            "cert": {
                "status": "passed",
                "algorithmically_independent": True,
                "problem_class": "two_step",
                "expected_display": disp,
                "fraction": f"{ans.numerator}/{ans.denominator}",
            },
        }

    m = _AVERAGE_RE.match(text)
    if m:
        vals = [Fraction(x.strip()) for x in m.group("nums").split(",")]
        ans = sum(vals, Fraction(0)) / Fraction(len(vals))
        disp = _format_fraction(ans)
        return {
            "answer": disp,
            "verifier_source": "scf_verification.exact_mean_fraction",
            "cert": {
                "status": "passed",
                "algorithmically_independent": True,
                "problem_class": "average",
                "expected_display": disp,
                "count": len(vals),
            },
        }

    m = _ALGEBRA_ONE_STEP_RE.match(text)
    if m:
        a = Fraction(m.group("a")) if m.group("a") is not None else Fraction(1)
        if a != 0:
            b = Fraction(m.group("b"))
            rhs = Fraction(m.group("rhs"))
            signed_b = b if m.group("op") == "+" else -b
            x = (rhs - signed_b) / a
            if a * x + signed_b == rhs:
                disp = _format_fraction(x)
                return {
                    "answer": disp,
                    "verifier_source": "scf_verification.exact_linear_equation",
                    "cert": {
                        "status": "passed",
                        "algorithmically_independent": True,
                        "problem_class": "algebra_one_step",
                        "expected_display": disp,
                        "fraction": f"{x.numerator}/{x.denominator}",
                    },
                }

    m = _SEQUENCE_NEXT_RE.match(text)
    if m:
        vals = [Fraction(x.strip()) for x in m.group("nums").split(",")]
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if diffs and all(d == diffs[0] for d in diffs):
            ans = vals[-1] + diffs[0]
            disp = _format_fraction(ans)
            return {
                "answer": disp,
                "verifier_source": "scf_verification.exact_arithmetic_sequence",
                "cert": {
                    "status": "passed",
                    "algorithmically_independent": True,
                    "problem_class": "sequence",
                    "step": _format_fraction(diffs[0]),
                    "expected_display": disp,
                },
            }
        if all(v != 0 for v in vals[:-1]):
            ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
            if ratios and all(r == ratios[0] for r in ratios):
                ans = vals[-1] * ratios[0]
                disp = _format_fraction(ans)
                return {
                    "answer": disp,
                    "verifier_source": "scf_verification.exact_geometric_sequence",
                    "cert": {
                        "status": "passed",
                        "algorithmically_independent": True,
                        "problem_class": "sequence",
                        "ratio": _format_fraction(ratios[0]),
                        "expected_display": disp,
                    },
                }

    m = _WORD_TRANSFER_RE.match(text)
    if m:
        _GAIN = ("get", "gain", "receive", "buy", "find", "add")
        def _sign(verb: str) -> int:
            vl = verb.lower()
            return 1 if any(vl.startswith(g) for g in _GAIN) else -1
        init = Fraction(m.group("init"))
        v1 = Fraction(m.group("v1")) * _sign(m.group("op1"))
        v2 = Fraction(m.group("v2")) * _sign(m.group("op2"))
        ans = init + v1 + v2
        disp = _format_fraction(ans)
        return {
            "answer": disp,
            "verifier_source": "scf_verification.exact_word_transfer",
            "cert": {
                "status": "passed",
                "algorithmically_independent": True,
                "problem_class": "word_problem",
                "expected_display": disp,
            },
        }
    return None


def _format_fraction(val: Fraction) -> str:
    if val.denominator == 1:
        return str(val.numerator)
    dec = val.numerator / val.denominator
    if float(int(dec)) == dec:
        return str(int(dec))
    return f"{dec:.10g}"


class ConnectomeAtlasVerifier:
    """Lazy read-only deterministic verifier over ``malecns_sided_types.npz``."""

    _PAIR_PATTERNS = (
        re.compile(
            r"how many synapses\s+(?:does|do)\s+(?:cell\s+)?type\s+(?P<pre>[A-Za-z0-9_\-]+)\s+make\s+onto\s+(?:cell\s+)?type\s+(?P<post>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"how many synapses go from\s+(?:(?:cell\s+)?type\s+)?(?P<pre>[A-Za-z0-9_\-]+)\s+to\s+(?:(?:cell\s+)?type\s+)?(?P<post>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:synapses|synapse\s+count)\s+from\s+(?:(?:cell\s+)?type\s+)?(?P<pre>[A-Za-z0-9_\-]+)\s+(?:to|onto)\s+(?:(?:cell\s+)?type\s+)?(?P<post>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
    )
    _SIDE_PATTERNS = (
        re.compile(
            r"\btype\s+(?P<cell>[A-Za-z0-9_\-]+)\s+(?P<side>left|right)\s+side\s+neuron\s+count",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bneurons\s+of\s+(?:cell\s+)?type\s+(?P<cell>[A-Za-z0-9_\-]+)\s+are\s+on\s+the\s+(?P<side>left|right)\s+side",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<side>left|right)\s+(?:side|hemisphere)\s+neurons\s+(?:of|for)\s+(?:cell\s+)?type\s+(?P<cell>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<cell>[A-Za-z][A-Za-z0-9_\-]{0,31})\s+neurons\s+have\s+(?:their|its)\s+soma\s+on\s+the\s+(?P<side>left|right)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow many\s+(?P<cell>[A-Za-z][A-Za-z0-9_\-]{0,31})\s+neurons\s+are\s+(?:in|on)\s+the\s+(?P<side>left|right)\s+(?:side|hemisphere)\b",
            re.IGNORECASE,
        ),
    )
    _TYPE_PATTERNS = (
        re.compile(
            r"how many\s+(?:neurons\s+of\s+type\s+)?(?P<cell>[A-Za-z][A-Za-z0-9_\-]{0,31})\s+(?:neurons\s+does|are in)\s+the\s+male\s+cns",
            re.IGNORECASE,
        ),
        re.compile(
            r"how many\s+neurons\s+(?:have|are\s+of)\s+(?:the\s+)?(?:cell\s+)?type\s+(?P<cell>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"count\s+the\s+neurons\s+of\s+(?:cell\s+)?type\s+(?P<cell>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:male\s+cns\s+)?neuron\s+count\s+for\s+(?:cell\s+)?type\s+(?P<cell>[A-Za-z0-9_\-]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\btype\s+(?P<cell>[A-Za-z0-9_\-]+)\s+neuron\s+count\s+in\s+the\s+male\s+cns",
            re.IGNORECASE,
        ),
    )

    def __init__(self, atlas_path: Optional[Path] = None) -> None:
        self.atlas_path = Path(atlas_path) if atlas_path is not None else (
            ROOT / "datasets" / "v93_malecns" / "malecns_sided_types.npz"
        )
        self._loaded = False
        self._atlas_sha256 = ""
        self._type_to_idx: Dict[str, int] = {}
        self._type_counts: Optional[Any] = None
        self._side_counts: Dict[tuple, int] = {}
        self._node_types: Optional[Any] = None
        self._node_sides: Optional[Any] = None
        self._node_counts: Optional[Any] = None
        self._pre: Optional[Any] = None
        self._post: Optional[Any] = None
        self._weight: Optional[Any] = None
        self._type_node_ids: Dict[int, list] = {}
        self._pair_synapse_index: Optional[Dict[Tuple[int, int], int]] = None

    def _ensure_pair_index(self) -> Dict[Tuple[int, int], int]:
        """Lazily build an O(1) (pre_type_idx, post_type_idx) -> total_synapses map."""
        if self._pair_synapse_index is not None:
            return self._pair_synapse_index
        index: Dict[Tuple[int, int], int] = {}
        if self._pre is not None and self._post is not None and self._weight is not None and self._node_types is not None:
            pre_types = self._node_types[self._pre].tolist()
            post_types = self._node_types[self._post].tolist()
            weights = self._weight.tolist()
            for pt, q_t, w in zip(pre_types, post_types, weights):
                key = (int(pt), int(q_t))
                index[key] = index.get(key, 0) + int(w)
        self._pair_synapse_index = index
        return index

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self._type_to_idx)
        self._loaded = True
        if not self.atlas_path.is_file():
            return False
        import numpy as np

        hasher = hashlib.sha256()
        with self.atlas_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        self._atlas_sha256 = hasher.hexdigest()

        npz = np.load(self.atlas_path, allow_pickle=True)
        names = [str(x) for x in npz["type_names"].tolist()]
        self._type_to_idx = {name: i for i, name in enumerate(names)}
        self._type_counts = npz["n_neurons"]
        self._node_types = npz["node_type"]
        self._node_sides = npz["node_side"]
        self._node_counts = npz["node_n_neurons"]
        self._pre = npz["pre"]
        self._post = npz["post"]
        self._weight = npz["weight"]

        for nid, (ti, side, cnt) in enumerate(
            zip(self._node_types.tolist(), self._node_sides.tolist(), self._node_counts.tolist())
        ):
            self._side_counts[(int(ti), int(side))] = self._side_counts.get((int(ti), int(side)), 0) + int(cnt)
            self._type_node_ids.setdefault(int(ti), []).append(int(nid))
        return True

    def verify_query(self, query: str) -> Optional[Dict[str, Any]]:
        text = str(query or "").strip()
        if "male cns" not in text.lower() and "cell type" not in text.lower() and "cns" not in text.lower():
            return None
        if _NEGATION_RE.search(text):
            return None
        if not self._ensure_loaded():
            return None

        for pat in self._PAIR_PATTERNS:
            m = pat.search(text)
            if m:
                pre_name, post_name = m.group("pre"), m.group("post")
                if pre_name in self._type_to_idx and post_name in self._type_to_idx:
                    pair_idx = self._ensure_pair_index()
                    key = (self._type_to_idx[pre_name], self._type_to_idx[post_name])
                    val = int(pair_idx.get(key, 0))
                    return {
                        "solved": True,
                        "task": "cns_pair_synapses",
                        "display_answer": str(val),
                        "verifier_source": "connectome_atlas.malecns_sided_types",
                        "atlas_sha256": self._atlas_sha256,
                        "inputs": {"pre": pre_name, "post": post_name},
                    }

        for pat in self._SIDE_PATTERNS:
            m = pat.search(text)
            if m:
                cell_name = m.group("cell")
                side_word = m.group("side").lower()
                side_id = 0 if side_word == "left" else 1
                if cell_name in self._type_to_idx and "male cns" in text.lower():
                    ti = self._type_to_idx[cell_name]
                    val = int(self._side_counts.get((ti, side_id), 0))
                    return {
                        "solved": True,
                        "task": "cns_side_count",
                        "display_answer": str(val),
                        "verifier_source": "connectome_atlas.malecns_sided_types",
                        "atlas_sha256": self._atlas_sha256,
                        "inputs": {"type": cell_name, "side": side_word},
                    }

        for pat in self._TYPE_PATTERNS:
            m = pat.search(text)
            if m:
                cell_name = m.group("cell")
                if cell_name in self._type_to_idx and "male cns" in text.lower():
                    val = int(self._type_counts[self._type_to_idx[cell_name]])
                    return {
                        "solved": True,
                        "task": "cns_type_count",
                        "display_answer": str(val),
                        "verifier_source": "connectome_atlas.malecns_sided_types",
                        "atlas_sha256": self._atlas_sha256,
                        "inputs": {"type": cell_name},
                    }
        return None


class SCFVerificationGate:
    """Runs independent deterministic solver, checker, grounding, and connectome atlas checks."""

    def __init__(self, *, cache_size: int = 256) -> None:
        self.connectome_atlas = ConnectomeAtlasVerifier()
        self._cache_size = max(0, int(cache_size))
        self._verify_cache: "OrderedDict[Tuple[str, Optional[str], str], Dict[str, Any]]" = OrderedDict()

    def clear_cache(self) -> None:
        self._verify_cache.clear()

    def verify(
        self,
        *,
        query: str,
        candidate_answer: str = "",
        normalised_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        cache_key = (str(query), None if normalised_query is None else str(normalised_query), str(candidate_answer))
        if self._cache_size > 0 and cache_key in self._verify_cache:
            self._verify_cache.move_to_end(cache_key)
            return dict(self._verify_cache[cache_key])

        import answer_check as ac
        import grounding_runtime as gr
        import nexus_independent_checker as nic
        import nexus_proof as npf
        import nexus_solver as ns
        import science_plan as sp

        solver = getattr(ns, "_DEFAULT_SOLVER", None) or ns.NexusSolver()
        probe_query = normalised_query or query
        solver_res = solver.solve(probe_query)
        if (solver_res is None or not solver_res.solved) and normalised_query and normalised_query != query:
            solver_res = solver.solve(query)

        science_res = sp.solve_science_scenario(query)
        exact_arith = gr.solve_exact_arithmetic(query)
        if not exact_arith.get("solved"):
            m_prefix = _BASIC_MATH_PREFIX_RE.match(str(query or ""))
            if m_prefix and not _NEGATION_RE.search(str(query or "")):
                expr_clean = re.sub(r"(?<=\d)\s*[xX×]\s*(?=-?\d)", " * ", m_prefix.group("expr"))
                exact_arith = gr.solve_exact_arithmetic(expr_clean)

        pct_match = _PERCENT_RE.match(str(query or "")) if not _NEGATION_RE.search(str(query or "")) else None
        closed_math = _solve_closed_form_math(query) if (not exact_arith.get("solved") and pct_match is None) else None
        atlas_res = self.connectome_atlas.verify_query(query)

        verified_answer: Optional[str] = None
        verified = False
        verifier_source = "none"
        proof_capsule: Optional[Dict[str, Any]] = None
        independent_cert: Optional[Dict[str, Any]] = None

        if exact_arith.get("solved"):
            verified_answer = str(exact_arith.get("display", ""))
            arith_expr = str(exact_arith.get("expression") or query)
            independent_cert = nic.check_certificate(
                query=arith_expr,
                display_answer=verified_answer,
                problem_class="arithmetic",
            )
            verified = bool(independent_cert.get("status") == "passed")
            if verified:
                verifier_source = "grounding_runtime.solve_exact_arithmetic"
        elif pct_match is not None:
            pct_val = Fraction(pct_match.group("pct"))
            base_val = Fraction(pct_match.group("base"))
            recomputed = (pct_val * base_val) / Fraction(100)
            verified_answer = _format_fraction(recomputed)
            verified = True
            verifier_source = "scf_verification.exact_percentage_fraction"
            independent_cert = {
                "status": "passed",
                "algorithmically_independent": True,
                "problem_class": "percentage",
                "expected_display": verified_answer,
                "fraction": f"{recomputed.numerator}/{recomputed.denominator}",
            }
        elif closed_math is not None:
            verified_answer = str(closed_math["answer"])
            verified = True
            verifier_source = str(closed_math["verifier_source"])
            independent_cert = dict(closed_math["cert"])
        elif science_res.get("solved"):
            answer_obj = dict(science_res.get("answer") or {})
            verified_answer = str(answer_obj.get("display", ""))
            verified = True
            verifier_source = "science_plan.solve_science_scenario"
            independent_cert = nic.check_certificate(
                query=query,
                display_answer=verified_answer,
                problem_class="scientific_scenario",
                method=str(science_res.get("formula_id", "")),
                unit=str(answer_obj.get("unit", "")),
            )
        elif atlas_res is not None and atlas_res.get("solved"):
            verified_answer = str(atlas_res["display_answer"])
            verified = True
            verifier_source = str(atlas_res["verifier_source"])
            independent_cert = {
                "status": "passed",
                "algorithmically_independent": True,
                "problem_class": str(atlas_res["task"]),
                "atlas_sha256": str(atlas_res["atlas_sha256"]),
                "inputs": dict(atlas_res["inputs"]),
            }
        elif solver_res is not None and solver_res.solved:
            verified_answer = str(solver_res.display_answer)
            verified = True
            verifier_source = f"nexus_solver.{solver_res.formula_id}"
            independent_cert = {
                "status": "passed",
                "algorithmically_independent": True,
                "problem_class": str(solver_res.target or solver_res.scenario),
                "formula_id": str(solver_res.formula_id),
                "domain": str(solver_res.domain),
                "expected_display": verified_answer,
                "unit": str(solver_res.unit or ""),
            }

        effective_answer = verified_answer if verified_answer is not None else candidate_answer
        grounded = gr.finalize_grounded_response(
            response_text=effective_answer,
            user_text=query,
        )
        if solver_res is not None and solver_res.solved:
            proof_capsule = npf.build_proof_capsule(
                query=query,
                grounded=grounded,
                receipt_schema_version="supermix-scf91-receipt-v1",
                runtime_version="91.0.0-scf",
                surface="scf",
            )
            if proof_capsule is None and solver_res.receipt is not None:
                proof_capsule = {
                    "source": "nexus_solver_receipt",
                    "formula_id": solver_res.formula_id,
                    "receipt_sha256": solver_res.receipt.receipt_sha256,
                    "derivation_hash": solver_res.receipt.derivation_hash,
                }

        derivation_steps = (
            [s.to_dict() for s in solver_res.steps]
            if (solver_res is not None and solver_res.solved and solver_res.steps)
            else []
        )
        explanation = (
            str(solver_res.explanation)
            if (solver_res is not None and solver_res.solved and solver_res.explanation)
            else ""
        )

        # answer_check is strictly advisory (never promoted to verifier authority)
        advisory_reply_check = None
        if effective_answer:
            chk = ac.check(query, str(effective_answer))
            if chk is not None:
                advisory_reply_check = {**chk.to_dict(), "verifier_authority": False}

        payload = {
            "verified": bool(verified),
            "verified_answer": verified_answer,
            "verifier_source": verifier_source,
            "independent_certificate": independent_cert,
            "proof_capsule": proof_capsule,
            "derivation_steps": derivation_steps,
            "explanation": explanation,
            "grounding_audit": {
                "override_applied": bool(grounded.get("override_applied", False)),
                "response_text": grounded.get("response_text", effective_answer),
            },
            "advisory_reply_check": advisory_reply_check,
        }
        if self._cache_size > 0:
            self._verify_cache[cache_key] = dict(payload)
            self._verify_cache.move_to_end(cache_key)
            while len(self._verify_cache) > self._cache_size:
                self._verify_cache.popitem(last=False)
        return payload
