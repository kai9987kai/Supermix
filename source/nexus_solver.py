"""NexusMind deterministic formula-pattern library.

This module provides broad deterministic math/science parsing for development,
audit, and bounded fixtures. A self-hashed `SolverReceipt` authenticates its
parse/result record but does not independently prove that the complete request
was consumed. User-facing answer authority belongs to the stricter fresh
grounding gate in `grounding_runtime.finalize_grounded_response`.

Available pattern families include:
1. **12+ Scientific & Mathematical Scenario Domains**:
   - Kinematics & Motion
   - Dynamics & Newton's Laws
   - Work, Energy & Power
   - Circular Motion & Gravitation
   - Thermodynamics & Heat
   - Hydrostatics & Fluid Dynamics
   - Electromagnetism & DC Circuits
   - Waves & Optics
   - Chemistry Stoichiometry & Solutions
   - Pure Algebra & Systems
   - Geometry & Trigonometry
   - Combinatorics, Probability & Statistics
2. **Exact Rational SI Arithmetic**:
   - Fractional precision (`Fraction` and `Decimal`), dimension vectors, zero float drift.
   - Comprehensive unit conversions for metric & imperial units.
3. **Formal LaTeX Derivations**:
   - Generates step-by-step mathematical proofs and substitutions.
4. **Deterministic integrity metadata**:
   - Produces `SolverReceipt` with a SHA-256 audit digest; the receipt is not
     factual, permission, safety, routing, activation, or promotion authority.
"""

from __future__ import annotations

import ast
import hashlib
import heapq
import json
import math
import operator as _op
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import science_plan as v71_science


__all__ = [
    "SolverReceipt",
    "DerivationStep",
    "SolverResult",
    "NexusSolver",
    "solve_problem",
]


SOLVER_RECEIPT_SCHEMA = "nexus-solver-receipt-v2"
MAX_QUERY_CHARS = 4096


# Standard Fundamental Physical Constants in SI (Exact / CODATA 2019)
CONSTANTS: Dict[str, Dict[str, Any]] = {
    "c": {"symbol": "c", "value": Fraction(299792458, 1), "unit": "m/s", "name": "speed_of_light"},
    "h": {"symbol": "h", "value": Fraction(662607015, 10**42), "unit": "J*s", "name": "planck_constant"},
    "hbar": {"symbol": "hbar", "value": Fraction(662607015, 10**42) / Fraction(2 * 3141592653589793, 10**15), "unit": "J*s", "name": "reduced_planck"},
    "G": {"symbol": "G", "value": Fraction(667430, 10**16), "unit": "N*m^2/kg^2", "name": "gravitational_constant"},
    "g": {"symbol": "g", "value": Fraction(980665, 100000), "unit": "m/s^2", "name": "standard_gravity"},
    "k_B": {"symbol": "k_B", "value": Fraction(1380649, 10**29), "unit": "J/K", "name": "boltzmann_constant"},
    "N_A": {"symbol": "N_A", "value": Fraction(602214076, 10**15) * Fraction(10**23, 1), "unit": "1/mol", "name": "avogadro_number"},
    "R": {"symbol": "R", "value": Fraction(831446261815324, 10**14), "unit": "J/(mol*K)", "name": "gas_constant"},
    "F": {"symbol": "F", "value": Fraction(964853321233100184, 10**13), "unit": "C/mol", "name": "faraday_constant"},
    "e": {"symbol": "e", "value": Fraction(1602176634, 10**28), "unit": "C", "name": "elementary_charge"},
    "k_e": {"symbol": "k_e", "value": Fraction(89875517923, 10), "unit": "N*m^2/C^2", "name": "coulomb_constant"},
    "pi": {"symbol": "pi", "value": Fraction(3141592653589793, 10**15), "unit": "dimensionless", "name": "pi"},
}


@dataclass
class DerivationStep:
    """A single step in a formal mathematical derivation."""

    step_index: int
    description: str
    formula_latex: str
    substitution_latex: str
    evaluated_value: str
    unit: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverReceipt:
    """Cryptographic audit receipt for a deterministic solver execution."""

    schema_version: str = SOLVER_RECEIPT_SCHEMA
    scenario: str = ""
    domain: str = ""
    target: str = ""
    formula_id: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    raw_result_fraction: str = ""
    display_result: str = ""
    unit: str = ""
    derivation_hash: str = ""
    receipt_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolverResult:
    """Master result container returned by the NexusSolver."""

    solved: bool
    query: str
    domain: str
    scenario: str
    target: str
    formula_id: str
    answer_value: Optional[float]
    display_answer: str
    unit: str
    steps: List[DerivationStep] = field(default_factory=list)
    receipt: Optional[SolverReceipt] = None
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solved": self.solved,
            "query": self.query,
            "domain": self.domain,
            "scenario": self.scenario,
            "target": self.target,
            "formula_id": self.formula_id,
            "answer_value": self.answer_value,
            "display_answer": self.display_answer,
            "unit": self.unit,
            "steps": [s.to_dict() for s in self.steps],
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "explanation": self.explanation,
        }


class NexusSolver:
    """Next-generation deterministic math and science problem solver."""

    def __init__(self):
        self._setup_unit_converters()

    def _setup_unit_converters(self):
        self.mass_to_kg = {
            "kg": Fraction(1), "g": Fraction(1, 1000), "mg": Fraction(1, 1000000),
            "lb": Fraction(45359237, 100000000), "oz": Fraction(28349523125, 1000000000000),
            "tonne": Fraction(1000), "ton": Fraction(1000),
        }
        self.length_to_m = {
            "m": Fraction(1), "km": Fraction(1000), "cm": Fraction(1, 100), "mm": Fraction(1, 1000),
            "um": Fraction(1, 1000000), "nm": Fraction(1, 1000000000), "pm": Fraction(1, 10**12),
            "mi": Fraction(1609344, 1000), "mile": Fraction(1609344, 1000), "miles": Fraction(1609344, 1000),
            "yd": Fraction(9144, 10000), "yard": Fraction(9144, 10000),
            "ft": Fraction(3048, 10000), "foot": Fraction(3048, 10000), "feet": Fraction(3048, 10000),
            "in": Fraction(254, 10000), "inch": Fraction(254, 10000), "inches": Fraction(254, 10000),
            "ly": Fraction(9460730472580800, 1), "au": Fraction(149597870700, 1),
        }
        self.time_to_s = {
            "s": Fraction(1), "sec": Fraction(1), "second": Fraction(1), "seconds": Fraction(1),
            "ms": Fraction(1, 1000), "us": Fraction(1, 1000000), "ns": Fraction(1, 1000000000),
            "min": Fraction(60), "minute": Fraction(60), "minutes": Fraction(60),
            "h": Fraction(3600), "hr": Fraction(3600), "hour": Fraction(3600), "hours": Fraction(3600),
            "d": Fraction(86400), "day": Fraction(86400), "days": Fraction(86400),
            "yr": Fraction(31557600), "year": Fraction(31557600), "years": Fraction(31557600),
        }
        self.vel_to_mps = {
            "m/s": Fraction(1), "mps": Fraction(1), "km/h": Fraction(1000, 3600),
            "kph": Fraction(1000, 3600), "mph": Fraction(1609344, 3600000),
            "knot": Fraction(1852, 3600), "knots": Fraction(1852, 3600),
        }
        self.force_to_n = {
            "n": Fraction(1), "kn": Fraction(1000), "mn": Fraction(1000000),
            "lbf": Fraction(4448222, 1000000), "dyn": Fraction(1, 100000),
        }
        self.energy_to_j = {
            "j": Fraction(1), "kj": Fraction(1000), "mj": Fraction(1000000), "gj": Fraction(10**9),
            "cal": Fraction(4184, 1000), "kcal": Fraction(4184), "ev": Fraction(1602176634, 10**28),
            "kev": Fraction(1602176634, 10**25), "mev": Fraction(1602176634, 10**22),
            "kwh": Fraction(3600000), "btu": Fraction(1055056, 1000),
        }
        self.power_to_w = {
            "w": Fraction(1), "kw": Fraction(1000), "mw": Fraction(1000000), "gw": Fraction(10**9),
            "hp": Fraction(745699872, 1000000),
        }
        self.pressure_to_pa = {
            "pa": Fraction(1), "kpa": Fraction(1000), "mpa": Fraction(1000000),
            "bar": Fraction(100000), "mbar": Fraction(100), "atm": Fraction(101325),
            "psi": Fraction(6894757, 1000), "torr": Fraction(101325, 760), "mmhg": Fraction(101325, 760),
        }

    def _normalize_num(self, val_str: str) -> Fraction:
        """Parse clean fractional or decimal numbers."""
        clean = val_str.strip().replace(",", "")
        if "/" in clean:
            num, den = clean.split("/", 1)
            return Fraction(int(num), int(den))
        if "." in clean:
            return Fraction(Decimal(clean))
        return Fraction(int(clean))

    def _format_fraction(self, f: Fraction, max_decimals: int = 4) -> str:
        """Format exact fraction to clean display string."""
        if f.denominator == 1:
            return str(f.numerator)
        val_float = float(f)
        if abs(val_float) < 1e-4 and val_float != 0:
            return f"{val_float:.4e}"
        # If fraction terminates cleanly in small decimal
        s = f"{val_float:.{max_decimals}f}".rstrip("0").rstrip(".")
        return s

    @staticmethod
    def _bridge_answer_value(ans_dict: Dict[str, Any]) -> Optional[float]:
        """The numeric answer of a `science_plan` result, or None.

        `science_plan.solve_science_scenario` writes its answer under
        ``"exact"`` as a fraction string (``"32"``, ``"161/5"``); this bridge
        read only ``"fraction"``, a key that result never carries, so every
        query the v71 path solved came back with ``answer_value=None``. Nothing
        noticed until v93 added `final_velocity` (``v = u + a t``, the one
        registry formula `build_omni_corpus` uses): `verify` compares
        ``answer_value`` to the generator's answer, so a None dropped 100% of
        the task at build time -- 0 of 200 smoke draws verified before this
        line, 200 of 200 after. Both keys are read so a result that carries
        either still bridges; an unparseable value stays None rather than
        raising, which keeps the fall-through to the extended solvers intact.
        """

        for key in ("fraction", "exact"):
            if key in ans_dict:
                try:
                    return float(Fraction(str(ans_dict[key])))
                except (ValueError, ZeroDivisionError, TypeError):
                    return None
        return None

    def solve(self, query: str) -> SolverResult:
        """Master solve dispatcher across all scientific and mathematical domains."""
        query_clean = query.strip()

        # Check v71 closed-world solver first for backward compatibility
        v71_res = v71_science.solve_science_scenario(query_clean)
        if v71_res.get("solved") is True:
            ans_dict = v71_res.get("answer", {})
            receipt_dict = v71_res.get("receipt", {})
            return SolverResult(
                solved=True,
                query=query_clean,
                domain="physics",
                scenario=v71_res.get("scenario", ""),
                target=v71_res.get("target", ""),
                formula_id=v71_res.get("formula_id", ""),
                answer_value=self._bridge_answer_value(ans_dict),
                display_answer=ans_dict.get("display", ""),
                unit=ans_dict.get("unit", ""),
                steps=[
                    DerivationStep(
                        step_index=1,
                        description=f"Verified registry evaluation [{v71_res.get('formula_id')}]",
                        formula_latex=v71_res.get("formula_id", ""),
                        substitution_latex=f"{ans_dict.get('display')} {ans_dict.get('unit')}",
                        evaluated_value=ans_dict.get("display", ""),
                        unit=ans_dict.get("unit", ""),
                    )
                ],
                receipt=SolverReceipt(
                    schema_version=SOLVER_RECEIPT_SCHEMA,
                    scenario=v71_res.get("scenario", ""),
                    domain="physics",
                    target=v71_res.get("target", ""),
                    formula_id=v71_res.get("formula_id", ""),
                    inputs=receipt_dict.get("inputs", {}),
                    raw_result_fraction=str(ans_dict.get("fraction", "")),
                    display_result=ans_dict.get("display", ""),
                    unit=ans_dict.get("unit", ""),
                    receipt_sha256=receipt_dict.get("receipt_sha256", ""),
                ),
                explanation=f"Exact verified scientific solution: {ans_dict.get('display')} {ans_dict.get('unit')}",
            )

        # Dispatch across extended scenario solvers
        solvers = [
            self._solve_kinematics_extended,
            self._solve_dynamics_newton,
            self._solve_work_energy_power,
            self._solve_circular_gravitation,
            self._solve_thermo_heat,
            self._solve_fluids_hydrostatics,
            self._solve_circuits_electromagnetism,
            self._solve_waves_optics,
            self._solve_chemistry_solutions,
            self._solve_quadratic_algebra,
            self._solve_linear_system_3x3,
            self._solve_linear_system_2x2,
            self._solve_matrix_linear_algebra,
            self._solve_compound_interest,
            self._solve_series_progressions,
            self._solve_geometry_trig,
            self._solve_combinatorics_stats,
            self._solve_natural_combinatorics,
            self._solve_general_sequences_and_laws,
            self._solve_calculus_stats_number_theory,
            self._solve_number_theory_and_probability,
            self._solve_code_expression,
            self._solve_python_ast_trace,
            self._solve_safe_ast_math,
        ]

        for s_func in solvers:
            try:
                res = s_func(query_clean)
                if res and res.solved:
                    return res
            except Exception:
                continue

        return SolverResult(
            solved=False,
            query=query_clean,
            domain="unresolved",
            scenario="none",
            target="none",
            formula_id="none",
            answer_value=None,
            display_answer="",
            unit="",
            explanation="Could not match deterministic formula pattern with complete parameters.",
        )

    def _solve_natural_combinatorics(self, q: str) -> Optional[SolverResult]:
        """Solve natural-language C(n, k) and P(n, k) combinatorics queries."""
        lower = q.lower()
        if any(neg in lower for neg in ("do not", "don't", "check proof", "verify proof")):
            return None

        # Explicit permutations: "permutations n=5 k=2"
        perm_explicit = re.fullmatch(
            r"\s*(?:please\s+)?permutations\s+n\s*=\s*(\d+)\s*(?:,\s*)?k\s*=\s*(\d+)\s*[?.!]?",
            lower,
        )
        if perm_explicit:
            n_val, k_val = int(perm_explicit.group(1)), int(perm_explicit.group(2))
            if 0 <= k_val <= n_val <= 1000:
                val = Fraction(math.perm(n_val, k_val))
                disp = self._format_fraction(val)
                steps = [
                    DerivationStep(1, "Compute permutations P(n, k)", "P(n, k) = \\frac{n!}{(n-k)!}", f"P({n_val}, {k_val}) = {disp}", disp)
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "permutations", "math.permutations", {"n": str(n_val), "k": str(k_val)}, str(val), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "permutations", "math.permutations", float(val), disp, "", steps, receipt, f"Permutations P({n_val}, {k_val}) = {disp}.")

        # Natural permutation: "arranged (in order) from N" or "ordered arrangements of K ... from N"
        if re.search(r"\b(?:permut\w*|arrang(?:e|ed|ing|ements?)|in\s+order|ordered)\b", lower):
            m = re.search(r"\b(\d+)\s+items?\b.*?\b(?:from|of)\s+(\d+)\b", lower) or re.search(
                r"\b(?:arrangements?|permutations?|ordered)\s+of\s+(\d+)\b.*?\bfrom\s+(\d+)\b",
                lower,
            )
            if m:
                k_val, n_val = int(m.group(1)), int(m.group(2))
                if 0 <= k_val <= n_val <= 1000:
                    val = Fraction(math.perm(n_val, k_val))
                    disp = self._format_fraction(val)
                    steps = [
                        DerivationStep(1, "Compute permutations P(n, k)", "P(n, k) = \\frac{n!}{(n-k)!}", f"P({n_val}, {k_val}) = {disp}", disp)
                    ]
                    receipt = self._build_receipt("combinatorics", "mathematics", "permutations", "math.permutations", {"n": str(n_val), "k": str(k_val)}, str(val), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "combinatorics", "permutations", "math.permutations", float(val), disp, "", steps, receipt, f"Permutations P({n_val}, {k_val}) = {disp}.")

        # Natural combination: "Choosing K out of N" / "K items be chosen from N"
        m = re.search(r"\bchoosing\s+(\d+)\s+(?:out\s+of|from)\s+(\d+)\b", lower) or re.search(
            r"\b(\d+)\s+items?\b.*?\b(?:chosen|selected|picked)\s+from\s+(\d+)\b",
            lower,
        )
        if m:
            k_val, n_val = int(m.group(1)), int(m.group(2))
            if 0 <= k_val <= n_val <= 1000:
                val = Fraction(math.comb(n_val, k_val))
                disp = self._format_fraction(val)
                steps = [
                    DerivationStep(1, "Compute combinations C(n, k)", "\\binom{n}{k} = \\frac{n!}{k!(n-k)!}", f"C({n_val}, {k_val}) = {disp}", disp)
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "combinations", "math.combinations", {"n": str(n_val), "k": str(k_val)}, str(val), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "combinations", "math.combinations", float(val), disp, "", steps, receipt, f"Combinations C({n_val}, {k_val}) = {disp}.")

        return None

    def _solve_python_ast_trace(self, q: str) -> Optional[SolverResult]:
        """Safely execute bounded straight-line/loop Python snippets using AST interpretation (no eval/exec)."""
        lower = q.lower()
        if any(tok in lower for tok in ("do not", "don't", "check proof", "verify proof", "denoise thought", "import ", "__", "eval", "exec", "lambda", "solve for ")):
            return None

        # Require recognizable assignment syntax plus a code structural clue
        if not re.search(r"\b[a-zA-Z_]\w*\s*=\s*", q):
            return None
        if not any(tok in q for tok in ("\n", ";", "for ", "while ", "range(", "sum(", "min(", "max(", ".count(", "nums[", "%", "//")):
            return None

        target_var: Optional[str] = None
        body = q.strip()

        # Strip common leading instruction wrappers
        lead_patterns = [
            re.compile(r"^\s*Python\s*:\s*", re.IGNORECASE),
            re.compile(r"^\s*What\s+does\s+([a-zA-Z_]\w*)\s+hold(?:\s+at\s+the\s+end)?\s*\?\s*", re.IGNORECASE),
            re.compile(r"^\s*Trace\s+this(?:\s+Python)?(?:\s+and\s+give\s+([a-zA-Z_]\w*))?\s*[.:]?\s*", re.IGNORECASE),
        ]
        for pat in lead_patterns:
            m = pat.match(body)
            if m:
                if m.lastindex and m.group(1):
                    target_var = m.group(1)
                body = body[m.end():].strip()

        # Strip common trailing question wrappers
        trail_patterns = [
            re.compile(r"[\s.]*What\s+is\s+([a-zA-Z_]\w*)\s+after\s+this\s+runs\s*\?\s*$", re.IGNORECASE),
            re.compile(r"[\s.]*(?:What\s+is|Final|Give(?:\s+the\s+value\s+of)?)\s+([a-zA-Z_]\w*)\s*[?.!]?\s*$", re.IGNORECASE),
            re.compile(r"[\s.]+([a-zA-Z_]\w*)\s*\?\s*$"),
        ]
        for pat in trail_patterns:
            m = pat.search(body)
            if m:
                target_var = m.group(1)
                body = body[:m.start()].rstrip()

        if not body or len(body) > 800:
            return None

        try:
            tree = ast.parse(body, mode="exec")
        except SyntaxError:
            return None
        if not tree.body:
            return None

        env: Dict[str, Any] = {}
        budget = [500]
        last_assigned: Optional[str] = None

        _BIN_OPS = {
            ast.Add: _op.add,
            ast.Sub: _op.sub,
            ast.Mult: _op.mul,
            ast.FloorDiv: _op.floordiv,
            ast.Mod: _op.mod,
            ast.Div: _op.truediv,
        }
        _CMP_OPS = {
            ast.Eq: _op.eq,
            ast.NotEq: _op.ne,
            ast.Lt: _op.lt,
            ast.LtE: _op.le,
            ast.Gt: _op.gt,
            ast.GtE: _op.ge,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }

        def _tick() -> None:
            budget[0] -= 1
            if budget[0] <= 0:
                raise ValueError("AST step limit exceeded")

        def _eval_expr(node: ast.AST) -> Any:
            _tick()
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, bool)) and not isinstance(node.value, bool):
                    return node.value
                if isinstance(node.value, bool):
                    return node.value
                raise ValueError("Unsupported constant")
            if isinstance(node, ast.Name):
                if node.id not in env:
                    raise ValueError(f"Unbound name {node.id}")
                return env[node.id]
            if isinstance(node, ast.List):
                if len(node.elts) > 256:
                    raise ValueError("List literal too large")
                return [_eval_expr(e) for e in node.elts]
            if isinstance(node, ast.Tuple):
                if len(node.elts) > 256:
                    raise ValueError("Tuple literal too large")
                return tuple(_eval_expr(e) for e in node.elts)
            if isinstance(node, ast.UnaryOp):
                v = _eval_expr(node.operand)
                if isinstance(node.op, ast.USub):
                    return -v
                if isinstance(node.op, ast.UAdd):
                    return +v
                if isinstance(node.op, ast.Not):
                    return not v
                raise ValueError("Unsupported unary op")
            if isinstance(node, ast.BinOp):
                fn = _BIN_OPS.get(type(node.op))
                if fn is None:
                    raise ValueError("Unsupported binary op")
                l_val = _eval_expr(node.left)
                r_val = _eval_expr(node.right)
                res = fn(l_val, r_val)
                if isinstance(res, (int, float)) and abs(res) > 10**9:
                    raise ValueError("Numeric bound exceeded")
                return res
            if isinstance(node, ast.Compare):
                left = _eval_expr(node.left)
                for cmp_op, comp_node in zip(node.ops, node.comparators):
                    fn = _CMP_OPS.get(type(cmp_op))
                    if fn is None:
                        raise ValueError("Unsupported compare")
                    right = _eval_expr(comp_node)
                    if not fn(left, right):
                        return False
                    left = right
                return True
            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    for v in node.values:
                        if not _eval_expr(v):
                            return False
                    return True
                if isinstance(node.op, ast.Or):
                    for v in node.values:
                        if _eval_expr(v):
                            return True
                    return False
                raise ValueError("Unsupported bool op")
            if isinstance(node, ast.IfExp):
                return _eval_expr(node.body) if _eval_expr(node.test) else _eval_expr(node.orelse)
            if isinstance(node, ast.Subscript):
                seq = _eval_expr(node.value)
                if not isinstance(seq, (list, tuple)):
                    raise ValueError("Subscript target must be sequence")
                idx = _eval_expr(node.slice)
                if not isinstance(idx, int) or isinstance(idx, bool):
                    raise ValueError("Subscript index must be int")
                return seq[idx]
            if isinstance(node, ast.Call):
                if node.keywords:
                    raise ValueError("Keyword arguments unsupported")
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                    args = [_eval_expr(a) for a in node.args]
                    if fname == "range":
                        if not (1 <= len(args) <= 3) or any(not isinstance(a, int) or isinstance(a, bool) for a in args):
                            raise ValueError("Invalid range")
                        r_obj = range(*args)
                        if len(r_obj) > 500:
                            raise ValueError("Range too long")
                        return list(r_obj)
                    if fname in {"sum", "min", "max", "len"} and len(args) == 1:
                        seq = args[0]
                        if not isinstance(seq, (list, tuple)):
                            raise ValueError("Expected sequence")
                        return {"sum": sum, "min": min, "max": max, "len": len}[fname](seq)
                    if fname == "abs" and len(args) == 1 and isinstance(args[0], (int, float)):
                        return abs(args[0])
                    raise ValueError(f"Unsupported call {fname}")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "count" and len(node.args) == 1:
                    recv = _eval_expr(node.func.value)
                    if not isinstance(recv, (list, tuple)):
                        raise ValueError("Count target must be sequence")
                    return recv.count(_eval_expr(node.args[0]))
                raise ValueError("Unsupported call target")
            raise ValueError(f"Unsupported AST expression {type(node).__name__}")

        def _exec_stmt(stmt: ast.stmt) -> None:
            nonlocal last_assigned
            _tick()
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise ValueError("Single variable assign only")
                val = _eval_expr(stmt.value)
                name = stmt.targets[0].id
                env[name] = val
                last_assigned = name
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, ast.Name):
                    raise ValueError("Single variable augassign only")
                fn = _BIN_OPS.get(type(stmt.op))
                if fn is None or stmt.target.id not in env:
                    raise ValueError("Invalid augassign")
                env[stmt.target.id] = fn(env[stmt.target.id], _eval_expr(stmt.value))
                last_assigned = stmt.target.id
            elif isinstance(stmt, ast.For):
                if stmt.orelse or not isinstance(stmt.target, ast.Name):
                    raise ValueError("Invalid for loop")
                seq = _eval_expr(stmt.iter)
                if not isinstance(seq, (list, tuple)) or len(seq) > 500:
                    raise ValueError("Invalid loop iterable")
                loop_var = stmt.target.id
                for item in seq:
                    _tick()
                    env[loop_var] = item
                    for sub in stmt.body:
                        _exec_stmt(sub)
            elif isinstance(stmt, ast.While):
                if stmt.orelse:
                    raise ValueError("While-else unsupported")
                while _eval_expr(stmt.test):
                    _tick()
                    for sub in stmt.body:
                        _exec_stmt(sub)
            elif isinstance(stmt, ast.If):
                branch = stmt.body if _eval_expr(stmt.test) else stmt.orelse
                for sub in branch:
                    _exec_stmt(sub)
            else:
                raise ValueError(f"Unsupported AST statement {type(stmt).__name__}")

        try:
            for stmt in tree.body:
                _exec_stmt(stmt)
        except Exception:
            return None

        out_name = target_var if target_var in env else ("r" if "r" in env else last_assigned)
        if not out_name or out_name not in env:
            return None
        val = env[out_name]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return None

        frac = self._normalize_num(str(val))
        disp = self._format_fraction(frac)
        steps = [
            DerivationStep(1, f"Execute bounded Python AST trace ({500 - budget[0]} ops)", out_name, f"{out_name} = {disp}", disp)
        ]
        receipt = self._build_receipt("python_trace", "computer_science", out_name, "code.ast_trace", {"target": out_name}, str(frac), disp, "dimensionless", steps)
        return SolverResult(True, q, "computer_science", "python_trace", out_name, "code.ast_trace", float(frac), disp, "", steps, receipt, f"Result for {out_name} is {disp}.")

    # --------------------------------------------------------------------------
    # 1. Kinematics Extended Solver
    # --------------------------------------------------------------------------
    def _solve_kinematics_extended(self, q: str) -> Optional[SolverResult]:
        # Torricelli: v^2 = u^2 + 2as -> v = sqrt(u^2 + 2as)
        u_m = re.search(r"(?:initial velocity|u)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s", q, re.I)
        a_m = re.search(r"(?:acceleration|a)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s\^?2", q, re.I)
        s_m = re.search(r"(?:displacement|distance|s|d)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        t_m = re.search(r"(?:time|t)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*s\b", q, re.I)

        if u_m and a_m and s_m and not t_m and ("final velocity" in q.lower() or "speed" in q.lower()):
            u = self._normalize_num(u_m.group(1))
            a = self._normalize_num(a_m.group(1))
            s = self._normalize_num(s_m.group(1))
            v_sq = u*u + 2*a*s
            if v_sq >= 0:
                v_val = math.sqrt(float(v_sq))
                v_frac = Fraction(Decimal(f"{v_val:.6f}"))
                disp = self._format_fraction(v_frac)
                steps = [
                    DerivationStep(1, "Apply Torricelli's equation for constant acceleration", "v^2 = u^2 + 2as", f"v^2 = ({u})^2 + 2({a})({s}) = {v_sq}", f"{v_sq}", "m^2/s^2"),
                    DerivationStep(2, "Extract root for positive final velocity", "v = \\sqrt{u^2 + 2as}", f"v = \\sqrt{{{v_sq}}} = {disp}", disp, "m/s"),
                ]
                receipt = self._build_receipt("kinematics", "physics", "final_velocity", "torricelli.final_velocity", {"u": str(u), "a": str(a), "s": str(s)}, str(v_frac), disp, "m/s", steps)
                return SolverResult(True, q, "physics", "kinematics", "final_velocity", "torricelli.final_velocity", v_val, disp, "m/s", steps, receipt, f"Final velocity is {disp} m/s.")

        # Constant acceleration final velocity: v = u + a * t
        if u_m and a_m and t_m and not s_m and ("final velocity" in q.lower() or "velocity after" in q.lower() or "velocity" in q.lower()):
            u = self._normalize_num(u_m.group(1))
            a = self._normalize_num(a_m.group(1))
            t = self._normalize_num(t_m.group(1))
            v = u + a * t
            disp = self._format_fraction(v)
            steps = [
                DerivationStep(1, "Apply first kinematic equation for constant acceleration", "v = u + a t", f"v = {u} + ({a})({t}) = {disp}", disp, "m/s")
            ]
            receipt = self._build_receipt("kinematics", "physics", "final_velocity", "kinematics.final_velocity", {"u": str(u), "a": str(a), "t": str(t)}, str(v), disp, "m/s", steps)
            return SolverResult(True, q, "physics", "kinematics", "final_velocity", "kinematics.final_velocity", float(v), disp, "m/s", steps, receipt, f"Final velocity is {disp} m/s.")

        # Average velocity: v_avg = (u + v) / 2
        v_m = re.search(r"(?:final velocity|v)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s", q, re.I)
        if u_m and v_m and ("average velocity" in q.lower() or "mean velocity" in q.lower()):
            u = self._normalize_num(u_m.group(1))
            v = self._normalize_num(v_m.group(1))
            v_avg = (u + v) / 2
            disp = self._format_fraction(v_avg)
            steps = [
                DerivationStep(1, "Compute average velocity under constant acceleration", "v_{\\text{avg}} = \\frac{u + v}{2}", f"v_{{\\text{{avg}}}} = \\frac{{{u} + {v}}}{{2}} = {disp}", disp, "m/s")
            ]
            receipt = self._build_receipt("kinematics", "physics", "average_velocity", "kinematics.average_velocity", {"u": str(u), "v": str(v)}, str(v_avg), disp, "m/s", steps)
            return SolverResult(True, q, "physics", "kinematics", "average_velocity", "kinematics.average_velocity", float(v_avg), disp, "m/s", steps, receipt, f"Average velocity is {disp} m/s.")

        return None

    # --------------------------------------------------------------------------
    # 2. Dynamics & Newton's Laws (F=ma, p=mv, J=F*dt, f_k=mu*N)
    # --------------------------------------------------------------------------
    def _solve_dynamics_newton(self, q: str) -> Optional[SolverResult]:
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)
        a_m = re.search(r"(?:acceleration|a)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)\s*m/s\^?2", q, re.I)
        f_m = re.search(r"(?:force|F|net force)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(N|kN)\b", q, re.I)
        v_m = re.search(r"(?:velocity|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)
        t_m = re.search(r"(?:time|duration|dt|t)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*s\b", q, re.I)
        mu_m = re.search(r"(?:friction coefficient|coefficient of friction|mu|mu_k)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)", q, re.I)
        n_m = re.search(r"(?:normal force|N)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*N\b", q, re.I)

        # Force F = m * a
        if m_m and a_m and ("force" in q.lower() or "net force" in q.lower()) and not f_m:
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            acc = self._normalize_num(a_m.group(1))
            force = mass * acc
            disp = self._format_fraction(force)
            steps = [
                DerivationStep(1, "Apply Newton's Second Law of Motion", "F = m \\cdot a", f"F = ({mass}\\text{{ kg}}) \\cdot ({acc}\\text{{ m/s}}^2) = {disp}\\text{{ N}}", disp, "N")
            ]
            receipt = self._build_receipt("dynamics", "physics", "force", "newton.second_law_force", {"m": str(mass), "a": str(acc)}, str(force), disp, "N", steps)
            return SolverResult(True, q, "physics", "dynamics", "force", "newton.second_law_force", float(force), disp, "N", steps, receipt, f"Net force is {disp} N.")

        # Acceleration a = F / m
        if f_m and m_m and ("acceleration" in q.lower() or "find acceleration" in q.lower()) and not a_m:
            force = self._normalize_num(f_m.group(1)) * (Fraction(1000) if f_m.group(2).lower() == "kn" else Fraction(1))
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            if mass > 0:
                acc = force / mass
                disp = self._format_fraction(acc)
                steps = [
                    DerivationStep(1, "Solve Newton's Second Law for acceleration", "a = \\frac{F}{m}", f"a = \\frac{{{force}\\text{{ N}}}}{{{mass}\\text{{ kg}}}} = {disp}\\text{{ m/s}}^2", disp, "m/s^2")
                ]
                receipt = self._build_receipt("dynamics", "physics", "acceleration", "newton.second_law_accel", {"F": str(force), "m": str(mass)}, str(acc), disp, "m/s^2", steps)
                return SolverResult(True, q, "physics", "dynamics", "acceleration", "newton.second_law_accel", float(acc), disp, "m/s^2", steps, receipt, f"Acceleration is {disp} m/s^2.")

        # Momentum p = m * v
        if m_m and v_m and ("momentum" in q.lower() or "linear momentum" in q.lower()):
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            vel = self._normalize_num(v_m.group(1))
            mom = mass * vel
            disp = self._format_fraction(mom)
            steps = [
                DerivationStep(1, "Calculate linear momentum definition", "p = m \\cdot v", f"p = ({mass}\\text{{ kg}}) \\cdot ({vel}\\text{{ m/s}}) = {disp}\\text{{ kg\\cdot m/s}}", disp, "kg*m/s")
            ]
            receipt = self._build_receipt("dynamics", "physics", "momentum", "dynamics.momentum", {"m": str(mass), "v": str(vel)}, str(mom), disp, "kg*m/s", steps)
            return SolverResult(True, q, "physics", "dynamics", "momentum", "dynamics.momentum", float(mom), disp, "kg*m/s", steps, receipt, f"Momentum is {disp} kg*m/s.")

        # Impulse J = F * dt
        if f_m and t_m and ("impulse" in q.lower() or "change in momentum" in q.lower()):
            force = self._normalize_num(f_m.group(1)) * (Fraction(1000) if f_m.group(2).lower() == "kn" else Fraction(1))
            dt = self._normalize_num(t_m.group(1))
            imp = force * dt
            disp = self._format_fraction(imp)
            steps = [
                DerivationStep(1, "Calculate impulse imparted by force over duration", "J = F \\cdot \\Delta t", f"J = ({force}\\text{{ N}}) \\cdot ({dt}\\text{{ s}}) = {disp}\\text{{ N\\cdot s}}", disp, "N*s")
            ]
            receipt = self._build_receipt("dynamics", "physics", "impulse", "dynamics.impulse", {"F": str(force), "t": str(dt)}, str(imp), disp, "N*s", steps)
            return SolverResult(True, q, "physics", "dynamics", "impulse", "dynamics.impulse", float(imp), disp, "N*s", steps, receipt, f"Impulse is {disp} N*s.")

        # Friction f_k = mu_k * N
        if mu_m and n_m and ("friction" in q.lower() or "frictional force" in q.lower()):
            mu = self._normalize_num(mu_m.group(1))
            norm = self._normalize_num(n_m.group(1))
            fric = mu * norm
            disp = self._format_fraction(fric)
            steps = [
                DerivationStep(1, "Calculate kinetic friction force", "f_k = \\mu_k \\cdot N", f"f_k = ({mu}) \\cdot ({norm}\\text{{ N}}) = {disp}\\text{{ N}}", disp, "N")
            ]
            receipt = self._build_receipt("dynamics", "physics", "friction", "dynamics.friction", {"mu": str(mu), "N": str(norm)}, str(fric), disp, "N", steps)
            return SolverResult(True, q, "physics", "dynamics", "friction", "dynamics.friction", float(fric), disp, "N", steps, receipt, f"Friction force is {disp} N.")

        return None

    # --------------------------------------------------------------------------
    # 3. Work, Energy & Power (W=F*d, Ek=1/2mv^2, Ep=mgh, P=W/t, Es=1/2kx^2)
    # --------------------------------------------------------------------------
    def _solve_work_energy_power(self, q: str) -> Optional[SolverResult]:
        f_m = re.search(r"(?:force|F)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*N\b", q, re.I)
        d_m = re.search(r"(?:distance|displacement|d|height|h|s)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)
        v_m = re.search(r"(?:velocity|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)
        t_m = re.search(r"(?:time|t|duration)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*s\b", q, re.I)
        w_m = re.search(r"(?:work|W|energy)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(J|kJ)\b", q, re.I)
        k_m = re.search(r"(?:spring\s+(?:of\s+)?constant|k)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*N/m\b", q, re.I)
        x_m = re.search(r"(?:compression|compressed|extension|extended|stretched?|x)\s*(?:=|is|of|by)?\s*(\d+(?:\.\d+)?)\s*(m|cm)\b", q, re.I)

        # Kinetic Energy: Ek = 1/2 * m * v^2
        if m_m and v_m and ("kinetic energy" in q.lower() or "ke" in q.lower()):
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            vel = self._normalize_num(v_m.group(1))
            ke = Fraction(1, 2) * mass * vel * vel
            disp = self._format_fraction(ke)
            steps = [
                DerivationStep(1, "Calculate kinetic energy", "E_k = \\frac{1}{2} m v^2", f"E_k = \\frac{{1}}{{2}} ({mass}\\text{{ kg}}) ({vel}\\text{{ m/s}})^2 = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "kinetic_energy", "energy.kinetic", {"m": str(mass), "v": str(vel)}, str(ke), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "kinetic_energy", "energy.kinetic", float(ke), disp, "J", steps, receipt, f"Kinetic energy is {disp} J.")

        # Potential Energy: Ep = m * g * h
        if m_m and d_m and ("potential energy" in q.lower() or "gravitational potential energy" in q.lower() or "pe" in q.lower()):
            mass = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            h = self._normalize_num(d_m.group(1))
            g = CONSTANTS["g"]["value"]
            pe = mass * g * h
            disp = self._format_fraction(pe)
            steps = [
                DerivationStep(1, "Calculate gravitational potential energy (g=9.80665 m/s^2)", "E_p = m g h", f"E_p = ({mass})({g})({h}) = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "potential_energy", "energy.potential", {"m": str(mass), "h": str(h)}, str(pe), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "potential_energy", "energy.potential", float(pe), disp, "J", steps, receipt, f"Potential energy is {disp} J.")

        # Work: W = F * d
        if f_m and d_m and ("work done" in q.lower() or "work" in q.lower()) and not w_m:
            force = self._normalize_num(f_m.group(1))
            dist = self._normalize_num(d_m.group(1))
            work = force * dist
            disp = self._format_fraction(work)
            steps = [
                DerivationStep(1, "Calculate mechanical work done along displacement", "W = F \\cdot d", f"W = ({force}\\text{{ N}}) \\cdot ({dist}\\text{{ m}}) = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "work", "energy.work", {"F": str(force), "d": str(dist)}, str(work), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "work", "energy.work", float(work), disp, "J", steps, receipt, f"Work done is {disp} J.")

        # Power: P = W / t
        if w_m and t_m and ("power" in q.lower()):
            work = self._normalize_num(w_m.group(1)) * (Fraction(1000) if w_m.group(2).lower() == "kj" else Fraction(1))
            time_val = self._normalize_num(t_m.group(1))
            if time_val > 0:
                power = work / time_val
                disp = self._format_fraction(power)
                steps = [
                    DerivationStep(1, "Calculate power as rate of energy transfer", "P = \\frac{W}{t}", f"P = \\frac{{{work}\\text{{ J}}}}{{{time_val}\\text{{ s}}}} = {disp}\\text{{ W}}", disp, "W")
                ]
                receipt = self._build_receipt("energy", "physics", "power", "energy.power", {"W": str(work), "t": str(time_val)}, str(power), disp, "W", steps)
                return SolverResult(True, q, "physics", "energy", "power", "energy.power", float(power), disp, "W", steps, receipt, f"Power is {disp} W.")

        # Spring Potential Energy: Es = 1/2 * k * x^2
        if k_m and x_m and ("spring energy" in q.lower() or "elastic potential" in q.lower()):
            k = self._normalize_num(k_m.group(1))
            x = self._normalize_num(x_m.group(1)) * (Fraction(1) if x_m.group(2).lower() == "m" else Fraction(1, 100))
            e_s = Fraction(1, 2) * k * x * x
            disp = self._format_fraction(e_s)
            steps = [
                DerivationStep(1, "Calculate elastic potential energy of Hookean spring", "E_s = \\frac{1}{2} k x^2", f"E_s = \\frac{{1}}{{2}} ({k}\\text{{ N/m}}) ({x}\\text{{ m}})^2 = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("energy", "physics", "spring_energy", "energy.spring", {"k": str(k), "x": str(x)}, str(e_s), disp, "J", steps)
            return SolverResult(True, q, "physics", "energy", "spring_energy", "energy.spring", float(e_s), disp, "J", steps, receipt, f"Elastic potential energy is {disp} J.")

        return None

    # --------------------------------------------------------------------------
    # 4. Circular Motion & Gravitation (ac=v^2/r, Fc=mv^2/r, Fg=G*m1*m2/r^2)
    # --------------------------------------------------------------------------
    def _solve_circular_gravitation(self, q: str) -> Optional[SolverResult]:
        v_m = re.search(r"(?:velocity|speed|v)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m/s\b", q, re.I)
        r_m = re.search(r"(?:radius|distance|r)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m|km)\b", q, re.I)
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)

        # Centripetal Acceleration: a_c = v^2 / r
        if v_m and r_m and ("centripetal acceleration" in q.lower() or "radial acceleration" in q.lower()):
            v = self._normalize_num(v_m.group(1))
            r = self._normalize_num(r_m.group(1)) * (Fraction(1000) if r_m.group(2).lower() == "km" else Fraction(1))
            if r > 0:
                ac = (v * v) / r
                disp = self._format_fraction(ac)
                steps = [
                    DerivationStep(1, "Calculate centripetal acceleration", "a_c = \\frac{v^2}{r}", f"a_c = \\frac{{({v}\\text{{ m/s}})^2}}{{{r}\\text{{ m}}}} = {disp}\\text{{ m/s}}^2", disp, "m/s^2")
                ]
                receipt = self._build_receipt("gravitation", "physics", "centripetal_acceleration", "circular.centripetal_accel", {"v": str(v), "r": str(r)}, str(ac), disp, "m/s^2", steps)
                return SolverResult(True, q, "physics", "gravitation", "centripetal_acceleration", "circular.centripetal_accel", float(ac), disp, "m/s^2", steps, receipt, f"Centripetal acceleration is {disp} m/s^2.")

        # Centripetal Force: F_c = m * v^2 / r
        if m_m and v_m and r_m and ("centripetal force" in q.lower()):
            m = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            v = self._normalize_num(v_m.group(1))
            r = self._normalize_num(r_m.group(1)) * (Fraction(1000) if r_m.group(2).lower() == "km" else Fraction(1))
            if r > 0:
                fc = (m * v * v) / r
                disp = self._format_fraction(fc)
                steps = [
                    DerivationStep(1, "Calculate centripetal force", "F_c = \\frac{m v^2}{r}", f"F_c = \\frac{{({m})({v})^2}}{{{r}}} = {disp}\\text{{ N}}", disp, "N")
                ]
                receipt = self._build_receipt("gravitation", "physics", "centripetal_force", "circular.centripetal_force", {"m": str(m), "v": str(v), "r": str(r)}, str(fc), disp, "N", steps)
                return SolverResult(True, q, "physics", "gravitation", "centripetal_force", "circular.centripetal_force", float(fc), disp, "N", steps, receipt, f"Centripetal force is {disp} N.")

        return None

    # --------------------------------------------------------------------------
    # 5. Thermodynamics & Heat (Q = mc*dT, Q = mL, eta = 1 - Tc/Th)
    # --------------------------------------------------------------------------
    def _solve_thermo_heat(self, q: str) -> Optional[SolverResult]:
        m_m = re.search(r"(?:mass|m)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(kg|g)\b", q, re.I)
        c_m = re.search(r"(?:specific heat|specific heat capacity|c)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*J/(?:kg\*K|kg\*C|kg K|kg C)\b", q, re.I)
        dt_m = re.search(r"(?:temperature change|delta T|dT|change in temperature)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:K|C|degrees)?\b", q, re.I)
        th_m = re.search(r"(?:hot temperature|hot reservoir|T_H|Th)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*K\b", q, re.I)
        tc_m = re.search(r"(?:cold temperature|cold reservoir|T_C|Tc)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*K\b", q, re.I)

        # Heat transfer: Q = m * c * dT
        if m_m and c_m and dt_m and ("heat" in q.lower() or "thermal energy" in q.lower()):
            m = self._normalize_num(m_m.group(1)) * (Fraction(1) if m_m.group(2).lower() == "kg" else Fraction(1, 1000))
            c = self._normalize_num(c_m.group(1))
            dt = self._normalize_num(dt_m.group(1))
            q_val = m * c * dt
            disp = self._format_fraction(q_val)
            steps = [
                DerivationStep(1, "Calculate sensible heat transfer", "Q = m \\cdot c \\cdot \\Delta T", f"Q = ({m}\\text{{ kg}}) ({c}\\text{{ J/kg\\cdot K}}) ({dt}\\text{{ K}}) = {disp}\\text{{ J}}", disp, "J")
            ]
            receipt = self._build_receipt("thermodynamics", "physics", "heat", "thermo.sensible_heat", {"m": str(m), "c": str(c), "dT": str(dt)}, str(q_val), disp, "J", steps)
            return SolverResult(True, q, "physics", "thermodynamics", "heat", "thermo.sensible_heat", float(q_val), disp, "J", steps, receipt, f"Heat transferred is {disp} J.")

        # Carnot Efficiency: eta = 1 - Tc / Th
        if th_m and tc_m and ("carnot efficiency" in q.lower() or "maximum efficiency" in q.lower() or "efficiency" in q.lower()):
            th = self._normalize_num(th_m.group(1))
            tc = self._normalize_num(tc_m.group(1))
            if th > tc and tc >= 0:
                eta = Fraction(1) - (tc / th)
                eta_pct = eta * 100
                disp = self._format_fraction(eta_pct)
                steps = [
                    DerivationStep(1, "Calculate ideal Carnot heat engine efficiency", "\\eta = 1 - \\frac{T_C}{T_H}", f"\\eta = 1 - \\frac{{{tc}}}{{{th}}} = {self._format_fraction(eta)} = {disp}\\%", f"{disp}%", "%")
                ]
                receipt = self._build_receipt("thermodynamics", "physics", "efficiency", "thermo.carnot_efficiency", {"Th": str(th), "Tc": str(tc)}, str(eta), f"{disp}%", "%", steps)
                return SolverResult(True, q, "physics", "thermodynamics", "efficiency", "thermo.carnot_efficiency", float(eta), f"{disp}%", "%", steps, receipt, f"Carnot efficiency is {disp}%.")

        return None

    # --------------------------------------------------------------------------
    # 6. Hydrostatics & Fluids (P = rho*g*h, Fb = rho*V*g, A1*v1 = A2*v2)
    # --------------------------------------------------------------------------
    def _solve_fluids_hydrostatics(self, q: str) -> Optional[SolverResult]:
        rho_m = re.search(r"(?:density|rho)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*kg/m\^?3\b", q, re.I)
        h_m = re.search(r"(?:depth|height|h)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        vol_m = re.search(r"(?:volume|V)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m\^?3|L|liters)\b", q, re.I)

        # Hydrostatic Pressure: P = rho * g * h
        if rho_m and h_m and ("hydrostatic pressure" in q.lower() or "gauge pressure" in q.lower() or "pressure at depth" in q.lower()):
            rho = self._normalize_num(rho_m.group(1))
            h = self._normalize_num(h_m.group(1))
            g = CONSTANTS["g"]["value"]
            p = rho * g * h
            disp = self._format_fraction(p)
            steps = [
                DerivationStep(1, "Calculate hydrostatic gauge pressure", "P = \\rho g h", f"P = ({rho}\\text{{ kg/m}}^3)({g}\\text{{ m/s}}^2)({h}\\text{{ m}}) = {disp}\\text{{ Pa}}", disp, "Pa")
            ]
            receipt = self._build_receipt("fluids", "physics", "hydrostatic_pressure", "fluids.hydrostatic_pressure", {"rho": str(rho), "h": str(h)}, str(p), disp, "Pa", steps)
            return SolverResult(True, q, "physics", "fluids", "hydrostatic_pressure", "fluids.hydrostatic_pressure", float(p), disp, "Pa", steps, receipt, f"Hydrostatic pressure is {disp} Pa.")

        # Buoyant Force: Fb = rho * V * g
        if rho_m and vol_m and ("buoyant force" in q.lower() or "buoyancy" in q.lower() or "archimedes" in q.lower()):
            rho = self._normalize_num(rho_m.group(1))
            vol = self._normalize_num(vol_m.group(1)) * (Fraction(1, 1000) if vol_m.group(2).lower() in ("l", "liters") else Fraction(1))
            g = CONSTANTS["g"]["value"]
            fb = rho * vol * g
            disp = self._format_fraction(fb)
            steps = [
                DerivationStep(1, "Apply Archimedes' principle for buoyant force", "F_b = \\rho \\cdot V \\cdot g", f"F_b = ({rho})({vol})({g}) = {disp}\\text{{ N}}", disp, "N")
            ]
            receipt = self._build_receipt("fluids", "physics", "buoyant_force", "fluids.buoyant_force", {"rho": str(rho), "V": str(vol)}, str(fb), disp, "N", steps)
            return SolverResult(True, q, "physics", "fluids", "buoyant_force", "fluids.buoyant_force", float(fb), disp, "N", steps, receipt, f"Buoyant force is {disp} N.")

        return None

    # --------------------------------------------------------------------------
    # 7. Electromagnetism & Circuits (V=IR, P=VI=I^2R=V^2/R, Req series/parallel)
    # --------------------------------------------------------------------------
    def _solve_circuits_electromagnetism(self, q: str) -> Optional[SolverResult]:
        v_m = re.search(r"(?:voltage|V|potential difference)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(V|mV|kV)\b", q, re.I) or (re.search(r"\b(\d+(?:\.\d+)?)\s*(V|mV|kV)\b", q, re.I) if any(k in q.lower() for k in ("current", "ohm", "resistor", "power")) else None)
        i_m = re.search(r"(?:current|I)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(A|mA)\b", q, re.I)
        r_m = re.search(r"(?:resistance|R)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I) or (re.search(r"\b(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I) if any(k in q.lower() for k in ("current", "voltage", "potential difference", "resistor")) else None)
        r1_m = re.search(r"R1\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I)
        r2_m = re.search(r"R2\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(ohm|ohms|Ω|kohm|kΩ)\b", q, re.I)

        # Ohm's Law: V = I * R -> Voltage
        if i_m and r_m and ("voltage" in q.lower() or "potential difference" in q.lower()) and not v_m:
            curr = self._normalize_num(i_m.group(1)) * (Fraction(1, 1000) if i_m.group(2).lower() == "ma" else Fraction(1))
            res = self._normalize_num(r_m.group(1)) * (Fraction(1000) if "k" in r_m.group(2).lower() else Fraction(1))
            volt = curr * res
            disp = self._format_fraction(volt)
            steps = [
                DerivationStep(1, "Apply Ohm's Law for voltage", "V = I \\cdot R", f"V = ({curr}\\text{{ A}}) \\cdot ({res}\\text{{ \\Omega}}) = {disp}\\text{{ V}}", disp, "V")
            ]
            receipt = self._build_receipt("circuits", "physics", "voltage", "circuits.ohms_law_voltage", {"I": str(curr), "R": str(res)}, str(volt), disp, "V", steps)
            return SolverResult(True, q, "physics", "circuits", "voltage", "circuits.ohms_law_voltage", float(volt), disp, "V", steps, receipt, f"Voltage is {disp} V.")

        # Ohm's Law: I = V / R -> Current
        if v_m and r_m and ("current" in q.lower()) and not i_m:
            volt = self._normalize_num(v_m.group(1)) * (Fraction(1000) if v_m.group(2).lower() == "kv" else (Fraction(1, 1000) if v_m.group(2).lower() == "mv" else Fraction(1)))
            res = self._normalize_num(r_m.group(1)) * (Fraction(1000) if "k" in r_m.group(2).lower() else Fraction(1))
            if res > 0:
                curr = volt / res
                disp = self._format_fraction(curr)
                steps = [
                    DerivationStep(1, "Apply Ohm's Law for current", "I = \\frac{V}{R}", f"I = \\frac{{{volt}\\text{{ V}}}}{{{res}\\text{{ \\Omega}}}} = {disp}\\text{{ A}}", disp, "A")
                ]
                receipt = self._build_receipt("circuits", "physics", "current", "circuits.ohms_law_current", {"V": str(volt), "R": str(res)}, str(curr), disp, "A", steps)
                return SolverResult(True, q, "physics", "circuits", "current", "circuits.ohms_law_current", float(curr), disp, "A", steps, receipt, f"Current is {disp} A.")

        # Electrical Power: P = V * I
        if v_m and i_m and ("electrical power" in q.lower() or "power dissipated" in q.lower() or "power" in q.lower()):
            volt = self._normalize_num(v_m.group(1)) * (Fraction(1000) if v_m.group(2).lower() == "kv" else (Fraction(1, 1000) if v_m.group(2).lower() == "mv" else Fraction(1)))
            curr = self._normalize_num(i_m.group(1)) * (Fraction(1, 1000) if i_m.group(2).lower() == "ma" else Fraction(1))
            power = volt * curr
            disp = self._format_fraction(power)
            steps = [
                DerivationStep(1, "Calculate electric power", "P = V \\cdot I", f"P = ({volt}\\text{{ V}}) \\cdot ({curr}\\text{{ A}}) = {disp}\\text{{ W}}", disp, "W")
            ]
            receipt = self._build_receipt("circuits", "physics", "power", "circuits.electric_power", {"V": str(volt), "I": str(curr)}, str(power), disp, "W", steps)
            return SolverResult(True, q, "physics", "circuits", "power", "circuits.electric_power", float(power), disp, "W", steps, receipt, f"Electrical power is {disp} W.")

        # Parallel Resistors: Req = (R1 * R2) / (R1 + R2)
        if r1_m and r2_m and ("parallel" in q.lower()):
            r1 = self._normalize_num(r1_m.group(1)) * (Fraction(1000) if "k" in r1_m.group(2).lower() else Fraction(1))
            r2 = self._normalize_num(r2_m.group(1)) * (Fraction(1000) if "k" in r2_m.group(2).lower() else Fraction(1))
            req = (r1 * r2) / (r1 + r2)
            disp = self._format_fraction(req)
            steps = [
                DerivationStep(1, "Calculate equivalent parallel resistance", "R_{\\text{eq}} = \\frac{R_1 R_2}{R_1 + R_2}", f"R_{{\\text{{eq}}}} = \\frac{{{r1} \\cdot {r2}}}{{{r1} + {r2}}} = {disp}\\text{{ \\Omega}}", disp, "Ω")
            ]
            receipt = self._build_receipt("circuits", "physics", "equivalent_resistance", "circuits.parallel_resistance", {"R1": str(r1), "R2": str(r2)}, str(req), disp, "Ω", steps)
            return SolverResult(True, q, "physics", "circuits", "equivalent_resistance", "circuits.parallel_resistance", float(req), disp, "Ω", steps, receipt, f"Equivalent parallel resistance is {disp} Ω.")

        return None

    # --------------------------------------------------------------------------
    # 8. Waves & Optics (v = f * lambda, T = 1/f, E = hf)
    # --------------------------------------------------------------------------
    def _solve_waves_optics(self, q: str) -> Optional[SolverResult]:
        _wave_ctx = "wave" in q.lower() or "wavelength" in q.lower()
        f_m = re.search(r"(?:frequency|f)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz|GHz)\b", q, re.I) or (re.search(r"\b(\d+(?:\.\d+)?)\s*(Hz|kHz|MHz|GHz)\b", q, re.I) if _wave_ctx else None)
        lam_m = re.search(r"(?:wavelength|lambda)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(m|cm|mm|nm)\b", q, re.I) or (re.search(r"\b(\d+(?:\.\d+)?)\s*(m|cm|mm|nm)\b", q, re.I) if _wave_ctx else None)

        # Wave Speed: v = f * lambda
        if f_m and lam_m and ("wave speed" in q.lower() or "speed of wave" in q.lower() or "speed of a wave" in q.lower() or "velocity" in q.lower() or ("wave" in q.lower() and "speed" in q.lower())):
            f_scale = {"hz": Fraction(1), "khz": Fraction(1000), "mhz": Fraction(1000000), "ghz": Fraction(10**9)}
            lam_scale = {"m": Fraction(1), "cm": Fraction(1, 100), "mm": Fraction(1, 1000), "nm": Fraction(1, 10**9)}
            freq = self._normalize_num(f_m.group(1)) * f_scale[f_m.group(2).lower()]
            lam = self._normalize_num(lam_m.group(1)) * lam_scale[lam_m.group(2).lower()]
            vel = freq * lam
            disp = self._format_fraction(vel)
            steps = [
                DerivationStep(1, "Calculate wave propagation speed", "v = f \\cdot \\lambda", f"v = ({freq}\\text{{ Hz}}) \\cdot ({lam}\\text{{ m}}) = {disp}\\text{{ m/s}}", disp, "m/s")
            ]
            receipt = self._build_receipt("waves", "physics", "wave_speed", "waves.wave_speed", {"f": str(freq), "lambda": str(lam)}, str(vel), disp, "m/s", steps)
            return SolverResult(True, q, "physics", "waves", "wave_speed", "waves.wave_speed", float(vel), disp, "m/s", steps, receipt, f"Wave speed is {disp} m/s.")

        # Wave period: T = 1 / f
        if f_m and ("period" in q.lower() or "wave period" in q.lower()):
            f_scale = {"hz": Fraction(1), "khz": Fraction(1000), "mhz": Fraction(1000000), "ghz": Fraction(10**9)}
            freq = self._normalize_num(f_m.group(1)) * f_scale[f_m.group(2).lower()]
            if freq > 0:
                period = Fraction(1) / freq
                disp = self._format_fraction(period, max_decimals=6)
                steps = [
                    DerivationStep(1, "Calculate period as inverse frequency", "T = \\frac{1}{f}", f"T = \\frac{{1}}{{{freq}\\text{{ Hz}}}} = {disp}\\text{{ s}}", disp, "s")
                ]
                receipt = self._build_receipt("waves", "physics", "period", "waves.period", {"f": str(freq)}, str(period), disp, "s", steps)
                return SolverResult(True, q, "physics", "waves", "period", "waves.period", float(period), disp, "s", steps, receipt, f"Wave period is {disp} s.")

        return None

    # --------------------------------------------------------------------------
    # 9. Chemistry & Solutions (M = n/V, M1*V1 = M2*V2, N = n*NA)
    # --------------------------------------------------------------------------
    def _solve_chemistry_solutions(self, q: str) -> Optional[SolverResult]:
        m1_m = re.search(r"(?:initial concentration|M1|C1)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:M|mol/L)\b", q, re.I)
        v1_m = re.search(r"(?:initial volume|V1)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(mL|L|liters)\b", q, re.I)
        m2_m = re.search(r"(?:target concentration|final concentration|M2|C2)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:M|mol/L)\b", q, re.I)
        _chem_ctx = "molar" in q.lower() or "concentration" in q.lower()
        n_m = re.search(r"(?:moles|amount|n)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:moles?|mol)\b", q, re.I) or (re.search(r"\b(\d+(?:\.\d+)?)\s*(?:moles?|mol)\b", q, re.I) if _chem_ctx else None)
        vol_m = re.search(r"(?:volume|V)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(litres|litre|liters|liter|L|mL)\b", q, re.I) or (re.search(r"\b(\d+(?:\.\d+)?)\s*(litres|litre|liters|liter|L|mL)\b", q, re.I) if _chem_ctx else None)

        # Molarity: M = n / V
        if n_m and vol_m and ("molarity" in q.lower() or "concentration" in q.lower()) and not m1_m:
            n = self._normalize_num(n_m.group(1))
            v = self._normalize_num(vol_m.group(1)) * (Fraction(1, 1000) if vol_m.group(2).lower() == "ml" else Fraction(1))
            if v > 0:
                molar = n / v
                disp = self._format_fraction(molar)
                steps = [
                    DerivationStep(1, "Calculate solution molarity", "M = \\frac{n}{V}", f"M = \\frac{{{n}\\text{{ mol}}}}{{{v}\\text{{ L}}}} = {disp}\\text{{ M}}", disp, "M")
                ]
                receipt = self._build_receipt("chemistry", "chemistry", "molarity", "chem.molarity", {"n": str(n), "V": str(v)}, str(molar), disp, "M", steps)
                return SolverResult(True, q, "chemistry", "chemistry", "molarity", "chem.molarity", float(molar), disp, "M", steps, receipt, f"Molarity is {disp} M.")

        # Dilution: V2 = (M1 * V1) / M2
        if m1_m and v1_m and m2_m and ("dilution" in q.lower() or "final volume" in q.lower()):
            m1 = self._normalize_num(m1_m.group(1))
            v1 = self._normalize_num(v1_m.group(1)) * (Fraction(1, 1000) if v1_m.group(2).lower() == "ml" else Fraction(1))
            m2 = self._normalize_num(m2_m.group(1))
            if m2 > 0:
                v2 = (m1 * v1) / m2
                v2_ml = v2 * 1000
                disp = self._format_fraction(v2_ml)
                steps = [
                    DerivationStep(1, "Apply conservation of solute mass (dilution equation)", "M_1 V_1 = M_2 V_2 \\implies V_2 = \\frac{M_1 V_1}{M_2}", f"V_2 = \\frac{{({m1})({v1})}}{{{m2}}} = {self._format_fraction(v2)}\\text{{ L}} = {disp}\\text{{ mL}}", f"{disp} mL", "mL")
                ]
                receipt = self._build_receipt("chemistry", "chemistry", "dilution_volume", "chem.dilution", {"M1": str(m1), "V1": str(v1), "M2": str(m2)}, str(v2), f"{disp} mL", "mL", steps)
                return SolverResult(True, q, "chemistry", "chemistry", "dilution_volume", "chem.dilution", float(v2_ml), f"{disp} mL", "mL", steps, receipt, f"Required final volume is {disp} mL.")

        return None

    # --------------------------------------------------------------------------
    # 10. Pure Algebra: Quadratic Equation (ax^2 + bx + c = rhs)
    # --------------------------------------------------------------------------
    def _solve_quadratic_algebra(self, q: str) -> Optional[SolverResult]:
        if re.search(r"\b(?:not|never|approx(?:imately)?|verify\s+proof|check\s+proof)\b", q, re.I):
            return None
        q_norm = q.replace("²", "^2").replace("−", "-")
        a_m = re.search(r"\ba\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q_norm, re.I)
        b_m = re.search(r"\bb\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q_norm, re.I)
        c_m = re.search(r"\bc\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)", q_norm, re.I)

        quad_match = re.search(
            r"([+-]?\s*\d+(?:\.\d+)?|[+-])?\s*\*?\s*x\^2"
            r"(?:\s*([+-]\s*(?:\d+(?:\.\d+)?)?)\s*\*?\s*x(?!\^))?"
            r"(?:\s*([+-]\s*\d+(?:\.\d+)?))?"
            r"\s*=\s*(-?\d+(?:\.\d+)?)",
            q_norm,
            re.I,
        )

        def _parse_signed_coeff(raw: Optional[str], default_if_empty: str = "1") -> Fraction:
            cleaned = (raw or "").replace(" ", "")
            if not cleaned:
                return self._normalize_num(default_if_empty)
            if cleaned == "+":
                return Fraction(1)
            if cleaned == "-":
                return Fraction(-1)
            return self._normalize_num(cleaned)

        if (a_m and b_m and c_m and ("quadratic" in q_norm.lower() or "roots" in q_norm.lower())) or quad_match:
            if quad_match:
                a = _parse_signed_coeff(quad_match.group(1), "1")
                b = _parse_signed_coeff(quad_match.group(2), "0") if quad_match.group(2) is not None else Fraction(0)
                c_lhs = _parse_signed_coeff(quad_match.group(3), "0") if quad_match.group(3) is not None else Fraction(0)
                rhs = _parse_signed_coeff(quad_match.group(4), "0")
                c = c_lhs - rhs
            else:
                a = self._normalize_num(a_m.group(1))
                b = self._normalize_num(b_m.group(1))
                c = self._normalize_num(c_m.group(1))

            if a == 0:
                return None

            disc = b * b - 4 * a * c
            disc_float = float(disc)
            if disc_float >= 0:
                sqrt_disc = math.sqrt(disc_float)
                r1 = (-float(b) + sqrt_disc) / (2 * float(a))
                r2 = (-float(b) - sqrt_disc) / (2 * float(a))
                disp1 = f"{r1:.4f}".rstrip("0").rstrip(".") or "0"
                disp2 = f"{r2:.4f}".rstrip("0").rstrip(".") or "0"
                res_str = f"x1 = {disp1}, x2 = {disp2}" if disp1 != disp2 else f"x = {disp1}"
                steps = [
                    DerivationStep(1, "Calculate discriminant Delta = b^2 - 4ac", "\\Delta = b^2 - 4ac", f"\\Delta = ({b})^2 - 4({a})({c}) = {disc}", str(disc)),
                    DerivationStep(2, "Calculate roots via quadratic formula", "x = \\frac{-b \\pm \\sqrt{\\Delta}}{2a}", f"x = \\frac{{-({b}) \\pm \\sqrt{{{disc}}}}}{{2({a})}} \\implies {res_str}", res_str),
                ]
                receipt = self._build_receipt("algebra", "mathematics", "quadratic_roots", "math.quadratic", {"a": str(a), "b": str(b), "c": str(c)}, str(disc), res_str, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "algebra", "quadratic_roots", "math.quadratic", float(r1), res_str, "", steps, receipt, f"Roots of quadratic: {res_str}.")

        return None

    # --------------------------------------------------------------------------
    # 11. Linear System of 2 Equations (a1*x + b1*y = c1, a2*x + b2*y = c2)
    # --------------------------------------------------------------------------
    def _solve_linear_system_2x2(self, q: str) -> Optional[SolverResult]:
        # e.g., "solve system 2x + 3y = 8 and 4x - y = 2" or "x + y = 10, x - y = 2"
        if re.search(r"\b(?:not|never|approx(?:imately)?|verify\s+proof|check\s+proof)\b", q, re.I):
            return None
        q_norm = q.replace("−", "-")
        sys_m = re.search(
            r"([+-]?\s*(?:\d+(?:\.\d+)?)?)\s*\*?\s*x\s*([+-]\s*(?:\d+(?:\.\d+)?)?)\s*\*?\s*y\s*=\s*(-?\d+(?:\.\d+)?)"
            r"\s*(?:and|,|;)\s*"
            r"([+-]?\s*(?:\d+(?:\.\d+)?)?)\s*\*?\s*x\s*([+-]\s*(?:\d+(?:\.\d+)?)?)\s*\*?\s*y\s*=\s*(-?\d+(?:\.\d+)?)",
            q_norm,
            re.I,
        )
        if sys_m:
            def _coeff(raw: str) -> Fraction:
                cleaned = raw.replace(" ", "")
                if cleaned in ("", "+"):
                    return Fraction(1)
                if cleaned == "-":
                    return Fraction(-1)
                return self._normalize_num(cleaned)

            a1 = _coeff(sys_m.group(1))
            b1 = _coeff(sys_m.group(2))
            c1 = self._normalize_num(sys_m.group(3))
            a2 = _coeff(sys_m.group(4))
            b2 = _coeff(sys_m.group(5))
            c2 = self._normalize_num(sys_m.group(6))

            det = a1*b2 - a2*b1
            if det != 0:
                det_x = c1*b2 - c2*b1
                det_y = a1*c2 - a2*c1
                x = det_x / det
                y = det_y / det
                disp_x = self._format_fraction(x)
                disp_y = self._format_fraction(y)
                res_str = f"x = {disp_x}, y = {disp_y}"
                steps = [
                    DerivationStep(1, "Compute system determinant via Cramer's rule", "D = a_1 b_2 - a_2 b_1", f"D = ({a1})({b2}) - ({a2})({b1}) = {det}", str(det)),
                    DerivationStep(2, "Compute coordinate determinants and solve", "x = \\frac{D_x}{D}, \\quad y = \\frac{D_y}{D}", f"x = \\frac{{{det_x}}}{{{det}}} = {disp_x}, \\quad y = \\frac{{{det_y}}}{{{det}}} = {disp_y}", res_str),
                ]
                receipt = self._build_receipt("algebra", "mathematics", "linear_system_2x2", "math.cramer_2x2", {"eq1": f"{a1}x+{b1}y={c1}", "eq2": f"{a2}x+{b2}y={c2}"}, f"({x},{y})", res_str, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "algebra", "linear_system_2x2", "math.cramer_2x2", float(x), res_str, "", steps, receipt, f"Solution to linear system: {res_str}.")

        return None

    # --------------------------------------------------------------------------
    # 12. Compound Interest (A = P * (1 + r/n)^(n*t))
    # --------------------------------------------------------------------------
    def _solve_compound_interest(self, q: str) -> Optional[SolverResult]:
        p_m = re.search(r"(?:principal|principal amount|P)\s*(?:=|is|of)?\s*(\$?\d+(?:\.\d+)?)", q, re.I)
        r_m = re.search(r"(?:interest rate|rate|r)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*%", q, re.I)
        t_m = re.search(r"(?:time|t|duration|years)\s*(?:=|is|of)?\s*(\d+(?:\.\d+)?)\s*(?:years|yr|yrs)\b", q, re.I)
        n_m = re.search(r"(?:compounded|frequency|n)\s*(?:=|is)?\s*(\d+)\s*(?:times per year)?", q, re.I)

        if p_m and r_m and t_m and ("compound interest" in q.lower() or "future value" in q.lower()):
            p_clean = p_m.group(1).replace("$", "")
            principal = self._normalize_num(p_clean)
            rate = self._normalize_num(r_m.group(1)) / 100
            years = self._normalize_num(t_m.group(1))
            n = self._normalize_num(n_m.group(1)) if n_m else Fraction(1)  # annual by default

            # A = P * (1 + r/n)^(nt)
            rate_float = float(rate)
            n_float = float(n)
            t_float = float(years)
            p_float = float(principal)
            amount = p_float * ((1 + rate_float / n_float) ** (n_float * t_float))
            interest = amount - p_float
            disp_amt = f"{amount:.2f}"
            disp_int = f"{interest:.2f}"
            res_str = f"Final Amount: ${disp_amt} (Interest: ${disp_int})"

            steps = [
                DerivationStep(1, "Apply compound interest formula", "A = P \\left(1 + \\frac{r}{n}\\right)^{nt}", f"A = {principal} \\left(1 + \\frac{{{rate}}}{{{n}}}\\right)^{{({n})({years})}} = {disp_amt}", disp_amt, "$")
            ]
            receipt = self._build_receipt("finance", "mathematics", "compound_interest", "math.compound_interest", {"P": str(principal), "r": str(rate), "t": str(years), "n": str(n)}, str(amount), res_str, "$", steps)
            return SolverResult(True, q, "mathematics", "finance", "compound_interest", "math.compound_interest", amount, res_str, "$", steps, receipt, f"Compound interest outcome: {res_str}.")

        return None

    # --------------------------------------------------------------------------
    # 13. Series & Progressions (Arithmetic / Geometric Sum)
    # --------------------------------------------------------------------------
    def _solve_series_progressions(self, q: str) -> Optional[SolverResult]:
        # Arithmetic: Sn = n/2 * (2a + (n-1)d)
        a_m = re.search(
            r"(?:first\s+term|starts?\s+at|starting\s+at|a1|a)\s*(?:=|is)?\s*(-?\d+(?:\.\d+)?)",
            q,
            re.I,
        )
        d_m = re.search(
            r"(?:common\s+difference|difference|d)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)",
            q,
            re.I,
        )
        r_m = re.search(r"(?:common\s+ratio|ratio|r)\s*(?:=|is|of)?\s*(-?\d+(?:\.\d+)?)", q, re.I)
        n_m = re.search(
            r"(?:(?:number\s+of\s+terms|n)\s*(?:=|is)?\s*(\d+)|(?:sum\s+of\s+(?:the\s+)?first\s+|first\s+)(\d+)\s+terms?\b|\b(\d+)\s+terms?\b)",
            q,
            re.I,
        )
        n_group = next((g for g in n_m.groups() if g is not None), None) if n_m else None

        if a_m and d_m and n_group and (
            "arithmetic series" in q.lower()
            or "arithmetic progression" in q.lower()
            or "sum of arithmetic" in q.lower()
        ):
            a = self._normalize_num(a_m.group(1))
            d = self._normalize_num(d_m.group(1))
            n = self._normalize_num(n_group)
            # Sn = n/2 * (2a + (n-1)d)
            sn = (n / 2) * (2*a + (n-1)*d)
            disp = self._format_fraction(sn)
            steps = [
                DerivationStep(1, "Compute arithmetic series sum", "S_n = \\frac{n}{2} (2a + (n-1)d)", f"S_{{{n}}} = \\frac{{{n}}}{{2}} (2({a}) + ({n}-1)({d})) = {disp}", disp)
            ]
            receipt = self._build_receipt("series", "mathematics", "arithmetic_sum", "math.arithmetic_series", {"a": str(a), "d": str(d), "n": str(n)}, str(sn), disp, "dimensionless", steps)
            return SolverResult(True, q, "mathematics", "series", "arithmetic_sum", "math.arithmetic_series", float(sn), disp, "", steps, receipt, f"Sum of arithmetic series is {disp}.")

        if a_m and r_m and n_group and ("geometric series" in q.lower() or "geometric progression" in q.lower() or "sum of geometric" in q.lower()):
            a = self._normalize_num(a_m.group(1))
            r = self._normalize_num(r_m.group(1))
            n = int(n_group)
            if r != 1:
                sn = a * (1 - r**n) / (1 - r)
                disp = self._format_fraction(sn)
                steps = [
                    DerivationStep(1, "Compute finite geometric series sum", "S_n = a \\frac{1 - r^n}{1 - r}", f"S_{{{n}}} = {a} \\frac{{1 - ({r})^{{{n}}}}}{{1 - {r}}} = {disp}", disp)
                ]
                receipt = self._build_receipt("series", "mathematics", "geometric_sum", "math.geometric_series", {"a": str(a), "r": str(r), "n": str(n)}, str(sn), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "series", "geometric_sum", "math.geometric_series", float(sn), disp, "", steps, receipt, f"Sum of geometric series is {disp}.")

        return None

    # --------------------------------------------------------------------------
    # 14. Geometry & Trigonometry (Pythagorean, Circle, Sphere, Cylinder)
    # --------------------------------------------------------------------------
    def _solve_geometry_trig(self, q: str) -> Optional[SolverResult]:
        # Pythagorean: c = sqrt(a^2 + b^2)
        side_a_m = re.search(r"(?:side a|leg a|a)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        side_b_m = re.search(r"(?:side b|leg b|b)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        hyp_m = re.search(r"(?:hypotenuse|c)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        rad_m = re.search(r"(?:radius|r)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)
        h_m = re.search(r"(?:height|h)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\s*m\b", q, re.I)

        # Hypotenuse: c = sqrt(a^2 + b^2)
        if side_a_m and side_b_m and ("hypotenuse" in q.lower() or "pythagorean" in q.lower()) and not hyp_m:
            a = self._normalize_num(side_a_m.group(1))
            b = self._normalize_num(side_b_m.group(1))
            c_sq = a*a + b*b
            c_val = math.sqrt(float(c_sq))
            c_frac = Fraction(Decimal(f"{c_val:.6f}"))
            disp = self._format_fraction(c_frac)
            steps = [
                DerivationStep(1, "Apply Pythagorean Theorem", "c = \\sqrt{a^2 + b^2}", f"c = \\sqrt{{{a}^2 + {b}^2}} = \\sqrt{{{c_sq}}} = {disp}\\text{{ m}}", disp, "m")
            ]
            receipt = self._build_receipt("geometry", "mathematics", "hypotenuse", "geometry.pythagorean", {"a": str(a), "b": str(b)}, str(c_frac), disp, "m", steps)
            return SolverResult(True, q, "mathematics", "geometry", "hypotenuse", "geometry.pythagorean", c_val, disp, "m", steps, receipt, f"Hypotenuse is {disp} m.")

        # Sphere Volume: V = 4/3 * pi * r^3
        if rad_m and ("sphere volume" in q.lower() or "volume of sphere" in q.lower()):
            r = self._normalize_num(rad_m.group(1))
            vol = Fraction(4, 3) * CONSTANTS["pi"]["value"] * (r**3)
            disp = self._format_fraction(vol)
            steps = [
                DerivationStep(1, "Calculate sphere volume", "V = \\frac{4}{3} \\pi r^3", f"V = \\frac{{4}}{{3}} \\pi ({r}\\text{{ m}})^3 = {disp}\\text{{ m}}^3", disp, "m^3")
            ]
            receipt = self._build_receipt("geometry", "mathematics", "sphere_volume", "geometry.sphere_volume", {"r": str(r)}, str(vol), disp, "m^3", steps)
            return SolverResult(True, q, "mathematics", "geometry", "sphere_volume", "geometry.sphere_volume", float(vol), disp, "m^3", steps, receipt, f"Sphere volume is {disp} m^3.")

        # Cylinder Volume: V = pi * r^2 * h
        if rad_m and h_m and ("cylinder volume" in q.lower() or "volume of cylinder" in q.lower()):
            r = self._normalize_num(rad_m.group(1))
            h = self._normalize_num(h_m.group(1))
            vol = CONSTANTS["pi"]["value"] * (r**2) * h
            disp = self._format_fraction(vol)
            steps = [
                DerivationStep(1, "Calculate cylinder volume", "V = \\pi r^2 h", f"V = \\pi ({r}\\text{{ m}})^2 ({h}\\text{{ m}}) = {disp}\\text{{ m}}^3", disp, "m^3")
            ]
            receipt = self._build_receipt("geometry", "mathematics", "cylinder_volume", "geometry.cylinder_volume", {"r": str(r), "h": str(h)}, str(vol), disp, "m^3", steps)
            return SolverResult(True, q, "mathematics", "geometry", "cylinder_volume", "geometry.cylinder_volume", float(vol), disp, "m^3", steps, receipt, f"Cylinder volume is {disp} m^3.")

        return None

    # --------------------------------------------------------------------------
    # 15. Combinatorics, Probability & Statistics (P(n,k), C(n,k), Entropy)
    # --------------------------------------------------------------------------
    def _solve_combinatorics_stats(self, q: str) -> Optional[SolverResult]:
        n_m = re.search(r"\bn\s*(?:=|is)?\s*(\d+)", q, re.I)
        k_m = re.search(r"\b(?:k|r)\s*(?:=|is)?\s*(\d+)", q, re.I)

        # Combinations: C(n, k) = n! / (k! * (n-k)!)
        if n_m and k_m and "repetition" not in q.lower() and "multiset" not in q.lower() and ("combination" in q.lower() or "choose" in q.lower() or "c(n,k)" in q.lower() or "n choose k" in q.lower()):
            n = int(n_m.group(1))
            k = int(k_m.group(1))
            if 0 <= k <= n:
                ans = math.comb(n, k)
                disp = str(ans)
                steps = [
                    DerivationStep(1, "Apply combination binomial coefficient formula", "C(n,k) = \\binom{n}{k} = \\frac{n!}{k!(n-k)!}", f"\\binom{{{n}}}{{{k}}} = \\frac{{{n}!}}{{{k}! ({n}-{k})!}} = {disp}", disp)
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "combinations", "math.combinations", {"n": str(n), "k": str(k)}, str(ans), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "combinations", "math.combinations", float(ans), disp, "", steps, receipt, f"Number of combinations C({n},{k}) = {disp}.")

        # Permutations: P(n, k) = n! / (n-k)!
        if n_m and k_m and ("permutation" in q.lower() or "p(n,k)" in q.lower()):
            n = int(n_m.group(1))
            k = int(k_m.group(1))
            if 0 <= k <= n:
                ans = math.perm(n, k)
                disp = str(ans)
                steps = [
                    DerivationStep(1, "Apply permutation formula", "P(n,k) = \\frac{n!}{(n-k)!}", f"P({n},{k}) = \\frac{{{n}!}}{{({n}-{k})!}} = {disp}", disp)
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "permutations", "math.permutations", {"n": str(n), "k": str(k)}, str(ans), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "permutations", "math.permutations", float(ans), disp, "", steps, receipt, f"Number of permutations P({n},{k}) = {disp}.")

        return None

    # --------------------------------------------------------------------------
    # 16. Bounded Python Expression Evaluation (v93 Code Families, Zero eval())
    # --------------------------------------------------------------------------
    def _solve_code_expression(self, q: str) -> Optional[SolverResult]:
        # code_range_sum: r = sum(range(stop)) or r = sum(range(start, stop))
        range_m = re.search(r"\bsum\s*\(\s*range\s*\(\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?\)\s*\)", q)
        if range_m:
            if range_m.group(2) is None:
                start, stop = 0, int(range_m.group(1))
            else:
                start, stop = int(range_m.group(1)), int(range_m.group(2))
            if abs(stop - start) <= 10000:
                total = sum(range(start, stop))
                disp = str(total)
                steps = [
                    DerivationStep(1, "Evaluate integer arithmetic range summation", f"\\sum_{{i={start}}}^{{{stop - 1}}} i", f"sum(range({start}, {stop})) = {disp}", disp)
                ]
                receipt = self._build_receipt("code", "programming", "code_range_sum", "code.range_sum", {"start": str(start), "stop": str(stop)}, str(total), disp, "dimensionless", steps)
                return SolverResult(True, q, "programming", "code", "code_range_sum", "code.range_sum", float(total), disp, "", steps, receipt, f"Range sum evaluates to {disp}.")

        # code_list_count: nums = [a, b, c, ...] ... nums.count(v) or [...].count(v)
        count_m = re.search(
            r"(?:nums\s*=\s*\[([^\]]+)\].*?nums|[(\[]\s*([^\]]+)\s*\])\.count\s*\(\s*(-?\d+)\s*\)",
            q,
            flags=re.DOTALL,
        )
        if count_m:
            raw_list = count_m.group(1) or count_m.group(2) or ""
            target_val = int(count_m.group(3))
            if re.fullmatch(r"\s*-?\d+(?:\s*,\s*-?\d+)*\s*", raw_list):
                items = [int(x.strip()) for x in raw_list.split(",")]
                hits = sum(1 for item in items if item == target_val)
                disp = str(hits)
                steps = [
                    DerivationStep(1, "Count occurrences in integer list literal", f"[{raw_list}].count({target_val})", f"count = {disp}", disp)
                ]
                receipt = self._build_receipt("code", "programming", "code_list_count", "code.list_count", {"items": raw_list, "value": str(target_val)}, str(hits), disp, "dimensionless", steps)
                return SolverResult(True, q, "programming", "code", "code_list_count", "code.list_count", float(hits), disp, "", steps, receipt, f"List count evaluates to {disp}.")

        # code_neg_index: nums = [a, b, ...] ... r = nums[-k] (single RHS subscript only,
        # so multi-term expressions like `r = nums[1] + nums[2]` fall through to AST trace).
        neg_m = re.search(
            r"nums\s*=\s*\[([^\]]+)\][\s;,\n]*r\s*=\s*nums\s*\[\s*(-\d+)\s*\]\s*(?:$|(?:Final|What|Give)\b|[?.!])",
            q,
            flags=re.IGNORECASE,
        )
        if neg_m:
            raw_list, idx_raw = neg_m.group(1), int(neg_m.group(2))
            if re.fullmatch(r"\s*-?\d+(?:\s*,\s*-?\d+)*\s*", raw_list):
                items = [int(x.strip()) for x in raw_list.split(",")]
                if -len(items) <= idx_raw < len(items):
                    val = items[idx_raw]
                    disp = str(val)
                    steps = [
                        DerivationStep(1, "Index sequence element", f"nums[{idx_raw}]", f"[{raw_list}][{idx_raw}] = {disp}", disp)
                    ]
                    receipt = self._build_receipt("code", "programming", "code_neg_index", "code.neg_index", {"items": raw_list, "index": str(idx_raw)}, str(val), disp, "dimensionless", steps)
                    return SolverResult(True, q, "programming", "code", "code_neg_index", "code.neg_index", float(val), disp, "", steps, receipt, f"Indexed element is {disp}.")

        return None

    # --------------------------------------------------------------------------
    # 17. Generalized Sequence & Recurrence Discovery (Polynomial / Geometric / Fibonacci)
    # --------------------------------------------------------------------------
    def _solve_general_sequences_and_laws(self, q: str) -> Optional[SolverResult]:
        lower = q.lower()
        if any(neg in lower for neg in ("not", "never", "approx", "verify proof", "check proof")):
            return None
        if not any(kw in lower for kw in ("sequence", "next term", "comes next", "pattern", "progression")):
            return None

        nums_m = re.search(r"(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){2,})", q)
        if not nums_m:
            return None
        vals = [self._normalize_num(item.strip()) for item in nums_m.group(1).split(",")]
        if len(vals) < 3:
            return None

        # 1. Exact Geometric progression: t[i+1] / t[i] == r (non-zero constant ratio)
        if all(v != 0 for v in vals[:-1]):
            ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
            if ratios and all(r == ratios[0] for r in ratios) and ratios[0] != 1:
                nxt = vals[-1] * ratios[0]
                disp = self._format_fraction(nxt)
                steps = [
                    DerivationStep(1, "Identify constant geometric ratio r", "r = \\frac{a_{k+1}}{a_k}", f"r = {self._format_fraction(ratios[0])}", self._format_fraction(ratios[0])),
                    DerivationStep(2, "Multiply last term by common ratio", "a_{n+1} = a_n \\cdot r", f"{self._format_fraction(vals[-1])} \\cdot {self._format_fraction(ratios[0])} = {disp}", disp),
                ]
                receipt = self._build_receipt("sequence_discovery", "mathematics", "geometric_next_term", "math.sequence_geometric", {"terms": nums_m.group(1), "ratio": str(ratios[0])}, str(nxt), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "sequence_discovery", "geometric_next_term", "math.sequence_geometric", float(nxt), disp, "", steps, receipt, f"Geometric progression with ratio {self._format_fraction(ratios[0])}; next term is {disp}.")

        # 2. Newton forward-difference polynomial sequence up to degree 3 (requires degree < len(vals))
        max_order = min(3, len(vals) - 2) if len(vals) >= 4 else 1
        levels: List[List[Fraction]] = [list(vals)]
        matched_order: Optional[int] = None
        for order in range(1, max_order + 1):
            prev = levels[-1]
            diff = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
            levels.append(diff)
            if diff and all(d == diff[0] for d in diff):
                matched_order = order
                break

        if matched_order is not None:
            # Propagate bottom-up to compute next term
            nxt_val = Fraction(0)
            for lvl in reversed(levels):
                nxt_val = lvl[-1] + nxt_val
            disp = self._format_fraction(nxt_val)
            steps = [
                DerivationStep(
                    1,
                    f"Compute finite-difference table of order {matched_order}",
                    f"\\Delta^{matched_order} a_k = \\text{{const}}",
                    f"\\Delta^{matched_order} = {self._format_fraction(levels[matched_order][0])}",
                    self._format_fraction(levels[matched_order][0]),
                ),
                DerivationStep(
                    2,
                    "Propagate terminal differences to next term",
                    "a_{n+1} = \\sum_{m=0}^{d} \\Delta^m a_{n-m}",
                    f"a_{{n+1}} = {disp}",
                    disp,
                ),
            ]
            receipt = self._build_receipt(
                "sequence_discovery",
                "mathematics",
                "polynomial_sequence_next",
                f"math.sequence_poly_deg{matched_order}",
                {"terms": nums_m.group(1), "degree": str(matched_order)},
                str(nxt_val),
                disp,
                "dimensionless",
                steps,
            )
            return SolverResult(
                True,
                q,
                "mathematics",
                "sequence_discovery",
                "polynomial_sequence_next",
                f"math.sequence_poly_deg{matched_order}",
                float(nxt_val),
                disp,
                "",
                steps,
                receipt,
                f"Degree-{matched_order} finite-difference law gives next term {disp}.",
            )

        # 3. 2-step Fibonacci additive recurrence: a[k] = a[k-1] + a[k-2]
        if len(vals) >= 4 and all(vals[i] == vals[i - 1] + vals[i - 2] for i in range(2, len(vals))):
            nxt_val = vals[-1] + vals[-2]
            disp = self._format_fraction(nxt_val)
            steps = [
                DerivationStep(1, "Verify two-term additive recurrence", "a_k = a_{k-1} + a_{k-2}", f"{self._format_fraction(vals[-2])} + {self._format_fraction(vals[-1])} = {disp}", disp)
            ]
            receipt = self._build_receipt("sequence_discovery", "mathematics", "fibonacci_next", "math.sequence_recurrence", {"terms": nums_m.group(1)}, str(nxt_val), disp, "dimensionless", steps)
            return SolverResult(True, q, "mathematics", "sequence_discovery", "fibonacci_next", "math.sequence_recurrence", float(nxt_val), disp, "", steps, receipt, f"Two-term additive recurrence gives next term {disp}.")

        return None

    # --------------------------------------------------------------------------
    # 18. Calculus (Polynomial Derivatives/Definite Integrals), Number Theory & Stats
    # --------------------------------------------------------------------------
    @staticmethod
    def _parse_poly_terms(poly_str: str) -> Optional[Dict[int, Fraction]]:
        """Parse univariate polynomial in x like '3x^3 - 4x + 5' into {power: Fraction}."""
        cleaned = poly_str.replace("−", "-").replace(" ", "")
        if not cleaned or any(ch not in "0123456789.+-*^xX()" for ch in cleaned):
            return None
        cleaned = cleaned.replace("*", "")
        # Split into signed monomials
        raw_terms = re.findall(r"[+-]?[^+-]+", cleaned)
        if not raw_terms or "".join(raw_terms) != cleaned:
            return None
        coeffs: Dict[int, Fraction] = {}
        for term in raw_terms:
            m = re.fullmatch(r"([+-]?(?:\d+(?:\.\d+)?)?)x(?:\^(\d+))?", term, re.I)
            if m:
                c_str = m.group(1) or ""
                if c_str in ("", "+"):
                    c = Fraction(1)
                elif c_str == "-":
                    c = Fraction(-1)
                else:
                    c = Fraction(Decimal(c_str))
                deg = int(m.group(2)) if m.group(2) is not None else 1
                if deg > 20:
                    return None
                coeffs[deg] = coeffs.get(deg, Fraction(0)) + c
            else:
                m_const = re.fullmatch(r"[+-]?\d+(?:\.\d+)?", term)
                if not m_const:
                    return None
                c = Fraction(Decimal(term))
                coeffs[0] = coeffs.get(0, Fraction(0)) + c
        return {k: v for k, v in coeffs.items() if v != 0}

    def _format_poly(self, coeffs: Dict[int, Fraction]) -> str:
        if not coeffs:
            return "0"
        parts: List[str] = []
        for deg in sorted(coeffs.keys(), reverse=True):
            c = coeffs[deg]
            if c == 0:
                continue
            sign = "-" if c < 0 else "+"
            mag = abs(c)
            mag_s = self._format_fraction(mag)
            if deg == 0:
                body = mag_s
            elif deg == 1:
                body = "x" if mag == 1 else f"{mag_s}x"
            else:
                body = f"x^{deg}" if mag == 1 else f"{mag_s}x^{deg}"
            if not parts:
                parts.append(f"-{body}" if sign == "-" else body)
            else:
                parts.append(f" {sign} {body}")
        return "".join(parts)

    def _solve_calculus_stats_number_theory(self, q: str) -> Optional[SolverResult]:
        lower = q.lower()
        if any(neg in lower for neg in ("not", "never", "approx", "verify proof", "check proof")):
            return None

        # Polynomial Derivative: "derivative of 3x^3 - 4x + 5" (+ optional "at x = 2")
        deriv_m = re.search(
            r"(?:derivative\s+of|differentiate)\s+(?:f\(x\)\s*=\s*)?([0-9xX\^\+\-\.\*\s]+?)(?:\s+at\s+x\s*=\s*(-?\d+(?:\.\d+)?)|\s*[?!.]|$)",
            q,
            re.I,
        )
        if deriv_m:
            terms = self._parse_poly_terms(deriv_m.group(1))
            if terms:
                d_terms = {deg - 1: c * deg for deg, c in terms.items() if deg >= 1 and c * deg != 0}
                d_str = self._format_poly(d_terms)
                at_x = deriv_m.group(2)
                if at_x is not None:
                    xv = self._normalize_num(at_x)
                    val = sum((c * (xv ** deg) for deg, c in d_terms.items()), Fraction(0))
                    disp = self._format_fraction(val)
                    steps = [
                        DerivationStep(1, "Apply power rule to differentiate polynomial", "\\frac{d}{dx}(cx^n) = ncx^{n-1}", f"f'(x) = {d_str}", d_str),
                        DerivationStep(2, f"Evaluate derivative at x = {at_x}", "f'(x_0)", f"f'({at_x}) = {disp}", disp),
                    ]
                    receipt = self._build_receipt("calculus", "mathematics", "polynomial_derivative_eval", "calc.poly_derivative_at", {"poly": deriv_m.group(1).strip(), "x": at_x}, str(val), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "calculus", "polynomial_derivative_eval", "calc.poly_derivative_at", float(val), disp, "", steps, receipt, f"f'(x) = {d_str}; at x = {at_x}, f'({at_x}) = {disp}.")
                steps = [
                    DerivationStep(1, "Apply power rule term-by-term", "\\frac{d}{dx}\\sum c_n x^n = \\sum n c_n x^{n-1}", f"\\frac{{d}}{{dx}}({deriv_m.group(1).strip()}) = {d_str}", d_str)
                ]
                receipt = self._build_receipt("calculus", "mathematics", "polynomial_derivative", "calc.poly_derivative", {"poly": deriv_m.group(1).strip()}, d_str, d_str, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "calculus", "polynomial_derivative", "calc.poly_derivative", None, d_str, "", steps, receipt, f"The derivative is {d_str}.")

        # Definite Polynomial Integral: "integrate 6x^2 + 2x from x = 0 to x = 2"
        integ_m = re.search(
            r"(?:integral\s+of|integrate)\s+([0-9xX\^\+\-\.\*\s]+?)\s+from\s+(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)\s+to\s+(?:x\s*=\s*)?(-?\d+(?:\.\d+)?)",
            q,
            re.I,
        )
        if integ_m:
            terms = self._parse_poly_terms(integ_m.group(1))
            if terms is not None:
                a_lo = self._normalize_num(integ_m.group(2))
                b_hi = self._normalize_num(integ_m.group(3))
                anti = {deg + 1: c / Fraction(deg + 1) for deg, c in terms.items()}
                val_b = sum((c * (b_hi ** deg) for deg, c in anti.items()), Fraction(0))
                val_a = sum((c * (a_lo ** deg) for deg, c in anti.items()), Fraction(0))
                res = val_b - val_a
                disp = self._format_fraction(res)
                anti_s = self._format_poly(anti)
                steps = [
                    DerivationStep(1, "Find antiderivative F(x)", "\\int c x^n dx = \\frac{c}{n+1} x^{n+1}", f"F(x) = {anti_s}", anti_s),
                    DerivationStep(2, "Evaluate F(b) - F(a)", "F(b) - F(a)", f"({self._format_fraction(val_b)}) - ({self._format_fraction(val_a)}) = {disp}", disp),
                ]
                receipt = self._build_receipt("calculus", "mathematics", "definite_integral", "calc.poly_definite_integral", {"poly": integ_m.group(1).strip(), "a": str(a_lo), "b": str(b_hi)}, str(res), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "calculus", "definite_integral", "calc.poly_definite_integral", float(res), disp, "", steps, receipt, f"Definite integral evaluates to {disp}.")

        # GCD / LCM (defer extended gcd / Bezout requests to the extended Euclidean solver)
        gcd_m = re.search(r"\b(gcd|greatest common divisor|lcm|least common multiple)\b(?:\s+of)?\s*[:(]?\s*(-?\d+(?:\s*(?:,|and)\s*-?\d+)+)", q, re.I)
        if gcd_m and "extended" not in q.lower() and "bezout" not in q.lower() and "bézout" not in q.lower():
            kind = gcd_m.group(1).lower()
            nums = [abs(int(x)) for x in re.findall(r"-?\d+", gcd_m.group(2))]
            if len(nums) >= 2 and len(nums) <= 16:
                val = math.gcd(*nums) if ("gcd" in kind or "greatest" in kind) else math.lcm(*nums)
                fid = "math.gcd" if ("gcd" in kind or "greatest" in kind) else "math.lcm"
                disp = str(val)
                steps = [DerivationStep(1, f"Compute {fid.split('.')[-1].upper()}", f"\\text{{{fid.split('.')[-1]}}}({nums})", disp, disp)]
                receipt = self._build_receipt("number_theory", "mathematics", fid.split(".")[-1], fid, {"values": str(nums)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "number_theory", fid.split(".")[-1], fid, float(val), disp, "", steps, receipt, f"{fid.split('.')[-1].upper()} is {disp}.")

        # Descriptive statistics: median / variance / std dev
        stat_m = re.search(r"\b(median|population\s+variance|variance|standard\s+deviation|std\s*dev)\b\s+of(?:\s+these\s+numbers)?\s*[:\-]?\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)", q, re.I)
        if stat_m:
            kind = stat_m.group(1).lower()
            nums = [self._normalize_num(x.strip()) for x in stat_m.group(2).split(",")]
            if len(nums) >= 2:
                s_nums = sorted(nums)
                n = len(s_nums)
                if "median" in kind:
                    med = s_nums[n // 2] if n % 2 == 1 else (s_nums[n // 2 - 1] + s_nums[n // 2]) / 2
                    disp = self._format_fraction(med)
                    steps = [DerivationStep(1, "Sort values and select midpoint", "\\text{median}", f"sorted={[self._format_fraction(v) for v in s_nums]} -> {disp}", disp)]
                    receipt = self._build_receipt("statistics", "mathematics", "median", "stats.median", {"n": str(n)}, str(med), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "statistics", "median", "stats.median", float(med), disp, "", steps, receipt, f"Median is {disp}.")
                mean_val = sum(nums, Fraction(0)) / Fraction(n)
                var_val = sum(((v - mean_val) ** 2 for v in nums), Fraction(0)) / Fraction(n)
                if "variance" in kind:
                    disp = self._format_fraction(var_val)
                    steps = [DerivationStep(1, "Compute population variance", "\\sigma^2 = \\frac{1}{N}\\sum (x_i - \\mu)^2", f"\\mu={self._format_fraction(mean_val)}, \\sigma^2={disp}", disp)]
                    receipt = self._build_receipt("statistics", "mathematics", "variance", "stats.variance", {"n": str(n)}, str(var_val), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "statistics", "variance", "stats.variance", float(var_val), disp, "", steps, receipt, f"Population variance is {disp}.")
                std_float = math.sqrt(float(var_val))
                disp = f"{std_float:.4f}".rstrip("0").rstrip(".") or "0"
                steps = [DerivationStep(1, "Compute population standard deviation", "\\sigma = \\sqrt{\\sigma^2}", f"\\sigma = \\sqrt{{{float(var_val):.6g}}} = {disp}", disp)]
                receipt = self._build_receipt("statistics", "mathematics", "std_dev", "stats.std_dev", {"n": str(n)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "statistics", "std_dev", "stats.std_dev", std_float, disp, "", steps, receipt, f"Population standard deviation is {disp}.")

        return None

    # --------------------------------------------------------------------------
    # 19. Safe Zero-Eval AST Math Evaluator & E-Graph Algebraic Equivalence Check
    # --------------------------------------------------------------------------
    def _eval_safe_ast_expr(self, expr: str, env: Optional[Dict[str, Fraction]] = None) -> Tuple[Optional[Fraction], Optional[float], str]:
        """Evaluate arithmetic/scientific expression via ast.parse (zero eval/exec).

        Returns `(exact_fraction_or_none, float_value_or_none, mode)`.
        Inspired by kai9987kai/Calculators + Supermix-Poseidon/athenaeum.py.
        """
        text = str(expr or "").strip()
        if not text or len(text) > 256:
            return None, None, "invalid"
        cleaned = (
            text.replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("^", "**")
        )
        cleaned = re.sub(r"(?<=\d)\s*[xX]\s*(?=-?\d|\()", " * ", cleaned)
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError:
            return None, None, "syntax_error"

        rational_only = True
        scope = dict(env or {})
        scientific_consts = {"pi": math.pi, "e": math.e, "tau": math.tau}
        exact_funcs = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": lambda xs: sum(xs),
            "gcd": math.gcd,
            "lcm": math.lcm,
            "comb": math.comb,
            "perm": math.perm,
            "factorial": math.factorial,
        }
        float_funcs = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        bin_ops = {
            ast.Add: _op.add,
            ast.Sub: _op.sub,
            ast.Mult: _op.mul,
            ast.Div: _op.truediv,
            ast.FloorDiv: _op.floordiv,
            ast.Mod: _op.mod,
            ast.Pow: _op.pow,
        }

        def _walk_exact(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _walk_exact(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise ValueError("Unsupported constant")
                if len(str(node.value)) > 48:
                    raise ValueError("Constant too long")
                return Fraction(Decimal(str(node.value))) if isinstance(node.value, float) else Fraction(node.value)
            if isinstance(node, ast.Name):
                if node.id in scope:
                    return scope[node.id]
                raise ValueError(f"Non-rational symbol: {node.id}")
            if isinstance(node, ast.UnaryOp):
                val = _walk_exact(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return val
                if isinstance(node.op, ast.USub):
                    return -val
                raise ValueError("Unsupported unary op")
            if isinstance(node, ast.BinOp):
                left = _walk_exact(node.left)
                right = _walk_exact(node.right)
                fn = bin_ops.get(type(node.op))
                if fn is None:
                    raise ValueError("Unsupported binary op")
                if isinstance(node.op, ast.Pow):
                    if right.denominator != 1 or abs(right.numerator) > 64:
                        raise ValueError("Exponent out of exact rational bounds")
                    return left ** right.numerator
                if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                    raise ZeroDivisionError("division by zero")
                return fn(left, right)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fname = node.func.id
                if fname not in exact_funcs or node.keywords:
                    raise ValueError(f"Unsupported exact fn: {fname}")
                args = [_walk_exact(a) for a in node.args]
                if fname in ("gcd", "lcm", "comb", "perm", "factorial"):
                    int_args = [int(a) for a in args if a.denominator == 1]
                    if len(int_args) != len(args):
                        raise ValueError("Integer argument required")
                    if fname == "factorial" and (len(int_args) != 1 or not (0 <= int_args[0] <= 170)):
                        raise ValueError("Factorial out of bounds")
                    if fname in ("comb", "perm") and any(a < 0 or a > 1000 for a in int_args):
                        raise ValueError("Combinatoric bound exceeded")
                    return Fraction(exact_funcs[fname](*int_args))
                if fname == "sum" and len(args) == 1 and isinstance(args[0], (list, tuple)):
                    return Fraction(sum(args[0], Fraction(0)))
                return exact_funcs[fname](*args)
            if isinstance(node, (ast.List, ast.Tuple)):
                return [_walk_exact(e) for e in node.elts]
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

        def _walk_float(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _walk_float(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise ValueError("Unsupported constant")
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id in scope:
                    return float(scope[node.id])
                if node.id in scientific_consts:
                    return float(scientific_consts[node.id])
                raise ValueError(f"Unknown symbol: {node.id}")
            if isinstance(node, ast.UnaryOp):
                val = _walk_float(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return val
                if isinstance(node.op, ast.USub):
                    return -val
                raise ValueError("Unsupported unary op")
            if isinstance(node, ast.BinOp):
                left = _walk_float(node.left)
                right = _walk_float(node.right)
                fn = bin_ops.get(type(node.op))
                if fn is None:
                    raise ValueError("Unsupported binary op")
                if isinstance(node.op, ast.Pow) and abs(right) > 256:
                    raise ValueError("Exponent too large")
                if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                    raise ZeroDivisionError("division by zero")
                return float(fn(left, right))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                fname = node.func.id
                if node.keywords:
                    raise ValueError("Keywords unsupported")
                args = [_walk_float(a) for a in node.args]
                if fname in float_funcs:
                    return float(float_funcs[fname](*args))
                if fname in exact_funcs:
                    if fname in ("gcd", "lcm", "comb", "perm", "factorial"):
                        int_args = [int(a) for a in args]
                        if fname == "factorial" and (len(int_args) != 1 or not (0 <= int_args[0] <= 170)):
                            raise ValueError("Factorial out of bounds")
                        return float(exact_funcs[fname](*int_args))
                    return float(exact_funcs[fname](*args))
                raise ValueError(f"Unsupported fn: {fname}")
            if isinstance(node, (ast.List, ast.Tuple)):
                return [_walk_float(e) for e in node.elts]
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")

        try:
            frac = _walk_exact(tree)
            if isinstance(frac, Fraction):
                val_f = float(frac)
                if math.isfinite(val_f):
                    return frac, val_f, "exact_fraction"
        except ZeroDivisionError:
            return None, None, "zero_division"
        except Exception:
            rational_only = False

        if not rational_only:
            try:
                val_f = _walk_float(tree)
                if isinstance(val_f, float) and math.isfinite(val_f):
                    return None, val_f, "scientific_float"
            except Exception:
                return None, None, "eval_failed"
        return None, None, "unsupported"

    def _solve_safe_ast_math(self, q: str) -> Optional[SolverResult]:
        lower = q.lower()
        if any(
            bad in lower
            for bad in (
                "not", "never", "without", "approx", "verify proof", "check proof",
                "denoise thought", "import ", "__", "eval", "exec", "lambda",
                "sequence", "average", "mean", "cns", "male cns", "solve for ",
                "nums =", "range(", ".count(", "for ", "while ",
            )
        ):
            return None

        candidate = q.strip()
        prefix_m = re.match(
            r"^\s*(?:please\s+)?(?:what\s+is|compute|calculate|evaluate|solve(?:\s+this\s+(?:basic\s+)?math\s+problem)?)\s*[:\-]?\s*(.+?)\s*[?!.]?\s*$",
            candidate,
            re.I,
        )
        expr = prefix_m.group(1) if prefix_m else candidate.rstrip("?!. ")
        # Require either mathematical operators or whitelisted math functions, no prose sentences
        if not any(op in expr for op in ("+", "-", "*", "/", "^", "×", "÷", "%", "(")):
            return None
        stripped_fn = re.sub(r"\b(?:sqrt|sin|cos|tan|log10|log2|log|exp|abs|min|max|gcd|lcm|comb|perm|factorial|round|floor|ceil|pi|e|tau)\b", "", expr)
        if re.search(r"[a-zA-Z_]", stripped_fn):
            return None

        frac, val_f, mode = self._eval_safe_ast_expr(expr)
        if val_f is None:
            return None
        if frac is not None:
            disp = self._format_fraction(frac)
            raw_str = str(frac)
        else:
            disp = f"{val_f:.6f}".rstrip("0").rstrip(".") or "0"
            raw_str = disp
        steps = [
            DerivationStep(
                1,
                f"Evaluate mathematical expression via zero-eval AST ({mode})",
                expr,
                f"{expr} = {disp}",
                disp,
            )
        ]
        receipt = self._build_receipt(
            "ast_math",
            "mathematics",
            "expression_eval",
            f"math.safe_ast.{mode}",
            {"expression": expr},
            raw_str,
            disp,
            "dimensionless",
            steps,
        )
        return SolverResult(
            True,
            q,
            "mathematics",
            "ast_math",
            "expression_eval",
            f"math.safe_ast.{mode}",
            val_f,
            disp,
            "",
            steps,
            receipt,
            f"Exact AST evaluation: {expr} = {disp}.",
        )

    def verify_algebraic_equivalence(self, lhs_expr: str, rhs_expr: str) -> Dict[str, Any]:
        """Deterministically verify whether two algebraic expressions agree across rational probe points."""
        lhs = str(lhs_expr or "").strip().replace("^", "**")
        rhs = str(rhs_expr or "").strip().replace("^", "**")
        probes = [
            {"x": Fraction(2), "y": Fraction(3), "a": Fraction(5), "b": Fraction(7)},
            {"x": Fraction(-3), "y": Fraction(5), "a": Fraction(2), "b": Fraction(-4)},
            {"x": Fraction(7, 2), "y": Fraction(-1, 3), "a": Fraction(3, 5), "b": Fraction(4, 7)},
        ]
        diffs: List[float] = []
        for env in probes:
            _, v_l, m_l = self._eval_safe_ast_expr(lhs, env=env)
            _, v_r, m_r = self._eval_safe_ast_expr(rhs, env=env)
            if v_l is None or v_r is None:
                return {"equivalent": False, "reason": f"eval_failed:{m_l}:{m_r}", "answer_authority": False}
            diffs.append(abs(v_l - v_r))
        max_diff = max(diffs) if diffs else 1.0
        equivalent = max_diff <= 1e-9
        return {
            "equivalent": equivalent,
            "max_probe_diff": max_diff,
            "probes_checked": len(probes),
            "answer_authority": bool(equivalent),
        }

    def _solve_linear_system_3x3(self, q: str) -> Optional[SolverResult]:
        """Exact rational Gaussian elimination for 3x3 linear systems in x, y, z."""
        lower = q.lower()
        if any(neg in lower for neg in ("not", "never", "approx", "verify proof", "check proof")):
            return None
        if not all(v in lower for v in ("x", "y", "z")) or q.count("=") < 3:
            return None

        eq_pattern = re.compile(
            r"([+-]?\s*(?:\d+(?:\.\d+)?)?\s*\*?\s*x)\s*"
            r"([+-]\s*(?:\d+(?:\.\d+)?)?\s*\*?\s*y)\s*"
            r"([+-]\s*(?:\d+(?:\.\d+)?)?\s*\*?\s*z)\s*=\s*(-?\d+(?:\.\d+)?)",
            re.I,
        )
        matches = eq_pattern.findall(q)
        if len(matches) != 3:
            return None

        def _parse_coeff(term: str, var: str) -> Fraction:
            c = term.replace(" ", "").replace("*", "").lower().replace(var, "")
            if c in ("", "+"):
                return Fraction(1)
            if c == "-":
                return Fraction(-1)
            return self._normalize_num(c)

        aug: List[List[Fraction]] = []
        for tx, ty, tz, rhs_s in matches:
            aug.append([
                _parse_coeff(tx, "x"),
                _parse_coeff(ty, "y"),
                _parse_coeff(tz, "z"),
                self._normalize_num(rhs_s),
            ])
        orig = [list(row) for row in aug]

        # Exact rational Gaussian elimination with partial pivoting
        for col in range(3):
            pivot_row = max(range(col, 3), key=lambda r: abs(aug[r][col]))
            if aug[pivot_row][col] == 0:
                return None
            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            piv = aug[col][col]
            aug[col] = [val / piv for val in aug[col]]
            for r in range(3):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor != 0:
                    aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(4)]

        sol_x, sol_y, sol_z = aug[0][3], aug[1][3], aug[2][3]
        # Independent substitution check across all 3 equations
        for a, b, c, rhs in orig:
            if a * sol_x + b * sol_y + c * sol_z != rhs:
                return None

        disp = (
            f"x = {self._format_fraction(sol_x)}, "
            f"y = {self._format_fraction(sol_y)}, "
            f"z = {self._format_fraction(sol_z)}"
        )
        steps = [
            DerivationStep(1, "Form augmented 3x3 rational system [A|b]", "A\\mathbf{x} = \\mathbf{b}", "3 equations in (x, y, z)", disp),
            DerivationStep(2, "Perform exact Fraction Gauss-Jordan elimination and verify residuals = 0", "\\mathbf{x} = A^{-1}\\mathbf{b}", disp, disp),
        ]
        receipt = self._build_receipt(
            "linear_system_3x3",
            "mathematics",
            "solution_xyz",
            "alg.linear_system_3x3",
            {"equations": "3"},
            f"({sol_x},{sol_y},{sol_z})",
            disp,
            "dimensionless",
            steps,
        )
        return SolverResult(
            True,
            q,
            "mathematics",
            "linear_system_3x3",
            "solution_xyz",
            "alg.linear_system_3x3",
            float(sol_x),
            disp,
            "",
            steps,
            receipt,
            f"Exact 3x3 rational solution: {disp}.",
        )

    def _solve_matrix_linear_algebra(self, q: str) -> Optional[SolverResult]:
        """Exact rational matrix determinant, trace, 2x2 inverse, 2x2 eigenvalues & vector ops."""
        lower = q.lower()
        if any(neg in lower for neg in ("not", "never", "approx", "verify proof", "check proof")):
            return None

        # Vector dot product & 3D cross product (ast.literal_eval, zero eval)
        if "dot product" in lower or "cross product" in lower:
            vecs = re.findall(r"(\[\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*\])", q)
            if len(vecs) == 2:
                try:
                    raw_u = ast.literal_eval(vecs[0])
                    raw_v = ast.literal_eval(vecs[1])
                except Exception:
                    return None
                if (
                    isinstance(raw_u, list)
                    and isinstance(raw_v, list)
                    and len(raw_u) == len(raw_v)
                    and 1 <= len(raw_u) <= 16
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw_u + raw_v)
                ):
                    u = [Fraction(Decimal(str(x))) if isinstance(x, float) else Fraction(x) for x in raw_u]
                    v = [Fraction(Decimal(str(x))) if isinstance(x, float) else Fraction(x) for x in raw_v]
                    if "dot product" in lower:
                        val = sum((a * b for a, b in zip(u, v)), Fraction(0))
                        disp = self._format_fraction(val)
                        steps = [DerivationStep(1, "Compute exact Euclidean dot product", "\\mathbf{u}\\cdot\\mathbf{v} = \\sum_i u_i v_i", f"{vecs[0]} \\cdot {vecs[1]} = {disp}", disp)]
                        receipt = self._build_receipt("linear_algebra", "mathematics", "dot_product", "linalg.dot_product", {"dim": str(len(u))}, str(val), disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "linear_algebra", "dot_product", "linalg.dot_product", float(val), disp, "", steps, receipt, f"Exact dot product is {disp}.")
                    if "cross product" in lower and len(u) == 3:
                        cx = u[1] * v[2] - u[2] * v[1]
                        cy = u[2] * v[0] - u[0] * v[2]
                        cz = u[0] * v[1] - u[1] * v[0]
                        disp = f"[{self._format_fraction(cx)}, {self._format_fraction(cy)}, {self._format_fraction(cz)}]"
                        steps = [DerivationStep(1, "Compute exact 3D cross product", "\\mathbf{u}\\times\\mathbf{v}", f"{vecs[0]} \\times {vecs[1]} = {disp}", disp)]
                        receipt = self._build_receipt("linear_algebra", "mathematics", "cross_product", "linalg.cross_product", {"dim": "3"}, disp, disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "linear_algebra", "cross_product", "linalg.cross_product", None, disp, "", steps, receipt, f"Exact 3D cross product is {disp}.")

        if not any(k in lower for k in ("determinant", "det(", "det ", "trace", "inverse", "eigenvalue")):
            return None
        mat_m = re.search(r"(\[\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\]){1,2}\s*\])", q)
        if not mat_m:
            return None
        try:
            raw_mat = ast.literal_eval(mat_m.group(1))
        except Exception:
            return None
        if not isinstance(raw_mat, list) or len(raw_mat) not in (2, 3):
            return None
        n = len(raw_mat)
        mat: List[List[Fraction]] = []
        for row in raw_mat:
            if not isinstance(row, list) or len(row) != n:
                return None
            f_row: List[Fraction] = []
            for v in row:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return None
                f_row.append(Fraction(Decimal(str(v))) if isinstance(v, float) else Fraction(v))
            mat.append(f_row)

        if "inverse" in lower and n == 2:
            det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
            if det == 0:
                return None
            inv = [
                [mat[1][1] / det, -mat[0][1] / det],
                [-mat[1][0] / det, mat[0][0] / det],
            ]
            # Independent exact check: A * A^-1 == I
            if (
                mat[0][0] * inv[0][0] + mat[0][1] * inv[1][0] != 1
                or mat[0][0] * inv[0][1] + mat[0][1] * inv[1][1] != 0
                or mat[1][0] * inv[0][0] + mat[1][1] * inv[1][0] != 0
                or mat[1][0] * inv[0][1] + mat[1][1] * inv[1][1] != 1
            ):
                return None
            def _fstr(f: Fraction) -> str:
                return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"
            disp = f"[[{_fstr(inv[0][0])}, {_fstr(inv[0][1])}], [{_fstr(inv[1][0])}, {_fstr(inv[1][1])}]]"
            steps = [
                DerivationStep(1, "Compute 2x2 determinant", "\\det(A) = ad - bc", f"\\det(A) = {self._format_fraction(det)}", self._format_fraction(det)),
                DerivationStep(2, "Compute exact adjugate / det and verify A A^-1 = I", "A^{-1} = \\frac{1}{\\det(A)}\\begin{pmatrix}d & -b\\\\-c & a\\end{pmatrix}", disp, disp),
            ]
            receipt = self._build_receipt("linear_algebra", "mathematics", "matrix_inverse", "linalg.inv_2x2", {"dim": "2x2"}, disp, disp, "dimensionless", steps)
            return SolverResult(True, q, "mathematics", "linear_algebra", "matrix_inverse", "linalg.inv_2x2", None, disp, "", steps, receipt, f"Exact 2x2 matrix inverse is {disp}.")

        if "eigenvalue" in lower and n == 2:
            tr = mat[0][0] + mat[1][1]
            det = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
            disc = tr * tr - Fraction(4) * det
            if disc < 0:
                return None
            disc_f = float(disc)
            sqrt_d = math.sqrt(disc_f)
            l1 = (float(tr) + sqrt_d) / 2.0
            l2 = (float(tr) - sqrt_d) / 2.0
            def _fmt_ev(val: float) -> str:
                if abs(val - round(val)) < 1e-9:
                    return str(int(round(val)))
                return f"{val:.4f}".rstrip("0").rstrip(".") or "0"
            disp = f"lambda1 = {_fmt_ev(l1)}, lambda2 = {_fmt_ev(l2)}"
            steps = [
                DerivationStep(1, "Form characteristic polynomial", "\\lambda^2 - \\operatorname{tr}(A)\\lambda + \\det(A) = 0", f"\\operatorname{{tr}}={self._format_fraction(tr)}, \\det={self._format_fraction(det)}", disp),
            ]
            receipt = self._build_receipt("linear_algebra", "mathematics", "matrix_eigenvalues", "linalg.eig_2x2", {"dim": "2x2"}, disp, disp, "dimensionless", steps)
            return SolverResult(True, q, "mathematics", "linear_algebra", "matrix_eigenvalues", "linalg.eig_2x2", l1, disp, "", steps, receipt, f"Real eigenvalues are {disp}.")

        if "trace" in lower:
            val = sum((mat[i][i] for i in range(n)), Fraction(0))
            disp = self._format_fraction(val)
            steps = [DerivationStep(1, f"Sum diagonal entries of {n}x{n} matrix", "\\operatorname{tr}(A) = \\sum_i A_{ii}", f"\\operatorname{{tr}}(A) = {disp}", disp)]
            receipt = self._build_receipt("linear_algebra", "mathematics", "matrix_trace", f"linalg.trace_{n}x{n}", {"dim": f"{n}x{n}"}, str(val), disp, "dimensionless", steps)
            return SolverResult(True, q, "mathematics", "linear_algebra", "matrix_trace", f"linalg.trace_{n}x{n}", float(val), disp, "", steps, receipt, f"Matrix trace is {disp}.")

        if n == 2:
            val = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
            formula_tex = "\\det(A) = a d - b c"
        else:
            val = (
                mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
                + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0])
            )
            formula_tex = "\\det(A) = \\sum_{j} (-1)^j A_{0j} M_{0j}"
        disp = self._format_fraction(val)
        steps = [DerivationStep(1, f"Compute exact {n}x{n} determinant via cofactor expansion", formula_tex, f"\\det(A) = {disp}", disp)]
        receipt = self._build_receipt("linear_algebra", "mathematics", "matrix_determinant", f"linalg.det_{n}x{n}", {"dim": f"{n}x{n}"}, str(val), disp, "dimensionless", steps)
        return SolverResult(True, q, "mathematics", "linear_algebra", "matrix_determinant", f"linalg.det_{n}x{n}", float(val), disp, "", steps, receipt, f"Exact {n}x{n} determinant is {disp}.")

    def _solve_number_theory_and_probability(self, q: str) -> Optional[SolverResult]:
        """Exact modular arithmetic, prime factorization, Euler totient, binomial, Bayes, resistors & thin lens."""
        lower = q.lower()
        if any(neg in lower for neg in ("not", "never", "approx", "verify proof", "check proof")):
            return None

        # 1. Modular inverse: "modular inverse of 3 mod 11"
        inv_m = re.search(r"modular\s+inverse\s+of\s+(-?\d+)\s+(?:mod|modulo)\s+(\d+)", q, re.I)
        if inv_m:
            a, m = int(inv_m.group(1)), int(inv_m.group(2))
            if 2 <= m <= 10**12 and math.gcd(a, m) == 1:
                inv = pow(a, -1, m)
                disp = str(inv)
                steps = [DerivationStep(1, "Compute modular inverse via extended Euclidean algorithm", "a \\cdot a^{-1} \\equiv 1 \\pmod m", f"{a} \\cdot {inv} \\equiv 1 \\pmod{{{m}}}", disp)]
                receipt = self._build_receipt("number_theory", "mathematics", "mod_inverse", "nt.mod_inverse", {"a": str(a), "m": str(m)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "number_theory", "mod_inverse", "nt.mod_inverse", float(inv), disp, "", steps, receipt, f"Modular inverse of {a} mod {m} is {disp}.")

        # 2. Modular exponentiation: "7^13 mod 11" or "7**13 mod 11"
        modpow_m = re.search(r"(-?\d+)\s*(?:\^|\*\*)\s*(\d+)\s*(?:mod|modulo)\s*(\d+)", q, re.I)
        if modpow_m:
            base, exp, mod = int(modpow_m.group(1)), int(modpow_m.group(2)), int(modpow_m.group(3))
            if 1 <= mod <= 10**12 and 0 <= exp <= 10**9:
                val = pow(base, exp, mod)
                disp = str(val)
                steps = [DerivationStep(1, "Fast binary modular exponentiation", "a^b \\bmod m", f"{base}^{{{exp}}} \\bmod {mod} = {disp}", disp)]
                receipt = self._build_receipt("number_theory", "mathematics", "mod_pow", "nt.mod_pow", {"base": str(base), "exp": str(exp), "mod": str(mod)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "number_theory", "mod_pow", "nt.mod_pow", float(val), disp, "", steps, receipt, f"{base}^{exp} mod {mod} = {disp}.")

        # 3. Prime factorization: "prime factorization of 360"
        pf_m = re.search(r"prime\s+factori[sz]ation\s+of\s+(\d+)", q, re.I)
        if pf_m:
            n = int(pf_m.group(1))
            if 2 <= n <= 10**10:
                rem = n
                factors: List[Tuple[int, int]] = []
                d = 2
                while d * d <= rem:
                    if rem % d == 0:
                        cnt = 0
                        while rem % d == 0:
                            rem //= d
                            cnt += 1
                        factors.append((d, cnt))
                    d = 3 if d == 2 else d + 2
                if rem > 1:
                    factors.append((rem, 1))
                disp = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors)
                steps = [DerivationStep(1, "Trial-division exact prime factorization", "n = \\prod p_i^{e_i}", f"{n} = {disp}", disp)]
                receipt = self._build_receipt("number_theory", "mathematics", "prime_factorization", "nt.prime_factorization", {"n": str(n)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "number_theory", "prime_factorization", "nt.prime_factorization", float(factors[0][0]), disp, "", steps, receipt, f"Prime factorization of {n} is {disp}.")

        # 4. Euler's totient: "euler totient of 36" or "phi(36)"
        phi_m = re.search(r"(?:euler(?:'s)?\s+totient(?:\s+of)?\s+|phi\s*\(\s*)(\d+)\s*\)?", q, re.I)
        if phi_m:
            n = int(phi_m.group(1))
            if 1 <= n <= 10**9:
                result = n
                rem = n
                d = 2
                while d * d <= rem:
                    if rem % d == 0:
                        while rem % d == 0:
                            rem //= d
                        result -= result // d
                    d = 3 if d == 2 else d + 2
                if rem > 1:
                    result -= result // rem
                disp = str(result)
                steps = [DerivationStep(1, "Apply Euler product formula", "\\varphi(n) = n \\prod_{p|n} (1 - 1/p)", f"\\varphi({n}) = {disp}", disp)]
                receipt = self._build_receipt("number_theory", "mathematics", "euler_totient", "nt.euler_totient", {"n": str(n)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "number_theory", "euler_totient", "nt.euler_totient", float(result), disp, "", steps, receipt, f"Euler's totient phi({n}) = {disp}.")

        # 5. Exact Binomial Probability: "binomial probability of k=2 successes in n=5 trials with p=0.4"
        binom_m = re.search(
            r"binomial\s+probability.*?\bk\s*=\s*(\d+).*?\bn\s*=\s*(\d+).*?\bp\s*=\s*(\d+(?:\.\d+)?(?:/\d+)?)",
            q,
            re.I,
        ) or re.search(
            r"binomial\s+probability.*?\bn\s*=\s*(\d+).*?\bk\s*=\s*(\d+).*?\bp\s*=\s*(\d+(?:\.\d+)?(?:/\d+)?)",
            q,
            re.I,
        )
        if binom_m:
            first_is_k = bool(re.search(r"\bk\s*=\s*\d+.*?\bn\s*=\s*\d+", q, re.I))
            k_val = int(binom_m.group(1) if first_is_k else binom_m.group(2))
            n_val = int(binom_m.group(2) if first_is_k else binom_m.group(1))
            p_val = self._normalize_num(binom_m.group(3))
            if 0 <= k_val <= n_val <= 60 and Fraction(0) <= p_val <= Fraction(1):
                prob = Fraction(math.comb(n_val, k_val)) * (p_val ** k_val) * ((Fraction(1) - p_val) ** (n_val - k_val))
                disp = self._format_fraction(prob, max_decimals=6)
                steps = [
                    DerivationStep(1, "Evaluate exact binomial PMF", "P(X=k) = \\binom{n}{k} p^k (1-p)^{n-k}", f"\\binom{{{n_val}}}{{{k_val}}} ({p_val})^{{{k_val}}} ({Fraction(1)-p_val})^{{{n_val-k_val}}} = {disp}", disp)
                ]
                receipt = self._build_receipt("probability", "mathematics", "binomial_pmf", "prob.binomial_pmf", {"n": str(n_val), "k": str(k_val), "p": str(p_val)}, str(prob), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "probability", "binomial_pmf", "prob.binomial_pmf", float(prob), disp, "", steps, receipt, f"Exact binomial probability P(X={k_val}) = {disp}.")

        # 6. Exact Bayes' Theorem: "Bayes: prior 0.01, sensitivity 0.95, false positive rate 0.05"
        if "bayes" in lower and "prior" in lower:
            pr_m = re.search(r"prior\s*(?:=|:|of|is)?\s*(\d+(?:\.\d+)?(?:/\d+)?)", q, re.I)
            sn_m = re.search(r"(?:sensitivity|true\s+positive(?:\s+rate)?|likelihood)\s*(?:=|:|of|is)?\s*(\d+(?:\.\d+)?(?:/\d+)?)", q, re.I)
            fp_m = re.search(r"(?:false\s+positive(?:\s+rate)?|fpr)\s*(?:=|:|of|is)?\s*(\d+(?:\.\d+)?(?:/\d+)?)", q, re.I)
            if pr_m and sn_m and fp_m:
                prior = self._normalize_num(pr_m.group(1))
                sens = self._normalize_num(sn_m.group(1))
                fpr = self._normalize_num(fp_m.group(1))
                if Fraction(0) <= prior <= Fraction(1) and Fraction(0) <= sens <= Fraction(1) and Fraction(0) <= fpr <= Fraction(1):
                    num = sens * prior
                    den = num + fpr * (Fraction(1) - prior)
                    if den > 0:
                        post = num / den
                        disp = self._format_fraction(post, max_decimals=6)
                        steps = [
                            DerivationStep(1, "Apply exact Bayes' theorem", "P(H|E) = \\frac{P(E|H)P(H)}{P(E|H)P(H) + P(E|\\neg H)P(\\neg H)}", f"P(H|E) = \\frac{{{num}}}{{{den}}} = {disp}", disp)
                        ]
                        receipt = self._build_receipt("probability", "mathematics", "bayes_posterior", "prob.bayes_theorem", {"prior": str(prior), "sensitivity": str(sens), "fpr": str(fpr)}, str(post), disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "probability", "bayes_posterior", "prob.bayes_theorem", float(post), disp, "", steps, receipt, f"Exact Bayes posterior probability is {disp} ({post.numerator}/{post.denominator}).")

        # 7. Parallel resistors: "parallel resistors 6 ohm and 3 ohm"
        par_m = re.search(
            r"parallel\s+resistors?\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:ohm|ohms|Ω)?\s*(?:and|,)\s*(\d+(?:\.\d+)?)\s*(?:ohm|ohms|Ω)?",
            q,
            re.I,
        )
        if par_m:
            r1, r2 = self._normalize_num(par_m.group(1)), self._normalize_num(par_m.group(2))
            if r1 > 0 and r2 > 0:
                req = (r1 * r2) / (r1 + r2)
                disp = self._format_fraction(req)
                steps = [DerivationStep(1, "Compute parallel equivalent resistance", "R_{\\text{eq}} = \\frac{R_1 R_2}{R_1 + R_2}", f"\\frac{{{r1}\\cdot{r2}}}{{{r1}+{r2}}} = {disp}\\,\\Omega", disp, "ohm")]
                receipt = self._build_receipt("circuits", "physics", "parallel_resistance", "circuits.parallel_resistors", {"R1": str(r1), "R2": str(r2)}, str(req), disp, "ohm", steps)
                return SolverResult(True, q, "physics", "circuits", "parallel_resistance", "circuits.parallel_resistors", float(req), disp, "ohm", steps, receipt, f"Equivalent parallel resistance is {disp} ohm.")

        # 8. Thin-lens equation: "thin lens with f = 10 cm and u = 30 cm"
        lens_m = re.search(
            r"thin\s+lens.*?\bf\s*=\s*(-?\d+(?:\.\d+)?)\s*(cm|m|mm)?.*?\bu\s*=\s*(-?\d+(?:\.\d+)?)\s*(cm|m|mm)?",
            q,
            re.I,
        )
        if lens_m:
            f_val = self._normalize_num(lens_m.group(1))
            u_val = self._normalize_num(lens_m.group(3))
            unit = (lens_m.group(2) or lens_m.group(4) or "cm").lower()
            if f_val != 0 and u_val != 0 and (Fraction(1) / f_val - Fraction(1) / u_val) != 0:
                inv_v = (Fraction(1) / f_val) - (Fraction(1) / u_val)
                v_val = Fraction(1) / inv_v
                disp = self._format_fraction(v_val)
                steps = [DerivationStep(1, "Apply Gaussian thin-lens equation", "\\frac{1}{v} = \\frac{1}{f} - \\frac{1}{u}", f"v = {disp}\\,{unit}", disp, unit)]
                receipt = self._build_receipt("optics", "physics", "image_distance", "optics.thin_lens", {"f": str(f_val), "u": str(u_val)}, str(v_val), disp, unit, steps)
                return SolverResult(True, q, "physics", "optics", "image_distance", "optics.thin_lens", float(v_val), disp, unit, steps, receipt, f"Image distance v = {disp} {unit}.")

        # 9. Geometric series (finite and convergent infinite)
        if "geometric series" in lower:
            a_m = re.search(r"\ba\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", q, re.I)
            r_m = re.search(r"\br\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", q, re.I)
            n_m = re.search(r"\bn\s*=\s*(\d+)", q, re.I)
            if a_m and r_m:
                a_v = self._normalize_num(a_m.group(1))
                r_v = self._normalize_num(r_m.group(1))
                if "infinite" in lower and abs(r_v) < Fraction(1):
                    s_inf = a_v / (Fraction(1) - r_v)
                    disp = self._format_fraction(s_inf, max_decimals=6)
                    steps = [DerivationStep(1, "Evaluate convergent infinite geometric series", "S_\\infty = \\frac{a}{1 - r}", f"\\frac{{{a_v}}}{{1 - ({r_v})}} = {disp}", disp)]
                    receipt = self._build_receipt("series", "mathematics", "infinite_geometric_sum", "math.geom_series_inf", {"a": str(a_v), "r": str(r_v)}, str(s_inf), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "series", "infinite_geometric_sum", "math.geom_series_inf", float(s_inf), disp, "", steps, receipt, f"Convergent infinite geometric series sum is {disp}.")
                if n_m:
                    n_v = int(n_m.group(1))
                    if 1 <= n_v <= 64:
                        s_fin = a_v * n_v if r_v == 1 else a_v * (Fraction(1) - (r_v ** n_v)) / (Fraction(1) - r_v)
                        disp = self._format_fraction(s_fin, max_decimals=6)
                        steps = [DerivationStep(1, "Evaluate exact finite geometric series", "S_n = a\\frac{1 - r^n}{1 - r}", f"S_{{{n_v}}} = {disp}", disp)]
                        receipt = self._build_receipt("series", "mathematics", "finite_geometric_sum", "math.geom_series_fin", {"a": str(a_v), "r": str(r_v), "n": str(n_v)}, str(s_fin), disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "series", "finite_geometric_sum", "math.geom_series_fin", float(s_fin), disp, "", steps, receipt, f"Finite geometric series sum is {disp}.")

        # 10. Exact polynomial evaluation: "evaluate 2x^3 - 3x^2 + 5 at x = 4"
        poly_ev = re.search(r"evaluate\s+(?:f\(x\)\s*=\s*)?([0-9xX\^\+\-\.\*\s]+?)\s+at\s+x\s*=\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", q, re.I)
        if poly_ev and "x" in poly_ev.group(1).lower():
            terms = self._parse_poly_terms(poly_ev.group(1))
            if terms is not None:
                xv = self._normalize_num(poly_ev.group(2))
                val = sum((c * (xv ** deg) for deg, c in terms.items()), Fraction(0))
                disp = self._format_fraction(val, max_decimals=6)
                steps = [DerivationStep(1, f"Evaluate polynomial at x = {xv}", "f(x) = \\sum c_k x^k", f"f({xv}) = {disp}", disp)]
                receipt = self._build_receipt("algebra", "mathematics", "polynomial_eval", "alg.poly_eval", {"poly": poly_ev.group(1).strip(), "x": str(xv)}, str(val), disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "algebra", "polynomial_eval", "alg.poly_eval", float(val), disp, "", steps, receipt, f"Polynomial evaluates to {disp}.")

        # --- v104 Extended Zero-Eval Exact Solvers ---

        # 11. Exact Catalan number C_n = comb(2n, n) // (n + 1)
        cat_m = re.search(
            r"(?:(\d+)(?:st|nd|rd|th)?\s+catalan(?:\s+number)?|catalan(?:\s+number)?\s*(?:C_?|\()?\s*(\d+)\)?)",
            q,
            re.I,
        )
        if cat_m:
            n_val = int(cat_m.group(1) or cat_m.group(2))
            if 0 <= n_val <= 30:
                c_val = math.comb(2 * n_val, n_val) // (n_val + 1)
                disp = str(c_val)
                steps = [
                    DerivationStep(
                        1,
                        f"Compute exact Catalan number C_{n_val}",
                        "C_n = \\frac{1}{n+1}\\binom{2n}{n}",
                        f"C_{{{n_val}}} = \\frac{{1}}{{{n_val + 1}}}\\binom{{{2 * n_val}}}{{{n_val}}} = {disp}",
                        disp,
                    )
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "catalan_number", "comb.catalan", {"n": str(n_val)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "catalan_number", "comb.catalan", float(c_val), disp, "", steps, receipt, f"The Catalan number C_{n_val} is {disp}.")

        # 12. Exact Fibonacci number F_n (F_0 = 0, F_1 = 1)
        fib_m = re.search(
            r"(?:(\d+)(?:st|nd|rd|th)\s+fibonacci(?:\s+number)?|fibonacci(?:\s+number)?\s*(?:F_?|\()?\s*(\d+)\)?)",
            q,
            re.I,
        )
        if fib_m:
            n_val = int(fib_m.group(1) or fib_m.group(2))
            if 0 <= n_val <= 200:
                a_f, b_f = 0, 1
                for _ in range(n_val):
                    a_f, b_f = b_f, a_f + b_f
                disp = str(a_f)
                steps = [
                    DerivationStep(
                        1,
                        f"Compute exact Fibonacci number F_{n_val} (F_0=0, F_1=1)",
                        "F_n = F_{n-1} + F_{n-2}",
                        f"F_{{{n_val}}} = {disp}",
                        disp,
                    )
                ]
                receipt = self._build_receipt("combinatorics", "mathematics", "fibonacci_number", "comb.fibonacci", {"n": str(n_val)}, disp, disp, "dimensionless", steps)
                return SolverResult(True, q, "mathematics", "combinatorics", "fibonacci_number", "comb.fibonacci", float(a_f), disp, "", steps, receipt, f"The Fibonacci number F_{n_val} is {disp}.")

        # 13. Exact Chinese Remainder Theorem: "x = 2 (mod 3) and x = 3 (mod 5)"
        crt_pairs = re.findall(
            r"x\s*(?:=|≡|==|congruent\s+to)\s*(-?\d+)\s*\(?\s*mod(?:ulo)?\s*(\d+)\s*\)?",
            q,
            re.I,
        )
        if 2 <= len(crt_pairs) <= 4:
            rems = [int(a) for a, _ in crt_pairs]
            mods = [int(m) for _, m in crt_pairs]
            if all(m >= 2 for m in mods):
                pairwise_coprime = all(
                    math.gcd(mods[i], mods[j]) == 1
                    for i in range(len(mods))
                    for j in range(i + 1, len(mods))
                )
                if pairwise_coprime:
                    prod_m = 1
                    for m in mods:
                        prod_m *= m
                    total = 0
                    for a_i, m_i in zip(rems, mods):
                        big_m = prod_m // m_i
                        inv_m = pow(big_m, -1, m_i)
                        total += (a_i % m_i) * big_m * inv_m
                    x_sol = total % prod_m
                    if all(x_sol % m_i == (a_i % m_i) for a_i, m_i in zip(rems, mods)):
                        disp = f"{x_sol} (mod {prod_m})"
                        steps = [
                            DerivationStep(
                                1,
                                "Verify pairwise coprime moduli and apply CRT construction",
                                "x \\equiv \\sum a_i M_i (M_i^{-1} \\bmod m_i) \\pmod{M}",
                                f"M = {prod_m},\\; x = {x_sol}",
                                str(x_sol),
                            ),
                            DerivationStep(
                                2,
                                "Verify solution against every input congruence",
                                "\\forall i:\\; x \\bmod m_i = a_i \\bmod m_i",
                                "verified",
                                disp,
                            ),
                        ]
                        receipt = self._build_receipt("number_theory", "mathematics", "crt_solution", "numtheory.crt", {"congruences": str(crt_pairs)}, str(x_sol), disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "number_theory", "crt_solution", "numtheory.crt", float(x_sol), disp, "", steps, receipt, f"By the Chinese Remainder Theorem, x ≡ {disp}.")

        # 14. Shannon Entropy in bits: "shannon entropy of [0.5, 0.25, 0.25]"
        if "shannon entropy" in lower or ("entropy" in lower and "[" in q and "kl" not in lower):
            list_m = re.search(r"\[([^\[\]]+)\]", q)
            if list_m:
                try:
                    raw_items = [tok.strip() for tok in list_m.group(1).split(",") if tok.strip()]
                    probs = [self._normalize_num(tok) for tok in raw_items]
                except Exception:
                    probs = []
                if probs and all(p >= 0 for p in probs) and abs(float(sum(probs)) - 1.0) <= 1e-6:
                    h_bits = -sum(float(p) * math.log2(float(p)) for p in probs if p > 0)
                    disp = f"{h_bits:.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Validate probability simplex and compute Shannon entropy in bits",
                            "H(P) = -\\sum_i p_i \\log_2 p_i",
                            f"H(P) = {disp}\\text{{ bits}}",
                            disp,
                            "bits",
                        )
                    ]
                    receipt = self._build_receipt("information_theory", "mathematics", "shannon_entropy", "infotheory.shannon_entropy", {"p": str([str(p) for p in probs])}, disp, disp, "bits", steps)
                    return SolverResult(True, q, "mathematics", "information_theory", "shannon_entropy", "infotheory.shannon_entropy", h_bits, disp, "bits", steps, receipt, f"Shannon entropy is {disp} bits.")

        # 15. KL Divergence in bits: "kl divergence between [0.5, 0.5] and [0.25, 0.75]"
        if "kl divergence" in lower or "kullback" in lower:
            lists = re.findall(r"\[([^\[\]]+)\]", q)
            if len(lists) >= 2:
                try:
                    p_dist = [self._normalize_num(tok.strip()) for tok in lists[0].split(",") if tok.strip()]
                    q_dist = [self._normalize_num(tok.strip()) for tok in lists[1].split(",") if tok.strip()]
                except Exception:
                    p_dist, q_dist = [], []
                if (
                    p_dist
                    and len(p_dist) == len(q_dist)
                    and all(p >= 0 for p in p_dist)
                    and all(qv >= 0 for qv in q_dist)
                    and abs(float(sum(p_dist)) - 1.0) <= 1e-6
                    and abs(float(sum(q_dist)) - 1.0) <= 1e-6
                    and all(not (p > 0 and qv == 0) for p, qv in zip(p_dist, q_dist))
                ):
                    kl_bits = sum(
                        float(p) * math.log2(float(p) / float(qv))
                        for p, qv in zip(p_dist, q_dist)
                        if p > 0
                    )
                    disp = f"{kl_bits:.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Validate distributions and compute KL divergence D_KL(P || Q) in bits",
                            "D_{\\text{KL}}(P \\| Q) = \\sum_i p_i \\log_2\\frac{p_i}{q_i}",
                            f"D_{{\\text{{KL}}}} = {disp}\\text{{ bits}}",
                            disp,
                            "bits",
                        )
                    ]
                    receipt = self._build_receipt("information_theory", "mathematics", "kl_divergence", "infotheory.kl_divergence", {"P": str([str(p) for p in p_dist]), "Q": str([str(qv) for qv in q_dist])}, disp, disp, "bits", steps)
                    return SolverResult(True, q, "mathematics", "information_theory", "kl_divergence", "infotheory.kl_divergence", kl_bits, disp, "bits", steps, receipt, f"KL divergence D_KL(P || Q) is {disp} bits.")

        # 16. Poisson PMF: "poisson with lambda = 3 and k = 2"
        if "poisson" in lower:
            lam_m = re.search(r"(?:lambda|rate|mean)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            k_m = re.search(r"\bk\s*=\s*(\d+)", q, re.I)
            if lam_m and k_m:
                lam = float(lam_m.group(1))
                kv = int(k_m.group(1))
                if lam > 0 and 0 <= kv <= 100:
                    pmf = (lam ** kv) * math.exp(-lam) / float(math.factorial(kv))
                    disp = f"{pmf:.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate Poisson probability mass function",
                            "P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}",
                            f"P(X = {kv}) = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("probability", "mathematics", "poisson_pmf", "prob.poisson_pmf", {"lambda": str(lam), "k": str(kv)}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "probability", "poisson_pmf", "prob.poisson_pmf", pmf, disp, "", steps, receipt, f"Poisson probability P(X = {kv}) = {disp}.")

        # 17. Ideal Gas Law PV = nRT (SI units, R = 8.314462618 J/(mol*K))
        if "ideal gas" in lower or "pv = nrt" in lower or "pv=nrt" in lower:
            p_m = re.search(r"\bP\s*=\s*(\d+(?:\.\d+)?)", q)
            v_m = re.search(r"\bV\s*=\s*(\d+(?:\.\d+)?)", q)
            n_m = re.search(r"\bn\s*=\s*(\d+(?:\.\d+)?)", q)
            t_m = re.search(r"\bT\s*=\s*(\d+(?:\.\d+)?)", q)
            known = {
                "P": float(p_m.group(1)) if p_m else None,
                "V": float(v_m.group(1)) if v_m else None,
                "n": float(n_m.group(1)) if n_m else None,
                "T": float(t_m.group(1)) if t_m else None,
            }
            missing = [k for k, v in known.items() if v is None]
            if len(missing) == 1 and all(v > 0 for v in known.values() if v is not None):
                r_const = 8.314462618
                target_var = missing[0]
                if target_var == "P":
                    val, unit = (known["n"] * r_const * known["T"]) / known["V"], "Pa"
                elif target_var == "V":
                    val, unit = (known["n"] * r_const * known["T"]) / known["P"], "m^3"
                elif target_var == "n":
                    val, unit = (known["P"] * known["V"]) / (r_const * known["T"]), "mol"
                else:
                    val, unit = (known["P"] * known["V"]) / (known["n"] * r_const), "K"
                disp = f"{val:.6f}".rstrip("0").rstrip(".") or "0"
                steps = [
                    DerivationStep(
                        1,
                        f"Solve ideal gas law PV = nRT for {target_var}",
                        "PV = nRT,\\; R = 8.314462618\\text{ J/(mol}\\cdot\\text{K)}",
                        f"{target_var} = {disp}\\text{{ {unit}}}",
                        disp,
                        unit,
                    )
                ]
                receipt = self._build_receipt("thermodynamics", "physics", f"ideal_gas_{target_var}", "thermo.ideal_gas", {k: str(v) for k, v in known.items() if v is not None}, disp, disp, unit, steps)
                return SolverResult(True, q, "physics", "thermodynamics", f"ideal_gas_{target_var}", "thermo.ideal_gas", val, disp, unit, steps, receipt, f"By the ideal gas law PV = nRT, {target_var} = {disp} {unit}.")

        # 18. Capacitors in Series / Parallel: "capacitors 6 uF and 3 uF in series"
        if "capacitor" in lower and ("series" in lower or "parallel" in lower):
            cap_nums = re.findall(r"(\d+(?:\.\d+)?)\s*(?:u|µ|m|n|p)?f\b", q, re.I)
            if len(cap_nums) >= 2:
                caps = [self._normalize_num(c) for c in cap_nums[:8]]
                if all(c > 0 for c in caps):
                    if "series" in lower:
                        inv_sum = sum((Fraction(1) / c for c in caps), Fraction(0))
                        ceq = Fraction(1) / inv_sum
                        fid, eq = "circuits.capacitors_series", "\\frac{1}{C_{\\text{eq}}} = \\sum \\frac{1}{C_i}"
                    else:
                        ceq = sum(caps, Fraction(0))
                        fid, eq = "circuits.capacitors_parallel", "C_{\\text{eq}} = \\sum C_i"
                    disp = self._format_fraction(ceq)
                    steps = [DerivationStep(1, "Compute equivalent capacitance", eq, f"C_{{\\text{{eq}}}} = {disp}", disp)]
                    receipt = self._build_receipt("circuits", "physics", "equivalent_capacitance", fid, {"capacitors": str([str(c) for c in caps])}, str(ceq), disp, "F", steps)
                    return SolverResult(True, q, "physics", "circuits", "equivalent_capacitance", fid, float(ceq), disp, "", steps, receipt, f"Equivalent capacitance is {disp}.")

        # 19. Relativistic Lorentz Factor: "lorentz factor at v = 0.6c" or "beta = 0.8"
        if "lorentz factor" in lower or ("lorentz" in lower and ("beta" in lower or "0." in lower)):
            beta_m = re.search(r"(?:beta\s*=\s*|v\s*=\s*)?(\d+(?:\.\d+)?)\s*c?\b", q, re.I)
            if beta_m:
                beta = float(beta_m.group(1))
                if 0.0 <= beta < 1.0:
                    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
                    disp = f"{gamma:.6f}".rstrip("0").rstrip(".") or "1"
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate relativistic Lorentz factor",
                            "\\gamma = \\frac{1}{\\sqrt{1 - \\beta^2}}",
                            f"\\gamma({beta}) = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("relativity", "physics", "lorentz_factor", "relativity.lorentz_gamma", {"beta": str(beta)}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "physics", "relativity", "lorentz_factor", "relativity.lorentz_gamma", gamma, disp, "", steps, receipt, f"The Lorentz factor γ for β = {beta} is {disp}.")

        # 20. Exact Compound Interest: "compound interest P = 1000, r = 0.05, n = 4, t = 2"
        if "compound interest" in lower:
            p_m = re.search(r"\b(?:P|principal)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            r_m = re.search(r"\b(?:r|rate)\s*=\s*(\d+(?:\.\d+)?)(%?)", q, re.I)
            n_m = re.search(r"\b(?:n|compounds?)\s*=\s*(\d+)", q, re.I)
            t_m = re.search(r"\b(?:t|years?)\s*=\s*(\d+)", q, re.I)
            if p_m and r_m and t_m:
                p_v = self._normalize_num(p_m.group(1))
                r_v = self._normalize_num(r_m.group(1))
                if r_m.group(2) == "%":
                    r_v = r_v / Fraction(100)
                n_v = int(n_m.group(1)) if n_m else 1
                t_v = int(t_m.group(1))
                if p_v > 0 and r_v >= 0 and 1 <= n_v <= 365 and 0 <= t_v <= 100 and (n_v * t_v) <= 1000:
                    amount = p_v * ((Fraction(1) + r_v / Fraction(n_v)) ** (n_v * t_v))
                    disp = f"{float(amount):.2f}"
                    steps = [
                        DerivationStep(
                            1,
                            "Apply exact compound interest formula",
                            "A = P\\left(1 + \\frac{r}{n}\\right)^{nt}",
                            f"A = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("finance", "mathematics", "compound_interest", "finance.compound_interest", {"P": str(p_v), "r": str(r_v), "n": str(n_v), "t": str(t_v)}, str(amount), disp, "currency", steps)
                    return SolverResult(True, q, "mathematics", "finance", "compound_interest", "finance.compound_interest", float(amount), disp, "", steps, receipt, f"Final compound amount A = {disp}.")

        # 21. Exact Extended GCD & Bezout Coefficients: "extended gcd of 240 and 46"
        if "extended gcd" in lower or "bezout" in lower or "bézout" in lower:
            nums = re.findall(r"-?\d+", q)
            if len(nums) == 2:
                a_v, b_v = int(nums[0]), int(nums[1])
                if not (a_v == 0 and b_v == 0) and abs(a_v) <= 10**12 and abs(b_v) <= 10**12:
                    old_r, r = a_v, b_v
                    old_s, s = 1, 0
                    old_t, t = 0, 1
                    while r != 0:
                        quot = old_r // r
                        old_r, r = r, old_r - quot * r
                        old_s, s = s, old_s - quot * s
                        old_t, t = t, old_t - quot * t
                    if old_r < 0:
                        old_r, old_s, old_t = -old_r, -old_s, -old_t
                    if a_v * old_s + b_v * old_t == old_r:
                        disp = f"gcd={old_r}, s={old_s}, t={old_t}"
                        steps = [
                            DerivationStep(
                                1,
                                "Run iterative Extended Euclidean Algorithm and verify Bezout identity",
                                "a s + b t = \\gcd(a, b)",
                                f"{a_v}({old_s}) + {b_v}({old_t}) = {old_r}",
                                disp,
                            )
                        ]
                        receipt = self._build_receipt("number_theory", "mathematics", "extended_gcd", "numtheory.extended_gcd", {"a": str(a_v), "b": str(b_v)}, str(old_r), disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "number_theory", "extended_gcd", "numtheory.extended_gcd", float(old_r), disp, "", steps, receipt, f"Extended GCD: {disp}.")

        # 22. Exact Harmonic Number H_n = sum_{k=1}^n 1/k: "harmonic number H_4"
        if "harmonic number" in lower or re.search(r"\bH_?\(?\d+\)?\b", q):
            hm = re.search(
                r"(?:(\d+)(?:st|nd|rd|th)?\s+harmonic\s+number|harmonic\s+number\s*(?:H_?|\()?\s*(\d+)\)?)",
                q,
                re.I,
            )
            if hm:
                n_v = int(hm.group(1) or hm.group(2))
                if 1 <= n_v <= 500:
                    h_frac = sum((Fraction(1, k) for k in range(1, n_v + 1)), Fraction(0))
                    disp = str(h_frac.numerator) if h_frac.denominator == 1 else f"{h_frac.numerator}/{h_frac.denominator}"
                    steps = [
                        DerivationStep(
                            1,
                            f"Compute exact rational harmonic number H_{n_v}",
                            "H_n = \\sum_{k=1}^{n} \\frac{1}{k}",
                            f"H_{{{n_v}}} = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("series", "mathematics", "harmonic_number", "series.harmonic_number", {"n": str(n_v)}, str(h_frac), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "series", "harmonic_number", "series.harmonic_number", float(h_frac), disp, "", steps, receipt, f"Exact harmonic number H_{n_v} = {disp}.")

        # 23. Exact Multinomial Coefficient: "multinomial(6; 2, 2, 2)"
        if "multinomial" in lower:
            semi_m = re.search(r"multinomial\s*\(\s*(\d+)\s*;\s*([\d,\s]+)\)", q, re.I)
            plain_m = re.search(r"multinomial(?:\s+coefficient)?(?:\s+of)?\s*\(([\d,\s]+)\)", q, re.I)
            n_explicit: Optional[int] = None
            k_parts: List[int] = []
            if semi_m:
                n_explicit = int(semi_m.group(1))
                k_parts = [int(tok.strip()) for tok in semi_m.group(2).split(",") if tok.strip()]
            elif plain_m:
                k_parts = [int(tok.strip()) for tok in plain_m.group(1).split(",") if tok.strip()]
            if k_parts and all(k >= 0 for k in k_parts):
                k_sum = sum(k_parts)
                if k_sum <= 200 and (n_explicit is None or n_explicit == k_sum):
                    num = math.factorial(k_sum)
                    den = 1
                    for k_i in k_parts:
                        den *= math.factorial(k_i)
                    val_int = num // den
                    disp = str(val_int)
                    steps = [
                        DerivationStep(
                            1,
                            "Compute exact multinomial coefficient",
                            "\\binom{n}{k_1, \\dots, k_m} = \\frac{n!}{k_1! \\cdots k_m!}",
                            f"\\frac{{{k_sum}!}}{{{' \\cdot '.join(f'{k}!' for k in k_parts)}}} = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("combinatorics", "mathematics", "multinomial", "comb.multinomial", {"n": str(k_sum), "k": str(k_parts)}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "combinatorics", "multinomial", "comb.multinomial", float(val_int), disp, "", steps, receipt, f"Multinomial coefficient = {disp}.")

        # 24. Exact Euclidean / Manhattan Distance between Vectors
        if ("euclidean distance" in lower or "manhattan distance" in lower) and (
            "(" in q or "[" in q
        ):
            vecs = re.findall(r"[\(\[]([^\(\)\[\]]+)[\)\]]", q)
            if len(vecs) >= 2:
                try:
                    u_v = [self._normalize_num(tok.strip()) for tok in vecs[0].split(",") if tok.strip()]
                    v_v = [self._normalize_num(tok.strip()) for tok in vecs[1].split(",") if tok.strip()]
                except Exception:
                    u_v, v_v = [], []
                if u_v and len(u_v) == len(v_v):
                    if "manhattan" in lower:
                        dist_f = sum((abs(a - b) for a, b in zip(u_v, v_v)), Fraction(0))
                        disp = self._format_fraction(dist_f)
                        fid, eq = "geometry.manhattan_distance", "d_1(u, v) = \\sum_i |u_i - v_i|"
                        fval = float(dist_f)
                    else:
                        sq_sum = sum(((a - b) * (a - b) for a, b in zip(u_v, v_v)), Fraction(0))
                        fval = math.sqrt(float(sq_sum))
                        if sq_sum.denominator == 1:
                            r_int = math.isqrt(sq_sum.numerator)
                            if r_int * r_int == sq_sum.numerator:
                                disp = str(r_int)
                            else:
                                disp = f"{fval:.6f}".rstrip("0").rstrip(".") or "0"
                        else:
                            disp = f"{fval:.6f}".rstrip("0").rstrip(".") or "0"
                        fid, eq = "geometry.euclidean_distance", "d_2(u, v) = \\sqrt{\\sum_i (u_i - v_i)^2}"
                    steps = [DerivationStep(1, "Compute exact vector distance", eq, f"d = {disp}", disp)]
                    receipt = self._build_receipt("geometry", "mathematics", "vector_distance", fid, {"u": str([str(x) for x in u_v]), "v": str([str(x) for x in v_v])}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "geometry", "vector_distance", fid, fval, disp, "", steps, receipt, f"Distance = {disp}.")

        # 25. Exact Cosine Similarity between Vectors: "cosine similarity of [1, 0] and [1, 1]"
        if "cosine similarity" in lower:
            vecs = re.findall(r"[\(\[]([^\(\)\[\]]+)[\)\]]", q)
            if len(vecs) >= 2:
                try:
                    u_v = [self._normalize_num(tok.strip()) for tok in vecs[0].split(",") if tok.strip()]
                    v_v = [self._normalize_num(tok.strip()) for tok in vecs[1].split(",") if tok.strip()]
                except Exception:
                    u_v, v_v = [], []
                if u_v and len(u_v) == len(v_v):
                    dot = sum((a * b for a, b in zip(u_v, v_v)), Fraction(0))
                    nu2 = sum((a * a for a in u_v), Fraction(0))
                    nv2 = sum((b * b for b in v_v), Fraction(0))
                    if nu2 > 0 and nv2 > 0:
                        cos_val = float(dot) / math.sqrt(float(nu2) * float(nv2))
                        cos_val = max(-1.0, min(1.0, cos_val))
                        disp = f"{cos_val:.6f}".rstrip("0").rstrip(".") or "0"
                        steps = [
                            DerivationStep(
                                1,
                                "Compute exact vector cosine similarity",
                                "\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\|_2 \\|v\\|_2}",
                                f"\\cos(\\theta) = {disp}",
                                disp,
                            )
                        ]
                        receipt = self._build_receipt("linear_algebra", "mathematics", "cosine_similarity", "linalg.cosine_similarity", {"u": str([str(x) for x in u_v]), "v": str([str(x) for x in v_v])}, disp, disp, "dimensionless", steps)
                        return SolverResult(True, q, "mathematics", "linear_algebra", "cosine_similarity", "linalg.cosine_similarity", cos_val, disp, "", steps, receipt, f"Cosine similarity = {disp}.")

        # 26. Exact Standard Normal Z-Score: "z-score of x = 85 with mean = 70 and std = 10"
        if "z-score" in lower or "z score" in lower:
            x_m = re.search(r"\bx\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            mu_m = re.search(r"\b(?:mean|mu)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            sd_m = re.search(r"\b(?:std|sigma|sd)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if x_m and mu_m and sd_m:
                x_v = self._normalize_num(x_m.group(1))
                mu_v = self._normalize_num(mu_m.group(1))
                sd_v = self._normalize_num(sd_m.group(1))
                if sd_v > 0:
                    z_v = (x_v - mu_v) / sd_v
                    disp = self._format_fraction(z_v)
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate exact standard score z = (x - mu) / sigma",
                            "z = \\frac{x - \\mu}{\\sigma}",
                            f"z = \\frac{{{x_v} - {mu_v}}}{{{sd_v}}} = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("statistics", "mathematics", "z_score", "stats.z_score", {"x": str(x_v), "mean": str(mu_v), "std": str(sd_v)}, str(z_v), disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "statistics", "z_score", "stats.z_score", float(z_v), disp, "", steps, receipt, f"Standard z-score = {disp}.")

        # 27. Exact Radioactive Half-Life Decay: "half-life decay N0 = 100, t = 15, half_life = 5"
        if "half-life" in lower or "half life" in lower or "radioactive decay" in lower:
            n0_m = re.search(r"\b(?:N0|N_0|initial)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            t_m = re.search(r"\bt\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            hl_m = re.search(r"\b(?:half_life|half-life|hl|T_half|T1/2)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            if n0_m and t_m and hl_m:
                n0_v = self._normalize_num(n0_m.group(1))
                t_v = self._normalize_num(t_m.group(1))
                hl_v = self._normalize_num(hl_m.group(1))
                if n0_v >= 0 and t_v >= 0 and hl_v > 0:
                    ratio_f = t_v / hl_v
                    if ratio_f.denominator == 1 and ratio_f.numerator <= 200:
                        rem = n0_v / Fraction(1 << ratio_f.numerator)
                        disp = self._format_fraction(rem)
                        fval = float(rem)
                    else:
                        fval = float(n0_v) * (2.0 ** (-float(ratio_f)))
                        disp = f"{fval:.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate exponential radioactive decay N(t) = N_0 * 2^(-t / T_{1/2})",
                            "N(t) = N_0 \\cdot 2^{-t / T_{1/2}}",
                            f"N({t_v}) = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("nuclear_physics", "physics", "half_life_decay", "physics.half_life_decay", {"N0": str(n0_v), "t": str(t_v), "half_life": str(hl_v)}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "physics", "nuclear_physics", "half_life_decay", "physics.half_life_decay", fval, disp, "", steps, receipt, f"Remaining quantity after half-life decay is {disp}.")

        # 28. Exact Acoustic Doppler Shift: "doppler shift f0 = 500, v_obs = 0, v_src = 34, v = 340"
        if "doppler" in lower:
            f0_m = re.search(r"\b(?:f0|f_0|freq(?:uency)?)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            v_m = re.search(r"\bv\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            vo_m = re.search(r"\b(?:v_obs|vo)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            vs_m = re.search(r"\b(?:v_src|vs)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if f0_m and (vo_m or vs_m):
                f0_v = self._normalize_num(f0_m.group(1))
                v_v = self._normalize_num(v_m.group(1)) if v_m else Fraction(340)
                vo_v = self._normalize_num(vo_m.group(1)) if vo_m else Fraction(0)
                vs_v = self._normalize_num(vs_m.group(1)) if vs_m else Fraction(0)
                if f0_v > 0 and v_v > 0 and (v_v - vs_v) > 0 and (v_v + vo_v) >= 0:
                    f_obs = f0_v * (v_v + vo_v) / (v_v - vs_v)
                    disp = f"{float(f_obs):.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate exact acoustic Doppler observed frequency",
                            "f' = f_0 \\frac{v + v_{\\text{obs}}}{v - v_{\\text{src}}}",
                            f"f' = {disp}\\text{{ Hz}}",
                            disp,
                            "Hz",
                        )
                    ]
                    receipt = self._build_receipt("waves", "physics", "doppler_shift", "physics.doppler_shift", {"f0": str(f0_v), "v": str(v_v), "v_obs": str(vo_v), "v_src": str(vs_v)}, str(f_obs), disp, "Hz", steps)
                    return SolverResult(True, q, "physics", "waves", "doppler_shift", "physics.doppler_shift", float(f_obs), disp, "Hz", steps, receipt, f"Observed Doppler frequency f' = {disp} Hz.")

        # 29. Exact Amdahl's Law Speedup: "amdahl speedup p = 0.8, n = 4"
        if "amdahl" in lower:
            p_m = re.search(r"\bp\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            n_m = re.search(r"\b(?:n|s|cores?|processors?)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            if p_m and n_m:
                p_v = self._normalize_num(p_m.group(1))
                n_v = self._normalize_num(n_m.group(1))
                if Fraction(0) <= p_v <= Fraction(1) and n_v > 0:
                    speedup = Fraction(1) / ((Fraction(1) - p_v) + (p_v / n_v))
                    disp = self._format_fraction(speedup)
                    steps = [
                        DerivationStep(
                            1,
                            "Apply exact Amdahl's Law parallel speedup formula",
                            "S(n) = \\frac{1}{(1 - p) + p / n}",
                            f"S({n_v}) = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("systems", "computer_science", "amdahl_speedup", "cs.amdahl_speedup", {"p": str(p_v), "n": str(n_v)}, str(speedup), disp, "x", steps)
                    return SolverResult(True, q, "computer_science", "systems", "amdahl_speedup", "cs.amdahl_speedup", float(speedup), disp, "x", steps, receipt, f"Amdahl's Law speedup S = {disp}x.")

        # 30. Exact Integer Base Conversion (Radix 2..36): "convert 255 from base 10 to base 16"
        base_m = re.search(
            r"convert\s+(-?[0-9A-Za-z]+)\s+from\s+base\s+(\d+)\s+to\s+base\s+(\d+)",
            q,
            re.I,
        )
        if base_m:
            raw_digits, b_from, b_to = base_m.group(1).upper(), int(base_m.group(2)), int(base_m.group(3))
            if 2 <= b_from <= 36 and 2 <= b_to <= 36 and len(raw_digits) <= 64:
                alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                neg = raw_digits.startswith("-")
                body = raw_digits[1:] if neg else raw_digits
                if body and all(ch in alphabet[:b_from] for ch in body):
                    val_int = int(body, b_from)
                    if val_int == 0:
                        converted = "0"
                    else:
                        out_chars: List[str] = []
                        tmp = val_int
                        while tmp > 0:
                            tmp, rem = divmod(tmp, b_to)
                            out_chars.append(alphabet[rem])
                        converted = ("-" if neg else "") + "".join(reversed(out_chars))
                    steps = [
                        DerivationStep(
                            1,
                            f"Validate base-{b_from} digits and convert exact integer to base-{b_to}",
                            f"({raw_digits})_{{{b_from}}} \\to ({converted})_{{{b_to}}}",
                            f"{val_int}_{{10}} = ({converted})_{{{b_to}}}",
                            converted,
                        )
                    ]
                    receipt = self._build_receipt("number_systems", "computer_science", "base_conversion", "cs.base_conversion", {"value": raw_digits, "from_base": str(b_from), "to_base": str(b_to)}, str(-val_int if neg else val_int), converted, "", steps)
                    return SolverResult(True, q, "computer_science", "number_systems", "base_conversion", "cs.base_conversion", float(-val_int if neg else val_int), converted, "", steps, receipt, f"{raw_digits} (base {b_from}) = {converted} (base {b_to}).")

        # 31. Exact Rational Matrix Multiplication: "matrix multiply [[1, 2], [3, 4]] and [[5, 6], [7, 8]]"
        if "matrix multiply" in lower or "matrix product" in lower or "multiply matrices" in lower:
            raw_mats = re.findall(r"\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]", q)
            if len(raw_mats) == 2:
                try:
                    m_a_raw = json.loads(raw_mats[0])
                    m_b_raw = json.loads(raw_mats[1])
                except Exception:
                    m_a_raw, m_b_raw = None, None
                if (
                    isinstance(m_a_raw, list)
                    and isinstance(m_b_raw, list)
                    and 1 <= len(m_a_raw) <= 6
                    and 1 <= len(m_b_raw) <= 6
                    and all(isinstance(r, list) and 1 <= len(r) <= 6 for r in m_a_raw)
                    and all(isinstance(r, list) and 1 <= len(r) <= 6 for r in m_b_raw)
                ):
                    cols_a = len(m_a_raw[0])
                    cols_b = len(m_b_raw[0])
                    if (
                        all(len(r) == cols_a for r in m_a_raw)
                        and all(len(r) == cols_b for r in m_b_raw)
                        and cols_a == len(m_b_raw)
                    ):
                        try:
                            mat_a = [[self._normalize_num(str(x)) for x in row] for row in m_a_raw]
                            mat_b = [[self._normalize_num(str(x)) for x in row] for row in m_b_raw]
                        except Exception:
                            mat_a, mat_b = [], []
                        if mat_a and mat_b:
                            prod = [
                                [
                                    sum((mat_a[i][k] * mat_b[k][j] for k in range(cols_a)), Fraction(0))
                                    for j in range(cols_b)
                                ]
                                for i in range(len(mat_a))
                            ]
                            disp = "[" + ", ".join(
                                "[" + ", ".join(self._format_fraction(c) for c in row) + "]"
                                for row in prod
                            ) + "]"
                            mat_sum = sum((c for row in prod for c in row), Fraction(0))
                            steps = [
                                DerivationStep(
                                    1,
                                    f"Multiply {len(mat_a)}x{cols_a} and {len(mat_b)}x{cols_b} matrices in exact rational arithmetic",
                                    "(AB)_{ij} = \\sum_k A_{ik} B_{kj}",
                                    f"AB = {disp}",
                                    disp,
                                )
                            ]
                            receipt = self._build_receipt("linear_algebra", "mathematics", "matrix_multiply", "linalg.matrix_multiply", {"A": raw_mats[0], "B": raw_mats[1]}, disp, disp, "matrix", steps)
                            return SolverResult(True, q, "mathematics", "linear_algebra", "matrix_multiply", "linalg.matrix_multiply", float(mat_sum), disp, "", steps, receipt, f"Exact matrix product AB = {disp}.")

        # 32. Exact Rational Matrix Rank via RREF: "matrix rank of [[1, 2, 3], [2, 4, 6], [0, 1, 1]]"
        if "matrix rank" in lower:
            raw_mats = re.findall(r"\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]", q)
            if len(raw_mats) == 1:
                try:
                    m_raw = json.loads(raw_mats[0])
                except Exception:
                    m_raw = None
                if (
                    isinstance(m_raw, list)
                    and 1 <= len(m_raw) <= 8
                    and all(isinstance(r, list) and 1 <= len(r) <= 8 for r in m_raw)
                ):
                    n_cols = len(m_raw[0])
                    if all(len(r) == n_cols for r in m_raw):
                        try:
                            mat = [[self._normalize_num(str(x)) for x in row] for row in m_raw]
                        except Exception:
                            mat = []
                        if mat:
                            n_rows = len(mat)
                            rank = 0
                            row_idx = 0
                            for col in range(n_cols):
                                pivot = None
                                for r in range(row_idx, n_rows):
                                    if mat[r][col] != 0:
                                        pivot = r
                                        break
                                if pivot is None:
                                    continue
                                mat[row_idx], mat[pivot] = mat[pivot], mat[row_idx]
                                p_val = mat[row_idx][col]
                                mat[row_idx] = [v / p_val for v in mat[row_idx]]
                                for r in range(n_rows):
                                    if r != row_idx and mat[r][col] != 0:
                                        factor = mat[r][col]
                                        mat[r] = [mat[r][c] - factor * mat[row_idx][c] for c in range(n_cols)]
                                rank += 1
                                row_idx += 1
                                if row_idx == n_rows:
                                    break
                            disp = str(rank)
                            steps = [
                                DerivationStep(
                                    1,
                                    "Reduce matrix to Reduced Row Echelon Form (RREF) in exact Fraction arithmetic and count pivot rows",
                                    "\\operatorname{rank}(A) = \\text{number of non-zero pivots in } \\operatorname{RREF}(A)",
                                    f"\\operatorname{{rank}}(A) = {disp}",
                                    disp,
                                )
                            ]
                            receipt = self._build_receipt("linear_algebra", "mathematics", "matrix_rank", "linalg.matrix_rank", {"A": raw_mats[0]}, disp, disp, "dimensionless", steps)
                            return SolverResult(True, q, "mathematics", "linear_algebra", "matrix_rank", "linalg.matrix_rank", float(rank), disp, "", steps, receipt, f"Exact matrix rank = {disp}.")

        # 33. Exact Complex Arithmetic & Modulus: "complex (3+4i) * (1-2i)" or "modulus of 3+4i"
        if "complex" in lower or "modulus of" in lower:
            def _parse_complex_pair(expr: str) -> Optional[Tuple[Fraction, Fraction]]:
                s_clean = expr.replace(" ", "").replace("j", "i")
                cm = re.fullmatch(
                    r"([+-]?\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)i",
                    s_clean,
                    re.I,
                )
                if cm:
                    return self._normalize_num(cm.group(1)), self._normalize_num(cm.group(2))
                return None

            def _fmt_complex(re_v: Fraction, im_v: Fraction) -> str:
                re_s = self._format_fraction(re_v)
                if im_v == 0:
                    return re_s
                sign = "+" if im_v > 0 else "-"
                im_abs = self._format_fraction(abs(im_v))
                if re_v == 0:
                    return f"{'' if im_v > 0 else '-'}{im_abs}i"
                return f"{re_s}{sign}{im_abs}i"

            bin_m = re.search(
                r"\(\s*([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?\s*[ij])\s*\)\s*([+\-*/])\s*\(\s*([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?\s*[ij])\s*\)",
                q,
                re.I,
            )
            if bin_m:
                z1 = _parse_complex_pair(bin_m.group(1))
                op_ch = bin_m.group(2)
                z2 = _parse_complex_pair(bin_m.group(3))
                if z1 is not None and z2 is not None:
                    a1, b1 = z1
                    a2, b2 = z2
                    ok_op = True
                    if op_ch == "+":
                        rr, ri = a1 + a2, b1 + b2
                    elif op_ch == "-":
                        rr, ri = a1 - a2, b1 - b2
                    elif op_ch == "*":
                        rr, ri = a1 * a2 - b1 * b2, a1 * b2 + b1 * a2
                    else:
                        denom = a2 * a2 + b2 * b2
                        if denom == 0:
                            ok_op = False
                            rr, ri = Fraction(0), Fraction(0)
                        else:
                            rr = (a1 * a2 + b1 * b2) / denom
                            ri = (b1 * a2 - a1 * b2) / denom
                    if ok_op:
                        disp = _fmt_complex(rr, ri)
                        mod_val = math.hypot(float(rr), float(ri))
                        steps = [
                            DerivationStep(
                                1,
                                f"Perform exact rational complex '{op_ch}' operation",
                                f"({bin_m.group(1).strip()}) {op_ch} ({bin_m.group(3).strip()})",
                                f"= {disp}",
                                disp,
                            )
                        ]
                        receipt = self._build_receipt("complex_analysis", "mathematics", "complex_arithmetic", "math.complex_arithmetic", {"z1": bin_m.group(1).strip(), "op": op_ch, "z2": bin_m.group(3).strip()}, disp, disp, "complex", steps)
                        return SolverResult(True, q, "mathematics", "complex_analysis", "complex_arithmetic", "math.complex_arithmetic", mod_val, disp, "", steps, receipt, f"Complex result = {disp}.")
            mod_m = re.search(
                r"(?:modulus|magnitude|abs)\s*(?:of\s*)?\(?\s*([+-]?\d+(?:\.\d+)?\s*[+-]\s*\d+(?:\.\d+)?\s*[ij])\s*\)?",
                q,
                re.I,
            )
            if mod_m:
                z_val = _parse_complex_pair(mod_m.group(1))
                if z_val is not None:
                    a_c, b_c = z_val
                    sq_c = a_c * a_c + b_c * b_c
                    fval = math.sqrt(float(sq_c))
                    if sq_c.denominator == 1 and math.isqrt(sq_c.numerator) ** 2 == sq_c.numerator:
                        disp = str(math.isqrt(sq_c.numerator))
                    else:
                        disp = f"{fval:.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Compute exact complex modulus |a + bi| = sqrt(a^2 + b^2)",
                            "|a + bi| = \\sqrt{a^2 + b^2}",
                            f"|{mod_m.group(1).strip()}| = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("complex_analysis", "mathematics", "complex_modulus", "math.complex_modulus", {"z": mod_m.group(1).strip()}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "complex_analysis", "complex_modulus", "math.complex_modulus", fval, disp, "", steps, receipt, f"Complex modulus = {disp}.")

        # 34. Exact Dijkstra Shortest Path: "shortest path from A to D in graph A-B:4, A-C:1, C-B:2, B-D:1, C-D:5"
        if ("dijkstra" in lower or "shortest path" in lower) and "from" in lower and "to" in lower:
            ft_m = re.search(r"from\s+([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)", q, re.I)
            edge_matches = re.findall(
                r"([A-Za-z0-9_]+)\s*(->|-)\s*([A-Za-z0-9_]+)\s*:\s*(-?\d+(?:\.\d+)?)",
                q,
            )
            if ft_m and edge_matches and len(edge_matches) <= 128:
                src, dst = ft_m.group(1), ft_m.group(2)
                adj: Dict[str, List[Tuple[str, Fraction]]] = {}
                valid_edges = True
                for u_n, arrow, v_n, w_s in edge_matches:
                    w_f = self._normalize_num(w_s)
                    if w_f < 0:
                        valid_edges = False
                        break
                    adj.setdefault(u_n, []).append((v_n, w_f))
                    adj.setdefault(v_n, [])
                    if arrow == "-":
                        adj[v_n].append((u_n, w_f))
                if valid_edges and src in adj and dst in adj:
                    dist: Dict[str, Fraction] = {src: Fraction(0)}
                    prev_node: Dict[str, str] = {}
                    pq: List[Tuple[Fraction, str]] = [(Fraction(0), src)]
                    while pq:
                        d_u, u_cur = heapq.heappop(pq)
                        if d_u > dist.get(u_cur, Fraction(10**18)):
                            continue
                        if u_cur == dst:
                            break
                        for v_nxt, w_uv in adj.get(u_cur, []):
                            cand = d_u + w_uv
                            if v_nxt not in dist or cand < dist[v_nxt]:
                                dist[v_nxt] = cand
                                prev_node[v_nxt] = u_cur
                                heapq.heappush(pq, (cand, v_nxt))
                    if dst in dist:
                        path_nodes: List[str] = [dst]
                        cur = dst
                        while cur != src and cur in prev_node:
                            cur = prev_node[cur]
                            path_nodes.append(cur)
                        path_nodes.reverse()
                        d_exact = dist[dst]
                        disp_num = self._format_fraction(d_exact)
                        path_str = " -> ".join(path_nodes)
                        steps = [
                            DerivationStep(
                                1,
                                f"Run Dijkstra's algorithm from {src} to {dst} over {len(edge_matches)} non-negative edges",
                                "d(v) = \\min_{(u,v)\\in E} (d(u) + w(u,v))",
                                f"\\text{{path: }} {path_str},\\; d = {disp_num}",
                                disp_num,
                            )
                        ]
                        receipt = self._build_receipt("graph_algorithms", "computer_science", "dijkstra_shortest_path", "cs.dijkstra_shortest_path", {"src": src, "dst": dst, "edges": str(len(edge_matches))}, str(d_exact), disp_num, "weight", steps)
                        return SolverResult(True, q, "computer_science", "graph_algorithms", "dijkstra_shortest_path", "cs.dijkstra_shortest_path", float(d_exact), disp_num, "", steps, receipt, f"Shortest path distance from {src} to {dst} is {disp_num} via {path_str}.")

        # 35. Exact Master Theorem Recurrence Solver: "master theorem T(n) = 2T(n/2) + n^1" or "a=4, b=2, d=1"
        if "master theorem" in lower:
            rec_m = re.search(
                r"T\s*\(\s*n\s*\)\s*=\s*(\d+(?:\.\d+)?)\s*\*?\s*T\s*\(\s*n\s*/\s*(\d+(?:\.\d+)?)\s*\)\s*\+\s*(?:O\s*\(\s*)?n(?:\^(\d+(?:\.\d+)?))?\)?",
                q,
                re.I,
            )
            a_v: Optional[float] = None
            b_v: Optional[float] = None
            d_v: Optional[float] = None
            if rec_m:
                a_v = float(rec_m.group(1))
                b_v = float(rec_m.group(2))
                d_v = float(rec_m.group(3)) if rec_m.group(3) is not None else 1.0
            else:
                am = re.search(r"\ba\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
                bm = re.search(r"\bb\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
                dm = re.search(r"\bd\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
                if am and bm and dm:
                    a_v, b_v, d_v = float(am.group(1)), float(bm.group(1)), float(dm.group(1))
            if a_v is not None and b_v is not None and d_v is not None and a_v >= 1.0 and b_v > 1.0 and d_v >= 0.0:
                c_crit = math.log(a_v, b_v)

                def _fmt_exp(e: float) -> str:
                    if abs(e - round(e)) < 1e-9:
                        ei = int(round(e))
                        return "" if ei == 1 else f"^{ei}"
                    return f"^{e:.4f}".rstrip("0").rstrip(".")

                if d_v < c_crit - 1e-9:
                    case_num = 1
                    disp = f"Theta(n{_fmt_exp(c_crit)})"
                elif abs(d_v - c_crit) <= 1e-9:
                    case_num = 2
                    if abs(d_v) < 1e-9:
                        disp = "Theta(log n)"
                    else:
                        disp = f"Theta(n{_fmt_exp(d_v)} log n)"
                else:
                    case_num = 3
                    disp = f"Theta(n{_fmt_exp(d_v)})"
                steps = [
                    DerivationStep(
                        1,
                        f"Compare d = {d_v} against critical exponent log_b(a) = {c_crit:.6f} (Master Theorem Case {case_num})",
                        "c_{\\text{crit}} = \\log_b(a)",
                        f"\\text{{Case {case_num}: }} T(n) = {disp}",
                        disp,
                    )
                ]
                receipt = self._build_receipt("algorithms", "computer_science", "master_theorem", "cs.master_theorem", {"a": str(a_v), "b": str(b_v), "d": str(d_v)}, f"{c_crit:.6f}", disp, "asymptotic", steps)
                return SolverResult(True, q, "computer_science", "algorithms", "master_theorem", "cs.master_theorem", round(c_crit, 6), disp, "", steps, receipt, f"By Master Theorem Case {case_num} (log_b(a) = {c_crit:.4f}), T(n) = {disp}.")

        # 36. Exact Beer-Lambert Absorbance Law: "beer-lambert absorbance epsilon = 150, l = 1, c = 0.02"
        if "beer-lambert" in lower or "beer lambert" in lower:
            eps_m = re.search(r"\b(?:epsilon|eps)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            l_m = re.search(r"\b(?:l|path|length)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            c_m = re.search(r"\b(?:c|conc(?:entration)?)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            if eps_m and l_m and c_m:
                eps_v = self._normalize_num(eps_m.group(1))
                l_v = self._normalize_num(l_m.group(1))
                c_v = self._normalize_num(c_m.group(1))
                if eps_v > 0 and l_v > 0 and c_v > 0:
                    abs_f = eps_v * l_v * c_v
                    disp = self._format_fraction(abs_f)
                    trans = 10.0 ** (-float(abs_f))
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate exact Beer-Lambert absorbance A = epsilon * l * c",
                            "A = \\varepsilon \\cdot l \\cdot c,\\quad T = 10^{-A}",
                            f"A = {eps_v} \\cdot {l_v} \\cdot {c_v} = {disp},\\; T \\approx {trans:.6f}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("spectroscopy", "chemistry", "beer_lambert_absorbance", "chemistry.beer_lambert", {"epsilon": str(eps_v), "l": str(l_v), "c": str(c_v)}, str(abs_f), disp, "dimensionless", steps)
                    return SolverResult(True, q, "chemistry", "spectroscopy", "beer_lambert_absorbance", "chemistry.beer_lambert", float(abs_f), disp, "", steps, receipt, f"Beer-Lambert absorbance A = {disp} (transmittance T = {trans:.6f}).")

        # 37. Exact Arrhenius Rate Constant: "arrhenius rate A = 1000, Ea = 8314.46261815324, T = 1000"
        if "arrhenius" in lower:
            a_m = re.search(r"\bA\s*=\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", q, re.I)
            ea_m = re.search(r"\b(?:Ea|E_a)\s*=\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)", q, re.I)
            t_m = re.search(r"\bT\s*=\s*(\d+(?:\.\d+)?)", q)
            if a_m and ea_m and t_m:
                a_val = float(a_m.group(1))
                ea_val = float(ea_m.group(1))
                t_val = float(t_m.group(1))
                if a_val > 0.0 and ea_val >= 0.0 and t_val > 0.0:
                    r_val = float(CONSTANTS["R"]["value"])
                    exponent = -ea_val / (r_val * t_val)
                    if exponent >= -700.0:
                        k_val = a_val * math.exp(exponent)
                        disp = f"{k_val:.6f}".rstrip("0").rstrip(".") if (abs(k_val) >= 1e-4 and abs(k_val) < 1e9) else f"{k_val:.6e}"
                        steps = [
                            DerivationStep(
                                1,
                                "Evaluate Arrhenius rate constant k = A * exp(-Ea / (R * T)) using CODATA R = 8.314462618 J/(mol*K)",
                                "k = A \\exp\\left(-\\frac{E_a}{RT}\\right)",
                                f"k = {disp}\\text{{ s}}^{{-1}}",
                                disp,
                                "1/s",
                            )
                        ]
                        receipt = self._build_receipt("kinetics", "chemistry", "arrhenius_rate", "chemistry.arrhenius_rate", {"A": str(a_val), "Ea": str(ea_val), "T": str(t_val)}, disp, disp, "1/s", steps)
                        return SolverResult(True, q, "chemistry", "kinetics", "arrhenius_rate", "chemistry.arrhenius_rate", k_val, disp, "1/s", steps, receipt, f"Arrhenius rate constant k = {disp} s^-1.")

        # 38. Exact Present Value of Ordinary Annuity: "annuity present value pmt = 1000, r = 0.05, n = 10"
        if "annuity" in lower and ("present value" in lower or "pv" in lower):
            pmt_m = re.search(r"\b(?:pmt|payment|c)\s*=\s*(\d+(?:\.\d+)?)", q, re.I)
            r_m = re.search(r"\b(?:r|rate)\s*=\s*(\d+(?:\.\d+)?)(%?)", q, re.I)
            n_m = re.search(r"\b(?:n|periods?)\s*=\s*(\d+)", q, re.I)
            if pmt_m and r_m and n_m:
                pmt_v = self._normalize_num(pmt_m.group(1))
                r_v = self._normalize_num(r_m.group(1))
                if r_m.group(2) == "%":
                    r_v = r_v / Fraction(100)
                n_v = int(n_m.group(1))
                if pmt_v > 0 and r_v >= 0 and 1 <= n_v <= 600:
                    if r_v == 0:
                        pv_f = float(pmt_v * n_v)
                        raw_str = str(pmt_v * n_v)
                    elif n_v <= 60:
                        pv_exact = pmt_v * (Fraction(1) - (Fraction(1) + r_v) ** (-n_v)) / r_v
                        pv_f = float(pv_exact)
                        raw_str = str(pv_exact)
                    else:
                        rf = float(r_v)
                        pv_f = float(pmt_v) * (1.0 - (1.0 + rf) ** (-n_v)) / rf
                        raw_str = f"{pv_f:.8f}"
                    disp = f"{pv_f:.2f}"
                    steps = [
                        DerivationStep(
                            1,
                            "Apply ordinary annuity present value formula",
                            "PV = \\text{PMT} \\cdot \\frac{1 - (1 + r)^{-n}}{r}",
                            f"PV = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("finance", "mathematics", "annuity_pv", "finance.annuity_pv", {"pmt": str(pmt_v), "r": str(r_v), "n": str(n_v)}, raw_str, disp, "currency", steps)
                    return SolverResult(True, q, "mathematics", "finance", "annuity_pv", "finance.annuity_pv", round(pv_f, 6), disp, "", steps, receipt, f"Present value of ordinary annuity PV = {disp}.")

        # 39. Exact Combinations with Repetition (Stars & Bars): "combinations with repetition n = 5, k = 3"
        if "combinations with repetition" in lower or "stars and bars" in lower or "multiset combination" in lower:
            n_m = re.search(r"\bn\s*=\s*(\d+)", q, re.I)
            k_m = re.search(r"\b(?:k|r)\s*=\s*(\d+)", q, re.I)
            if n_m and k_m:
                n_v, k_v = int(n_m.group(1)), int(k_m.group(1))
                if 1 <= n_v <= 500 and 0 <= k_v <= 500:
                    val_int = math.comb(n_v + k_v - 1, k_v)
                    disp = str(val_int)
                    steps = [
                        DerivationStep(
                            1,
                            "Apply stars-and-bars multiset combination formula",
                            "\\left(\\!\\!\\binom{n}{k}\\!\\!\\right) = \\binom{n + k - 1}{k}",
                            f"\\binom{{{n_v} + {k_v} - 1}}{{{k_v}}} = \\binom{{{n_v + k_v - 1}}}{{{k_v}}} = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("combinatorics", "mathematics", "multiset_combinations", "comb.multiset_combinations", {"n": str(n_v), "k": str(k_v)}, disp, disp, "dimensionless", steps)
                    return SolverResult(True, q, "mathematics", "combinatorics", "multiset_combinations", "comb.multiset_combinations", float(val_int), disp, "", steps, receipt, f"Combinations with repetition = {disp}.")

        # 40. Exact Jensen-Shannon Divergence (bits): "jensen-shannon divergence between [0.5, 0.5] and [0.25, 0.75]"
        if "jensen-shannon" in lower or "jensen shannon" in lower:
            vecs = re.findall(r"\[([^\[\]]+)\]", q)
            if len(vecs) >= 2:
                try:
                    p_dist = [self._normalize_num(tok.strip()) for tok in vecs[0].split(",") if tok.strip()]
                    q_dist = [self._normalize_num(tok.strip()) for tok in vecs[1].split(",") if tok.strip()]
                except Exception:
                    p_dist, q_dist = [], []
                if (
                    p_dist
                    and len(p_dist) == len(q_dist)
                    and all(p >= 0 for p in p_dist)
                    and all(qv >= 0 for qv in q_dist)
                    and abs(float(sum(p_dist, Fraction(0))) - 1.0) <= 1e-6
                    and abs(float(sum(q_dist, Fraction(0))) - 1.0) <= 1e-6
                ):
                    jsd_bits = 0.0
                    for p_i, q_i in zip(p_dist, q_dist):
                        m_i = (p_i + q_i) / Fraction(2)
                        if p_i > 0:
                            jsd_bits += 0.5 * float(p_i) * math.log2(float(p_i / m_i))
                        if q_i > 0:
                            jsd_bits += 0.5 * float(q_i) * math.log2(float(q_i / m_i))
                    disp = f"{jsd_bits:.6f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Validate discrete distributions and compute symmetric Jensen-Shannon divergence in bits",
                            "\\text{JSD}(P \\| Q) = \\frac{1}{2} D_{\\text{KL}}(P \\| M) + \\frac{1}{2} D_{\\text{KL}}(Q \\| M),\\; M = \\frac{P+Q}{2}",
                            f"\\text{{JSD}}(P \\| Q) = {disp}\\text{{ bits}}",
                            disp,
                            "bits",
                        )
                    ]
                    receipt = self._build_receipt("information_theory", "mathematics", "jensen_shannon_divergence", "infotheory.jensen_shannon_divergence", {"P": str([str(p) for p in p_dist]), "Q": str([str(qv) for qv in q_dist])}, disp, disp, "bits", steps)
                    return SolverResult(True, q, "mathematics", "information_theory", "jensen_shannon_divergence", "infotheory.jensen_shannon_divergence", jsd_bits, disp, "bits", steps, receipt, f"Jensen-Shannon divergence JSD(P || Q) = {disp} bits.")

        # 41. Exact Nernst Equilibrium Potential (mV): "Nernst potential z = 1, T = 310, out = 5, in = 140"
        if "nernst" in lower:
            z_m = re.search(r"\bz\s*=\s*([+-]?\d+)", q, re.I)
            t_m = re.search(r"\bT\s*=\s*(-?\d+(?:\.\d+)?)", q)
            out_m = re.search(r"\b(?:out|c_out|ext(?:racellular)?)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            in_m = re.search(r"\b(?:in|c_in|int(?:racellular)?)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if z_m and t_m and out_m and in_m:
                z_v = int(z_m.group(1))
                t_v = float(t_m.group(1))
                c_out = float(out_m.group(1))
                c_in = float(in_m.group(1))
                if z_v != 0 and t_v > 0.0 and c_out > 0.0 and c_in > 0.0:
                    r_c = float(CONSTANTS["R"]["value"])
                    f_c = float(CONSTANTS["F"]["value"])
                    e_mv = (r_c * t_v / (float(z_v) * f_c)) * math.log(c_out / c_in) * 1000.0
                    disp = f"{e_mv:.4f}".rstrip("0").rstrip(".")
                    if disp == "-0":
                        disp = "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate Nernst equilibrium potential E = (R*T / (z*F)) * ln([out]/[in]) in mV",
                            "E_{\\text{ion}} = \\frac{RT}{zF} \\ln\\!\\left(\\frac{[\\text{ion}]_{\\text{out}}}{[\\text{ion}]_{\\text{in}}}\\right)",
                            f"E = {disp}\\text{{ mV}}",
                            disp,
                            "mV",
                        )
                    ]
                    receipt = self._build_receipt("electrophysiology", "neuroscience", "nernst_potential", "neuroscience.nernst_potential", {"z": str(z_v), "T": str(t_v), "out": str(c_out), "in": str(c_in)}, disp, disp, "mV", steps)
                    return SolverResult(True, q, "neuroscience", "electrophysiology", "nernst_potential", "neuroscience.nernst_potential", round(e_mv, 6), disp, "mV", steps, receipt, f"Nernst equilibrium potential E = {disp} mV.")

        # 42. Exact Goldman-Hodgkin-Katz (GHK) Membrane Potential (mV):
        # "GHK potential T = 310, pK = 1, K_out = 5, K_in = 140, pNa = 0.04, Na_out = 145, Na_in = 12, pCl = 0.45, Cl_out = 110, Cl_in = 4"
        if "ghk" in lower or "goldman-hodgkin-katz" in lower or "goldman hodgkin katz" in lower:
            t_m = re.search(r"\bT\s*=\s*(-?\d+(?:\.\d+)?)", q)
            pk_m = re.search(r"\bp_?K\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            ko_m = re.search(r"\bK_?out\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            ki_m = re.search(r"\bK_?in\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            pna_m = re.search(r"\bp_?Na\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            nao_m = re.search(r"\bNa_?out\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            nai_m = re.search(r"\bNa_?in\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            pcl_m = re.search(r"\bp_?Cl\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            clo_m = re.search(r"\bCl_?out\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            cli_m = re.search(r"\bCl_?in\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if t_m and pk_m and ko_m and ki_m and pna_m and nao_m and nai_m:
                t_v = float(t_m.group(1))
                pk, ko, ki = float(pk_m.group(1)), float(ko_m.group(1)), float(ki_m.group(1))
                pna, nao, nai = float(pna_m.group(1)), float(nao_m.group(1)), float(nai_m.group(1))
                pcl = float(pcl_m.group(1)) if pcl_m else 0.0
                clo = float(clo_m.group(1)) if clo_m else 1.0
                cli = float(cli_m.group(1)) if cli_m else 1.0
                if (
                    t_v > 0.0
                    and pk >= 0.0
                    and pna >= 0.0
                    and pcl >= 0.0
                    and min(ko, ki, nao, nai, clo, cli) > 0.0
                ):
                    numer = pk * ko + pna * nao + pcl * cli
                    denom = pk * ki + pna * nai + pcl * clo
                    if numer > 0.0 and denom > 0.0:
                        r_c = float(CONSTANTS["R"]["value"])
                        f_c = float(CONSTANTS["F"]["value"])
                        vm_mv = (r_c * t_v / f_c) * math.log(numer / denom) * 1000.0
                        disp = f"{vm_mv:.4f}".rstrip("0").rstrip(".")
                        steps = [
                            DerivationStep(
                                1,
                                "Evaluate Goldman-Hodgkin-Katz voltage equation (note Cl- valence inversion)",
                                "V_m = \\frac{RT}{F} \\ln\\!\\left(\\frac{P_K[K^+]_o + P_{Na}[Na^+]_o + P_{Cl}[Cl^-]_i}{P_K[K^+]_i + P_{Na}[Na^+]_i + P_{Cl}[Cl^-]_o}\\right)",
                                f"V_m = {disp}\\text{{ mV}}",
                                disp,
                                "mV",
                            )
                        ]
                        receipt = self._build_receipt("electrophysiology", "neuroscience", "ghk_potential", "neuroscience.ghk_potential", {"T": str(t_v), "pK": str(pk), "pNa": str(pna), "pCl": str(pcl)}, disp, disp, "mV", steps)
                        return SolverResult(True, q, "neuroscience", "electrophysiology", "ghk_potential", "neuroscience.ghk_potential", round(vm_mv, 6), disp, "mV", steps, receipt, f"GHK resting membrane potential V_m = {disp} mV.")

        # 43. Exact Rall Passive Cable Constants: "Cable constants rm = 4000, ri = 1000, cm = 0.001"
        if "cable constant" in lower or "rall cable" in lower:
            rm_m = re.search(r"\br_?m\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            ri_m = re.search(r"\br_?i\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            cm_m = re.search(r"\bc_?m\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if rm_m and ri_m and cm_m:
                rm_v = self._normalize_num(rm_m.group(1))
                ri_v = self._normalize_num(ri_m.group(1))
                cm_v = self._normalize_num(cm_m.group(1))
                if rm_v > 0 and ri_v > 0 and cm_v > 0:
                    ratio = rm_v / ri_v
                    tau_exact = rm_v * cm_v
                    lam_f = math.sqrt(float(ratio))
                    if ratio.denominator == 1 and math.isqrt(ratio.numerator) ** 2 == ratio.numerator:
                        lam_disp = str(math.isqrt(ratio.numerator))
                    else:
                        lam_disp = f"{lam_f:.6f}".rstrip("0").rstrip(".")
                    tau_disp = self._format_fraction(tau_exact)
                    disp = f"lambda={lam_disp}, tau={tau_disp}"
                    steps = [
                        DerivationStep(
                            1,
                            "Compute passive dendritic cable space constant lambda = sqrt(r_m / r_i) and time constant tau_m = r_m * c_m",
                            "\\lambda = \\sqrt{\\frac{r_m}{r_i}},\\quad \\tau_m = r_m c_m",
                            f"\\lambda = {lam_disp},\\; \\tau_m = {tau_disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("cable_theory", "neuroscience", "cable_constants", "neuroscience.cable_constants", {"rm": str(rm_v), "ri": str(ri_v), "cm": str(cm_v)}, str(tau_exact), disp, "", steps)
                    return SolverResult(True, q, "neuroscience", "cable_theory", "cable_constants", "neuroscience.cable_constants", round(lam_f, 6), disp, "", steps, receipt, f"Passive cable space constant lambda = {lam_disp}, membrane time constant tau = {tau_disp}.")

        # 44. Exact Michaelis-Menten Enzyme Kinetics: "Michaelis-Menten Vmax = 120, Km = 4, S = 12"
        if "michaelis-menten" in lower or "michaelis menten" in lower:
            vm_m = re.search(r"\bV_?max\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            km_m = re.search(r"\bK_?m\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            s_m = re.search(r"\bS\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if vm_m and km_m and s_m:
                vmax_v = self._normalize_num(vm_m.group(1))
                km_v = self._normalize_num(km_m.group(1))
                s_v = self._normalize_num(s_m.group(1))
                if vmax_v > 0 and km_v > 0 and s_v >= 0:
                    v_exact = (vmax_v * s_v) / (km_v + s_v)
                    disp = self._format_fraction(v_exact)
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate exact Michaelis-Menten reaction velocity v = (Vmax * [S]) / (Km + [S])",
                            "v = \\frac{V_{\\max}[S]}{K_m + [S]}",
                            f"v = \\frac{{{vmax_v} \\cdot {s_v}}}{{{km_v} + {s_v}}} = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("enzyme_kinetics", "biochemistry", "michaelis_menten", "biochemistry.michaelis_menten", {"Vmax": str(vmax_v), "Km": str(km_v), "S": str(s_v)}, str(v_exact), disp, "", steps)
                    return SolverResult(True, q, "biochemistry", "enzyme_kinetics", "michaelis_menten", "biochemistry.michaelis_menten", float(v_exact), disp, "", steps, receipt, f"Michaelis-Menten reaction velocity v = {disp}.")

        # 45. Exact Henderson-Hasselbalch Buffer pH: "Henderson-Hasselbalch pKa = 4.76, base = 0.2, acid = 0.02"
        if "henderson-hasselbalch" in lower or "henderson hasselbalch" in lower:
            pka_m = re.search(r"\bpK_?a\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            base_m = re.search(r"\b(?:base|conjugate_base|A_minus)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            acid_m = re.search(r"\b(?:acid|weak_acid|HA)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if pka_m and base_m and acid_m:
                pka_v = float(pka_m.group(1))
                base_v = float(base_m.group(1))
                acid_v = float(acid_m.group(1))
                if base_v > 0.0 and acid_v > 0.0:
                    ph_val = pka_v + math.log10(base_v / acid_v)
                    disp = f"{ph_val:.4f}".rstrip("0").rstrip(".") or "0"
                    steps = [
                        DerivationStep(
                            1,
                            "Apply Henderson-Hasselbalch buffer equation pH = pKa + log10([base]/[acid])",
                            "\\text{pH} = \\text{p}K_a + \\log_{10}\\!\\left(\\frac{[\\text{A}^-]}{[\\text{HA}]}\\right)",
                            f"\\text{{pH}} = {pka_v} + \\log_{{10}}({base_v}/{acid_v}) = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("acid_base", "chemistry", "henderson_hasselbalch", "chemistry.henderson_hasselbalch", {"pKa": str(pka_v), "base": str(base_v), "acid": str(acid_v)}, disp, disp, "pH", steps)
                    return SolverResult(True, q, "chemistry", "acid_base", "henderson_hasselbalch", "chemistry.henderson_hasselbalch", round(ph_val, 6), disp, "pH", steps, receipt, f"Henderson-Hasselbalch buffer pH = {disp}.")

        # 46. Exact Shannon-Hartley Channel Capacity (bps): "Shannon-Hartley bandwidth = 1000, snr = 15"
        if "shannon-hartley" in lower or "shannon hartley" in lower:
            bw_m = re.search(r"\b(?:bandwidth|B)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            snr_m = re.search(r"\bsnr\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if bw_m and snr_m:
                bw_v = float(bw_m.group(1))
                snr_v = float(snr_m.group(1))
                if bw_v > 0.0 and snr_v >= 0.0:
                    cap = bw_v * math.log2(1.0 + snr_v)
                    disp = str(int(round(cap))) if abs(cap - round(cap)) < 1e-9 else f"{cap:.4f}".rstrip("0").rstrip(".")
                    steps = [
                        DerivationStep(
                            1,
                            "Evaluate Shannon-Hartley AWGN channel capacity C = B * log2(1 + SNR)",
                            "C = B \\log_2(1 + \\text{SNR})",
                            f"C = {bw_v} \\cdot \\log_2(1 + {snr_v}) = {disp}\\text{{ bps}}",
                            disp,
                            "bps",
                        )
                    ]
                    receipt = self._build_receipt("information_theory", "mathematics", "shannon_hartley", "infotheory.shannon_hartley", {"bandwidth": str(bw_v), "snr": str(snr_v)}, disp, disp, "bps", steps)
                    return SolverResult(True, q, "mathematics", "information_theory", "shannon_hartley", "infotheory.shannon_hartley", round(cap, 6), disp, "bps", steps, receipt, f"Shannon-Hartley channel capacity C = {disp} bps.")

        # 47. Exact M/M/1 Queueing Steady-State Mean System Size: "M/M/1 queue lambda = 3, mu = 5"
        if "m/m/1" in lower or "mm1 queue" in lower:
            lam_m = re.search(r"\b(?:lambda|arrival)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            mu_m = re.search(r"\b(?:mu|service)\s*=\s*(-?\d+(?:\.\d+)?)", q, re.I)
            if lam_m and mu_m:
                lam_v = self._normalize_num(lam_m.group(1))
                mu_v = self._normalize_num(mu_m.group(1))
                if lam_v > 0 and mu_v > 0 and lam_v < mu_v:
                    rho_exact = lam_v / mu_v
                    l_exact = lam_v / (mu_v - lam_v)
                    w_exact = Fraction(1) / (mu_v - lam_v)
                    disp = self._format_fraction(l_exact)
                    w_disp = self._format_fraction(w_exact)
                    rho_disp = self._format_fraction(rho_exact)
                    steps = [
                        DerivationStep(
                            1,
                            "Verify stability rho = lambda/mu < 1 and compute steady-state mean customers L = lambda / (mu - lambda)",
                            "\\rho = \\frac{\\lambda}{\\mu} < 1,\\quad L = \\frac{\\lambda}{\\mu - \\lambda},\\quad W = \\frac{1}{\\mu - \\lambda}",
                            f"\\rho = {rho_disp},\\; L = {disp},\\; W = {w_disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("queueing_theory", "computer_science", "mm1_queue", "cs.mm1_queue", {"lambda": str(lam_v), "mu": str(mu_v)}, str(l_exact), disp, "customers", steps)
                    return SolverResult(True, q, "computer_science", "queueing_theory", "mm1_queue", "cs.mm1_queue", float(l_exact), disp, "customers", steps, receipt, f"M/M/1 steady-state mean customers L = {disp} (utilization rho = {rho_disp}, mean sojourn W = {w_disp}).")

        # 48. Exact 2-State Ergodic Markov Stationary Distribution: "Markov stationary distribution of [[0.7, 0.3], [0.4, 0.6]]"
        if "markov" in lower and "stationary" in lower:
            row_matches = re.findall(r"\[\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:/\d+)?)\s*\]", q)
            if len(row_matches) == 2:
                try:
                    p00, p01 = self._normalize_num(row_matches[0][0]), self._normalize_num(row_matches[0][1])
                    p10, p11 = self._normalize_num(row_matches[1][0]), self._normalize_num(row_matches[1][1])
                except Exception:
                    p00 = Fraction(-1)
                    p01 = p10 = p11 = Fraction(0)
                if (
                    min(p00, p01, p10, p11) >= 0
                    and p00 + p01 == Fraction(1)
                    and p10 + p11 == Fraction(1)
                    and (p01 + p10) > 0
                ):
                    pi0 = p10 / (p01 + p10)
                    pi1 = p01 / (p01 + p10)
                    disp = f"[{self._format_fraction(pi0)}, {self._format_fraction(pi1)}]"
                    steps = [
                        DerivationStep(
                            1,
                            "Solve pi * P = pi with pi_0 + pi_1 = 1: pi = [p10/(p01+p10), p01/(p01+p10)]",
                            "\\pi = \\left[\\frac{p_{10}}{p_{01}+p_{10}},\\; \\frac{p_{01}}{p_{01}+p_{10}}\\right]",
                            f"\\pi = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("stochastic_processes", "mathematics", "markov_stationary", "math.markov_stationary", {"P": f"[[{p00},{p01}],[{p10},{p11}]]"}, f"{pi0},{pi1}", disp, "", steps)
                    return SolverResult(True, q, "mathematics", "stochastic_processes", "markov_stationary", "math.markov_stationary", float(pi0), disp, "", steps, receipt, f"Stationary distribution pi = {disp}.")

        # 49. Exact Quadratic Form x^T A x: "Quadratic form x = [1, 2], A = [[3, 1], [1, 2]]"
        if "quadratic form" in lower:
            x_m = re.search(r"\bx\s*=\s*\[([^\[\]]+)\]", q, re.I)
            a_m = re.search(r"\bA\s*=\s*(\[\s*\[.*?\]\s*\])", q, re.I)
            if x_m and a_m:
                try:
                    x_vec = [self._normalize_num(tok.strip()) for tok in x_m.group(1).split(",") if tok.strip()]
                    row_strs = re.findall(r"\[([^\[\]]+)\]", a_m.group(1))
                    a_mat = [
                        [self._normalize_num(tok.strip()) for tok in r_s.split(",") if tok.strip()]
                        for r_s in row_strs
                    ]
                except Exception:
                    x_vec, a_mat = [], []
                n_q = len(x_vec)
                if 1 <= n_q <= 8 and len(a_mat) == n_q and all(len(row) == n_q for row in a_mat):
                    qf_val = Fraction(0)
                    for i_q in range(n_q):
                        for j_q in range(n_q):
                            qf_val += x_vec[i_q] * a_mat[i_q][j_q] * x_vec[j_q]
                    disp = self._format_fraction(qf_val)
                    steps = [
                        DerivationStep(
                            1,
                            f"Evaluate exact {n_q}x{n_q} quadratic form x^T A x = sum_{{i,j}} x_i A_{{ij}} x_j",
                            "q(x) = x^\\top A x = \\sum_{i=1}^{n}\\sum_{j=1}^{n} x_i A_{ij} x_j",
                            f"x^\\top A x = {disp}",
                            disp,
                        )
                    ]
                    receipt = self._build_receipt("linear_algebra", "mathematics", "quadratic_form", "linalg.quadratic_form", {"n": str(n_q)}, str(qf_val), disp, "", steps)
                    return SolverResult(True, q, "mathematics", "linear_algebra", "quadratic_form", "linalg.quadratic_form", float(qf_val), disp, "", steps, receipt, f"Quadratic form x^T A x = {disp}.")

        # 50. Exact European Option Put-Call Parity: "Put-call parity S = 100, K = 100, r = 0.05, T = 1, P = 5.57"
        if "put-call parity" in lower or "put call parity" in lower:
            s_m = re.search(r"\bS\s*=\s*(-?\d+(?:\.\d+)?)", q)
            k_m = re.search(r"\bK\s*=\s*(-?\d+(?:\.\d+)?)", q)
            r_m = re.search(r"\br\s*=\s*(-?\d+(?:\.\d+)?)", q)
            t_m = re.search(r"\bT\s*=\s*(-?\d+(?:\.\d+)?)", q)
            p_m = re.search(r"\bP\s*=\s*(-?\d+(?:\.\d+)?)", q)
            c_m = re.search(r"\bC\s*=\s*(-?\d+(?:\.\d+)?)", q)
            if s_m and k_m and r_m and t_m and (p_m is not None) ^ (c_m is not None):
                s_v = float(s_m.group(1))
                k_v = float(k_m.group(1))
                r_v = float(r_m.group(1))
                t_v = float(t_m.group(1))
                pv_k = k_v * math.exp(-r_v * t_v)
                if s_v > 0.0 and k_v > 0.0 and t_v >= 0.0:
                    if p_m is not None:
                        p_v = float(p_m.group(1))
                        if p_v >= 0.0:
                            c_fair = p_v + s_v - pv_k
                            if c_fair >= -1e-9:
                                c_fair = max(0.0, c_fair)
                                disp = f"{c_fair:.4f}".rstrip("0").rstrip(".") or "0"
                                steps = [
                                    DerivationStep(
                                        1,
                                        "Apply European put-call parity C = P + S - K * exp(-r*T)",
                                        "C = P + S - K e^{-rT}",
                                        f"C = {p_v} + {s_v} - {k_v} e^{{-{r_v}\\cdot {t_v}}} = {disp}",
                                        disp,
                                    )
                                ]
                                receipt = self._build_receipt("derivatives", "mathematics", "put_call_parity", "finance.put_call_parity", {"S": str(s_v), "K": str(k_v), "r": str(r_v), "T": str(t_v), "P": str(p_v)}, disp, disp, "currency", steps)
                                return SolverResult(True, q, "mathematics", "derivatives", "put_call_parity", "finance.put_call_parity", round(c_fair, 6), disp, "", steps, receipt, f"By European put-call parity, fair call value C = {disp}.")
                    else:
                        assert c_m is not None
                        c_v = float(c_m.group(1))
                        if c_v >= 0.0:
                            p_fair = c_v - s_v + pv_k
                            if p_fair >= -1e-9:
                                p_fair = max(0.0, p_fair)
                                disp = f"{p_fair:.4f}".rstrip("0").rstrip(".") or "0"
                                steps = [
                                    DerivationStep(
                                        1,
                                        "Apply European put-call parity P = C - S + K * exp(-r*T)",
                                        "P = C - S + K e^{-rT}",
                                        f"P = {c_v} - {s_v} + {k_v} e^{{-{r_v}\\cdot {t_v}}} = {disp}",
                                        disp,
                                    )
                                ]
                                receipt = self._build_receipt("derivatives", "mathematics", "put_call_parity", "finance.put_call_parity", {"S": str(s_v), "K": str(k_v), "r": str(r_v), "T": str(t_v), "C": str(c_v)}, disp, disp, "currency", steps)
                                return SolverResult(True, q, "mathematics", "derivatives", "put_call_parity", "finance.put_call_parity", round(p_fair, 6), disp, "", steps, receipt, f"By European put-call parity, fair put value P = {disp}.")

        return None

    def _build_receipt(
        self,
        scenario: str,
        domain: str,
        target: str,
        formula_id: str,
        inputs: Dict[str, str],
        raw_result_fraction: str,
        display_result: str,
        unit: str,
        steps: List[DerivationStep],
    ) -> SolverReceipt:
        """Construct a deterministic cryptographic audit receipt."""
        derivation_text = json.dumps([s.to_dict() for s in steps], sort_keys=True)
        deriv_hash = hashlib.sha256(derivation_text.encode("utf-8")).hexdigest()

        payload = {
            "schema_version": SOLVER_RECEIPT_SCHEMA,
            "scenario": scenario,
            "domain": domain,
            "target": target,
            "formula_id": formula_id,
            "inputs": inputs,
            "raw_result_fraction": raw_result_fraction,
            "display_result": display_result,
            "unit": unit,
            "derivation_hash": deriv_hash,
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        receipt_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return SolverReceipt(
            schema_version=SOLVER_RECEIPT_SCHEMA,
            scenario=scenario,
            domain=domain,
            target=target,
            formula_id=formula_id,
            inputs=inputs,
            raw_result_fraction=raw_result_fraction,
            display_result=display_result,
            unit=unit,
            derivation_hash=deriv_hash,
            receipt_sha256=receipt_sha256,
        )


_DEFAULT_SOLVER = NexusSolver()


def solve_problem(query: str) -> SolverResult:
    """Convenience functional interface to NexusSolver."""
    return _DEFAULT_SOLVER.solve(query)
