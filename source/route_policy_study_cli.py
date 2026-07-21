"""Export a read-only adjacent-route rehearsal charter as JSON.

The CLI consumes an explicit prompt-free post-filter support snapshot.  It does
not discover models, sample an assignment, touch the route ledger, or execute a
route.  Redirect stdout or use ``--output`` to retain the rehearsal artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from .route_policy_explorer import plan_adjacent_route_study
except ImportError:  # Direct ``python source/route_policy_study_cli.py`` use.
    from route_policy_explorer import plan_adjacent_route_study


_INPUT_KEYS = {
    "baseline_mode",
    "source_contract",
    "post_filter_candidates",
    "post_filter_exclusions",
    "exploration_rate",
    "planned_routes",
    "scenario_confidence",
    "assumed_feedback_rate",
    "target_observed_labels",
}


def _example_payload() -> Dict[str, Any]:
    return {
        "baseline_mode": "collective",
        "source_contract": {
            "policy_id": "auto-route-v2",
            "policy_version": "2.0.0",
            "feature_schema_version": "route-context-v1",
            "support_schema_version": "route-support-v1",
            "candidate_set_hash": "a" * 64,
            "distribution_hash": "b" * 64,
            "outcome_contract_schema_version": "route-outcome-contract-v1",
        },
        "post_filter_candidates": [
            {
                "action": "off",
                "estimated_cost_units": 1.0,
                "estimated_model_calls": 1,
                "planned_loop_steps": 0,
                "latency_tier": "low",
                "selected": False,
            },
            {
                "action": "collective",
                "estimated_cost_units": 3.0,
                "estimated_model_calls": 3,
                "planned_loop_steps": 0,
                "latency_tier": "moderate",
                "selected": True,
            },
            {
                "action": "loop",
                "estimated_cost_units": 12.0,
                "estimated_model_calls": 12,
                "planned_loop_steps": 4,
                "latency_tier": "frontier",
                "selected": False,
            },
            {
                "action": "collective_loop",
                "estimated_cost_units": 24.0,
                "estimated_model_calls": 24,
                "planned_loop_steps": 4,
                "latency_tier": "frontier",
                "selected": False,
            },
        ],
        "post_filter_exclusions": [],
        "exploration_rate": 0.10,
        "planned_routes": 2_000,
        "scenario_confidence": 0.95,
        "assumed_feedback_rate": 0.30,
        "target_observed_labels": 20,
    }


def _load_payload(path: str) -> Dict[str, Any]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(path).read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    unknown = set(payload) - _INPUT_KEYS
    if unknown:
        raise ValueError(
            "input contains unsupported or non-prompt-free fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    missing = {
        "baseline_mode",
        "source_contract",
        "post_filter_candidates",
        "post_filter_exclusions",
    } - set(payload)
    if missing:
        raise ValueError("input is missing required fields: " + ", ".join(sorted(missing)))
    return payload


def build_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the CLI payload and return the shared pure planner result."""

    return plan_adjacent_route_study(
        payload["baseline_mode"],
        payload["post_filter_candidates"],
        payload["post_filter_exclusions"],
        source_contract=payload["source_contract"],
        exploration_rate=payload.get("exploration_rate", 0.10),
        planned_routes=payload.get("planned_routes", 2_000),
        scenario_confidence=payload.get("scenario_confidence", 0.95),
        assumed_feedback_rate=payload.get("assumed_feedback_rate", 0.30),
        target_observed_labels=payload.get("target_observed_labels", 20),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a prompt-free, non-executing adjacent-route study rehearsal."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSON support snapshot path, or - for stdin")
    source.add_argument(
        "--example",
        action="store_true",
        help="render a documented synthetic example; it is not live route evidence",
    )
    parser.add_argument("--output", help="optional JSON output path; stdout is always supported")
    parser.add_argument("--compact", action="store_true", help="emit compact canonical JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = _example_payload() if args.example else _load_payload(str(args.input))
        plan = build_plan(payload)
        rendered = json.dumps(
            plan,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":") if args.compact else None,
            indent=None if args.compact else 2,
        )
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except (OSError, ValueError) as exc:
        print(f"route study error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
