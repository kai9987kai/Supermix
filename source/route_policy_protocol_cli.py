"""Export or audit a prompt-free stateful route experiment preflight.

It can also create a portable review bundle that carries the complete canonical
source plans needed for semantic reconstruction.  Neither surface signs, seals,
assigns, touches the ledger, runs inference, or enables promotion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from .route_policy_protocol import (
        PROTOCOL_BUILD_INPUT_KEYS,
        audit_route_study_review_bundle,
        audit_route_study_protocol,
        build_route_study_protocol_from_input,
        build_route_study_review_bundle_from_input,
    )
    from .route_policy_study_cli import _example_payload, build_plan
except ImportError:  # pragma: no cover - direct ``python source/...`` use
    from route_policy_protocol import (
        PROTOCOL_BUILD_INPUT_KEYS,
        audit_route_study_review_bundle,
        audit_route_study_protocol,
        build_route_study_protocol_from_input,
        build_route_study_review_bundle_from_input,
    )
    from route_policy_study_cli import _example_payload, build_plan


_INPUT_KEYS = set(PROTOCOL_BUILD_INPUT_KEYS)


def _example_protocol_input() -> Dict[str, Any]:
    return {
        "study_plans": [build_plan(_example_payload())],
        "target_policy_profile": "balanced",
        "design_mode": "sticky_session_cluster",
        "carryover_scope": "unknown",
        "interference_scope": "unknown",
        "temporal_variation": "unknown",
        "population_rule_id": "interactive-auto-route-opt-in",
        "population_rule_version": "1",
        "cluster_key_schema_version": "session-hash-v1",
        "planned_clusters": 200,
        "max_routes_per_cluster": 20,
        "analysis_every_clusters": 50,
        "block_length_routes": 20,
        "washout_routes": 0,
    }


def _example_bundle_input() -> Dict[str, Any]:
    first_payload = _example_payload()
    second_payload = _example_payload()
    second_payload["source_contract"] = {
        **second_payload["source_contract"],
        "candidate_set_hash": "c" * 64,
        "distribution_hash": "d" * 64,
    }
    payload = _example_protocol_input()
    payload["study_plans"] = [build_plan(first_payload), build_plan(second_payload)]
    return payload


def _read_json(path: str) -> Any:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input is not valid JSON: {exc.msg}") from exc


def _load_build_input(path: str) -> Dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    unknown = set(payload) - _INPUT_KEYS
    if unknown:
        raise ValueError(
            "input contains unsupported or non-prompt-free fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    if "study_plans" not in payload:
        raise ValueError("input is missing required field: study_plans")
    return payload


def build_protocol(payload: Dict[str, Any]) -> Dict[str, Any]:
    return build_route_study_protocol_from_input(payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or audit a prompt-free, fail-closed route experiment preflight."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="protocol-input JSON path, or - for stdin")
    source.add_argument(
        "--example",
        action="store_true",
        help="render a synthetic protocol draft; it is not live study evidence",
    )
    source.add_argument("--audit", help="audit an existing protocol draft JSON path, or -")
    source.add_argument(
        "--bundle-input",
        help="prompt-free protocol-input JSON path to bundle, or - for stdin",
    )
    source.add_argument(
        "--example-bundle",
        action="store_true",
        help="render a synthetic two-stratum semantic-review bundle",
    )
    source.add_argument(
        "--audit-bundle",
        help="fully reconstruct and audit an existing review bundle JSON path, or -",
    )
    parser.add_argument("--output", help="optional JSON output path; stdout is the default")
    parser.add_argument("--compact", action="store_true", help="emit compact canonical JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.audit_bundle:
            result = audit_route_study_review_bundle(_read_json(str(args.audit_bundle)))
        elif args.audit:
            result = audit_route_study_protocol(_read_json(str(args.audit)))
        elif args.example_bundle:
            result = build_route_study_review_bundle_from_input(_example_bundle_input())
        elif args.bundle_input:
            result = build_route_study_review_bundle_from_input(
                _load_build_input(str(args.bundle_input))
            )
        else:
            payload = _example_protocol_input() if args.example else _load_build_input(str(args.input))
            result = build_protocol(payload)
        rendered = json.dumps(
            result,
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
    except (OSError, TypeError, ValueError) as exc:
        print(f"route protocol error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
