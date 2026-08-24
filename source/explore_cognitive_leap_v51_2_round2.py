"""Run a predeclared development-only interpolation search for v51.2.

This round consumes only the already-burned development cohort. It starts from
the exact unpromoted v51.1 candidate and interpolates toward the strongest
round-one v51.2 soup. No final seed, activation pointer, catalog entry, or Store
package is touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

import run_cognitive_leap_v51_2 as runner


SCHEMA = "supermix-cognitive-leap-v51.2-round2-development-v1"
BETAS = (0.25, 0.50, 0.75)
ROUND_ONE_ALPHA = 0.30


def _bound_json(path: Path) -> dict[str, Any]:
    return {
        "path": runner.relative_path(path),
        "sha256": runner.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_parent(parent: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = parent / "protocol.json"
    selection_path = parent / "selection_receipt.json"
    replay_path = parent / "development_replay_verification.json"
    protocol = runner.load_json_strict(protocol_path)
    selection = runner.load_json_strict(selection_path)
    replay = runner.load_json_strict(replay_path)
    if (
        protocol.get("protocol_sha256") != runner.protocol_digest(protocol)
        or protocol.get("evaluation_profile_sha256")
        != runner.canonical_evaluation_profile_sha256()
        or selection.get("selection_sha256") != runner.selection_digest(selection)
        or selection.get("passed") is not False
        or selection.get("decision") != "no_development_candidate_passed"
        or selection.get("selected", {}).get("baseline_blend_alpha")
        != ROUND_ONE_ALPHA
        or replay.get("passed") is not False
        or replay.get("protocol_sha256") != protocol["protocol_sha256"]
        or replay.get("selection_sha256") != selection["selection_sha256"]
    ):
        raise ValueError("Round-one rejection evidence is missing or changed")
    replay_payload = dict(replay)
    verification_id = replay_payload.pop("verification_id", None)
    if verification_id != runner.sha256_bytes(
        runner.canonical_json_bytes(replay_payload)
    ):
        raise ValueError("Round-one replay verification digest mismatch")
    return protocol, selection


def run_search(parent: Path, output_dir: Path, device: torch.device) -> Path:
    protocol, selection = _validate_parent(parent)
    if output_dir.exists():
        raise FileExistsError(f"Round-two output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    specification = {
        "schema": SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(runner.AUTHORITY),
        "claim_scope": dict(runner.CLAIM_SCOPE),
        "cohort_role": "development",
        "seeds": list(runner.DEV_SEEDS),
        "samples_per_seed": 2_000,
        "betas": list(BETAS),
        "round_one_alpha": ROUND_ONE_ALPHA,
        "release_continuity_criteria": dict(runner.DEVELOPMENT_CRITERIA),
        "prior_candidate_superiority_criteria": dict(
            runner.PRIOR_CANDIDATE_CRITERIA
        ),
        "parent": {
            "protocol": _bound_json(parent / "protocol.json"),
            "protocol_content_sha256": protocol["protocol_sha256"],
            "selection": _bound_json(parent / "selection_receipt.json"),
            "selection_content_sha256": selection["selection_sha256"],
            "development_replay": _bound_json(
                parent / "development_replay_verification.json"
            ),
        },
        "code": {
            "runner_sha256": runner.sha256_file(
                Path(runner.__file__).resolve()
            ),
            "generator_sha256": runner.sha256_file(
                runner.SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
            ),
            "search_sha256": runner.sha256_file(Path(__file__).resolve()),
        },
    }
    specification["specification_sha256"] = runner.sha256_bytes(
        runner.canonical_json_bytes(specification)
    )
    runner.write_json_exclusive(output_dir / "search_specification.json", specification)

    baseline_state = runner.load_state(
        runner.resolve_repo_relative(str(protocol["baseline"]["path"]))
    )
    prior_state = runner.load_state(
        runner.resolve_repo_relative(str(protocol["prior_candidate"]["path"]))
    )
    member_states = {
        name: runner.load_state(
            runner.resolve_repo_relative(str(receipt["artifact"]["path"]))
        )
        for name, receipt in selection["member_receipts"].items()
    }
    round_one_members = [str(value) for value in selection["selected"]["members"]]
    round_one_soup = runner.average_states(
        [member_states[name] for name in round_one_members],
        selection["selected"]["member_weights"],
    )
    round_one_state = runner.blend_with_baseline(
        baseline_state,
        round_one_soup,
        ROUND_ONE_ALPHA,
    )
    cohort = runner.build_cohort(
        runner.DEV_SEEDS,
        2_000,
        cohort_role="development",
    )
    evaluator = runner.ChampionNetCognitiveLeapUltraExpert().to(device)
    evaluator.load_state_dict(baseline_state, strict=True)
    baseline_predictions = runner.predict_cohort(evaluator, cohort, device)
    evaluator.load_state_dict(prior_state, strict=True)
    prior_predictions = runner.predict_cohort(evaluator, cohort, device)

    rows: list[dict[str, Any]] = []
    states: dict[float, dict[str, torch.Tensor]] = {}
    for beta in BETAS:
        state = runner.average_states(
            [prior_state, round_one_state],
            [1.0 - beta, beta],
        )
        states[beta] = state
        evaluator.load_state_dict(state, strict=True)
        predictions = runner.predict_cohort(evaluator, cohort, device)
        release_comparison = runner.compare_predictions(
            baseline_predictions,
            predictions,
            cohort,
            runner.DEVELOPMENT_CRITERIA,
        )
        prior_comparison = runner.compare_predictions(
            prior_predictions,
            predictions,
            cohort,
            runner.PRIOR_CANDIDATE_CRITERIA,
        )
        score = runner.dual_selection_score(release_comparison, prior_comparison)
        row = {
            "name": f"prior_v51_1_to_round1_alpha30__beta_{beta:.2f}",
            "beta": beta,
            "passed": bool(score[0]),
            "selection_score": list(score),
            "comparisons": {
                "release_continuity": release_comparison,
                "prior_candidate_superiority": prior_comparison,
            },
            "canonical_state_sha256": runner.state_dict_summary(state)[
                "canonical_state_sha256"
            ],
        }
        rows.append(row)
        print(
            f"round2 beta={beta:.2f} pass={row['passed']} "
            f"vs_v51={release_comparison['summary']['accuracy_delta']:+.5f} "
            f"vs_v51.1={prior_comparison['summary']['accuracy_delta']:+.5f} "
            f"checks={sum(release_comparison['checks'].values()) + sum(prior_comparison['checks'].values())}/12",
            flush=True,
        )
    selected = max(rows, key=lambda row: tuple(row["selection_score"]))
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(runner.AUTHORITY),
        "specification": _bound_json(output_dir / "search_specification.json"),
        "specification_content_sha256": specification["specification_sha256"],
        "development_dataset_sha256": cohort["dataset_sha256"],
        "candidates": rows,
        "selected": selected,
        "passed": bool(selected["passed"]),
        "decision": (
            "development_candidate_found"
            if selected["passed"]
            else "no_development_candidate_passed"
        ),
    }
    if selected["passed"]:
        artifact = runner.save_state(
            output_dir / "selected" / "cognitive_leap_ultra_v51_2_round2.pth",
            states[float(selected["beta"])],
        )
        receipt["selected"] = {**selected, "artifact": artifact}
    receipt["receipt_id"] = runner.sha256_bytes(
        runner.canonical_json_bytes(receipt)
    )
    receipt_path = output_dir / "round2_development_receipt.json"
    runner.write_json_atomic(receipt_path, receipt)
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    device = runner.configure_runtime(args.torch_threads)
    print(run_search(args.parent.resolve(), args.output_dir.resolve(), device))


if __name__ == "__main__":
    main()
