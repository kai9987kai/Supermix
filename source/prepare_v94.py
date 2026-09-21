"""Prepare a frozen v94 experiment bundle; never train, load or promote a model.

Optional corpus/evaluation JSONL rows require row_id, semantic_id, family,
prompt and response. semantic_id must come from the task generator and group
all paraphrases of one problem; this tool cannot infer semantic equivalence.
No corpus is required to write the research/readiness plan while v93 runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import unicodedata


ROOT = Path(__file__).resolve().parents[1]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed during hashing: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": hasher.hexdigest()}


def _prompt_key(prompt):
    return " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())


def _rows(rows):
    indexed, prompts, groups = {}, {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("corpus rows must be objects")
        for field in ("row_id", "semantic_id", "family", "prompt", "response"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"row requires a nonempty {field}")
        key = row["row_id"]
        if key in indexed:
            raise ValueError(f"duplicate row_id: {key}")
        group = row["semantic_id"]
        if group in groups and groups[group] != row["family"]:
            raise ValueError("semantic group spans different families")
        groups[group] = row["family"]
        prompt = _prompt_key(row["prompt"])
        if prompt in prompts and prompts[prompt] != group:
            raise ValueError("equivalent normalized prompts use different semantic groups")
        prompts[prompt] = group
        indexed[key] = {"row_id": key, "semantic_id": group, "family": row["family"],
                        "content_sha256": digest(row)}
    if not indexed:
        raise ValueError("empty corpus")
    return indexed, prompts, groups


def freeze_cohorts(training_rows, evaluation_rows, *, seed=94, calibration_fraction=0.1,
                   dev_fraction=0.1) -> dict:
    """Group-disjoint, family-stratified cohorts with externally frozen evaluation.

Calibration is adaptively used training material, not held-out evidence.
The manifest fingerprints membership and content, not just a file name.
"""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value in (calibration_fraction, dev_fraction):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 0.5:
            raise ValueError("split fractions must be finite and in (0, 0.5)")
    train, prompts, groups = _rows(training_rows)
    final, final_prompts, final_groups = _rows(evaluation_rows)
    if train.keys() & final.keys() or groups.keys() & final_groups.keys() or prompts.keys() & final_prompts.keys():
        raise ValueError("training/evaluation overlap in row IDs, semantic groups or normalized prompts")
    roles = {"gradient_train": [], "growth_calibration": [], "selection_dev": [],
             "final_eval": list(final.values())}
    group_roles = {}
    for family in sorted(set(groups.values())):
        keys = [key for key, label in groups.items() if label == family]
        keys.sort(key=lambda key: (digest([seed, family, key]), key))
        n_cal = max(1, math.ceil(len(keys) * calibration_fraction))
        n_dev = max(1, math.ceil(len(keys) * dev_fraction))
        if n_cal + n_dev >= len(keys):
            raise ValueError(f"family {family!r} needs more independent semantic groups")
        for index, key in enumerate(keys):
            group_roles[key] = ("growth_calibration" if index < n_cal else
                                "selection_dev" if index < n_cal + n_dev else "gradient_train")
    for row in train.values():
        roles[group_roles[row["semantic_id"]]].append(row)
    cohorts = {}
    for role, rows in roles.items():
        rows.sort(key=lambda row: row["row_id"])
        cohorts[role] = {"membership_sha256": digest(rows), "row_count": len(rows),
                         "group_count": len({row["semantic_id"] for row in rows}),
                         "rows": rows}
    payload = {"schema": "supermix-v94-cohorts-v1", "seed": seed,
               "calibration_fraction": calibration_fraction, "dev_fraction": dev_fraction,
               "semantic_equivalence": "generator-supplied groups plus normalized exact prompt check",
               "cohorts": cohorts}
    return {**payload, "manifest_sha256": digest(payload)}


def experiment_plan() -> dict:
    return {
        "schema": "supermix-v94-experiment-plan-v1",
        "status": "prepared_not_launchable",
        "hypothesis": "Mature utility-guided growth improves learning/retention at a bounded active budget.",
        "arms": [
            {"id": "A", "treatment": "fixed-topology continuation of completed v93"},
            {"id": "B", "treatment": "v93 growth heuristics on training calibration"},
            {"id": "C", "treatment": "v94 maturity, utility and family witness guards"},
        ],
        "common": ["identical parent checkpoint/tokenizer hashes", "identical initial capacity/top-k",
                   "same ordered training rows, supervised-token budget and old/new mixture schedule",
                   "same optimizer/scheduler and paired seeds 94, 95, 96",
                   "same training-calibration membership and diagnostic cadence",
                   "external final evaluation frozen before training"],
        "comparisons": {"C_vs_B": "policy package", "B_vs_A": "growth heuristic package",
                        "C_vs_A": "combined policy benefit"},
        "reported": ["paired old-family accuracy and regression counts", "new-family accuracy",
                     "per-family token loss", "tokens/second and total wall-clock including probes",
                     "peak memory and allocated versus active capacity", "birth shock and newborn gradient/update norms",
                     "all-gates-off, commissure and per-hemisphere deletion sensitivity"],
        "limitations": ["B changes v93 feedback from dev to training calibration; it is not an exact rerun",
                        "C_vs_B combines maturity, utility and witness rules; split factors if it wins",
                        "grown-component deletion is not a no-growth training counterfactual",
                        "masked dense recurrence does not promise sparsity speedups",
                        "paired percentile intervals are descriptive; no automatic promotion threshold"],
        "deferred": ["signed-gradient edge proposals", "learned cross-hemisphere gates",
                     "random newcomer versus cloned-child ablation", "static final-capacity control"],
        "launch_enabled": False,
    }


def _jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{number}") from error


def prepare(output: Path, *, corpus=None, evaluation=None, parent_checkpoint=None) -> dict:
    if bool(corpus) != bool(evaluation):
        raise ValueError("provide both corpus and evaluation to freeze cohorts")
    if output.exists():
        raise FileExistsError("use a new output directory; frozen bundles are not overwritten")
    manifest = freeze_cohorts(_jsonl(corpus), _jsonl(evaluation)) if corpus else None
    parent = file_identity(Path(parent_checkpoint)) if parent_checkpoint else None
    # These are observed source hashes, not a reconstruction of another run's code.
    names = ["source/mimomix_core.py", "source/neurogenesis.py",
             "source/train_mimomix_generalisation.py", "source/train_mimomix_talk.py",
             "source/v94_growth_policy.py", "source/v94_connectome_audit.py", "source/prepare_v94.py",
             "test_v94_growth_policy.py", "test_v94_connectome_audit.py", "test_prepare_v94.py",
             "docs/V94_RESEARCH_REVIEW.md", "docs/V94_PREPARATION.md"]
    identities = {name: file_identity(ROOT / name) for name in names if (ROOT / name).is_file()}
    blockers = ["trainer integration and transaction rollback are intentionally not installed",
                "completed parent run receipt, exact tokenizer and parent evaluation need verification",
                "utility probes and matched compute preflight are not yet measured"]
    if parent is None:
        blockers.append("completed v93 parent checkpoint has not been bound")
    if manifest is None:
        blockers.append("training calibration and external evaluation memberships have not been frozen")
    report = {"schema": "supermix-v94-readiness-v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "research_and_tools_prepared", "launch_enabled": False,
              "parent_checkpoint": parent, "observed_source_files": identities,
              "cohort_manifest_sha256": manifest["manifest_sha256"] if manifest else None,
              "blockers": blockers,
              "training_started": False, "active_pointer_modified": False}
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"readiness.json": report, "experiment_plan.json": experiment_plan()}
    if manifest:
        artifacts["cohort_manifest.json"] = manifest
    for name, value in artifacts.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    args = parser.parse_args()
    report = prepare(args.output, corpus=args.corpus, evaluation=args.evaluation,
                     parent_checkpoint=args.parent_checkpoint)
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
                      "launch_enabled": report["launch_enabled"], "blockers": report["blockers"]}, indent=2))


if __name__ == "__main__":
    main()
