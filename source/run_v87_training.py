"""Check a prepared v87 bundle; launch only with an explicit --train flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from eval_prompt_robustness import load_evaluation
from prepare_v87_training import SOURCE_FILES, TASKS_V86, sha256_file
from v87_reasoning import digest_json

ROOT = Path(__file__).resolve().parent.parent


def preflight(bundle: Path) -> tuple[dict, list[str]]:
    from train_mimomix_generalisation import build_parser, validate_selection_settings
    from v87_frozen_split import load_frozen_split

    bundle = bundle.resolve()
    manifest, _ = load_evaluation(bundle)
    for name, field in (("train.jsonl", "train_sha256"), ("frozen_split.json", "frozen_split_sha256")):
        if sha256_file(bundle / name) != manifest[field]:
            raise ValueError(f"bundle file changed: {name}")
    if sha256_file(Path(manifest["source"])) != manifest["source_sha256"]:
        raise ValueError("original source corpus changed")
    expected_code = {name: sha256_file(ROOT / "source" / name) for name in SOURCE_FILES}
    if expected_code != manifest["source_code_sha256"]:
        raise ValueError("source code changed since preparation; prepare a fresh bundle")
    command = manifest["trainer_args"]
    if (not isinstance(command, list) or not all(isinstance(t, str) for t in command)
            or digest_json(command) != manifest["trainer_args_sha256"]):
        raise ValueError("training arguments changed since preparation")
    args = build_parser().parse_args(command)
    validate_selection_settings(args)
    if (Path(args.corpus_jsonl).resolve() != bundle / "train.jsonl"
            or Path(args.frozen_split).resolve() != bundle / "frozen_split.json"
            or args.accuracy_task != list(TASKS_V86) or args.seed != manifest["seed"]
            or args.init_from or args.start_step or not args.strict):
        raise ValueError("training arguments do not match the prepared experiment")
    split, proof = load_frozen_split(bundle / "train.jsonl", bundle / "frozen_split.json",
                                     min_response_characters=args.min_response_characters, limit=args.pairs)
    if proof["partition_sha256"] != manifest["partition_sha256"]:
        raise ValueError("training partition differs from preparation")
    return {"checked": True, "rehearsal": manifest["rehearsal"],
            "training_launch_ready": not manifest["rehearsal"],
            "train_rows": len(split.train), "dev_rows": len(split.dev),
            "test_rows": sum(len(rows) for _, rows in split.tiers()),
            "output_dir": str((ROOT / args.output_dir).resolve()),
            "training_started": False, "promotion_authorized": False,
            "note": "Integrity preflight only; it does not grant privacy/provenance clearance or establish model quality."}, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    report, command = preflight(args.bundle)
    print(json.dumps(report, indent=2), flush=True)
    if not args.train:
        return 0
    if report["rehearsal"]:
        raise ValueError("refusing long training on a rehearsal corpus")
    if Path(report["output_dir"]).exists():
        raise FileExistsError("training output already exists; refusing to overwrite or implicitly resume")
    return subprocess.call([sys.executable, str(ROOT / "source" / "train_supervised.py"), "--", *command], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
