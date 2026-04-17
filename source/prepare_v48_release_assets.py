from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _default_notes(existing: object) -> list[str]:
    if isinstance(existing, list):
        return [str(item) for item in existing]
    return []


def build_v48_meta(meta_template: Path) -> dict:
    payload = _load_json(meta_template)
    payload["architecture_version"] = 48
    payload["family"] = "omni_collective_v48"
    payload["prompt_understanding_mode"] = "adaptive_got_hierarchical_moe_v48"
    notes = _default_notes(payload.get("notes"))
    release_note = (
        "v48 extends the v47 scaffold with Hierarchical Mixture-of-Experts (H-MoE) routing, "
        "Adaptive Graph-of-Thoughts (AGoT), and DPO-aligned frontier tuning."
    )
    if release_note not in notes:
        notes.append(release_note)
    payload["notes"] = notes
    return payload


def build_v48_summary(summary_template: Path, weights_name: str) -> dict:
    payload = _load_json(summary_template)
    payload["family"] = "omni_collective_v48"
    payload["weights_name"] = weights_name
    payload["release_stage"] = "v48_frontier_bundle"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage Supermix v48 release assets.")
    parser.add_argument("--weights", required=True, help="Path to the v48 weights file to bundle.")
    parser.add_argument("--dest", required=True, help="Destination directory for staged v48 assets.")
    parser.add_argument(
        "--meta-template",
        required=True,
        help="Template metadata JSON used to seed the staged v48 metadata file.",
    )
    parser.add_argument(
        "--summary-template",
        required=True,
        help="Template summary JSON used to seed the staged v48 summary file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    dest_dir = Path(args.dest).expanduser().resolve()
    meta_template = Path(args.meta_template).expanduser().resolve()
    summary_template = Path(args.summary_template).expanduser().resolve()

    if not weights_path.is_file():
        raise SystemExit(f"weights file does not exist: {weights_path}")
    if not meta_template.is_file():
        raise SystemExit(f"meta template does not exist: {meta_template}")
    if not summary_template.is_file():
        raise SystemExit(f"summary template does not exist: {summary_template}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    staged_weights = dest_dir / "omni_collective_v48_frontier.pth"
    staged_meta = dest_dir / "omni_collective_v48_frontier_meta.json"
    staged_summary = dest_dir / "omni_collective_v48_frontier_summary.json"

    shutil.copy2(weights_path, staged_weights)
    _write_json(staged_meta, build_v48_meta(meta_template))
    _write_json(staged_summary, build_v48_summary(summary_template, staged_weights.name))

    print(staged_weights)
    print(staged_meta)
    print(staged_summary)


if __name__ == "__main__":
    main()
