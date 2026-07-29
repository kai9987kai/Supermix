"""Create a deterministic, fully materialized v52 initialization checkpoint."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from model_variants import (  # noqa: E402
    build_model,
    detect_model_size_from_state_dict,
    load_weights_for_model,
)
from run import safe_load_state_dict  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _state_sha256(state: dict) -> str:
    """Hash tensor content independently of torch.save container metadata."""
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        digest.update(key.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest().upper()


def materialize(input_path: Path, output_path: Path, seed: int = 52) -> dict:
    torch.manual_seed(int(seed))
    state = safe_load_state_dict(str(input_path))
    source_size = detect_model_size_from_state_dict(state)
    model = build_model("cognitive_leap_v52_expert", dropout=0.0)
    missing, unexpected = load_weights_for_model(
        model,
        state,
        model_size="cognitive_leap_v52_expert",
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Compatibility load failed. Missing={missing}, unexpected={unexpected}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    reloaded = safe_load_state_dict(str(output_path))
    detected = detect_model_size_from_state_dict(reloaded)
    if detected != "cognitive_leap_v52_expert":
        raise RuntimeError(f"Materialized checkpoint detected as {detected!r}")
    check_model = build_model("cognitive_leap_v52_expert", dropout=0.0)
    check_model.load_state_dict(reloaded, strict=True)

    return {
        "created_by": Path(__file__).name,
        "source_checkpoint": str(input_path),
        "source_model_size": source_size,
        "model_size": detected,
        "seed": int(seed),
        "output_checkpoint": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
        "state_sha256": _state_sha256(reloaded),
        "trained_v52": False,
        "purpose": "deterministic initialization for v52 fine-tuning and runtime integration",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth",
    )
    parser.add_argument(
        "--output",
        default="artifacts/v52_initialization/champion_model_chat_v52_initialized.pth",
    )
    parser.add_argument(
        "--manifest",
        default="artifacts/v52_initialization/v52_initialization_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=52)
    args = parser.parse_args()

    payload = materialize(Path(args.input), Path(args.output), seed=args.seed)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
