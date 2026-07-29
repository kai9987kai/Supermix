import json
import os
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(ROOT / "source"))

from materialize_v52_from_v50 import materialize  # noqa: E402
from model_variants import build_model, detect_model_size_from_state_dict  # noqa: E402


# The donor checkpoint is a `.pth`, which .gitignore excludes, so it is present
# in a working tree that has run the v50 line but absent from a fresh clone.
# Skipping keeps the suite green for anyone who clones; the manifest test below
# still runs, because the manifest itself is checked in.
DONOR = ROOT / "artifacts" / "v52_initialization" / "champion_model_chat_v50_cognitive_leap.pth"


@pytest.mark.skipif(not DONOR.exists(), reason=f"donor checkpoint not present: {DONOR.name}")
def test_materialized_v52_checkpoint_is_deterministic_and_strict_loadable(tmp_path):
    donor = DONOR
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"

    first_manifest = materialize(donor, first, seed=52)
    second_manifest = materialize(donor, second, seed=52)

    assert first_manifest["state_sha256"] == second_manifest["state_sha256"]
    assert first_manifest["model_size"] == "cognitive_leap_v52_expert"
    state = torch.load(first, map_location="cpu", weights_only=True)
    second_state = torch.load(second, map_location="cpu", weights_only=True)
    assert state.keys() == second_state.keys()
    assert all(torch.equal(state[key], second_state[key]) for key in state)
    assert detect_model_size_from_state_dict(state) == "cognitive_leap_v52_expert"
    build_model("cognitive_leap_v52_expert", dropout=0.0).load_state_dict(
        state, strict=True
    )


def test_checked_in_initialization_manifest_is_explicitly_untrained():
    manifest_path = ROOT / "artifacts" / "v52_initialization" / "v52_initialization_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["model_size"] == "cognitive_leap_v52_expert"
    assert payload["trained_v52"] is False
    assert payload["seed"] == 52
