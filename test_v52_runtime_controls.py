"""The v52 launcher, the CLI parsers, and the model forward must agree.

The v52 model was merged into main before its control surface was, so
`launch_v52_unified_chat.ps1` passed `--core_top_k 2` to a parser that did not
define it and the launcher aborted with argparse exit code 2. These tests pin
the three contracts that failure crossed:

1. every flag the launcher passes is accepted by the app it launches;
2. the controls actually reach the v52 forward; and
3. pre-v52 checkpoints are bit-identical whether or not the controls are set.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
LAUNCHER = ROOT / "launch_v52_unified_chat.ps1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chat_app():
    sys.path.insert(0, str(SOURCE))
    try:
        yield _load("v52_controls_chat_app", SOURCE / "chat_app.py")
    finally:
        sys.path.remove(str(SOURCE))


@pytest.fixture(scope="module")
def model_variants():
    sys.path.insert(0, str(SOURCE))
    try:
        yield _load("v52_controls_model_variants", SOURCE / "model_variants.py")
    finally:
        sys.path.remove(str(SOURCE))


def _launcher_flags() -> list[str]:
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r'"(--[a-z0-9_]+)"', text)))


def _launcher_target() -> str:
    text = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'"source[\\/]([a-z0-9_]+\.py)"', text)
    assert match, "launcher no longer names a source entrypoint"
    return match.group(1)


def test_launcher_passes_at_least_one_v52_flag() -> None:
    """Guards the guard: if the launcher stops exercising v52, say so loudly."""

    assert "--core_top_k" in _launcher_flags()
    assert _launcher_target() == "chat_web_app.py"


def test_every_launcher_flag_is_accepted_by_its_target_app() -> None:
    """The exact failure that shipped: a launcher flag the parser rejects."""

    target = SOURCE / _launcher_target()
    help_text = subprocess.run(
        [sys.executable, str(target), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
    ).stdout

    unknown = [flag for flag in _launcher_flags() if flag not in help_text]
    assert not unknown, f"{target.name} does not accept launcher flags: {unknown}"


def test_v52_controls_reach_the_forward(chat_app, model_variants) -> None:
    torch.manual_seed(0)
    model = model_variants.build_model("cognitive_leap_v52_expert", dropout=0.0)
    model.eval()
    x = torch.randn(1, 1, 128)

    with torch.no_grad():
        _, dense = chat_app.forward_with_runtime_compute(
            model, x, reasoning_cycles=3, adaptive_compute=True
        )
        _, sparse = chat_app.forward_with_runtime_compute(
            model, x, reasoning_cycles=3, adaptive_compute=True, core_top_k=2
        )
        _, verified = chat_app.forward_with_runtime_compute(
            model,
            x,
            reasoning_cycles=3,
            adaptive_compute=True,
            verifier_adaptive_compute=True,
            verifier_continue_threshold=0.1,
        )

    assert dense["core_routing_mode"] == "dense"
    assert dense["core_top_k"] is None
    assert sparse["core_routing_mode"] == "sparse"
    assert sparse["core_top_k"] == 2
    assert verified["verifier_adaptive_compute"] is True
    assert dense["verifier_adaptive_compute"] is False

    metrics = chat_app.collect_runtime_compute_metrics(model)
    for key in (
        "router_load_balance",
        "active_cores",
        "quality_score",
        "continue_probability",
        "calibrated_entropy",
    ):
        assert metrics[key] is not None, f"v52 telemetry {key} was dropped"


def test_out_of_range_control_values_are_clamped_not_crashed(chat_app, model_variants) -> None:
    torch.manual_seed(0)
    model = model_variants.build_model("cognitive_leap_v52_expert", dropout=0.0)
    model.eval()
    x = torch.randn(1, 1, 128)

    with torch.no_grad():
        _, diagnostics = chat_app.forward_with_runtime_compute(
            model,
            x,
            reasoning_cycles=2,
            adaptive_compute=True,
            core_top_k=9999,
            verifier_adaptive_compute=True,
            verifier_continue_threshold=7.5,
            max_verifier_cycles=10**9,
        )

    assert 0 < diagnostics["core_top_k"] <= chat_app.MAX_RUNTIME_CORE_TOP_K
    assert 0.0 <= diagnostics["verifier_continue_threshold"] <= 1.0
    assert 0 <= diagnostics["max_verifier_cycles"] <= chat_app.MAX_RUNTIME_REASONING_CYCLES

    for bad in ("not-a-number", None, float("nan")):
        with torch.no_grad():
            chat_app.forward_with_runtime_compute(
                model,
                x,
                reasoning_cycles=1,
                adaptive_compute=True,
                core_top_k=bad,
                verifier_continue_threshold=bad,
                max_verifier_cycles=bad,
            )


def test_pre_v52_checkpoints_are_bit_identical_with_the_new_controls(
    chat_app, model_variants
) -> None:
    """Legacy variants must not even notice the v52 kwargs exist."""

    torch.manual_seed(0)
    model = model_variants.build_model("cognitive_leap_ultra_expert", dropout=0.0)
    model.eval()
    x = torch.randn(1, 1, 128)

    with torch.no_grad():
        baseline, _ = chat_app.forward_with_runtime_compute(
            model, x, reasoning_cycles=3, adaptive_compute=True
        )
        with_controls, diagnostics = chat_app.forward_with_runtime_compute(
            model,
            x,
            reasoning_cycles=3,
            adaptive_compute=True,
            core_top_k=2,
            verifier_adaptive_compute=True,
        )

    assert torch.equal(baseline, with_controls)
    # The head does not accept core_top_k, so the control is reported as unused
    # rather than silently claimed.
    assert diagnostics["core_routing_mode"] == "dense"
    assert diagnostics["core_top_k"] is None


def test_web_app_settings_registry_carries_the_v52_controls() -> None:
    sys.path.insert(0, str(SOURCE))
    try:
        web = _load("v52_controls_chat_web_app", SOURCE / "chat_web_app.py")
    finally:
        sys.path.remove(str(SOURCE))

    defaults = web._normalize_runtime_compute_defaults(web._library_runtime_compute_defaults())
    for key in (
        "core_top_k",
        "verifier_adaptive_compute",
        "verifier_continue_threshold",
        "max_verifier_cycles",
    ):
        assert key in web._RUNTIME_COMPUTE_DEFAULT_KEYS
        assert key in defaults

    # Off by default: sparse dispatch is not always faster and the verifier head
    # is untrained on imported v50 weights.
    assert defaults["core_top_k"] is None
    assert defaults["verifier_adaptive_compute"] is False

    clamped = web._normalize_runtime_compute_defaults(
        {"core_top_k": 10**6, "verifier_continue_threshold": -3.0, "max_verifier_cycles": -1}
    )
    assert 0 < clamped["core_top_k"] <= web.chat_app.MAX_RUNTIME_CORE_TOP_K
    assert clamped["verifier_continue_threshold"] == 0.0
    assert clamped["max_verifier_cycles"] == 0
