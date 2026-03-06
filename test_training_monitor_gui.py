import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "source"))

from training_monitor_gui import (
    RunSnapshot,
    _build_health_summary,
    _build_launch_command,
    _compute_display_progress_percent,
    _derive_stage_monitor_fields,
    _parse_log,
)


def _make_snapshot(**overrides):
    now = time.time()
    payload = {
        "run_name": "train_sample_20260305_010101",
        "out_log": Path("train_sample.out.log"),
        "err_log": None,
        "pid_file": None,
        "pid": None,
        "pid_alive": False,
        "status": "running",
        "stage": "data",
        "sft_step": 0,
        "pref_step": 0,
        "pref_pairs": 0,
        "loss": None,
        "lr": None,
        "beta": None,
        "margin": None,
        "checkpoint_count": 0,
        "last_checkpoint_stage": "-",
        "last_checkpoint_step": 0,
        "save_every_steps": 20,
        "sft_target_steps": 470,
        "pref_target_steps": 130,
        "has_distill_stage": True,
        "has_pref_mining_stage": True,
        "progress_units": 0.0,
        "total_units": 600.0,
        "progress_percent": 0.0,
        "eta_seconds": None,
        "checkpoint_eta_seconds": None,
        "step_rate_per_hour": None,
        "stage_progress_label": "21000/42000 pairs",
        "stage_progress_percent": 50.0,
        "stage_rate_label": "900.00/s",
        "stage_eta_seconds": 12.0,
        "out_size": 100,
        "out_last_write_ts": now,
        "stale_minutes": 0.0,
        "err_size": 0,
        "err_last_write_ts": None,
        "err_signal": "ok",
        "err_summary": "-",
        "launch_hint": "",
        "command_line": "",
        "launch_command": "",
        "health_summary": "healthy",
        "data_summary": "pairs=21000/42000 raw=37572 kept=24041 rate=900.00/s",
        "sft_filter_summary": "-",
        "distill_summary": "visited=120/650 generated=42 rate=14.00/s",
        "pref_mining_summary": "accepted=512/3000 visited=900/48082 rate=6.75/s",
        "pref_selection_summary": "-",
        "tail_lines": [],
        "err_tail_lines": [],
    }
    payload.update(overrides)
    return RunSnapshot(**payload)


def smoke_test_training_monitor_parsing():
    sample_lines = [
        "[data] progress: pairs=42000/42000 raw=75145 kept=48082 rate=1021.34/s",
        "[data] quality filter: raw=75145 kept=48082 empty=338 placeholder=5183 filtered=15531 deduped=6011 source_cap=0 synthetic_cap=0 prompt_cap=6082 cap_relax=0",
        "[data] synthetic_kept=3084/42000",
        "[distill] progress: visited=120/650 generated=42 rate=14.00/s",
        "[sft] quality filter: threshold=0.90 kept=41928 dropped_quality=53 dropped_short=0 exempt_sources=3",
        "[pref] mining config: mode=auto generation=on target_pairs=3000 candidates=48082 max_attempts=60000 selection=capacity_aware keep_ratio=0.620 max_seconds=4500.0",
        "[pref] mining progress: visited=900/48082 accepted=512 rate=6.75/s",
        "[pref] pair selection: strategy=capacity_aware keep=1860/3000 keep_ratio=0.620 gap=0.312->0.401 sim=0.441->0.382 selected_score_mean=1.1180",
        "[pref] mining complete: pairs=1860 mined=3000 visited=9000 generation_failures=3 elapsed=245.2s",
        "[pref] pairs=1860",
        "[pref] step=7 loss=1.2345 lr=1.4e-05 beta=1.9 margin=0.18",
        "[checkpoint] saved stage=preference step=20",
    ]

    with tempfile.TemporaryDirectory() as td:
        out_log = Path(td) / "train_sample.out.log"
        out_log.write_text("\n".join(sample_lines), encoding="utf-8")
        parsed = _parse_log(out_log)

    assert parsed.stage == "preference"
    assert parsed.pref_step == 7
    assert parsed.pref_pairs == 1860
    assert parsed.last_checkpoint_stage == "preference"
    assert parsed.last_checkpoint_step == 20
    assert parsed.data_pairs_current == 42000
    assert parsed.data_pairs_total == 42000
    assert "synthetic=3084/42000" in parsed.data_summary
    assert parsed.distill_generated == 42
    assert parsed.distill_total == 650
    assert "threshold=0.90" in parsed.sft_filter_summary
    assert parsed.pref_mining_target_pairs == 3000
    assert parsed.pref_mining_candidates == 48082
    assert parsed.pref_mining_accepted == 1860
    assert parsed.pref_mining_generation_failures == 3
    assert "strategy=capacity_aware keep=1860/3000" in parsed.pref_selection_summary

    parsed.stage = "data"
    label, pct, rate, eta = _derive_stage_monitor_fields(parsed)
    assert label == "42000/42000 pairs"
    assert pct == 100.0
    assert rate == "1021.34/s"
    assert eta == 0.0

    parsed.stage = "preference_mining"
    label, pct, rate, eta = _derive_stage_monitor_fields(parsed)
    assert label == "1860/3000 acc"
    assert pct is not None and 18.0 < pct < 19.0
    assert rate == "6.75/s"
    assert eta is not None and eta > 0

    parsed.stage = "preference"
    label, pct, rate, eta = _derive_stage_monitor_fields(parsed)
    assert label == "1860"
    assert pct is None
    assert rate == "-"
    assert eta is None

    parsed.stage = "sft_setup"
    label, pct, rate, eta = _derive_stage_monitor_fields(parsed)
    assert label == "setup"
    assert pct is None
    assert rate == "-"
    assert eta is None

    parsed.stage = "sft_filter"
    label, pct, rate, eta = _derive_stage_monitor_fields(parsed)
    assert label == "quality filter"
    assert pct is None
    assert rate == "-"
    assert eta is None

    data_snap = _make_snapshot()
    data_pct = _compute_display_progress_percent(data_snap)
    assert data_pct is not None and 4.9 < data_pct < 5.1

    sft_snap = _make_snapshot(
        stage="sft",
        sft_step=100,
        progress_units=100.0,
        progress_percent=(100.0 / 600.0) * 100.0,
        stage_progress_label="-",
        stage_progress_percent=None,
        stage_rate_label="-",
        stage_eta_seconds=None,
    )
    sft_pct = _compute_display_progress_percent(sft_snap)
    assert sft_pct is not None and sft_pct > sft_snap.progress_percent
    assert 35.0 < sft_pct < 36.0

    launch_cmd = _build_launch_command(
        root_dir=Path.cwd(),
        launch_hint=str(Path.cwd() / "source" / "run_train_qwen_supermix_v25_selective_pref.ps1"),
        command_line="",
    )
    assert "powershell -ExecutionPolicy Bypass -File" in launch_cmd
    assert "source" in launch_cmd

    health = _build_health_summary(
        status="stopped",
        stage="sft",
        pid_file=Path("train_sample.pid"),
        pid_alive=False,
        err_signal="warn",
        err_summary="warning: gradient checkpointing disabled",
        stale_minutes=48.5,
    )
    assert "stopped before completion" in health
    assert "pid file is stale" in health
    assert "gradient checkpointing disabled" in health

    print("Training monitor parser smoke test PASSED")


if __name__ == "__main__":
    smoke_test_training_monitor_parsing()
