import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import ttk

if os.name == "nt":
    import ctypes


TRAIN_STEP_RE = re.compile(r"\[train\] step=(\d+) loss=([0-9eE+\-.]+) lr=([0-9eE+\-.]+)")
PREF_STEP_RE = re.compile(
    r"\[pref\] step=(\d+) loss=([0-9eE+\-.]+) lr=([0-9eE+\-.]+)(?: beta=([0-9eE+\-.]+) margin=([0-9eE+\-.]+))?"
)
PREF_PAIRS_RE = re.compile(r"\[pref\] pairs=(\d+)")
CHECKPOINT_RE = re.compile(r"\[checkpoint\] saved stage=(sft|preference) step=(\d+)")
DATA_PROGRESS_RE = re.compile(
    r"\[data\] progress: pairs=(\d+)/(\d+) raw=(\d+) kept=(\d+) rate=([0-9eE+\-.]+)/s"
)
DATA_QUALITY_RE = re.compile(
    r"\[data\] quality filter: raw=(\d+) kept=(\d+) empty=(\d+) placeholder=(\d+) filtered=(\d+) deduped=(\d+) "
    r"source_cap=(\d+) synthetic_cap=(\d+) prompt_cap=(\d+) cap_relax=(\d+)"
)
DATA_SYNTHETIC_RE = re.compile(r"\[data\] synthetic_kept=(\d+)/(\d+)")
DISTILL_PROGRESS_RE = re.compile(
    r"\[distill\] progress: visited=(\d+)/(\d+) generated=(\d+) rate=([0-9eE+\-.]+)/s"
)
DISTILL_COMPLETE_RE = re.compile(
    r"\[distill\] complete: generated=(\d+) visited=(\d+)/(\d+) elapsed=([0-9eE+\-.]+)s"
)
SFT_QUALITY_RE = re.compile(
    r"\[sft\] quality filter(?: fallback)?: threshold=([0-9eE+\-.]+) kept=(\d+) dropped_quality=(\d+) "
    r"dropped_short=(\d+) exempt_sources=(\d+)"
)
PREF_MINING_CONFIG_RE = re.compile(
    r"\[pref\] mining config: mode=(\S+) generation=(\S+) target_pairs=(\d+) candidates=(\d+) max_attempts=(\d+) "
    r"selection=(\S+) keep_ratio=([0-9eE+\-.]+) max_seconds=(\S+)"
)
PREF_MINING_PROGRESS_RE = re.compile(
    r"\[pref\] mining progress: visited=(\d+)/(\d+) accepted=(\d+) rate=([0-9eE+\-.]+)/s"
)
PREF_MINING_COMPLETE_RE = re.compile(
    r"\[pref\] mining complete: pairs=(\d+) mined=(\d+) visited=(\d+) generation_failures=(\d+) elapsed=([0-9eE+\-.]+)s"
)
PREF_SELECTION_RE = re.compile(
    r"\[pref\] pair selection: strategy=(\S+) keep=(\d+)/(\d+) keep_ratio=([0-9eE+\-.]+) "
    r"gap=([0-9eE+\-.]+)->([0-9eE+\-.]+) sim=([0-9eE+\-.]+)->([0-9eE+\-.]+) "
    r"selected_score_mean=([0-9eE+\-.]+)"
)
MAX_STEPS_RE = re.compile(r"--max_steps(?:\s+|=)(\d+)")
PREF_STEPS_RE = re.compile(r"--preference_steps(?:\s+|=)(\d+)")
SAVE_EVERY_STEPS_RE = re.compile(r"--save_every_steps(?:\s+|=)(\d+)")
PS1_FILE_RE = re.compile(r"-File\s+(\"[^\"]+\\.ps1\"|'[^']+\\.ps1'|[^\s]+\\.ps1)", flags=re.IGNORECASE)
RUN_CORE_RE = re.compile(r"^train_(.+?)_(\d{8}_\d{6})$")

PROCESS_CMD_CACHE: Dict[int, Tuple[float, Optional[str]]] = {}
PS1_TARGET_CACHE: Dict[str, Tuple[float, Optional[int], Optional[int], Optional[int], bool, bool]] = {}


@dataclass
class RunSnapshot:
    run_name: str
    out_log: Path
    err_log: Optional[Path]
    pid_file: Optional[Path]
    pid: Optional[int]
    pid_alive: bool
    status: str
    stage: str
    sft_step: int
    pref_step: int
    pref_pairs: int
    loss: Optional[float]
    lr: Optional[float]
    beta: Optional[float]
    margin: Optional[float]
    checkpoint_count: int
    last_checkpoint_stage: str
    last_checkpoint_step: int
    save_every_steps: Optional[int]
    sft_target_steps: Optional[int]
    pref_target_steps: Optional[int]
    has_distill_stage: bool
    has_pref_mining_stage: bool
    progress_units: float
    total_units: Optional[float]
    progress_percent: Optional[float]
    eta_seconds: Optional[float]
    checkpoint_eta_seconds: Optional[float]
    step_rate_per_hour: Optional[float]
    stage_progress_label: str
    stage_progress_percent: Optional[float]
    stage_rate_label: str
    stage_eta_seconds: Optional[float]
    out_size: int
    out_last_write_ts: float
    stale_minutes: float
    err_size: int
    err_last_write_ts: Optional[float]
    err_signal: str
    err_summary: str
    launch_hint: str
    command_line: str
    launch_command: str
    health_summary: str
    data_summary: str
    sft_filter_summary: str
    distill_summary: str
    pref_mining_summary: str
    pref_selection_summary: str
    tail_lines: List[str]
    err_tail_lines: List[str]


@dataclass
class ParsedLog:
    stage: str = "unknown"
    sft_step: int = 0
    pref_step: int = 0
    pref_pairs: int = 0
    loss: Optional[float] = None
    lr: Optional[float] = None
    beta: Optional[float] = None
    margin: Optional[float] = None
    checkpoint_count: int = 0
    last_checkpoint_stage: str = "-"
    last_checkpoint_step: int = 0
    data_pairs_current: Optional[int] = None
    data_pairs_total: Optional[int] = None
    data_raw_count: Optional[int] = None
    data_kept_count: Optional[int] = None
    data_rate_per_sec: Optional[float] = None
    data_synthetic_kept: Optional[int] = None
    data_synthetic_total: Optional[int] = None
    data_summary: str = "-"
    sft_filter_summary: str = "-"
    distill_generated: Optional[int] = None
    distill_visited: Optional[int] = None
    distill_total: Optional[int] = None
    distill_rate_per_sec: Optional[float] = None
    distill_summary: str = "-"
    pref_mining_target_pairs: Optional[int] = None
    pref_mining_candidates: Optional[int] = None
    pref_mining_accepted: Optional[int] = None
    pref_mining_visited: Optional[int] = None
    pref_mining_rate_per_sec: Optional[float] = None
    pref_mining_generation_failures: Optional[int] = None
    pref_mining_summary: str = "-"
    pref_selection_summary: str = "-"
    tail_lines: List[str] = field(default_factory=list)


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is unreliable on some Windows Python builds (WinError 87).
        # Query a real process handle instead.
        access = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
        handle = ctypes.windll.kernel32.OpenProcess(access, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        err = ctypes.GetLastError()
        # Access denied can still indicate that the PID exists.
        if err == 5:
            return True
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pid(pid_file: Path) -> Optional[int]:
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _read_tail_lines(path: Path, max_bytes: int = 2_000_000, max_lines: int = 2400) -> List[str]:
    try:
        size = path.stat().st_size
    except Exception:
        return []
    if size <= 0:
        return []

    try:
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read()
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return []

    lines = text.splitlines()
    if len(lines) > max_lines:
        return lines[-max_lines:]
    return lines


def _query_process_cmdline(pid: int) -> Optional[str]:
    now = time.time()
    cached = PROCESS_CMD_CACHE.get(pid)
    if cached is not None and (now - cached[0]) <= 20.0:
        return cached[1]

    result: Optional[str] = None
    try:
        if os.name == "nt":
            query = f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\"; if($p){{$p.CommandLine}}"
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-Command", query],
                capture_output=True,
                text=True,
                timeout=4,
            )
            out = cp.stdout.strip()
            if out:
                result = out
        else:
            proc_path = Path(f"/proc/{pid}/cmdline")
            if proc_path.exists():
                raw = proc_path.read_bytes()
                if raw:
                    result = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        result = None

    PROCESS_CMD_CACHE[pid] = (now, result)
    return result


def _extract_int_arg(command_line: str, pattern: re.Pattern[str]) -> Optional[int]:
    if not command_line:
        return None
    m = pattern.search(command_line)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_ps1_path(command_line: str, root_dir: Path) -> Optional[Path]:
    if not command_line:
        return None
    m = PS1_FILE_RE.search(command_line)
    if not m:
        return None
    raw = m.group(1).strip().strip("\"'")
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (root_dir / p).resolve()
    if p.exists():
        return p
    return None


def _parse_ps1_targets(ps1_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int], bool, bool]:
    key = str(ps1_path).lower()
    try:
        mtime = ps1_path.stat().st_mtime
    except Exception:
        return None, None, None, False, False

    cached = PS1_TARGET_CACHE.get(key)
    if cached is not None and abs(cached[0] - mtime) < 1e-6:
        return cached[1], cached[2], cached[3], cached[4], cached[5]

    try:
        text = ps1_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None, None, False, False

    sft_target: Optional[int] = None
    pref_target: Optional[int] = None
    save_every: Optional[int] = None
    text_low = text.lower()
    has_distill = "--supermix_distill_" in text_low
    has_pref_mining = "--preference_mining_" in text_low
    m1 = MAX_STEPS_RE.findall(text)
    if m1:
        try:
            sft_target = int(m1[-1])
        except Exception:
            sft_target = None
    m2 = PREF_STEPS_RE.findall(text)
    if m2:
        try:
            pref_target = int(m2[-1])
        except Exception:
            pref_target = None

    m3 = SAVE_EVERY_STEPS_RE.findall(text)
    if m3:
        try:
            save_every = int(m3[-1])
        except Exception:
            save_every = None

    PS1_TARGET_CACHE[key] = (mtime, sft_target, pref_target, save_every, has_distill, has_pref_mining)
    return sft_target, pref_target, save_every, has_distill, has_pref_mining


def _run_core(run_name: str) -> str:
    m = RUN_CORE_RE.match(run_name)
    if m:
        return m.group(1).strip().lower()
    return run_name.replace("train_", "", 1).strip().lower()


def _guess_ps1_from_run_name(run_name: str, root_dir: Path) -> Optional[Path]:
    core = _run_core(run_name)
    source_dir = root_dir / "source"
    if not source_dir.exists():
        return None
    candidates = list(source_dir.glob("run_train_qwen_supermix_*.ps1"))
    if not candidates:
        return None

    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        stem_core = p.stem.replace("run_train_qwen_supermix_", "", 1).strip().lower()
        score = 0
        if stem_core and stem_core in core:
            score = len(stem_core)
        elif core and core in stem_core:
            score = len(core)
        elif stem_core and any(tok in core for tok in stem_core.split("_")):
            score = 1
        if score > 0:
            scored.append((score, p))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _infer_targets(
    run_name: str,
    root_dir: Path,
    command_line: str,
) -> Tuple[Optional[int], Optional[int], Optional[int], str, bool, bool]:
    sft_target = _extract_int_arg(command_line, MAX_STEPS_RE)
    pref_target = _extract_int_arg(command_line, PREF_STEPS_RE)
    save_every = _extract_int_arg(command_line, SAVE_EVERY_STEPS_RE)
    launch_hint = ""
    command_line_low = command_line.lower()
    has_distill = "--supermix_distill_" in command_line_low
    has_pref_mining = "--preference_mining_" in command_line_low

    ps1_from_cmd = _resolve_ps1_path(command_line, root_dir) if command_line else None
    if ps1_from_cmd is not None:
        launch_hint = str(ps1_from_cmd)
        ps1_sft, ps1_pref, ps1_save, ps1_distill, ps1_pref_mining = _parse_ps1_targets(ps1_from_cmd)
        if sft_target is None:
            sft_target = ps1_sft
        if pref_target is None:
            pref_target = ps1_pref
        if save_every is None:
            save_every = ps1_save
        has_distill = has_distill or ps1_distill
        has_pref_mining = has_pref_mining or ps1_pref_mining

    if sft_target is None or pref_target is None or save_every is None:
        ps1_guess = _guess_ps1_from_run_name(run_name, root_dir)
        if ps1_guess is not None:
            if not launch_hint:
                launch_hint = str(ps1_guess)
            guess_sft, guess_pref, guess_save, guess_distill, guess_pref_mining = _parse_ps1_targets(ps1_guess)
            if sft_target is None:
                sft_target = guess_sft
            if pref_target is None:
                pref_target = guess_pref
            if save_every is None:
                save_every = guess_save
            has_distill = has_distill or guess_distill
            has_pref_mining = has_pref_mining or guess_pref_mining

    return sft_target, pref_target, save_every, launch_hint, has_distill, has_pref_mining


def _path_display(path: Path, root_dir: Path) -> str:
    try:
        root_resolved = root_dir.resolve()
        path_resolved = path.resolve()
        try:
            return str(path_resolved.relative_to(root_resolved))
        except Exception:
            return str(path_resolved)
    except Exception:
        return str(path)


def _quote_powershell_arg(text: str) -> str:
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or '"' in text:
        return '"' + text.replace('"', '`"') + '"'
    return text


def _build_launch_command(root_dir: Path, launch_hint: str, command_line: str) -> str:
    live_cmd = str(command_line or "").strip()
    if live_cmd:
        return live_cmd

    raw_hint = str(launch_hint or "").strip()
    if not raw_hint:
        return ""

    hint_path = Path(raw_hint)
    display_path = _path_display(hint_path, root_dir)
    suffix = hint_path.suffix.lower()
    if suffix == ".ps1":
        return f"powershell -ExecutionPolicy Bypass -File {_quote_powershell_arg(display_path)}"
    if suffix == ".bat":
        return _quote_powershell_arg(display_path)
    return display_path


def _infer_stage(line: str, current: str) -> str:
    if line.startswith("[data]"):
        return "data"
    if line.startswith("[distill]"):
        return "distill"
    if line.startswith("[sft]"):
        return "sft_filter"
    if line.startswith("[train] stage="):
        return "sft_setup"
    if line.startswith("[train] step="):
        return "sft"
    if line.startswith("[pref] building") or line.startswith("[pref] mining"):
        return "preference_mining"
    if line.startswith("[pref] step="):
        return "preference"
    if line.startswith("[eval]"):
        return "eval"
    if line.startswith("[done]"):
        return "done"
    return current


def _percent_complete(current: Optional[int], total: Optional[int]) -> Optional[float]:
    if current is None or total is None or total <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * float(current) / float(total)))


def _eta_from_rate(current: Optional[int], total: Optional[int], rate_per_sec: Optional[float]) -> Optional[float]:
    if current is None or total is None or total <= 0 or rate_per_sec is None or rate_per_sec <= 0:
        return None
    remaining = max(0.0, float(total - current))
    return float(remaining / float(rate_per_sec))


def _fmt_rate_per_sec(rate_per_sec: Optional[float]) -> str:
    if rate_per_sec is None or rate_per_sec <= 0:
        return "-"
    return f"{rate_per_sec:.2f}/s"


def _derive_stage_monitor_fields(parsed: ParsedLog) -> Tuple[str, Optional[float], str, Optional[float]]:
    stage = str(parsed.stage or "unknown").strip().lower()

    if stage == "data" and parsed.data_pairs_total is not None:
        current = parsed.data_pairs_current if parsed.data_pairs_current is not None else parsed.data_kept_count
        total = parsed.data_pairs_total
        label = f"{current}/{total} pairs" if current is not None else "-"
        return (
            label,
            _percent_complete(current, total),
            _fmt_rate_per_sec(parsed.data_rate_per_sec),
            _eta_from_rate(current, total, parsed.data_rate_per_sec),
        )

    if stage == "distill" and parsed.distill_total is not None:
        current = parsed.distill_visited
        total = parsed.distill_total
        generated = parsed.distill_generated
        if generated is not None and current is not None and total is not None:
            label = f"{generated} gen | {current}/{total}"
        elif current is not None:
            label = f"{current}/{total}"
        else:
            label = "-"
        return (
            label,
            _percent_complete(current, total),
            _fmt_rate_per_sec(parsed.distill_rate_per_sec),
            _eta_from_rate(current, total, parsed.distill_rate_per_sec),
        )

    if stage == "preference_mining":
        current = parsed.pref_mining_visited
        total = parsed.pref_mining_candidates
        accepted = parsed.pref_mining_accepted
        if accepted is not None and parsed.pref_mining_target_pairs is not None:
            label = f"{accepted}/{parsed.pref_mining_target_pairs} acc"
        elif current is not None and total is not None:
            label = f"{current}/{total} seen"
        else:
            label = "-"
        return (
            label,
            _percent_complete(current, total),
            _fmt_rate_per_sec(parsed.pref_mining_rate_per_sec),
            _eta_from_rate(current, total, parsed.pref_mining_rate_per_sec),
        )

    if stage == "sft_setup":
        return ("setup", None, "-", None)

    if stage == "sft_filter":
        return ("quality filter", None, "-", None)

    if parsed.pref_pairs > 0:
        return (str(parsed.pref_pairs), None, "-", None)

    return ("-", None, "-", None)


def _parse_log(
    out_log: Path,
) -> ParsedLog:
    parsed = ParsedLog()
    lines = _read_tail_lines(out_log)
    for line in lines:
        parsed.stage = _infer_stage(line, parsed.stage)
        m_train = TRAIN_STEP_RE.search(line)
        if m_train:
            parsed.sft_step = max(parsed.sft_step, int(m_train.group(1)))
            parsed.loss = float(m_train.group(2))
            parsed.lr = float(m_train.group(3))
        m_pref = PREF_STEP_RE.search(line)
        if m_pref:
            parsed.pref_step = max(parsed.pref_step, int(m_pref.group(1)))
            parsed.loss = float(m_pref.group(2))
            parsed.lr = float(m_pref.group(3))
            parsed.beta = float(m_pref.group(4)) if m_pref.group(4) is not None else parsed.beta
            parsed.margin = float(m_pref.group(5)) if m_pref.group(5) is not None else parsed.margin
        m_pairs = PREF_PAIRS_RE.search(line)
        if m_pairs:
            parsed.pref_pairs = max(parsed.pref_pairs, int(m_pairs.group(1)))
        m_ckpt = CHECKPOINT_RE.search(line)
        if m_ckpt:
            parsed.checkpoint_count += 1
            parsed.last_checkpoint_stage = m_ckpt.group(1)
            parsed.last_checkpoint_step = int(m_ckpt.group(2))

        m_data = DATA_PROGRESS_RE.search(line)
        if m_data:
            parsed.data_pairs_current = int(m_data.group(1))
            parsed.data_pairs_total = int(m_data.group(2))
            parsed.data_raw_count = int(m_data.group(3))
            parsed.data_kept_count = int(m_data.group(4))
            parsed.data_rate_per_sec = float(m_data.group(5))
            parsed.data_summary = (
                f"pairs={parsed.data_pairs_current}/{parsed.data_pairs_total} raw={parsed.data_raw_count} "
                f"kept={parsed.data_kept_count} rate={parsed.data_rate_per_sec:.2f}/s"
            )

        m_data_quality = DATA_QUALITY_RE.search(line)
        if m_data_quality:
            raw = int(m_data_quality.group(1))
            kept = int(m_data_quality.group(2))
            empty = int(m_data_quality.group(3))
            placeholder = int(m_data_quality.group(4))
            filtered = int(m_data_quality.group(5))
            deduped = int(m_data_quality.group(6))
            source_cap = int(m_data_quality.group(7))
            synthetic_cap = int(m_data_quality.group(8))
            prompt_cap = int(m_data_quality.group(9))
            cap_relax = int(m_data_quality.group(10))
            parsed.data_summary = (
                f"raw={raw} kept={kept} filtered={filtered} deduped={deduped} empty={empty} "
                f"placeholder={placeholder} source_cap={source_cap} synthetic_cap={synthetic_cap} "
                f"prompt_cap={prompt_cap} cap_relax={cap_relax}"
            )

        m_data_synth = DATA_SYNTHETIC_RE.search(line)
        if m_data_synth:
            parsed.data_synthetic_kept = int(m_data_synth.group(1))
            parsed.data_synthetic_total = int(m_data_synth.group(2))
            suffix = f" synthetic={parsed.data_synthetic_kept}/{parsed.data_synthetic_total}"
            parsed.data_summary = parsed.data_summary + suffix if parsed.data_summary != "-" else suffix.strip()

        m_distill = DISTILL_PROGRESS_RE.search(line)
        if m_distill:
            parsed.distill_visited = int(m_distill.group(1))
            parsed.distill_total = int(m_distill.group(2))
            parsed.distill_generated = int(m_distill.group(3))
            parsed.distill_rate_per_sec = float(m_distill.group(4))
            parsed.distill_summary = (
                f"visited={parsed.distill_visited}/{parsed.distill_total} generated={parsed.distill_generated} "
                f"rate={parsed.distill_rate_per_sec:.2f}/s"
            )

        m_distill_complete = DISTILL_COMPLETE_RE.search(line)
        if m_distill_complete:
            parsed.distill_generated = int(m_distill_complete.group(1))
            parsed.distill_visited = int(m_distill_complete.group(2))
            parsed.distill_total = int(m_distill_complete.group(3))
            elapsed = float(m_distill_complete.group(4))
            parsed.distill_summary = (
                f"generated={parsed.distill_generated} visited={parsed.distill_visited}/{parsed.distill_total} "
                f"elapsed={elapsed:.1f}s"
            )

        m_sft_quality = SFT_QUALITY_RE.search(line)
        if m_sft_quality:
            threshold = float(m_sft_quality.group(1))
            kept = int(m_sft_quality.group(2))
            dropped_quality = int(m_sft_quality.group(3))
            dropped_short = int(m_sft_quality.group(4))
            exempt_sources = int(m_sft_quality.group(5))
            parsed.sft_filter_summary = (
                f"threshold={threshold:.2f} kept={kept} dropped_quality={dropped_quality} "
                f"dropped_short={dropped_short} exempt_sources={exempt_sources}"
            )

        m_pref_cfg = PREF_MINING_CONFIG_RE.search(line)
        if m_pref_cfg:
            mode = m_pref_cfg.group(1)
            generation = m_pref_cfg.group(2)
            parsed.pref_mining_target_pairs = int(m_pref_cfg.group(3))
            parsed.pref_mining_candidates = int(m_pref_cfg.group(4))
            selection = m_pref_cfg.group(6)
            keep_ratio = float(m_pref_cfg.group(7))
            max_seconds = m_pref_cfg.group(8)
            parsed.pref_mining_summary = (
                f"mode={mode} generation={generation} target_pairs={parsed.pref_mining_target_pairs} "
                f"candidates={parsed.pref_mining_candidates} selection={selection} keep_ratio={keep_ratio:.3f} "
                f"max_seconds={max_seconds}"
            )

        m_pref_progress = PREF_MINING_PROGRESS_RE.search(line)
        if m_pref_progress:
            parsed.pref_mining_visited = int(m_pref_progress.group(1))
            parsed.pref_mining_candidates = int(m_pref_progress.group(2))
            parsed.pref_mining_accepted = int(m_pref_progress.group(3))
            parsed.pref_mining_rate_per_sec = float(m_pref_progress.group(4))
            target_txt = (
                str(parsed.pref_mining_target_pairs)
                if parsed.pref_mining_target_pairs is not None and parsed.pref_mining_target_pairs > 0
                else "-"
            )
            parsed.pref_mining_summary = (
                f"accepted={parsed.pref_mining_accepted}/{target_txt} visited={parsed.pref_mining_visited}/"
                f"{parsed.pref_mining_candidates} rate={parsed.pref_mining_rate_per_sec:.2f}/s"
            )

        m_pref_complete = PREF_MINING_COMPLETE_RE.search(line)
        if m_pref_complete:
            parsed.pref_mining_accepted = int(m_pref_complete.group(1))
            mined_pairs = int(m_pref_complete.group(2))
            parsed.pref_mining_visited = int(m_pref_complete.group(3))
            parsed.pref_mining_generation_failures = int(m_pref_complete.group(4))
            elapsed = float(m_pref_complete.group(5))
            target_txt = (
                str(parsed.pref_mining_target_pairs)
                if parsed.pref_mining_target_pairs is not None and parsed.pref_mining_target_pairs > 0
                else "-"
            )
            candidates_txt = (
                str(parsed.pref_mining_candidates)
                if parsed.pref_mining_candidates is not None and parsed.pref_mining_candidates > 0
                else "-"
            )
            parsed.pref_mining_summary = (
                f"pairs={parsed.pref_mining_accepted}/{target_txt} mined={mined_pairs} visited={parsed.pref_mining_visited}/"
                f"{candidates_txt} generation_failures={parsed.pref_mining_generation_failures} elapsed={elapsed:.1f}s"
            )

        m_pref_selection = PREF_SELECTION_RE.search(line)
        if m_pref_selection:
            strategy = m_pref_selection.group(1)
            keep_n = int(m_pref_selection.group(2))
            total = int(m_pref_selection.group(3))
            keep_ratio = float(m_pref_selection.group(4))
            gap_before = float(m_pref_selection.group(5))
            gap_after = float(m_pref_selection.group(6))
            sim_before = float(m_pref_selection.group(7))
            sim_after = float(m_pref_selection.group(8))
            score_mean = float(m_pref_selection.group(9))
            parsed.pref_selection_summary = (
                f"strategy={strategy} keep={keep_n}/{total} keep_ratio={keep_ratio:.3f} "
                f"gap={gap_before:.3f}->{gap_after:.3f} sim={sim_before:.3f}->{sim_after:.3f} "
                f"score_mean={score_mean:.4f}"
            )

    parsed.tail_lines = lines[-80:]
    return parsed


def _compute_progress(
    sft_step: int,
    pref_step: int,
    stage: str,
    sft_target_steps: Optional[int],
    pref_target_steps: Optional[int],
) -> Tuple[float, Optional[float], Optional[float]]:
    progress_units = float(max(0, sft_step) + max(0, pref_step))

    sft_target = sft_target_steps if (sft_target_steps is not None and sft_target_steps > 0) else None
    pref_target = pref_target_steps if (pref_target_steps is not None and pref_target_steps > 0) else None

    if sft_target is None and pref_target is None:
        return progress_units, None, None

    total_units: float
    if sft_target is not None and pref_target is not None:
        total_units = float(sft_target + pref_target)
        pct = 100.0 * progress_units / max(1.0, total_units)
        return progress_units, total_units, max(0.0, min(100.0, pct))

    if sft_target is not None:
        total_units = float(sft_target)
        pct = 100.0 * float(sft_step) / max(1.0, total_units)
        if pref_step == 0 and stage not in {"preference", "preference_mining", "done", "eval"}:
            pct = min(99.0, pct)
        return progress_units, total_units, max(0.0, min(100.0, pct))

    total_units = float(pref_target)
    pct = 100.0 * float(pref_step) / max(1.0, total_units)
    return progress_units, total_units, max(0.0, min(100.0, pct))


def collect_run_snapshots(root_dir: Path, stale_minutes_threshold: float) -> List[RunSnapshot]:
    now = time.time()
    out_logs = sorted(root_dir.glob("train_*.out.log"))
    snapshots: List[RunSnapshot] = []

    for out_log in out_logs:
        stem = out_log.name.replace(".out.log", "")
        err_log = root_dir / f"{stem}.err.log"
        if not err_log.exists():
            err_log = None
        pid_file = root_dir / f"{stem}.pid"
        if not pid_file.exists():
            pid_file = None
        pid = _read_pid(pid_file) if pid_file is not None else None
        pid_alive = _is_pid_alive(pid) if pid is not None else False

        command_line = _query_process_cmdline(pid) if (pid_alive and pid is not None) else None
        command_line = command_line or ""
        sft_target_steps, pref_target_steps, save_every_steps, launch_hint, has_distill_stage, has_pref_mining_stage = _infer_targets(
            stem,
            root_dir,
            command_line,
        )

        parsed = _parse_log(out_log)

        out_stat = out_log.stat()
        out_mtime = out_stat.st_mtime
        out_size = out_stat.st_size
        stale_mins = max(0.0, (now - out_mtime) / 60.0)

        err_size = 0
        err_last_write_ts: Optional[float] = None
        err_tail_lines: List[str] = []
        if err_log is not None:
            try:
                err_stat = err_log.stat()
                err_size = int(err_stat.st_size)
                err_last_write_ts = float(err_stat.st_mtime)
            except Exception:
                err_size = 0
                err_last_write_ts = None
            err_tail_lines = _read_tail_lines(err_log, max_bytes=350_000, max_lines=400)[-60:]
        err_signal, err_summary = _summarize_err_tail(err_tail_lines)
        launch_command = _build_launch_command(root_dir=root_dir, launch_hint=launch_hint, command_line=command_line)

        progress_units, total_units, progress_percent = _compute_progress(
            sft_step=parsed.sft_step,
            pref_step=parsed.pref_step,
            stage=parsed.stage,
            sft_target_steps=sft_target_steps,
            pref_target_steps=pref_target_steps,
        )
        stage_progress_label, stage_progress_percent, stage_rate_label, stage_eta_seconds = _derive_stage_monitor_fields(
            parsed
        )

        if pid_alive:
            status = "running"
            if stale_mins >= float(stale_minutes_threshold):
                status = "stalled"
        else:
            if parsed.stage == "done":
                status = "finished"
            elif out_size > 0:
                status = "stopped"
            else:
                status = "unknown"

        health_summary = _build_health_summary(
            status=status,
            stage=parsed.stage,
            pid_file=pid_file,
            pid_alive=pid_alive,
            err_signal=err_signal,
            err_summary=err_summary,
            stale_minutes=stale_mins,
        )

        snapshots.append(
            RunSnapshot(
                run_name=stem,
                out_log=out_log,
                err_log=err_log,
                pid_file=pid_file,
                pid=pid,
                pid_alive=pid_alive,
                status=status,
                stage=parsed.stage,
                sft_step=parsed.sft_step,
                pref_step=parsed.pref_step,
                pref_pairs=parsed.pref_pairs,
                loss=parsed.loss,
                lr=parsed.lr,
                beta=parsed.beta,
                margin=parsed.margin,
                checkpoint_count=parsed.checkpoint_count,
                last_checkpoint_stage=parsed.last_checkpoint_stage,
                last_checkpoint_step=parsed.last_checkpoint_step,
                save_every_steps=save_every_steps,
                sft_target_steps=sft_target_steps,
                pref_target_steps=pref_target_steps,
                has_distill_stage=has_distill_stage,
                has_pref_mining_stage=has_pref_mining_stage,
                progress_units=progress_units,
                total_units=total_units,
                progress_percent=progress_percent,
                eta_seconds=None,
                checkpoint_eta_seconds=None,
                step_rate_per_hour=None,
                stage_progress_label=stage_progress_label,
                stage_progress_percent=stage_progress_percent,
                stage_rate_label=stage_rate_label,
                stage_eta_seconds=stage_eta_seconds,
                out_size=out_size,
                out_last_write_ts=out_mtime,
                stale_minutes=stale_mins,
                err_size=err_size,
                err_last_write_ts=err_last_write_ts,
                err_signal=err_signal,
                err_summary=err_summary,
                launch_hint=launch_hint,
                command_line=command_line,
                launch_command=launch_command,
                health_summary=health_summary,
                data_summary=parsed.data_summary,
                sft_filter_summary=parsed.sft_filter_summary,
                distill_summary=parsed.distill_summary,
                pref_mining_summary=parsed.pref_mining_summary,
                pref_selection_summary=parsed.pref_selection_summary,
                tail_lines=parsed.tail_lines,
                err_tail_lines=err_tail_lines,
            )
        )

    snapshots.sort(key=lambda x: x.out_last_write_ts, reverse=True)
    return snapshots


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    sec = max(0, int(seconds))
    days, rem = divmod(sec, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def _summarize_err_tail(err_lines: Sequence[str]) -> Tuple[str, str]:
    if not err_lines:
        return "ok", "-"

    for raw in reversed(list(err_lines)):
        line = str(raw).strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("loading weights:"):
            continue
        if "traceback" in low:
            return "error", "Traceback detected"
        if "error" in low or "exception" in low:
            if "userwarning" in low:
                return "warn", line[:140]
            return "error", line[:140]
        if "warning" in low:
            return "warn", line[:140]
    return "ok", "-"


def _build_health_summary(
    status: str,
    stage: str,
    pid_file: Optional[Path],
    pid_alive: bool,
    err_signal: str,
    err_summary: str,
    stale_minutes: float,
) -> str:
    notes: List[str] = []
    status_low = str(status or "").strip().lower()
    stage_low = str(stage or "").strip().lower()

    if status_low == "stalled":
        notes.append(f"stalled for {stale_minutes:.1f}m")
    elif status_low == "stopped" and stage_low != "done":
        notes.append("stopped before completion")
    elif status_low == "unknown":
        notes.append("run state unknown")

    if pid_file is not None and not pid_alive and status_low in {"stopped", "unknown"}:
        notes.append("pid file is stale")

    if err_signal == "error":
        notes.append(err_summary if err_summary and err_summary != "-" else "error detected in err log")
    elif err_signal == "warn":
        notes.append(err_summary if err_summary and err_summary != "-" else "warning detected in err log")

    if not notes:
        return "healthy"
    return "; ".join(notes)


def _next_checkpoint_step(stage: str, save_every_steps: Optional[int], sft_step: int, pref_step: int) -> Optional[int]:
    if save_every_steps is None or save_every_steps <= 0:
        return None
    interval = int(save_every_steps)
    if stage in {"preference", "preference_mining"}:
        current = max(0, int(pref_step))
    else:
        current = max(0, int(sft_step))
    if current <= 0:
        return interval
    return ((current // interval) + 1) * interval


def _stage_rank(stage: str) -> int:
    return {
        "unknown": 0,
        "data": 1,
        "distill": 2,
        "sft_setup": 3,
        "sft_filter": 4,
        "sft": 5,
        "preference_mining": 6,
        "preference": 7,
        "eval": 8,
        "done": 9,
    }.get(str(stage or "").strip().lower(), 0)


def _clamp_fraction(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _phase_weight_plan(snap: RunSnapshot) -> List[Tuple[str, float]]:
    phases: List[Tuple[str, float]] = []

    if snap.data_summary != "-" or snap.stage in {"data", "distill", "sft_setup", "sft_filter", "sft", "preference_mining", "preference", "eval", "done"}:
        phases.append(("data", 10.0))

    if snap.has_distill_stage or snap.distill_summary != "-" or snap.stage == "distill":
        phases.append(("distill", 10.0))

    has_sft = (
        (snap.sft_target_steps is not None and snap.sft_target_steps > 0)
        or snap.sft_step > 0
        or snap.stage in {"sft_setup", "sft_filter", "sft", "preference_mining", "preference", "eval", "done"}
    )
    if has_sft:
        phases.append(("sft_setup", 5.0))
        phases.append(("sft", 50.0))

    has_pref = (
        (snap.pref_target_steps is not None and snap.pref_target_steps > 0)
        or snap.pref_step > 0
        or snap.pref_pairs > 0
        or snap.stage in {"preference_mining", "preference", "eval", "done"}
    )
    if has_pref and (snap.has_pref_mining_stage or snap.pref_mining_summary != "-" or snap.stage == "preference_mining"):
        phases.append(("preference_mining", 10.0))
    if has_pref:
        phases.append(("preference", 15.0))

    return phases


def _phase_completion(snap: RunSnapshot, phase: str) -> Optional[float]:
    current_rank = _stage_rank(snap.stage)
    phase_rank = _stage_rank(phase)

    if current_rank > phase_rank:
        return 1.0
    if current_rank < phase_rank:
        return 0.0

    if phase == "data":
        return _clamp_fraction(
            None if snap.stage_progress_percent is None or snap.stage != "data" else snap.stage_progress_percent / 100.0
        )

    if phase == "distill":
        return _clamp_fraction(
            None if snap.stage_progress_percent is None or snap.stage != "distill" else snap.stage_progress_percent / 100.0
        )

    if phase == "sft_setup":
        if snap.stage == "sft_setup":
            return 0.5
        if snap.stage == "sft_filter":
            return 1.0
        return 0.0

    if phase == "sft":
        if snap.sft_target_steps is not None and snap.sft_target_steps > 0:
            return _clamp_fraction(float(snap.sft_step) / float(snap.sft_target_steps))
        return 1.0 if current_rank > phase_rank else 0.0

    if phase == "preference_mining":
        return _clamp_fraction(
            None
            if snap.stage_progress_percent is None or snap.stage != "preference_mining"
            else snap.stage_progress_percent / 100.0
        )

    if phase == "preference":
        if snap.pref_target_steps is not None and snap.pref_target_steps > 0:
            return _clamp_fraction(float(snap.pref_step) / float(snap.pref_target_steps))
        return 1.0 if current_rank > phase_rank else 0.0

    return None


def _compute_display_progress_percent(snap: RunSnapshot) -> Optional[float]:
    if snap.stage == "done" or snap.status == "finished":
        return 100.0

    phases = _phase_weight_plan(snap)
    if not phases:
        if snap.progress_percent is not None:
            return snap.progress_percent
        return snap.stage_progress_percent

    total_weight = sum(weight for _, weight in phases)
    if total_weight <= 0:
        if snap.progress_percent is not None:
            return snap.progress_percent
        return snap.stage_progress_percent

    completed = 0.0
    for phase, weight in phases:
        frac = _phase_completion(snap, phase)
        if frac is None:
            continue
        completed += weight * frac

    pct = max(0.0, min(100.0, 100.0 * completed / total_weight))
    if pct > 0.0:
        return pct
    if snap.progress_percent is not None:
        return snap.progress_percent
    return snap.stage_progress_percent


class TrainingMonitorApp:
    def __init__(self, root: tk.Tk, root_dir: Path, refresh_seconds: float, stale_minutes: float) -> None:
        self.root = root
        self.root.title("Supermix Training Monitor")
        self.root.geometry("1520x920")
        self.root_dir = root_dir
        self.refresh_seconds = max(1.0, float(refresh_seconds))
        self.stale_minutes = max(1.0, float(stale_minutes))
        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.only_active_var = tk.BooleanVar(value=False)
        self.only_issues_var = tk.BooleanVar(value=False)
        self.filter_status_var = tk.StringVar(value="all")
        self.search_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.selected_progress_var = tk.StringVar(value="Progress: -")
        self.selected_eta_var = tk.StringVar(value="ETA: -")
        self.selected_ckpt_eta_var = tk.StringVar(value="Next Ckpt: -")
        self.selected_rate_var = tk.StringVar(value="Rate: -")
        self.selected_eta_confidence_var = tk.StringVar(value="ETA Confidence: -")
        self.fleet_progress_var = tk.StringVar(value="Fleet Progress: -")
        self.fleet_eta_var = tk.StringVar(value="Fleet ETA: -")
        self.trend_metric_var = tk.StringVar(value="progress")
        self.current_snapshots: Dict[str, RunSnapshot] = {}
        self.progress_history: Dict[str, List[Tuple[float, float]]] = {}
        self.display_progress_history: Dict[str, List[Tuple[float, float]]] = {}
        self.loss_history: Dict[str, List[Tuple[float, float]]] = {}
        self.lr_history: Dict[str, List[Tuple[float, float]]] = {}
        self.rate_history: Dict[str, List[Tuple[float, float]]] = {}
        self.sort_col = "updated"
        self.sort_reverse = True

        self._build_ui()
        self._load_settings()
        self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_refresh()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Workspace").pack(side=tk.LEFT)
        self.root_entry = ttk.Entry(top, width=62)
        self.root_entry.insert(0, str(self.root_dir))
        self.root_entry.pack(side=tk.LEFT, padx=(8, 10))

        ttk.Label(top, text="Stall mins").pack(side=tk.LEFT)
        self.stale_entry = ttk.Entry(top, width=7)
        self.stale_entry.insert(0, str(int(self.stale_minutes)))
        self.stale_entry.pack(side=tk.LEFT, padx=(8, 10))

        ttk.Label(top, text="Refresh s").pack(side=tk.LEFT)
        self.refresh_entry = ttk.Entry(top, width=7)
        self.refresh_entry.insert(0, str(int(self.refresh_seconds)))
        self.refresh_entry.pack(side=tk.LEFT, padx=(8, 10))

        ttk.Checkbutton(top, text="Auto", variable=self.auto_refresh_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(top, text="Refresh Now", command=self.refresh).pack(side=tk.LEFT)

        filter_row = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        filter_row.pack(fill=tk.X)
        ttk.Label(filter_row, text="Status").pack(side=tk.LEFT)
        self.status_combo = ttk.Combobox(
            filter_row,
            width=10,
            state="readonly",
            values=("all", "running", "stalled", "finished", "stopped", "unknown"),
            textvariable=self.filter_status_var,
        )
        self.status_combo.pack(side=tk.LEFT, padx=(8, 12))
        self.status_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh())

        ttk.Checkbutton(filter_row, text="Only Active", variable=self.only_active_var, command=self.refresh).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )
        ttk.Checkbutton(filter_row, text="Only Issues", variable=self.only_issues_var, command=self.refresh).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )
        ttk.Label(filter_row, text="Search").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(filter_row, width=34, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, padx=(8, 12))
        self.search_entry.bind("<Return>", lambda _e: self.refresh())
        ttk.Button(filter_row, text="Clear Filters", command=self._clear_filters).pack(side=tk.LEFT, padx=(0, 16))

        ttk.Button(filter_row, text="Open OUT Log", command=self._open_selected_out_log).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Open ERR Log", command=self._open_selected_err_log).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Open Run Dir", command=self._open_selected_run_dir).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Copy CMD/Launch", command=self._copy_selected_command).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Copy Details", command=self._copy_detail_to_clipboard).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Next Issue", command=self._select_next_issue).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Export JSON", command=self._export_snapshots_json).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Export CSV", command=self._export_snapshots_csv).pack(side=tk.LEFT)

        summary = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        summary.pack(fill=tk.X)
        self.progress_bar = ttk.Progressbar(summary, mode="determinate", maximum=100, length=420)
        self.progress_bar.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(summary, textvariable=self.selected_progress_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(summary, textvariable=self.selected_eta_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(summary, textvariable=self.selected_ckpt_eta_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(summary, textvariable=self.selected_rate_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(summary, textvariable=self.selected_eta_confidence_var).pack(side=tk.LEFT)

        fleet = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        fleet.pack(fill=tk.X)
        self.fleet_progress_bar = ttk.Progressbar(fleet, mode="determinate", maximum=100, length=420)
        self.fleet_progress_bar.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(fleet, textvariable=self.fleet_progress_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(fleet, textvariable=self.fleet_eta_var).pack(side=tk.LEFT)

        trend = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        trend.pack(fill=tk.X)
        ttk.Label(trend, text="Selected Run Trend (last hour):").pack(side=tk.LEFT, padx=(0, 8))
        self.trend_metric_combo = ttk.Combobox(
            trend,
            width=12,
            state="readonly",
            values=("progress", "loss", "lr", "rate"),
            textvariable=self.trend_metric_var,
        )
        self.trend_metric_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.trend_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._draw_selected_trend(self._selected_run_name()))
        self.trend_canvas = tk.Canvas(trend, width=620, height=80, background="#ffffff", highlightthickness=1)
        self.trend_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        table_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = (
            "run",
            "status",
            "stage",
            "sft",
            "pref",
            "pairs",
            "loss",
            "lr",
            "prog",
            "eta",
            "eta_conf",
            "ckpt_eta",
            "rate",
            "err",
            "stale",
            "updated",
            "pid",
        )
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=16)
        for col, width in (
            ("run", 290),
            ("status", 90),
            ("stage", 130),
            ("sft", 80),
            ("pref", 80),
            ("pairs", 120),
            ("loss", 90),
            ("lr", 95),
            ("prog", 90),
            ("eta", 90),
            ("eta_conf", 90),
            ("ckpt_eta", 90),
            ("rate", 95),
            ("err", 85),
            ("stale", 75),
            ("updated", 165),
            ("pid", 80),
        ):
            heading = "WORK" if col == "pairs" else col.upper()
            self.tree.heading(col, text=heading, command=lambda c=col: self._sort_by_column(c))
            self.tree.column(col, width=width, anchor=tk.CENTER)
        self.tree.column("run", anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        detail = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        detail.pack(fill=tk.BOTH, expand=True)
        self.detail_text = tk.Text(detail, wrap=tk.NONE, height=22, font=("Consolas", 10))
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll = ttk.Scrollbar(detail, orient=tk.VERTICAL, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)

        self.tree.tag_configure("running", background="#e9f7ef")
        self.tree.tag_configure("stalled", background="#fdecea")
        self.tree.tag_configure("finished", background="#eef4fd")
        self.tree.tag_configure("stopped", background="#fff8e6")
        self.tree.tag_configure("err_error", foreground="#9b1c1c")
        self.tree.tag_configure("err_warn", foreground="#8a6d3b")

    def _schedule_refresh(self) -> None:
        self.root.after(int(self.refresh_seconds * 1000), self._tick)

    def _tick(self) -> None:
        if self.auto_refresh_var.get():
            self.refresh()
        self._schedule_refresh()

    def _clear_filters(self) -> None:
        self.filter_status_var.set("all")
        self.only_active_var.set(False)
        self.only_issues_var.set(False)
        self.search_var.set("")
        self.refresh()

    def _selected_snapshot(self) -> Optional[RunSnapshot]:
        run_name = self._selected_run_name()
        if not run_name:
            return None
        return self.current_snapshots.get(run_name)

    def _open_path(self, path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif os.name == "posix":
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass

    def _open_selected_out_log(self) -> None:
        snap = self._selected_snapshot()
        if snap is not None:
            self._open_path(snap.out_log)

    def _open_selected_err_log(self) -> None:
        snap = self._selected_snapshot()
        if snap is not None and snap.err_log is not None:
            self._open_path(snap.err_log)

    def _open_selected_run_dir(self) -> None:
        snap = self._selected_snapshot()
        if snap is not None:
            self._open_path(snap.out_log.parent)

    def _settings_path(self) -> Path:
        return self.root_dir / ".training_monitor_gui_state.json"

    def _load_settings(self) -> None:
        path = self._settings_path()
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        try:
            status = str(raw.get("filter_status", "all"))
            if status in {"all", "running", "stalled", "finished", "stopped", "unknown"}:
                self.filter_status_var.set(status)
            self.only_active_var.set(bool(raw.get("only_active", False)))
            self.only_issues_var.set(bool(raw.get("only_issues", False)))
            self.search_var.set(str(raw.get("search", "")))
            self.auto_refresh_var.set(bool(raw.get("auto_refresh", True)))
            trend_metric = str(raw.get("trend_metric", "progress")).strip().lower()
            if trend_metric in {"progress", "loss", "lr", "rate"}:
                self.trend_metric_var.set(trend_metric)

            sort_col = str(raw.get("sort_col", self.sort_col))
            if sort_col:
                self.sort_col = sort_col
            self.sort_reverse = bool(raw.get("sort_reverse", self.sort_reverse))

            root_saved = str(raw.get("root_dir", "")).strip()
            if root_saved:
                self.root_entry.delete(0, tk.END)
                self.root_entry.insert(0, root_saved)
                self.root_dir = Path(root_saved)

            refresh_val = raw.get("refresh_seconds")
            if refresh_val is not None:
                self.refresh_entry.delete(0, tk.END)
                self.refresh_entry.insert(0, str(refresh_val))
            stale_val = raw.get("stale_minutes")
            if stale_val is not None:
                self.stale_entry.delete(0, tk.END)
                self.stale_entry.insert(0, str(stale_val))
        except Exception:
            return

    def _save_settings(self) -> None:
        try:
            payload = {
                "root_dir": str(self.root_entry.get().strip() or self.root_dir),
                "refresh_seconds": self.refresh_entry.get().strip(),
                "stale_minutes": self.stale_entry.get().strip(),
                "filter_status": str(self.filter_status_var.get()),
                "only_active": bool(self.only_active_var.get()),
                "only_issues": bool(self.only_issues_var.get()),
                "search": str(self.search_var.get()),
                "auto_refresh": bool(self.auto_refresh_var.get()),
                "trend_metric": str(self.trend_metric_var.get()),
                "sort_col": self.sort_col,
                "sort_reverse": bool(self.sort_reverse),
            }
            self._settings_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    def _copy_selected_command(self) -> None:
        snap = self._selected_snapshot()
        if snap is None:
            return
        payload = snap.command_line if snap.command_line else snap.launch_command
        if not payload:
            self.status_var.set(f"No command available for {snap.run_name}")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(payload)
            self.status_var.set(f"Copied command/launch line for {snap.run_name}")
        except Exception:
            pass

    def _copy_detail_to_clipboard(self) -> None:
        payload = self.detail_text.get("1.0", tk.END).strip()
        if not payload:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(payload)
            run_name = self._selected_run_name() or "selection"
            self.status_var.set(f"Copied detail panel for {run_name}")
        except Exception:
            pass

    def _is_issue_snapshot(self, snap: RunSnapshot) -> bool:
        if snap.status in {"stalled", "stopped"}:
            return True
        if snap.status == "unknown":
            return True
        if snap.err_signal in {"error", "warn"}:
            return True
        return False

    def _select_next_issue(self) -> None:
        iids = [str(x) for x in self.tree.get_children()]
        if not iids:
            return
        selected = self._selected_run_name()
        try:
            start_idx = iids.index(selected) + 1 if selected in iids else 0
        except Exception:
            start_idx = 0

        ordered = iids[start_idx:] + iids[:start_idx]
        for iid in ordered:
            snap = self.current_snapshots.get(iid)
            if snap is None:
                continue
            if not self._is_issue_snapshot(snap):
                continue
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
            self.on_select()
            self.status_var.set(f"Selected issue run: {iid}")
            return
        self.status_var.set("No issue runs in current view")

    def _display_progress_percent(self, snap: RunSnapshot) -> Optional[float]:
        return _compute_display_progress_percent(snap)

    def _display_eta_seconds(self, snap: RunSnapshot) -> Optional[float]:
        if snap.eta_seconds is not None:
            return snap.eta_seconds
        return snap.stage_eta_seconds

    def _display_rate_text(self, snap: RunSnapshot) -> str:
        if snap.step_rate_per_hour is not None:
            return f"{snap.step_rate_per_hour:.2f}/h"
        return snap.stage_rate_label

    def _display_work_text(self, snap: RunSnapshot) -> str:
        if snap.stage_progress_label and snap.stage_progress_label != "-":
            return snap.stage_progress_label
        if snap.pref_pairs > 0:
            return str(snap.pref_pairs)
        return "-"

    def _snapshot_row(self, snap: RunSnapshot) -> Dict[str, object]:
        return {
            "run": snap.run_name,
            "status": snap.status,
            "stage": snap.stage,
            "pid": snap.pid,
            "sft_step": snap.sft_step,
            "sft_target_steps": snap.sft_target_steps,
            "pref_step": snap.pref_step,
            "pref_target_steps": snap.pref_target_steps,
            "has_distill_stage": snap.has_distill_stage,
            "has_pref_mining_stage": snap.has_pref_mining_stage,
            "pref_pairs": snap.pref_pairs,
            "loss": snap.loss,
            "lr": snap.lr,
            "beta": snap.beta,
            "margin": snap.margin,
            "progress_units": snap.progress_units,
            "total_units": snap.total_units,
            "progress_percent": snap.progress_percent,
            "display_progress_percent": self._display_progress_percent(snap),
            "stage_progress_label": snap.stage_progress_label,
            "stage_progress_percent": snap.stage_progress_percent,
            "eta_seconds": snap.eta_seconds,
            "stage_eta_seconds": snap.stage_eta_seconds,
            "checkpoint_eta_seconds": snap.checkpoint_eta_seconds,
            "step_rate_per_hour": snap.step_rate_per_hour,
            "stage_rate_label": snap.stage_rate_label,
            "checkpoint_count": snap.checkpoint_count,
            "last_checkpoint_stage": snap.last_checkpoint_stage,
            "last_checkpoint_step": snap.last_checkpoint_step,
            "save_every_steps": snap.save_every_steps,
            "stale_minutes": snap.stale_minutes,
            "out_log": str(snap.out_log),
            "err_log": str(snap.err_log) if snap.err_log is not None else "",
            "err_signal": snap.err_signal,
            "err_summary": snap.err_summary,
            "health_summary": snap.health_summary,
            "out_last_write": _fmt_ts(snap.out_last_write_ts),
            "err_last_write": _fmt_ts(snap.err_last_write_ts) if snap.err_last_write_ts is not None else "",
            "launch_hint": snap.launch_hint,
            "command_line": snap.command_line,
            "launch_command": snap.launch_command,
            "data_summary": snap.data_summary,
            "sft_filter_summary": snap.sft_filter_summary,
            "distill_summary": snap.distill_summary,
            "pref_mining_summary": snap.pref_mining_summary,
            "pref_selection_summary": snap.pref_selection_summary,
        }

    def _export_snapshots_json(self) -> None:
        rows = [self._snapshot_row(s) for s in self.current_snapshots.values()]
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_path = self.root_dir / f"training_monitor_snapshot_{ts}.json"
        try:
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            self.status_var.set(f"Exported JSON snapshot: {out_path}")
        except Exception as e:
            self.status_var.set(f"JSON export failed: {e}")

    def _export_snapshots_csv(self) -> None:
        rows = [self._snapshot_row(s) for s in self.current_snapshots.values()]
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_path = self.root_dir / f"training_monitor_snapshot_{ts}.csv"
        if not rows:
            self.status_var.set("CSV export skipped: no visible rows")
            return
        try:
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            self.status_var.set(f"Exported CSV snapshot: {out_path}")
        except Exception as e:
            self.status_var.set(f"CSV export failed: {e}")

    def _update_fleet_summary(self, all_snapshots: Sequence[RunSnapshot]) -> None:
        runs = [s for s in all_snapshots if s.status in {"running", "stalled"}]
        if not runs:
            self.fleet_progress_bar["value"] = 0.0
            self.fleet_progress_var.set("Fleet Progress: -")
            self.fleet_eta_var.set("Fleet ETA: -")
            return

        progress_rows: List[Tuple[float, float]] = []
        for snap in runs:
            pct = self._display_progress_percent(snap)
            if pct is None:
                continue
            weight = float(snap.total_units) if snap.total_units is not None and snap.total_units > 0 else 1.0
            progress_rows.append((pct, max(1.0, weight)))

        if progress_rows:
            total_weight = sum(weight for _, weight in progress_rows)
            pct = sum(pct * weight for pct, weight in progress_rows) / max(1.0, total_weight)
            self.fleet_progress_bar["value"] = pct
            self.fleet_progress_var.set(f"Fleet Progress: {pct:.1f}% ({len(runs)} active)")
        else:
            self.fleet_progress_bar["value"] = 0.0
            self.fleet_progress_var.set(f"Fleet Progress: unknown ({len(runs)} active)")

        etas = [self._display_eta_seconds(s) for s in runs]
        etas = [eta for eta in etas if eta is not None]
        if etas:
            fleet_eta = max(float(eta) for eta in etas)
            self.fleet_eta_var.set(f"Fleet ETA: {_fmt_eta(fleet_eta)}")
        else:
            self.fleet_eta_var.set("Fleet ETA: -")

    def _draw_selected_trend(self, run_name: Optional[str]) -> None:
        self.trend_canvas.delete("all")
        if not run_name:
            self.trend_canvas.create_text(10, 40, anchor="w", text="No run selected", fill="#777777")
            return

        metric = str(self.trend_metric_var.get() or "progress").strip().lower()
        if metric == "loss":
            history = self.loss_history.get(run_name, [])
            label = "loss"
            formatter = lambda v: f"{v:.4f}"
        elif metric == "lr":
            history = self.lr_history.get(run_name, [])
            label = "lr"
            formatter = lambda v: f"{v:.3g}"
        elif metric == "rate":
            history = self.rate_history.get(run_name, [])
            label = "steps/h"
            formatter = lambda v: f"{v:.2f}"
        else:
            history = self.display_progress_history.get(run_name, [])
            label = "progress"
            formatter = lambda v: f"{v:.1f}%"

        if len(history) < 2:
            self.trend_canvas.create_text(10, 40, anchor="w", text="Collecting trend data...", fill="#777777")
            return

        w = int(self.trend_canvas.winfo_width() or 620)
        h = int(self.trend_canvas.winfo_height() or 80)
        pad = 6
        now = time.time()
        start_ts = now - 3600.0
        points = [(t, v) for (t, v) in history if t >= start_ts]
        if len(points) < 2:
            points = history[-2:]
        t_min = points[0][0]
        t_max = points[-1][0]
        v_min = min(v for _, v in points)
        v_max = max(v for _, v in points)
        if t_max - t_min < 1e-6:
            t_max = t_min + 1.0
        if v_max - v_min < 1e-6:
            v_max = v_min + 1.0

        for frac in (0.25, 0.5, 0.75):
            y = pad + (h - 2 * pad) * frac
            self.trend_canvas.create_line(pad, y, w - pad, y, fill="#f0f0f0")
        self.trend_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline="#dddddd")
        xy: List[float] = []
        for t, v in points:
            x = pad + (w - 2 * pad) * ((t - t_min) / (t_max - t_min))
            y = (h - pad) - (h - 2 * pad) * ((v - v_min) / (v_max - v_min))
            xy.extend([x, y])
        if len(xy) >= 4:
            self.trend_canvas.create_line(*xy, fill="#2c7be5", width=2, smooth=True)
        self.trend_canvas.create_text(
            w - pad - 4,
            pad + 2,
            anchor="ne",
            text=f"{label}: {formatter(points[-1][1])} | min {formatter(v_min)} max {formatter(v_max)}",
            fill="#444444",
        )

    def _append_history_point(
        self,
        store: Dict[str, List[Tuple[float, float]]],
        run_name: str,
        ts: float,
        value: float,
        keep_seconds: float = 3600.0,
        keep_points: int = 600,
    ) -> List[Tuple[float, float]]:
        history = store.setdefault(run_name, [])
        history.append((float(ts), float(value)))
        min_ts = float(ts) - float(keep_seconds)
        history[:] = [x for x in history if x[0] >= min_ts]
        if len(history) > keep_points:
            history[:] = history[-keep_points:]
        return history

    def _eta_confidence_for_snapshot(self, snap: RunSnapshot) -> str:
        if snap.eta_seconds is None:
            return "-"
        history = self.progress_history.get(snap.run_name, [])
        if len(history) < 4:
            return "low"

        rates: List[float] = []
        for idx in range(1, len(history)):
            t0, v0 = history[idx - 1]
            t1, v1 = history[idx]
            dt = t1 - t0
            dv = v1 - v0
            if dt >= 15.0 and dv > 0:
                rates.append(dv / dt)

        if len(rates) < 3:
            return "low"
        mean_rate = sum(rates) / float(len(rates))
        if mean_rate <= 0:
            return "low"
        var = sum((r - mean_rate) ** 2 for r in rates) / float(len(rates))
        cv = (var ** 0.5) / mean_rate
        if cv < 0.15:
            return "high"
        if cv < 0.45:
            return "medium"
        return "low"

    def _apply_filters(self, snapshots: Sequence[RunSnapshot]) -> List[RunSnapshot]:
        out: List[RunSnapshot] = []
        status_filter = str(self.filter_status_var.get() or "all").strip().lower()
        search = str(self.search_var.get() or "").strip().lower()
        only_active = bool(self.only_active_var.get())
        only_issues = bool(self.only_issues_var.get())
        for snap in snapshots:
            if status_filter != "all" and snap.status != status_filter:
                continue
            if only_active and snap.status not in {"running", "stalled"}:
                continue
            if only_issues and not self._is_issue_snapshot(snap):
                continue
            if search:
                hay = " ".join(
                    [
                        snap.run_name.lower(),
                        snap.stage.lower(),
                        snap.status.lower(),
                        (snap.command_line or "").lower(),
                        (snap.launch_command or "").lower(),
                        (snap.health_summary or "").lower(),
                        (snap.err_summary or "").lower(),
                        (snap.data_summary or "").lower(),
                        (snap.sft_filter_summary or "").lower(),
                        (snap.distill_summary or "").lower(),
                        (snap.pref_mining_summary or "").lower(),
                        (snap.pref_selection_summary or "").lower(),
                    ]
                )
                if search not in hay:
                    continue
            out.append(snap)
        return out

    def _sort_key(self, snap: RunSnapshot, col: str):
        if col == "run":
            return snap.run_name.lower()
        if col == "status":
            return snap.status
        if col == "stage":
            return snap.stage
        if col == "sft":
            return snap.sft_step
        if col == "pref":
            return snap.pref_step
        if col == "pairs":
            prog = self._display_progress_percent(snap)
            if prog is not None:
                return prog
            return snap.pref_pairs
        if col == "loss":
            return -1e18 if snap.loss is None else snap.loss
        if col == "lr":
            return -1e18 if snap.lr is None else snap.lr
        if col == "prog":
            prog = self._display_progress_percent(snap)
            return -1e18 if prog is None else prog
        if col == "eta":
            eta = self._display_eta_seconds(snap)
            return 1e18 if eta is None else eta
        if col == "eta_conf":
            score = {"high": 3, "medium": 2, "low": 1, "-": 0}.get(self._eta_confidence_for_snapshot(snap), 0)
            return score
        if col == "ckpt_eta":
            return 1e18 if snap.checkpoint_eta_seconds is None else snap.checkpoint_eta_seconds
        if col == "rate":
            return -1e18 if snap.step_rate_per_hour is None else snap.step_rate_per_hour
        if col == "err":
            score = {"error": 2, "warn": 1, "ok": 0}.get(snap.err_signal, -1)
            return score
        if col == "stale":
            return snap.stale_minutes
        if col == "updated":
            return snap.out_last_write_ts
        if col == "pid":
            return -1 if snap.pid is None else snap.pid
        return snap.run_name.lower()

    def _sort_by_column(self, col: str) -> None:
        if self.sort_col == col:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_col = col
            self.sort_reverse = True if col in {"updated", "prog", "rate", "loss", "sft", "pref", "eta_conf"} else False
        self.refresh()

    def _apply_eta_and_rate(self, snapshots: Sequence[RunSnapshot]) -> None:
        now = time.time()
        for snap in snapshots:
            history = self._append_history_point(
                store=self.progress_history,
                run_name=snap.run_name,
                ts=now,
                value=float(snap.progress_units),
            )
            display_progress = self._display_progress_percent(snap)
            if display_progress is not None:
                self._append_history_point(
                    store=self.display_progress_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(display_progress),
                )
            if snap.loss is not None:
                self._append_history_point(
                    store=self.loss_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.loss),
                )
            if snap.lr is not None:
                self._append_history_point(
                    store=self.lr_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.lr),
                )

            rate_per_sec = 0.0
            if len(history) >= 2:
                t1, v1 = history[-1]
                for t0, v0 in history[:-1]:
                    dt = t1 - t0
                    dv = v1 - v0
                    if dt >= 60.0 and dv > 0:
                        rate_per_sec = dv / dt
                        break

            if rate_per_sec > 0:
                snap.step_rate_per_hour = float(rate_per_sec * 3600.0)
                self._append_history_point(
                    store=self.rate_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.step_rate_per_hour),
                )
            else:
                snap.step_rate_per_hour = None

            if snap.stage == "done" or snap.status == "finished":
                snap.eta_seconds = 0.0
                snap.checkpoint_eta_seconds = 0.0
                continue
            if snap.total_units is not None and snap.total_units > 0 and rate_per_sec > 0:
                remaining = max(0.0, float(snap.total_units) - float(snap.progress_units))
                snap.eta_seconds = float(remaining / rate_per_sec)
            else:
                snap.eta_seconds = None

            next_ckpt = _next_checkpoint_step(
                stage=snap.stage,
                save_every_steps=snap.save_every_steps,
                sft_step=snap.sft_step,
                pref_step=snap.pref_step,
            )
            if next_ckpt is None or rate_per_sec <= 0:
                snap.checkpoint_eta_seconds = None
            else:
                if snap.stage in {"preference", "preference_mining"}:
                    remaining_steps = max(0, next_ckpt - snap.pref_step)
                else:
                    remaining_steps = max(0, next_ckpt - snap.sft_step)
                snap.checkpoint_eta_seconds = float(remaining_steps / rate_per_sec)

    def refresh(self) -> None:
        root_text = self.root_entry.get().strip()
        stale_text = self.stale_entry.get().strip()
        refresh_text = self.refresh_entry.get().strip()

        if root_text:
            self.root_dir = Path(root_text)
        try:
            self.stale_minutes = max(1.0, float(stale_text))
        except Exception:
            pass
        try:
            self.refresh_seconds = max(1.0, float(refresh_text))
        except Exception:
            pass

        all_snapshots = collect_run_snapshots(self.root_dir, stale_minutes_threshold=self.stale_minutes)
        self._apply_eta_and_rate(all_snapshots)
        filtered = self._apply_filters(all_snapshots)
        snapshots = sorted(
            filtered,
            key=lambda s: self._sort_key(s, self.sort_col),
            reverse=bool(self.sort_reverse),
        )
        self.current_snapshots = {s.run_name: s for s in snapshots}

        selected_name = self._selected_run_name()
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        for snap in snapshots:
            loss_txt = "-" if snap.loss is None else f"{snap.loss:.4f}"
            lr_txt = "-" if snap.lr is None else f"{snap.lr:.3g}"
            display_progress = self._display_progress_percent(snap)
            prog_txt = "-" if display_progress is None else f"{display_progress:.1f}%"
            eta_txt = _fmt_eta(self._display_eta_seconds(snap))
            eta_conf_txt = self._eta_confidence_for_snapshot(snap)
            if eta_conf_txt != "-":
                eta_conf_txt = eta_conf_txt.upper()
            ckpt_eta_txt = _fmt_eta(snap.checkpoint_eta_seconds)
            rate_txt = self._display_rate_text(snap)
            err_txt = {"error": "ERROR", "warn": "WARN", "ok": "OK"}.get(snap.err_signal, "-")
            pid_txt = "-" if snap.pid is None else str(snap.pid)
            updated = _fmt_ts(snap.out_last_write_ts)

            sft_txt = str(snap.sft_step)
            if snap.sft_target_steps is not None and snap.sft_target_steps > 0:
                sft_txt = f"{snap.sft_step}/{snap.sft_target_steps}"
            pref_txt = str(snap.pref_step)
            if snap.pref_target_steps is not None and snap.pref_target_steps > 0:
                pref_txt = f"{snap.pref_step}/{snap.pref_target_steps}"

            row_tags = [snap.status]
            if snap.err_signal == "error":
                row_tags.append("err_error")
            elif snap.err_signal == "warn":
                row_tags.append("err_warn")

            self.tree.insert(
                "",
                tk.END,
                iid=snap.run_name,
                values=(
                    snap.run_name,
                    snap.status,
                    snap.stage,
                    sft_txt,
                    pref_txt,
                    self._display_work_text(snap),
                    loss_txt,
                    lr_txt,
                    prog_txt,
                    eta_txt,
                    eta_conf_txt,
                    ckpt_eta_txt,
                    rate_txt,
                    err_txt,
                    f"{snap.stale_minutes:.1f}",
                    updated,
                    pid_txt,
                ),
                tags=tuple(row_tags),
            )

        if selected_name and selected_name in self.current_snapshots:
            self.tree.selection_set(selected_name)
            self.on_select()
        elif snapshots:
            self.tree.selection_set(snapshots[0].run_name)
            self.on_select()
        else:
            self.progress_bar["value"] = 0.0
            self.selected_progress_var.set("Progress: -")
            self.selected_eta_var.set("ETA: -")
            self.selected_ckpt_eta_var.set("Next Ckpt: -")
            self.selected_rate_var.set("Rate: -")
            self.selected_eta_confidence_var.set("ETA Confidence: -")

        self._update_fleet_summary(all_snapshots)

        running_count = sum(1 for s in all_snapshots if s.status == "running")
        stalled_count = sum(1 for s in all_snapshots if s.status == "stalled")
        finished_count = sum(1 for s in all_snapshots if s.status == "finished")
        error_count = sum(1 for s in all_snapshots if s.err_signal == "error")
        warn_count = sum(1 for s in all_snapshots if s.err_signal == "warn")
        issue_count = sum(1 for s in all_snapshots if self._is_issue_snapshot(s))
        best_eta = None
        for s in all_snapshots:
            eta_val = self._display_eta_seconds(s)
            if eta_val is None:
                continue
            if best_eta is None or eta_val < best_eta:
                best_eta = eta_val

        self.status_var.set(
            f"Visible: {len(snapshots)} / Total: {len(all_snapshots)} | Running: {running_count} "
            f"Stalled: {stalled_count} Finished: {finished_count} | Issues: {issue_count} "
            f"(Err {error_count}/Warn {warn_count}) | Best ETA: {_fmt_eta(best_eta)} "
            f"| Last refresh: {_fmt_ts(time.time())} | Root: {self.root_dir}"
        )
        self._draw_selected_trend(self._selected_run_name())

    def _selected_run_name(self) -> Optional[str]:
        sel = self.tree.selection()
        if not sel:
            return None
        return str(sel[0])

    def on_select(self, _event=None) -> None:
        run_name = self._selected_run_name()
        if not run_name:
            self.selected_progress_var.set("Progress: -")
            self.selected_eta_var.set("ETA: -")
            self.selected_ckpt_eta_var.set("Next Ckpt: -")
            self.selected_rate_var.set("Rate: -")
            self.selected_eta_confidence_var.set("ETA Confidence: -")
            self._draw_selected_trend(None)
            return
        snap = self.current_snapshots.get(run_name)
        if snap is None:
            self.selected_progress_var.set("Progress: -")
            self.selected_eta_var.set("ETA: -")
            self.selected_ckpt_eta_var.set("Next Ckpt: -")
            self.selected_rate_var.set("Rate: -")
            self.selected_eta_confidence_var.set("ETA Confidence: -")
            self._draw_selected_trend(None)
            return

        display_progress = self._display_progress_percent(snap)
        if display_progress is not None:
            self.progress_bar["value"] = max(0.0, min(100.0, display_progress))
            if snap.stage_progress_percent is not None and snap.stage in {"data", "distill", "preference_mining"}:
                self.selected_progress_var.set(
                    f"Progress: {display_progress:.2f}% overall | {snap.stage_progress_percent:.1f}% {snap.stage}"
                )
            else:
                self.selected_progress_var.set(f"Progress: {display_progress:.2f}%")
        else:
            self.progress_bar["value"] = 0.0
            self.selected_progress_var.set("Progress: -")
        if snap.eta_seconds is not None:
            self.selected_eta_var.set(f"ETA: {_fmt_eta(snap.eta_seconds)}")
        elif snap.stage_eta_seconds is not None:
            self.selected_eta_var.set(f"ETA: {_fmt_eta(snap.stage_eta_seconds)} ({snap.stage})")
        else:
            self.selected_eta_var.set("ETA: -")
        self.selected_ckpt_eta_var.set(f"Next Ckpt: {_fmt_eta(snap.checkpoint_eta_seconds)}")
        if snap.step_rate_per_hour is None:
            if snap.stage_rate_label != "-" and snap.stage_rate_label:
                self.selected_rate_var.set(f"Rate: {snap.stage_rate_label} ({snap.stage})")
            else:
                self.selected_rate_var.set("Rate: -")
        else:
            self.selected_rate_var.set(f"Rate: {snap.step_rate_per_hour:.2f} steps/hour")
        eta_conf = self._eta_confidence_for_snapshot(snap)
        self.selected_eta_confidence_var.set(
            f"ETA Confidence: {'-' if eta_conf == '-' else eta_conf.upper()}"
        )

        out_last = _fmt_ts(snap.out_last_write_ts)
        err_last = "-" if snap.err_last_write_ts is None else _fmt_ts(snap.err_last_write_ts)
        total_txt = "-" if snap.total_units is None else f"{snap.total_units:.0f}"
        prog_units_txt = f"{snap.progress_units:.0f}"
        display_progress_txt = self._display_progress_percent(snap)
        command_line = snap.command_line if snap.command_line else "-"
        launch_hint = snap.launch_hint if snap.launch_hint else "-"
        launch_command = snap.launch_command if snap.launch_command else "-"

        lines = [
            f"run: {snap.run_name}",
            f"status: {snap.status}",
            f"health: {snap.health_summary}",
            f"stage: {snap.stage}",
            f"pid: {snap.pid if snap.pid is not None else '-'} (alive={snap.pid_alive})",
            f"sft_step: {snap.sft_step} / {snap.sft_target_steps if snap.sft_target_steps is not None else '-'}",
            f"pref_step: {snap.pref_step} / {snap.pref_target_steps if snap.pref_target_steps is not None else '-'}",
            f"pref_pairs: {snap.pref_pairs}",
            f"loss: {snap.loss if snap.loss is not None else '-'}",
            f"lr: {snap.lr if snap.lr is not None else '-'}",
            f"beta: {snap.beta if snap.beta is not None else '-'}",
            f"margin: {snap.margin if snap.margin is not None else '-'}",
            f"last_checkpoint: stage={snap.last_checkpoint_stage} step={snap.last_checkpoint_step}",
            f"save_every_steps: {snap.save_every_steps if snap.save_every_steps is not None else '-'}",
            f"checkpoints_seen: {snap.checkpoint_count}",
            f"progress_units: {prog_units_txt} / {total_txt}",
            f"progress_percent: {snap.progress_percent if snap.progress_percent is not None else '-'}",
            f"display_progress_percent: {display_progress_txt if display_progress_txt is not None else '-'}",
            f"stage_progress: {snap.stage_progress_label}",
            f"stage_progress_percent: {snap.stage_progress_percent if snap.stage_progress_percent is not None else '-'}",
            f"eta: {_fmt_eta(snap.eta_seconds)}",
            f"stage_eta: {_fmt_eta(snap.stage_eta_seconds)}",
            f"eta_confidence: {eta_conf}",
            f"next_checkpoint_eta: {_fmt_eta(snap.checkpoint_eta_seconds)}",
            f"step_rate_per_hour: {snap.step_rate_per_hour if snap.step_rate_per_hour is not None else '-'}",
            f"stage_rate: {snap.stage_rate_label}",
            f"data_summary: {snap.data_summary}",
            f"sft_filter_summary: {snap.sft_filter_summary}",
            f"distill_summary: {snap.distill_summary}",
            f"pref_mining_summary: {snap.pref_mining_summary}",
            f"pref_selection_summary: {snap.pref_selection_summary}",
            f"trend_metric: {self.trend_metric_var.get()}",
            f"stale_minutes: {snap.stale_minutes:.2f}",
            f"err_signal: {snap.err_signal}",
            f"err_summary: {snap.err_summary}",
            f"out_log: {snap.out_log}",
            f"err_log: {snap.err_log if snap.err_log is not None else '-'}",
            f"pid_file: {snap.pid_file if snap.pid_file is not None else '-'}",
            f"launch_hint: {launch_hint}",
            f"launch_command: {launch_command}",
            f"command_line: {command_line}",
            f"out_size_bytes: {snap.out_size}",
            f"out_last_write: {out_last}",
            f"err_size_bytes: {snap.err_size}",
            f"err_last_write: {err_last}",
            "",
            "out tail:",
            "---------",
        ]
        lines.extend(snap.tail_lines[-40:])
        lines.append("")
        lines.append("err tail:")
        lines.append("---------")
        if snap.err_tail_lines:
            lines.extend(snap.err_tail_lines[-30:])
        else:
            lines.append("(empty)")

        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "\n".join(lines))
        self._draw_selected_trend(run_name)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Desktop GUI monitor for Supermix Qwen training logs.")
    ap.add_argument("--root", default=".", help="Project root containing train_*.out.log files.")
    ap.add_argument("--refresh_seconds", type=float, default=4.0, help="Auto-refresh interval.")
    ap.add_argument("--stale_minutes", type=float, default=20.0, help="Minutes without log updates to mark stalled.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root).resolve()
    root = tk.Tk()
    TrainingMonitorApp(
        root=root,
        root_dir=root_dir,
        refresh_seconds=float(args.refresh_seconds),
        stale_minutes=float(args.stale_minutes),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
