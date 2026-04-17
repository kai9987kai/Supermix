import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import time
import datetime
import threading
import winsound
import psutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import ttk

if os.name == "nt":
    import ctypes


TRAIN_STEP_RE = re.compile(
    r"\[train\] step=(\d+) loss=([0-9eE+\-.]+) lr=([0-9eE+\-.]+)(?: rdrop=([0-9eE+\-.]+))?"
)
PREF_STEP_RE = re.compile(
    r"\[pref\] step=(\d+) loss=([0-9eE+\-.]+) lr=([0-9eE+\-.]+)"
    r"(?: beta=([0-9eE+\-.]+) margin=([0-9eE+\-.]+))?"
    r"(?: wpo_std=([0-9eE+\-.]+))?"
)
PREF_PAIRS_RE = re.compile(r"\[pref\] pairs=(\d+)")
CHECKPOINT_RE = re.compile(r"\[checkpoint\] saved stage=(sft|preference) step=(\d+)")
RUNTIME_CONFIG_RE = re.compile(r"\[train\] runtime config: (.+)")
MATCHED_LORA_CONFIG_RE = re.compile(r"\[train\] matched LoRA config to init adapter: (.+)")
NEFTUNE_CONFIG_RE = re.compile(
    r"\[train\] NEFTune config: sft_noise_alpha=([0-9eE+\-.]+) preference_noise_alpha=([0-9eE+\-.]+)"
)
LORA_PLUS_RE = re.compile(r"\[train\] LoRA\+ optimizer groups: ratio=([0-9eE+\-.]+) base_group=(\d+) fast_group=(\d+)")
SOURCE_BALANCE_RE = re.compile(r"\[sft\] source balance factors: (.+)")
PREF_BUILD_RE = re.compile(r"\[pref\] building preference pairs \(mode=(\S+)\)\.\.\.")
PREF_OBJECTIVE_RE = re.compile(r"\[pref\] objective schedule: (.+)")
PREF_REFERENCE_RE = re.compile(r"\[pref\] reference margins cached for (\d+) pairs")
DATA_PROGRESS_RE = re.compile(
    r"\[data\] progress: pairs=(\d+)/(\d+) raw=(\d+) kept=(\d+) rate=([0-9eE+\-.]+)/s"
)
DATA_QUALITY_RE = re.compile(
    r"\[data\] quality filter: raw=(\d+) kept=(\d+) empty=(\d+) placeholder=(\d+) filtered=(\d+) deduped=(\d+) "
    r"source_cap=(\d+) synthetic_cap=(\d+) prompt_cap=(\d+) cap_relax=(\d+)"
)
DATA_SPLIT_RE = re.compile(r"\[data\] train=(\d+) eval=(\d+)(?: \(raw_eval=(\d+)\))?")
DATA_SYNTHETIC_RE = re.compile(r"\[data\] synthetic_kept=(\d+)/(\d+)")
DISTILL_CONFIG_RE = re.compile(
    r"\[distill\] config: target=(\d+) ratio=([0-9eE+\-.]+) max_seconds=(\S+) best_of=(\d+)"
    r"(?: min_gain=([0-9eE+\-.]+))?(?: density_bias=([0-9eE+\-.]+))?"
)
DISTILL_PROGRESS_RE = re.compile(
    r"\[distill\] progress: visited=(\d+)/(\d+) generated=(\d+) rate=([0-9eE+\-.]+)/s"
)
DISTILL_COMPLETE_RE = re.compile(
    r"\[distill\] complete: generated=(\d+) visited=(\d+)/(\d+) elapsed=([0-9eE+\-.]+)s"
)
SFT_QUALITY_RE = re.compile(
    r"\[sft\] quality filter(?: fallback)?: threshold=([0-9eE+\-.]+) kept=(\d+) dropped_quality=(\d+) "
    r"dropped_short=(\d+)(?: dropped_synthetic=(\d+))? exempt_sources=(\d+)"
)
SFT_QUALITY_FALLBACK_RE = re.compile(
    r"\[sft\] quality filter fallback: kept=(\d+) too small; using unfiltered set=(\d+)"
)
EVAL_FILTER_RE = re.compile(
    r"\[eval\] quality filter: threshold=([0-9eE+\-.]+) kept=(\d+) dropped_quality=(\d+) dropped_synthetic=(\d+)"
)
EVAL_FILTER_FALLBACK_RE = re.compile(
    r"\[eval\] quality filter fallback: kept=(\d+) after ranking top non-synthetic pairs "
    r"\(threshold=([0-9eE+\-.]+), dropped_quality=(\d+), dropped_synthetic=(\d+)\)"
)
PREF_MINING_CONFIG_RE = re.compile(
    r"\[pref\] mining config: mode=(\S+) generation=(\S+) target_pairs=(\d+) candidates=(\d+) max_attempts=(\d+)"
    r"(?: self_play_budget=(\d+) self_play_curriculum=(\S+) self_play_max_new_tokens=(\d+))? "
    r"selection=(\S+) keep_ratio=([0-9eE+\-.]+) max_seconds=(\S+)"
)
PREF_MINING_PROGRESS_RE = re.compile(
    r"\[pref\] mining progress: visited=(\d+)/(\d+) accepted=(\d+) rate=([0-9eE+\-.]+)/s"
)
PREF_MINING_COMPLETE_RE = re.compile(
    r"\[pref\] mining complete: pairs=(\d+) mined=(\d+) visited=(\d+) generation_failures=(\d+)"
    r"(?: brevity_filtered=(\d+) stop_reject_variants=(\d+))?"
    r"(?: self_play_prompts=(\d+) self_play_candidates=(\d+) self_play_failures=(\d+))?"
    r" elapsed=([0-9eE+\-.]+)s"
)
PREF_SELECTION_RE = re.compile(
    r"\[pref\] pair selection: strategy=(\S+) keep=(\d+)/(\d+) keep_ratio=([0-9eE+\-.]+) "
    r"gap=([0-9eE+\-.]+)->([0-9eE+\-.]+) sim=([0-9eE+\-.]+)->([0-9eE+\-.]+) "
    r"(?:conv=([0-9eE+\-.]+)->([0-9eE+\-.]+) )?"
    r"(?:reason=([0-9eE+\-.]+)->([0-9eE+\-.]+) )?"
    r"(?:creative=([0-9eE+\-.]+)->([0-9eE+\-.]+) )?"
    r"(?:density=([0-9eE+\-.]+)->([0-9eE+\-.]+) )?"
    r"selected_score_mean=([0-9eE+\-.]+)"
)
MAX_STEPS_RE = re.compile(r"--max_steps(?:\s+|=)(\d+)")
PREF_STEPS_RE = re.compile(r"--preference_steps(?:\s+|=)(\d+)")
SAVE_EVERY_STEPS_RE = re.compile(r"--save_every_steps(?:\s+|=)(\d+)")
PS1_FILE_RE = re.compile(r"-File\s+(\"[^\"]+\\.ps1\"|'[^']+\\.ps1'|[^\s]+\\.ps1)", flags=re.IGNORECASE)
RUN_CORE_RE = re.compile(r"^train_(.+?)_(\d{8}_\d{6})$")
RUN_TIMESTAMP_RE = re.compile(r"^(.*)_(\d{8}_\d{6})$")
SUMMARY_FRACTION_RE = re.compile(r"\b([a-z_]+)=(\d+)/(\d+)")
SUMMARY_COUNT_RE = re.compile(r"\b([a-z_]+)=(\d+)\b")
V46_MOD_RE = re.compile(r"\[v46\] mod_bypass=(\d+) skipped_blocks=(\d+) total_flops_saved=([0-9eE+\-.]+)")
V46_GOT_RE = re.compile(r"\[v46\] got_synthesis branches=(\d+) merge_confidence=([0-9eE+\-.]+) topology=(\w+)")
V46_CCOT_RE = re.compile(r"\[v46\] ccot_latent depth=(\d+) expansion_rate=([0-9eE+\-.]+) stability=([0-9eE+\-.]+)")

PROCESS_CMD_CACHE: Dict[int, Tuple[float, Optional[str]]] = {}
PS1_TARGET_CACHE: Dict[str, Tuple[float, Optional[int], Optional[int], Optional[int], bool, bool]] = {}
PROCESS_LIST_CACHE: Tuple[float, List["ProcessEntry"]] = (0.0, [])
GPU_CACHE: Tuple[float, List[Dict[str, str]]] = (0.0, [])
ACCELERATOR_CACHE: Tuple[float, Dict[str, Any]] = (0.0, {})
TORCH_BACKEND_CACHE: Tuple[float, Dict[str, str]] = (0.0, {})


def _module_available(module_name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _query_torch_backend_summary() -> Dict[str, str]:
    global TORCH_BACKEND_CACHE
    now = time.time()
    if now - TORCH_BACKEND_CACHE[0] < 15.0:
        return TORCH_BACKEND_CACHE[1]
    summary = {
        "torch": "unavailable",
        "resolved": "unknown",
        "cuda": "no",
        "dml": "yes" if _module_available("torch_directml") else "no",
        "npu": "yes" if _module_available("torch_npu") else "no",
        "qnn": "yes" if _module_available("onnxruntime_qnn") else "no",
    }
    try:
        import torch

        summary["torch"] = str(torch.__version__)
        resolved = "cpu"
        if torch.cuda.is_available():
            summary["cuda"] = "yes"
            resolved = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
            resolved = "npu"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            resolved = "xpu"
        elif summary["dml"] == "yes":
            resolved = "dml-ready"
        summary["resolved"] = resolved
    except Exception:
        pass
    TORCH_BACKEND_CACHE = (now, summary)
    return summary


def _coerce_json_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _run_powershell_json(script: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    raw = str(result.stdout or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_engine_instance(name: str) -> Tuple[str, str]:
    text = str(name or "").lower()
    phys_match = re.search(r"_phys_(\d+)", text)
    eng_match = re.search(r"_eng_(\d+)", text)
    engtype_match = re.search(r"_engtype_([^\\)]+)", text)
    phys_key = f"phys_{phys_match.group(1)}" if phys_match else "phys_0"
    eng_key = eng_match.group(1) if eng_match else "0"
    eng_type = engtype_match.group(1).replace("_", " ").strip() if engtype_match else "unknown"
    return phys_key, f"{eng_key}:{eng_type}"


def _extract_phys_key(name: str) -> str:
    text = str(name or "").lower()
    phys_match = re.search(r"_phys_(\d+)", text)
    return f"phys_{phys_match.group(1)}" if phys_match else "phys_0"


def _summarize_windows_accelerators(payload: Dict[str, Any]) -> Dict[str, Any]:
    adapters = _coerce_json_list(payload.get("adapters"))
    engines = _coerce_json_list(payload.get("engines"))
    memory = _coerce_json_list(payload.get("memory"))
    npu_devices = _coerce_json_list(payload.get("npu"))

    engine_util: Dict[str, Dict[str, float]] = {}
    for row in engines:
        phys_key, engine_key = _parse_engine_instance(str(row.get("Name", "")))
        util = 0.0
        try:
            util = max(0.0, float(row.get("UtilizationPercentage", 0.0) or 0.0))
        except Exception:
            util = 0.0
        phys_engines = engine_util.setdefault(phys_key, {})
        phys_engines[engine_key] = max(util, phys_engines.get(engine_key, 0.0))

    memory_by_phys: Dict[str, Dict[str, float]] = {}
    for row in memory:
        phys_key = _extract_phys_key(str(row.get("Name", "")))
        try:
            shared = float(row.get("SharedUsage", 0.0) or 0.0)
        except Exception:
            shared = 0.0
        try:
            committed = float(row.get("TotalCommitted", 0.0) or 0.0)
        except Exception:
            committed = 0.0
        try:
            dedicated = float(row.get("DedicatedUsage", 0.0) or 0.0)
        except Exception:
            dedicated = 0.0
        current = memory_by_phys.setdefault(phys_key, {"shared": 0.0, "committed": 0.0, "dedicated": 0.0})
        current["shared"] = max(current["shared"], shared)
        current["committed"] = max(current["committed"], committed)
        current["dedicated"] = max(current["dedicated"], dedicated)

    gpu_rows: List[Dict[str, Any]] = []
    adapter_names = [str(row.get("Name", "")).strip() for row in adapters if str(row.get("Name", "")).strip()]
    phys_keys = sorted(set(engine_util.keys()) | set(memory_by_phys.keys()) | {f"phys_{idx}" for idx in range(len(adapter_names))})
    for idx, phys_key in enumerate(phys_keys):
        engines_for_phys = engine_util.get(phys_key, {})
        total_util = max(engines_for_phys.values(), default=0.0)
        compute_util = max(
            (value for key, value in engines_for_phys.items() if key.endswith(":compute")),
            default=0.0,
        )
        graphics_util = max(
            (value for key, value in engines_for_phys.items() if key.endswith(":3d")),
            default=0.0,
        )
        video_util = max(
            (value for key, value in engines_for_phys.items() if "video" in key),
            default=0.0,
        )
        mem = memory_by_phys.get(phys_key, {})
        shared_gb = float(mem.get("shared", 0.0)) / (1024.0 ** 3)
        committed_gb = float(mem.get("committed", 0.0)) / (1024.0 ** 3)
        gpu_rows.append(
            {
                "name": adapter_names[idx] if idx < len(adapter_names) else phys_key.replace("_", " ").upper(),
                "util": total_util,
                "compute": compute_util,
                "graphics": graphics_util,
                "video": video_util,
                "shared_gb": shared_gb,
                "committed_gb": committed_gb,
            }
        )

    npu_rows = []
    for row in npu_devices:
        name = str(row.get("Name", "")).strip()
        if not name:
            continue
        npu_rows.append(
            {
                "name": name,
                "manufacturer": str(row.get("Manufacturer", "")).strip(),
                "status": str(row.get("Status", "")).strip() or "Unknown",
            }
        )

    return {
        "gpus": gpu_rows,
        "npus": npu_rows,
        "backend": _query_torch_backend_summary(),
    }


def _query_windows_accelerators() -> Dict[str, Any]:
    global ACCELERATOR_CACHE
    now = time.time()
    if now - ACCELERATOR_CACHE[0] < 3.0:
        return ACCELERATOR_CACHE[1]
    if os.name != "nt":
        ACCELERATOR_CACHE = (now, {})
        return {}
    script = """
$payload = [ordered]@{
  adapters = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
  memory = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | Select-Object Name, DedicatedUsage, SharedUsage, TotalCommitted)
  engines = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Select-Object Name, UtilizationPercentage)
  npu = @(Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'NPU|Hexagon|Ryzen AI|Intel AI Boost|Neural|Hailo|Movidius' } | Select-Object Name, Manufacturer, Status)
}
$payload | ConvertTo-Json -Compress -Depth 5
"""
    payload = _run_powershell_json(script)
    summary = _summarize_windows_accelerators(payload) if payload else {"backend": _query_torch_backend_summary(), "gpus": [], "npus": []}
    ACCELERATOR_CACHE = (now, summary)
    return summary


def _query_gpu_stats() -> List[Dict[str, str]]:
    """Query GPU stats via nvidia-smi. Returns a list of dicts per GPU."""
    global GPU_CACHE
    now = time.time()
    if now - GPU_CACHE[0] < 3.0:  # cache for 3 seconds
        return GPU_CACHE[1]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            GPU_CACHE = (now, [])
            return []
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "util": parts[2],
                    "mem_used": parts[3],
                    "mem_total": parts[4],
                    "temp": parts[5],
                    "power": parts[6],
                })
        GPU_CACHE = (now, gpus)
        return gpus
    except Exception:
        GPU_CACHE = (now, [])
        return []



@dataclass(frozen=True)
class ProcessEntry:
    pid: int
    parent_pid: Optional[int]
    name: str
    command_line: str


@dataclass(frozen=True)
class ResearchResult:
    timestamp: str
    commit: str
    run_tag: str
    output_dir: str
    benchmark_json: str
    token_f1_delta: float
    char_similarity_delta: float
    perplexity_delta: float
    avg_gen_seconds_delta: float
    status: str
    description: str


@dataclass(frozen=True)
class ResearchFailureInsight:
    summary_line: str
    prompt_line: str
    prediction_line: str
    sample_comparison_jsonl: str = ""


@dataclass
class RunSnapshot:
    run_name: str
    out_log: Path
    err_log: Optional[Path]
    pid_file: Optional[Path]
    pid: Optional[int]
    pid_source: str
    pid_alive: bool
    status: str
    stage: str
    sft_step: int
    pref_step: int
    pref_pairs: int
    loss: Optional[float]
    lr: Optional[float]
    rdrop: Optional[float]
    wpo_std: Optional[float]
    beta: Optional[float]
    margin: Optional[float]
    pref_objective: str
    pref_reference_pairs: Optional[int]
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
    cpu_percent: Optional[float]
    ram_gb: Optional[float]
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
    runtime_summary: str
    adapter_summary: str
    source_balance_summary: str
    objective_summary: str
    data_summary: str
    eval_summary: str
    sft_filter_summary: str
    distill_config_summary: str
    distill_summary: str
    pref_mining_summary: str
    pref_selection_summary: str
    tail_lines: List[str]
    err_tail_lines: List[str]
    v46_mod_summary: str = "-"
    v46_got_summary: str = "-"
    v46_ccot_summary: str = "-"
    state_file: Optional[Path] = None
    balance_loss: Optional[float] = None
    stage_source: str = "log"


@dataclass
class ParsedLog:
    stage: str = "unknown"
    sft_step: int = 0
    pref_step: int = 0
    pref_pairs: int = 0
    loss: Optional[float] = None
    lr: Optional[float] = None
    rdrop: Optional[float] = None
    wpo_std: Optional[float] = None
    beta: Optional[float] = None
    margin: Optional[float] = None
    pref_objective: str = "-"
    pref_reference_pairs: Optional[int] = None
    checkpoint_count: int = 0
    last_checkpoint_stage: str = "-"
    last_checkpoint_step: int = 0
    runtime_summary: str = "-"
    adapter_summary: str = "-"
    source_balance_summary: str = "-"
    objective_summary: str = "-"
    data_pairs_current: Optional[int] = None
    data_pairs_total: Optional[int] = None
    data_raw_count: Optional[int] = None
    data_kept_count: Optional[int] = None
    data_rate_per_sec: Optional[float] = None
    train_pairs_count: Optional[int] = None
    eval_pairs_count: Optional[int] = None
    raw_eval_pairs_count: Optional[int] = None
    data_synthetic_kept: Optional[int] = None
    data_synthetic_total: Optional[int] = None
    data_summary: str = "-"
    eval_summary: str = "-"
    sft_filter_summary: str = "-"
    distill_target: Optional[int] = None
    distill_config_summary: str = "-"
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
    pref_mining_brevity_filtered: Optional[int] = None
    pref_mining_stop_reject_variants: Optional[int] = None
    pref_mining_self_play_budget: Optional[int] = None
    pref_mining_self_play_curriculum: str = "-"
    pref_mining_self_play_max_new_tokens: Optional[int] = None
    pref_mining_self_play_prompts: Optional[int] = None
    pref_mining_self_play_candidates: Optional[int] = None
    pref_mining_self_play_failures: Optional[int] = None
    pref_mining_summary: str = "-"
    pref_selection_summary: str = "-"
    tail_lines: List[str] = field(default_factory=list)
    omni_batch_index: Optional[int] = None
    omni_total_batches: Optional[int] = None
    omni_elapsed_seconds: Optional[float] = None
    omni_eta_seconds: Optional[float] = None
    omni_avg_balance_loss: Optional[float] = None
    v46_mod_summary: str = "-"
    v46_got_summary: str = "-"
    v46_ccot_summary: str = "-"
    v46_mod_bypass: Optional[int] = None
    v46_got_merge_conf: Optional[float] = None
    v46_ccot_depth: Optional[int] = None


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


def _run_base_name(run_name: str) -> str:
    text = str(run_name or "").strip()
    m = RUN_TIMESTAMP_RE.match(text)
    return m.group(1) if m else text


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
        text = _decode_log_bytes(data)
    except Exception:
        return []

    lines = text.splitlines()
    if len(lines) > max_lines:
        return lines[-max_lines:]
    return lines


def _decode_log_bytes(data: bytes) -> str:
    if not data:
        return ""

    # Handle PowerShell/file redirection logs that sometimes end up UTF-16 encoded.
    if data.startswith((b"\xff\xfe", b"\xfe\xff")) or data.count(b"\x00") > max(8, len(data) // 10):
        for encoding in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return data.decode(encoding, errors="replace").replace("\x00", "")
            except Exception:
                continue

    try:
        return data.decode("utf-8-sig", errors="replace").replace("\x00", "")
    except Exception:
        return data.decode("latin-1", errors="replace").replace("\x00", "")


def _list_process_entries() -> List[ProcessEntry]:
    global PROCESS_LIST_CACHE

    now = time.time()
    cached_ts, cached_entries = PROCESS_LIST_CACHE
    if cached_entries and (now - cached_ts) <= 5.0:
        return cached_entries

    entries: List[ProcessEntry] = []
    try:
        if os.name == "nt":
            query = (
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
                "ConvertTo-Json -Compress"
            )
            cp = subprocess.run(
                ["powershell", "-NoProfile", "-Command", query],
                capture_output=True,
                text=True,
                timeout=6,
            )
            raw = cp.stdout.strip()
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    payload = [payload]
                if isinstance(payload, list):
                    for item in payload:
                        if not isinstance(item, dict):
                            continue
                        try:
                            pid = int(item.get("ProcessId"))
                        except Exception:
                            continue
                        parent_pid_raw = item.get("ParentProcessId")
                        try:
                            parent_pid = int(parent_pid_raw) if parent_pid_raw is not None else None
                        except Exception:
                            parent_pid = None
                        entries.append(
                            ProcessEntry(
                                pid=pid,
                                parent_pid=parent_pid,
                                name=str(item.get("Name") or ""),
                                command_line=str(item.get("CommandLine") or ""),
                            )
                        )
        else:
            cp = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,comm=,args=", "-ww"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            for raw_line in cp.stdout.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(None, 3)
                if len(parts) < 4:
                    continue
                try:
                    pid = int(parts[0])
                    parent_pid = int(parts[1])
                except Exception:
                    continue
                entries.append(
                    ProcessEntry(
                        pid=pid,
                        parent_pid=parent_pid,
                        name=str(parts[2]),
                        command_line=str(parts[3]),
                    )
                )
    except Exception:
        entries = []

    PROCESS_LIST_CACHE = (now, entries)
    return entries


def _signature_tokens(*parts: str) -> List[str]:
    tokens = set()
    for part in parts:
        for token in re.split(r"[^a-z0-9]+", str(part or "").lower()):
            if len(token) < 3:
                continue
            if token in {"train", "run", "source", "artifacts", "python", "powershell", "cmd", "exe"}:
                continue
            tokens.add(token)
    return sorted(tokens)


def _score_process_match(
    proc: ProcessEntry,
    run_name: str,
    out_log: Path,
    launch_hint: str,
) -> int:
    cmd_low = str(proc.command_line or "").lower()
    name_low = str(proc.name or "").lower()
    run_low = str(run_name or "").lower()
    out_name = out_log.name.lower()
    out_stem = out_log.stem.lower()
    hint_low = str(launch_hint or "").lower()
    hint_stem = Path(hint_low).stem.lower() if hint_low else ""
    core = _run_core(run_name)

    if not cmd_low:
        return -1
    if "training_monitor_gui.py" in cmd_low:
        return -1
    if "import training_monitor_gui" in cmd_low:
        return -1

    shell_wrapper = name_low.startswith("powershell") or name_low.startswith("cmd")

    score = 0
    if out_name and out_name in cmd_low:
        score += 80 if shell_wrapper else 220
    if out_stem and out_stem in cmd_low:
        score += 70 if shell_wrapper else 180
    if run_low and run_low in cmd_low:
        score += 60 if shell_wrapper else 150
    if core and len(core) >= 8 and core in cmd_low:
        score += 55 if shell_wrapper else 130
    if hint_stem and len(hint_stem) >= 8 and hint_stem in cmd_low:
        score += 45 if shell_wrapper else 110

    token_hits = 0
    for token in _signature_tokens(run_low, out_stem, core, hint_stem):
        if token in cmd_low:
            token_hits += 1
    score += token_hits * 12
    if token_hits >= 2:
        score += 25
    if token_hits >= 3:
        score += 40

    if name_low.startswith("python"):
        score += 35
    elif name_low.startswith("powershell"):
        score += 12
    elif name_low.startswith("cmd"):
        score += 4

    if "qwen_supermix_pipeline.py" in cmd_low or "finetune_chat.py" in cmd_low:
        score += 260

    if shell_wrapper:
        score = int(score * 0.7)

    return score


def _find_live_process_for_run(
    run_name: str,
    out_log: Path,
    launch_hint: str,
    processes: Optional[Sequence[ProcessEntry]] = None,
) -> Tuple[Optional[int], str, str]:
    proc_entries = list(processes) if processes is not None else _list_process_entries()
    best: Optional[ProcessEntry] = None
    best_score = -1

    for proc in proc_entries:
        score = _score_process_match(proc, run_name=run_name, out_log=out_log, launch_hint=launch_hint)
        if score > best_score:
            best = proc
            best_score = score

    if best is None or best_score < 60:
        return None, "", ""
    return int(best.pid), str(best.command_line or ""), "process_scan"


def _query_process_cmdline(pid: int) -> Optional[str]:
    for entry in _list_process_entries():
        if int(entry.pid) == int(pid):
            result = str(entry.command_line or "").strip()
            PROCESS_CMD_CACHE[pid] = (time.time(), result or None)
            return result or None

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
    base = _run_base_name(run_name)
    m = RUN_CORE_RE.match(base)
    if m:
        return m.group(1).strip().lower()
    if base.startswith("train_"):
        return base.replace("train_", "", 1).strip().lower()
    if base.endswith("_train"):
        return base[: -len("_train")].strip().lower()
    return base.strip().lower()


def _guess_ps1_from_run_name(run_name: str, root_dir: Path) -> Optional[Path]:
    core = _run_core(run_name)
    source_dir = root_dir / "source"
    if not source_dir.exists():
        return None
    candidates = list(source_dir.glob("run_train_*.ps1")) + list(source_dir.glob("run_*.ps1"))
    if not candidates:
        return None

    scored: List[Tuple[int, Path]] = []
    for p in candidates:
        stem_core = p.stem
        if stem_core.startswith("run_train_"):
            stem_core = stem_core.replace("run_train_", "", 1)
        elif stem_core.startswith("run_"):
            stem_core = stem_core.replace("run_", "", 1)
        stem_core = stem_core.strip().lower()
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

    if stage in {"dataset", "building_dataset", "building_dataset_from_resume"}:
        if parsed.omni_batch_index is not None and parsed.omni_total_batches is not None:
            return (
                f"{parsed.omni_batch_index}/{parsed.omni_total_batches} batches",
                _percent_complete(parsed.omni_batch_index, parsed.omni_total_batches),
                "-",
                parsed.omni_eta_seconds,
            )
        return ("dataset build", None, "-", parsed.omni_eta_seconds)

    if stage in {"stage1", "stage2"} and parsed.omni_total_batches is not None:
        current = parsed.omni_batch_index
        total = parsed.omni_total_batches
        rate_per_sec = None
        if current is not None and total is not None and parsed.omni_eta_seconds is not None:
            remaining = max(0.0, float(total) - float(current))
            if parsed.omni_eta_seconds > 0:
                rate_per_sec = remaining / float(parsed.omni_eta_seconds)
        if rate_per_sec is None and current is not None and parsed.omni_elapsed_seconds not in {None, 0.0}:
            rate_per_sec = float(current) / max(float(parsed.omni_elapsed_seconds or 0.0), 1.0)
        label = f"{current}/{total} batches" if current is not None else f"{total} batches"
        return (
            label,
            _percent_complete(current, total),
            _fmt_rate_per_sec(rate_per_sec),
            parsed.omni_eta_seconds,
        )

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

    if stage == "distill" and parsed.distill_target is not None:
        return (f"target={parsed.distill_target}", None, "-", None)

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

    if stage == "eval":
        kept = parsed.eval_pairs_count
        raw = parsed.raw_eval_pairs_count if parsed.raw_eval_pairs_count is not None else kept
        if kept is not None:
            label = f"{kept}/{raw} eval" if raw is not None else f"{kept} eval"
            return (label, _percent_complete(kept, raw), "-", None)
        return (parsed.eval_summary if parsed.eval_summary != "-" else "benchmark", None, "-", None)

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
            parsed.rdrop = float(m_train.group(4)) if m_train.group(4) is not None else parsed.rdrop
        m_pref = PREF_STEP_RE.search(line)
        if m_pref:
            parsed.pref_step = max(parsed.pref_step, int(m_pref.group(1)))
            parsed.loss = float(m_pref.group(2))
            parsed.lr = float(m_pref.group(3))
            parsed.beta = float(m_pref.group(4)) if m_pref.group(4) is not None else parsed.beta
            parsed.margin = float(m_pref.group(5)) if m_pref.group(5) is not None else parsed.margin
            parsed.wpo_std = float(m_pref.group(6)) if m_pref.group(6) is not None else parsed.wpo_std
        m_pairs = PREF_PAIRS_RE.search(line)
        if m_pairs:
            parsed.pref_pairs = max(parsed.pref_pairs, int(m_pairs.group(1)))
        m_ckpt = CHECKPOINT_RE.search(line)
        if m_ckpt:
            parsed.checkpoint_count += 1
            parsed.last_checkpoint_stage = m_ckpt.group(1)
            parsed.last_checkpoint_step = int(m_ckpt.group(2))
        m_runtime = RUNTIME_CONFIG_RE.search(line)
        if m_runtime:
            parsed.runtime_summary = m_runtime.group(1).strip()
        m_lora_match = MATCHED_LORA_CONFIG_RE.search(line)
        if m_lora_match:
            payload = m_lora_match.group(1).strip()
            parsed.adapter_summary = payload if parsed.adapter_summary == "-" else f"{parsed.adapter_summary} | {payload}"
        m_neftune = NEFTUNE_CONFIG_RE.search(line)
        if m_neftune:
            payload = (
                f"neftune_sft={float(m_neftune.group(1)):.3f} "
                f"neftune_pref={float(m_neftune.group(2)):.3f}"
            )
            parsed.adapter_summary = payload if parsed.adapter_summary == "-" else f"{parsed.adapter_summary} | {payload}"
        m_lora_plus = LORA_PLUS_RE.search(line)
        if m_lora_plus:
            payload = (
                f"lora_plus_ratio={float(m_lora_plus.group(1)):.2f} "
                f"base_group={int(m_lora_plus.group(2))} fast_group={int(m_lora_plus.group(3))}"
            )
            parsed.adapter_summary = payload if parsed.adapter_summary == "-" else f"{parsed.adapter_summary} | {payload}"
        m_source_balance = SOURCE_BALANCE_RE.search(line)
        if m_source_balance:
            parsed.source_balance_summary = m_source_balance.group(1).strip()
        m_pref_build = PREF_BUILD_RE.search(line)
        if m_pref_build:
            parsed.pref_objective = m_pref_build.group(1).strip()
        m_pref_obj = PREF_OBJECTIVE_RE.search(line)
        if m_pref_obj:
            parsed.objective_summary = m_pref_obj.group(1).strip()
        m_pref_ref = PREF_REFERENCE_RE.search(line)
        if m_pref_ref:
            parsed.pref_reference_pairs = int(m_pref_ref.group(1))

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

        m_data_split = DATA_SPLIT_RE.search(line)
        if m_data_split:
            parsed.train_pairs_count = int(m_data_split.group(1))
            parsed.eval_pairs_count = int(m_data_split.group(2))
            parsed.raw_eval_pairs_count = (
                int(m_data_split.group(3)) if m_data_split.group(3) is not None else parsed.eval_pairs_count
            )
            parsed.eval_summary = (
                f"train={parsed.train_pairs_count} eval={parsed.eval_pairs_count} "
                f"raw_eval={parsed.raw_eval_pairs_count}"
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

        m_distill_cfg = DISTILL_CONFIG_RE.search(line)
        if m_distill_cfg:
            parsed.distill_target = int(m_distill_cfg.group(1))
            ratio = float(m_distill_cfg.group(2))
            max_seconds = m_distill_cfg.group(3)
            best_of = int(m_distill_cfg.group(4))
            min_gain = float(m_distill_cfg.group(5)) if m_distill_cfg.group(5) is not None else 0.0
            density_bias = float(m_distill_cfg.group(6)) if m_distill_cfg.group(6) is not None else 0.0
            parsed.distill_config_summary = (
                f"target={parsed.distill_target} ratio={ratio:.3f} max_seconds={max_seconds} "
                f"best_of={best_of} min_gain={min_gain:.3f} density_bias={density_bias:.3f}"
            )

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
            dropped_synthetic = int(m_sft_quality.group(5)) if m_sft_quality.group(5) is not None else None
            exempt_sources = int(m_sft_quality.group(6))
            parsed.sft_filter_summary = (
                f"threshold={threshold:.2f} kept={kept} dropped_quality={dropped_quality} "
                f"dropped_short={dropped_short}"
            )
            if dropped_synthetic is not None:
                parsed.sft_filter_summary += f" dropped_synthetic={dropped_synthetic}"
            parsed.sft_filter_summary += f" exempt_sources={exempt_sources}"

        m_sft_fallback = SFT_QUALITY_FALLBACK_RE.search(line)
        if m_sft_fallback:
            kept = int(m_sft_fallback.group(1))
            unfiltered = int(m_sft_fallback.group(2))
            parsed.sft_filter_summary = f"fallback kept={kept} using_unfiltered={unfiltered}"

        m_eval_filter = EVAL_FILTER_RE.search(line)
        if m_eval_filter:
            threshold = float(m_eval_filter.group(1))
            kept = int(m_eval_filter.group(2))
            dropped_quality = int(m_eval_filter.group(3))
            dropped_synthetic = int(m_eval_filter.group(4))
            raw_eval = kept + dropped_quality + dropped_synthetic
            parsed.eval_pairs_count = kept
            parsed.raw_eval_pairs_count = raw_eval
            parsed.eval_summary = (
                f"threshold={threshold:.2f} kept={kept}/{raw_eval} "
                f"dropped_quality={dropped_quality} dropped_synthetic={dropped_synthetic}"
            )

        m_eval_fallback = EVAL_FILTER_FALLBACK_RE.search(line)
        if m_eval_fallback:
            kept = int(m_eval_fallback.group(1))
            threshold = float(m_eval_fallback.group(2))
            dropped_quality = int(m_eval_fallback.group(3))
            dropped_synthetic = int(m_eval_fallback.group(4))
            raw_eval = kept + dropped_quality + dropped_synthetic
            parsed.eval_pairs_count = kept
            parsed.raw_eval_pairs_count = raw_eval
            parsed.eval_summary = (
                f"fallback threshold={threshold:.2f} kept={kept}/{raw_eval} "
                f"dropped_quality={dropped_quality} dropped_synthetic={dropped_synthetic}"
            )

        if line.startswith("[eval] skipped"):
            parsed.eval_summary = line[len("[eval]") :].strip()

        m_pref_cfg = PREF_MINING_CONFIG_RE.search(line)
        if m_pref_cfg:
            mode = m_pref_cfg.group(1)
            generation = m_pref_cfg.group(2)
            parsed.pref_mining_target_pairs = int(m_pref_cfg.group(3))
            parsed.pref_mining_candidates = int(m_pref_cfg.group(4))
            parsed.pref_mining_self_play_budget = (
                int(m_pref_cfg.group(6)) if m_pref_cfg.group(6) is not None else None
            )
            parsed.pref_mining_self_play_curriculum = m_pref_cfg.group(7) if m_pref_cfg.group(7) is not None else "-"
            parsed.pref_mining_self_play_max_new_tokens = (
                int(m_pref_cfg.group(8)) if m_pref_cfg.group(8) is not None else None
            )
            selection = m_pref_cfg.group(9)
            keep_ratio = float(m_pref_cfg.group(10))
            max_seconds = m_pref_cfg.group(11)
            parsed.pref_mining_summary = (
                f"mode={mode} generation={generation} target_pairs={parsed.pref_mining_target_pairs} "
                f"candidates={parsed.pref_mining_candidates} selection={selection} keep_ratio={keep_ratio:.3f} "
                f"max_seconds={max_seconds}"
            )
            if parsed.pref_mining_self_play_budget is not None:
                parsed.pref_mining_summary += (
                    f" self_play={parsed.pref_mining_self_play_budget} "
                    f"curriculum={parsed.pref_mining_self_play_curriculum} "
                    f"gen_tokens={parsed.pref_mining_self_play_max_new_tokens}"
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
            parsed.pref_mining_brevity_filtered = int(m_pref_complete.group(5)) if m_pref_complete.group(5) is not None else None
            parsed.pref_mining_stop_reject_variants = int(m_pref_complete.group(6)) if m_pref_complete.group(6) is not None else None
            parsed.pref_mining_self_play_prompts = int(m_pref_complete.group(7)) if m_pref_complete.group(7) is not None else None
            parsed.pref_mining_self_play_candidates = int(m_pref_complete.group(8)) if m_pref_complete.group(8) is not None else None
            parsed.pref_mining_self_play_failures = int(m_pref_complete.group(9)) if m_pref_complete.group(9) is not None else None
            elapsed = float(m_pref_complete.group(10))
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
            if parsed.pref_mining_brevity_filtered is not None:
                parsed.pref_mining_summary += (
                    f" brevity_filtered={parsed.pref_mining_brevity_filtered}"
                    f" stop_reject_variants={parsed.pref_mining_stop_reject_variants if parsed.pref_mining_stop_reject_variants is not None else 0}"
                )
            if parsed.pref_mining_self_play_prompts is not None:
                parsed.pref_mining_summary += (
                    f" self_play_prompts={parsed.pref_mining_self_play_prompts}"
                    f" self_play_candidates={parsed.pref_mining_self_play_candidates if parsed.pref_mining_self_play_candidates is not None else 0}"
                    f" self_play_failures={parsed.pref_mining_self_play_failures if parsed.pref_mining_self_play_failures is not None else 0}"
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
            conv_before = float(m_pref_selection.group(9)) if m_pref_selection.group(9) is not None else None
            conv_after = float(m_pref_selection.group(10)) if m_pref_selection.group(10) is not None else None
            reason_before = float(m_pref_selection.group(11)) if m_pref_selection.group(11) is not None else None
            reason_after = float(m_pref_selection.group(12)) if m_pref_selection.group(12) is not None else None
            creative_before = float(m_pref_selection.group(13)) if m_pref_selection.group(13) is not None else None
            creative_after = float(m_pref_selection.group(14)) if m_pref_selection.group(14) is not None else None
            density_before = float(m_pref_selection.group(15)) if m_pref_selection.group(15) is not None else None
            density_after = float(m_pref_selection.group(16)) if m_pref_selection.group(16) is not None else None
            score_mean = float(m_pref_selection.group(17))
            parsed.pref_selection_summary = (
                f"strategy={strategy} keep={keep_n}/{total} keep_ratio={keep_ratio:.3f} "
                f"gap={gap_before:.3f}->{gap_after:.3f} sim={sim_before:.3f}->{sim_after:.3f} "
            )
            if conv_before is not None and conv_after is not None:
                parsed.pref_selection_summary += f"conv={conv_before:.3f}->{conv_after:.3f} "
            if reason_before is not None and reason_after is not None:
                parsed.pref_selection_summary += f"reason={reason_before:.3f}->{reason_after:.3f} "
            if creative_before is not None and creative_after is not None:
                parsed.pref_selection_summary += f"creative={creative_before:.3f}->{creative_after:.3f} "
            if density_before is not None and density_after is not None:
                parsed.pref_selection_summary += f"density={density_before:.3f}->{density_after:.3f} "
            parsed.pref_selection_summary += f"score_mean={score_mean:.4f}"
        m_v46_mod = V46_MOD_RE.search(line)
        if m_v46_mod:
            parsed.v46_mod_bypass = int(m_v46_mod.group(1))
            skipped = int(m_v46_mod.group(2))
            flops = float(m_v46_mod.group(3))
            parsed.v46_mod_summary = f"bypass={parsed.v46_mod_bypass} skipped={skipped} flops_saved={flops:.2e}"

        m_v46_got = V46_GOT_RE.search(line)
        if m_v46_got:
            branches = int(m_v46_got.group(1))
            parsed.v46_got_merge_conf = float(m_v46_got.group(2))
            topology = m_v46_got.group(3)
            parsed.v46_got_summary = f"branches={branches} conf={parsed.v46_got_merge_conf:.3f} topo={topology}"

        m_v46_ccot = V46_CCOT_RE.search(line)
        if m_v46_ccot:
            parsed.v46_ccot_depth = int(m_v46_ccot.group(1))
            expansion = float(m_v46_ccot.group(2))
            stability = float(m_v46_ccot.group(3))
            parsed.v46_ccot_summary = f"depth={parsed.v46_ccot_depth} expansion={expansion:.2f} stability={stability:.3f}"

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
    search_dirs = [root_dir]
    output_dir = root_dir / "output"
    if output_dir.exists():
        search_dirs.append(output_dir)
    out_logs = sorted(
        {
            *(path for base in search_dirs for path in base.glob("train_*.out.log")),
            *(path for base in search_dirs for path in base.glob("*_train_*.out.log")),
        }
    )
    snapshots: List[RunSnapshot] = []
    process_entries = _list_process_entries()

    for out_log in out_logs:
        log_dir = out_log.parent
        stem = out_log.name.replace(".out.log", "")
        err_log = log_dir / f"{stem}.err.log"
        if not err_log.exists():
            err_log = None
        base_stem = _run_base_name(stem)
        pid_candidates = [
            log_dir / f"{stem}.pid",
            log_dir / f"{base_stem}.worker.pid",
            log_dir / f"{base_stem}.pid",
        ]
        pid_file = next((path for path in pid_candidates if path.exists()), None)
        pid = _read_pid(pid_file) if pid_file is not None else None
        pid_alive = _is_pid_alive(pid) if pid is not None else False
        pid_source = "pid_file" if pid_alive and pid is not None else ""
        state_file = log_dir / f"{base_stem}_state.json"
        if not state_file.exists():
            state_file = None

        command_line = _query_process_cmdline(pid) if (pid_alive and pid is not None) else None
        command_line = command_line or ""
        process_launch_hint = ""
        if not command_line:
            guessed_ps1 = _guess_ps1_from_run_name(stem, root_dir)
            process_launch_hint = str(guessed_ps1) if guessed_ps1 is not None else ""
            inferred_pid, inferred_cmd, inferred_source = _find_live_process_for_run(
                run_name=stem,
                out_log=out_log,
                launch_hint=process_launch_hint,
                processes=process_entries,
            )
            if inferred_pid is not None:
                pid = inferred_pid
                pid_alive = True
                pid_source = inferred_source or "process_scan"
                command_line = inferred_cmd or ""
        sft_target_steps, pref_target_steps, save_every_steps, launch_hint, has_distill_stage, has_pref_mining_stage = _infer_targets(
            stem,
            root_dir,
            command_line,
        )
        if not launch_hint and process_launch_hint:
            launch_hint = process_launch_hint

        parsed = _parse_log(out_log)
        state_payload = _read_json_file(state_file) if state_file is not None else {}
        if state_payload:
            state_status = str(state_payload.get("status") or "").strip().lower()
            state_stage = str(state_payload.get("stage") or state_status or parsed.stage).strip().lower()
            parsed.stage = state_stage or parsed.stage
            try:
                parsed.omni_batch_index = int(state_payload.get("batch_index")) if state_payload.get("batch_index") is not None else parsed.omni_batch_index
            except Exception:
                pass
            try:
                parsed.omni_total_batches = int(state_payload.get("total_batches")) if state_payload.get("total_batches") is not None else parsed.omni_total_batches
            except Exception:
                pass
            try:
                parsed.omni_elapsed_seconds = float(state_payload.get("elapsed_seconds")) if state_payload.get("elapsed_seconds") is not None else parsed.omni_elapsed_seconds
            except Exception:
                pass
            try:
                parsed.omni_eta_seconds = float(state_payload.get("eta_seconds")) if state_payload.get("eta_seconds") is not None else parsed.omni_eta_seconds
            except Exception:
                pass
            try:
                parsed.omni_avg_balance_loss = float(state_payload.get("avg_balance_loss")) if state_payload.get("avg_balance_loss") is not None else parsed.omni_avg_balance_loss
            except Exception:
                pass
            try:
                parsed.loss = float(state_payload.get("avg_train_loss")) if state_payload.get("avg_train_loss") is not None else parsed.loss
            except Exception:
                pass
            try:
                parsed.lr = float(state_payload.get("lr")) if state_payload.get("lr") is not None else parsed.lr
            except Exception:
                pass

        out_stat = out_log.stat()
        out_mtime = out_stat.st_mtime
        out_size = out_stat.st_size
        if state_file is not None:
            try:
                out_mtime = max(out_mtime, float(state_file.stat().st_mtime))
            except Exception:
                pass
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
        stage_source = "state" if state_payload else "log"
        if parsed.stage in {"stage1", "stage2"} and parsed.omni_batch_index is not None:
            progress_units = float(parsed.omni_batch_index)
            total_units = float(parsed.omni_total_batches) if parsed.omni_total_batches is not None else total_units
            progress_percent = _percent_complete(parsed.omni_batch_index, parsed.omni_total_batches)

        cpu_percent = None
        ram_gb = None
        if parsed.stage == "done":
            status = "finished"
        elif pid_alive:
            status = "running"
            if stale_mins >= float(stale_minutes_threshold):
                status = "stalled"
            try:
                if pid is not None:
                    p = psutil.Process(pid)
                    cpu_percent = p.cpu_percent(interval=None)
                    ram_gb = p.memory_info().rss / (1024 ** 3)
            except Exception:
                pass
        else:
            if out_size > 0:
                status = "stopped"
            else:
                status = "unknown"

        health_summary = _build_health_summary(
            status=status,
            stage=parsed.stage,
            pid_file=pid_file,
            pid_source=pid_source,
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
                pid_source=pid_source,
                pid_alive=pid_alive,
                status=status,
                stage=parsed.stage,
                sft_step=parsed.sft_step,
                pref_step=parsed.pref_step,
                pref_pairs=parsed.pref_pairs,
                loss=parsed.loss,
                lr=parsed.lr,
                rdrop=parsed.rdrop,
                wpo_std=parsed.wpo_std,
                beta=parsed.beta,
                margin=parsed.margin,
                pref_objective=parsed.pref_objective,
                pref_reference_pairs=parsed.pref_reference_pairs,
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
                cpu_percent=cpu_percent,
                ram_gb=ram_gb,
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
                runtime_summary=parsed.runtime_summary,
                adapter_summary=parsed.adapter_summary,
                source_balance_summary=parsed.source_balance_summary,
                objective_summary=parsed.objective_summary,
                data_summary=parsed.data_summary,
                eval_summary=parsed.eval_summary,
                sft_filter_summary=parsed.sft_filter_summary,
                distill_config_summary=parsed.distill_config_summary,
                distill_summary=parsed.distill_summary,
                pref_mining_summary=parsed.pref_mining_summary,
                pref_selection_summary=parsed.pref_selection_summary,
                tail_lines=parsed.tail_lines,
                err_tail_lines=err_tail_lines,
                v46_mod_summary=parsed.v46_mod_summary,
                v46_got_summary=parsed.v46_got_summary,
                v46_ccot_summary=parsed.v46_ccot_summary,
                state_file=state_file,
                balance_loss=parsed.omni_avg_balance_loss,
                stage_source=stage_source,
            )
        )

    snapshots.sort(key=lambda x: x.out_last_write_ts, reverse=True)
    return snapshots


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_research_results(root_dir: Path) -> List[ResearchResult]:
    path = root_dir / "research" / "results.tsv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows: List[ResearchResult] = []
            for row in reader:
                if not isinstance(row, dict):
                    continue
                run_tag = str(row.get("run_tag", "") or "").strip()
                if not run_tag:
                    continue
                rows.append(
                    ResearchResult(
                        timestamp=str(row.get("timestamp", "") or "").strip(),
                        commit=str(row.get("commit", "") or "").strip(),
                        run_tag=run_tag,
                        output_dir=str(row.get("output_dir", "") or "").strip(),
                        benchmark_json=str(row.get("benchmark_json", "") or "").strip(),
                        token_f1_delta=_safe_float(row.get("token_f1_delta", 0.0)),
                        char_similarity_delta=_safe_float(row.get("char_similarity_delta", 0.0)),
                        perplexity_delta=_safe_float(row.get("perplexity_delta", 0.0)),
                        avg_gen_seconds_delta=_safe_float(row.get("avg_gen_seconds_delta", 0.0)),
                        status=str(row.get("status", "") or "").strip().lower() or "unknown",
                        description=str(row.get("description", "") or "").strip(),
                    )
                )
    except Exception:
        return []
    rows.sort(key=lambda x: x.timestamp, reverse=True)
    return rows


def _format_signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def _preview_research_text(value: Any, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return "-"
    if limit <= 3 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _resolve_benchmark_artifact_path(benchmark_json: Path, raw_path: str) -> Optional[Path]:
    candidate_raw = str(raw_path or "").strip()
    if not candidate_raw:
        return None
    candidate = Path(candidate_raw)
    if not candidate.is_absolute():
        candidate = (benchmark_json.parent / candidate).resolve()
    return candidate


def _research_failure_insight_from_row(
    row: Dict[str, Any],
    sample_comparison_jsonl: str = "",
) -> ResearchFailureInsight:
    sample_index = int(row.get("sample_index", 0) or 0)
    source = str(row.get("source", "") or "").strip() or "-"
    summary_line = (
        f"Top regression: #{sample_index} {source} "
        f"| dF1 {_format_signed(_safe_float(row.get('delta_token_f1', 0.0)))} "
        f"| dChar {_format_signed(_safe_float(row.get('delta_char_similarity', 0.0)))} "
        f"| dGen {_format_signed(_safe_float(row.get('delta_gen_seconds', 0.0), 0.0), 2)}s"
    )
    prompt_line = f"Prompt: {_preview_research_text(row.get('user_preview', row.get('user', '')), limit=180)}"
    prediction_line = (
        f"Tuned: {_preview_research_text(row.get('tuned_prediction_preview', row.get('tuned_prediction', '')), limit=150)} "
        f"| Ref: {_preview_research_text(row.get('reference_preview', row.get('reference', '')), limit=120)}"
    )
    return ResearchFailureInsight(
        summary_line=summary_line,
        prompt_line=prompt_line,
        prediction_line=prediction_line,
        sample_comparison_jsonl=sample_comparison_jsonl,
    )


def _load_research_failure_insight(benchmark_json: Path) -> ResearchFailureInsight:
    empty = ResearchFailureInsight(
        summary_line="Top regression: -",
        prompt_line="Prompt: -",
        prediction_line="Tuned: - | Ref: -",
    )
    if not benchmark_json.exists():
        return empty
    try:
        payload = json.loads(benchmark_json.read_text(encoding="utf-8"))
    except Exception:
        return empty
    if not isinstance(payload, dict):
        return empty

    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    sample_path = _resolve_benchmark_artifact_path(
        benchmark_json,
        str(artifacts.get("sample_comparison_jsonl", "") if isinstance(artifacts, dict) else ""),
    )
    sample_path_text = str(sample_path) if sample_path is not None else ""

    sample_summary = payload.get("sample_summary")
    if isinstance(sample_summary, dict):
        worst_regression = sample_summary.get("worst_regression")
        if isinstance(worst_regression, dict):
            return _research_failure_insight_from_row(
                worst_regression,
                sample_comparison_jsonl=sample_path_text,
            )

    fallback_path = sample_path if sample_path is not None else (benchmark_json.parent / "sample_comparison.jsonl")
    if not fallback_path.exists():
        return ResearchFailureInsight(
            summary_line="Top regression: unavailable for this run",
            prompt_line="Prompt: rerun the benchmark with detailed traces to generate sample_comparison.jsonl",
            prediction_line="Tuned: - | Ref: -",
        )
    try:
        with fallback_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    return _research_failure_insight_from_row(
                        row,
                        sample_comparison_jsonl=str(fallback_path),
                    )
                break
    except Exception:
        return empty
    return ResearchFailureInsight(
        summary_line="Top regression: sample trace file is empty",
        prompt_line="Prompt: no detailed benchmark rows were written for this run",
        prediction_line="Tuned: - | Ref: -",
        sample_comparison_jsonl=str(fallback_path),
    )


def _summarize_research_results(results: Sequence[ResearchResult]) -> Tuple[str, str, str]:
    if not results:
        return ("Research: no autoresearch results yet", "Best: -", "Latest: -")

    counts: Dict[str, int] = {}
    for row in results:
        counts[row.status] = counts.get(row.status, 0) + 1

    status_bits = []
    for key in ("keep", "discard", "crash", "unknown"):
        if counts.get(key, 0):
            status_bits.append(f"{key} {counts[key]}")
    summary = f"Research: {len(results)} runs | " + (" | ".join(status_bits) if status_bits else "no status data")

    benchmarked = [row for row in results if row.benchmark_json or row.status in {"keep", "discard"}]
    if benchmarked:
        best = max(
            benchmarked,
            key=lambda row: (
                float(row.token_f1_delta),
                float(row.char_similarity_delta),
                -float(row.avg_gen_seconds_delta),
            ),
        )
        best_line = (
            f"Best: {best.run_tag} [{best.status}] "
            f"f1Δ {_format_signed(best.token_f1_delta)} "
            f"charΔ {_format_signed(best.char_similarity_delta)} "
            f"genΔ {_format_signed(best.avg_gen_seconds_delta, 2)}s"
        )
    else:
        best_line = "Best: -"

    latest = results[0]
    latest_line = (
        f"Latest: {latest.run_tag} [{latest.status}] "
        f"f1Δ {_format_signed(latest.token_f1_delta)} "
        f"charΔ {_format_signed(latest.char_similarity_delta)} "
        f"| {latest.description or '-'}"
    )
    return summary, best_line, latest_line


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
        dur = f"{days}d {hours}h {mins}m"
    elif hours > 0:
        dur = f"{hours}h {mins}m"
    else:
        dur = f"{mins}m"
    eta_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    time_str = eta_time.strftime("%I:%M %p").lstrip("0")
    if days > 0:
        time_str += f" (+{days}d)"
    return f"{dur} ({time_str})"


def _fmt_age_minutes(minutes: Optional[float]) -> str:
    if minutes is None:
        return "-"
    mins = max(0.0, float(minutes))
    if mins < 0.7:
        return "just now"
    if mins < 60.0:
        return f"{mins:.1f}m"
    hours = mins / 60.0
    if hours < 24.0:
        return f"{hours:.1f}h"
    days = hours / 24.0
    return f"{days:.1f}d"


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
    pid_source: str,
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
    elif pid_source == "process_scan" and pid_alive:
        notes.append("live pid inferred from process scan")

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
        "dataset": 1,
        "building_dataset": 1,
        "building_dataset_from_resume": 1,
        "distill": 2,
        "sft_setup": 3,
        "sft_filter": 4,
        "sft": 5,
        "stage1": 5,
        "preference_mining": 6,
        "preference": 7,
        "stage2": 7,
        "eval": 8,
        "done": 9,
    }.get(str(stage or "").strip().lower(), 0)


def _clamp_fraction(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _phase_weight_plan(snap: RunSnapshot) -> List[Tuple[str, float]]:
    if snap.stage in {"dataset", "building_dataset", "building_dataset_from_resume", "stage1", "stage2"}:
        return [("dataset", 10.0), ("stage1", 45.0), ("stage2", 45.0)]

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

    if phase == "dataset":
        if snap.stage in {"dataset", "building_dataset", "building_dataset_from_resume"}:
            return 0.5
        return 1.0 if current_rank > phase_rank else 0.0

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

    if phase == "stage1":
        return _clamp_fraction(
            None if snap.stage_progress_percent is None or snap.stage != "stage1" else snap.stage_progress_percent / 100.0
        )

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

    if phase == "stage2":
        return _clamp_fraction(
            None if snap.stage_progress_percent is None or snap.stage != "stage2" else snap.stage_progress_percent / 100.0
        )

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


def _build_run_recommendation(snap: RunSnapshot) -> str:
    if snap.err_signal == "error":
        return "Review the ERR log first; the run is reporting a hard failure."
    if snap.status == "stalled":
        return f"Investigate the last {snap.stage or 'active'} stage and compare the latest OUT/ERR tails."
    if snap.status == "finished" or snap.stage == "done":
        return "Benchmark or promote the finished adapter, then archive the logs."
    if snap.stage == "data":
        return "Watch dataset throughput and filter counts before the training stages start."
    if snap.stage == "distill":
        return "Monitor teacher generation rate; lower best_of or max_seconds if this stage drags."
    if snap.stage == "sft_filter":
        return "Review the quality-filter summary to make sure useful pairs are not being dropped."
    if snap.stage == "sft":
        return "Track loss trend, rate, and checkpoint ETA; this is the main throughput stage."
    if snap.stage == "preference_mining":
        return "Check accepted pairs versus visited prompts; reduce mining cost if acceptance stays low."
    if snap.stage == "preference":
        return "Monitor loss, beta, and checkpoint cadence for preference stability."
    if snap.stage == "eval":
        return "Review eval filtering and benchmark readiness before using this adapter."
    return "Select a run and inspect the latest tails for the next concrete action."


def _summary_fractions(text: str) -> Dict[str, Tuple[int, int]]:
    values: Dict[str, Tuple[int, int]] = {}
    for name, left, right in SUMMARY_FRACTION_RE.findall(str(text or "").lower()):
        try:
            values[str(name)] = (int(left), int(right))
        except Exception:
            continue
    return values


def _summary_counts(text: str) -> Dict[str, int]:
    values: Dict[str, int] = {}
    for name, raw in SUMMARY_COUNT_RE.findall(str(text or "").lower()):
        try:
            values[str(name)] = int(raw)
        except Exception:
            continue
    return values


def _safe_ratio(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    if float(right) <= 0.0:
        return None
    return float(left) / float(right)


def _parse_rate_per_second(rate_label: str) -> Optional[float]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*/\s*s\b", str(rate_label or "").strip().lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _history_recent_values(
    history: Sequence[Tuple[float, float]],
    window_seconds: float = 7200.0,
) -> List[float]:
    if not history:
        return []
    latest_ts = float(history[-1][0])
    min_ts = latest_ts - float(window_seconds)
    return [float(value) for ts, value in history if float(ts) >= min_ts]


def _history_ratio(
    history: Sequence[Tuple[float, float]],
    window_seconds: float = 7200.0,
    recent_points: int = 6,
) -> Optional[float]:
    values = _history_recent_values(history, window_seconds=window_seconds)
    if len(values) < max(6, recent_points + 3):
        return None
    recent = values[-recent_points:]
    baseline = values[:-recent_points]
    if len(baseline) < 3:
        return None
    try:
        baseline_median = float(statistics.median(baseline))
        recent_median = float(statistics.median(recent))
    except Exception:
        return None
    if baseline_median <= 0.0:
        return None
    return recent_median / baseline_median


def _failure_lens(snap: RunSnapshot) -> str:
    corpus = " ".join(
        [snap.err_summary, snap.health_summary, *(snap.err_tail_lines or []), *(snap.tail_lines[-10:] if snap.tail_lines else [])]
    ).lower()
    if not corpus.strip():
        return "Failure Lens: no dominant failure pattern"
    if any(token in corpus for token in ["no space left on device", "pytorchstreamwriter", "unexpected pos", "permission denied", "access is denied", "disk full"]):
        return "Failure Lens: storage or checkpoint-write risk"
    if any(token in corpus for token in ["out of memory", "cuda oom", "cublas", "device-side assert", "cudnn", "driver"]):
        return "Failure Lens: accelerator memory or runtime failure"
    if any(token in corpus for token in ["keyerror", "attributeerror", "typeerror", "valueerror", "jsondecodeerror", "state_dict", "scheduler"]):
        return "Failure Lens: schema, resume, or trainer-state mismatch"
    if any(token in corpus for token in ["keyboardinterrupt", "terminated", "killed", "shutdown", "power", "interrupted"]):
        return "Failure Lens: external interruption or host shutdown risk"
    if snap.status == "stalled":
        return "Failure Lens: live process stalled without decisive error; inspect throughput and stale tails"
    return "Failure Lens: no dominant failure pattern"


def _build_run_stability_lab(
    snap: RunSnapshot,
    loss_history: Sequence[Tuple[float, float]],
    rate_history: Sequence[Tuple[float, float]],
    cpu_history: Sequence[Tuple[float, float]],
    stale_history: Optional[Sequence[Tuple[float, float]]] = None,
) -> str:
    notes: List[str] = []
    loss_ratio = _history_ratio(loss_history, window_seconds=7200.0, recent_points=6)
    if loss_ratio is not None:
        if loss_ratio >= 1.08:
            notes.append(f"loss drift +{(loss_ratio - 1.0) * 100.0:.0f}% vs recent baseline")
        elif loss_ratio <= 0.96:
            notes.append("loss still improving")
        else:
            notes.append("loss trend stable")

    rate_ratio = _history_ratio(rate_history, window_seconds=7200.0, recent_points=5)
    if rate_ratio is not None:
        if rate_ratio <= 0.70:
            notes.append(f"throughput down to {rate_ratio * 100.0:.0f}% of baseline")
        elif rate_ratio >= 1.15:
            notes.append("throughput above recent baseline")

    if snap.balance_loss is not None:
        if snap.balance_loss >= 0.25:
            notes.append(f"routing balance pressure {snap.balance_loss:.3f}")
        elif snap.balance_loss <= 0.12:
            notes.append(f"routing balance healthy {snap.balance_loss:.3f}")
        else:
            notes.append(f"routing balance watch {snap.balance_loss:.3f}")

    if snap.status == "running" and snap.stale_minutes >= 6.0:
        notes.append(f"log heartbeat delayed {snap.stale_minutes:.1f}m")
    elif stale_history:
        stale_ratio = _history_ratio(stale_history, window_seconds=3600.0, recent_points=5)
        if stale_ratio is not None and stale_ratio >= 1.5 and snap.status == "running":
            notes.append("log heartbeat slowing")

    cpu_ratio = _history_ratio(cpu_history, window_seconds=7200.0, recent_points=5)
    device = _runtime_device_value(snap)
    if cpu_ratio is not None and device in {"cuda", "dml", "npu"} and cpu_ratio >= 1.20 and (rate_ratio is not None and rate_ratio <= 0.80):
        notes.append("host-side bottleneck pressure")

    if snap.stage_source == "state":
        notes.append("structured state telemetry active")

    if not notes:
        return "Stability Lab: no strong instability signal"
    return "Stability Lab: " + " | ".join(notes[:4])


def _summarize_fleet_research_radar(
    snapshots: Sequence[RunSnapshot],
    loss_histories: Dict[str, List[Tuple[float, float]]],
    rate_histories: Dict[str, List[Tuple[float, float]]],
) -> str:
    if not snapshots:
        return "Research Radar: -"
    loss_drift = 0
    rate_drift = 0
    balance_alert = 0
    failure_buckets: Dict[str, int] = {}
    for snap in snapshots:
        ratio = _history_ratio(loss_histories.get(snap.run_name, []), window_seconds=7200.0, recent_points=6)
        if ratio is not None and ratio >= 1.08:
            loss_drift += 1
        rate_ratio = _history_ratio(rate_histories.get(snap.run_name, []), window_seconds=7200.0, recent_points=5)
        if rate_ratio is not None and rate_ratio <= 0.70:
            rate_drift += 1
        if snap.balance_loss is not None and snap.balance_loss >= 0.25:
            balance_alert += 1
        lens = _failure_lens(snap).replace("Failure Lens: ", "", 1)
        failure_buckets[lens] = failure_buckets.get(lens, 0) + 1

    top_failure = max(failure_buckets.items(), key=lambda item: item[1])[0] if failure_buckets else "no dominant failure pattern"
    return (
        "Research Radar: "
        f"{loss_drift} loss-drift | {rate_drift} throughput-drift | {balance_alert} balance-alert | dominant failure: {top_failure}"
    )


def _build_runtime_headline(snap: RunSnapshot) -> str:
    device = _runtime_device_value(snap)
    cpu_txt = "-" if snap.cpu_percent is None else f"{snap.cpu_percent:.0f}%"
    ram_txt = "-" if snap.ram_gb is None else f"{snap.ram_gb:.1f}G"
    stale_txt = f"{snap.stale_minutes:.1f}m"
    pid_txt = "live pid" if snap.pid_alive else "no live pid"
    return f"Runtime: {device} | CPU {cpu_txt} | RAM {ram_txt} | stale {stale_txt} | {pid_txt}"


def _build_recovery_outlook(snap: RunSnapshot) -> str:
    if snap.status == "running" and snap.pid_alive:
        if snap.checkpoint_count > 0 and snap.last_checkpoint_step > 0:
            return (
                f"Recovery: live run; latest durable save is {snap.last_checkpoint_stage} "
                f"step {snap.last_checkpoint_step}"
            )
        return "Recovery: live run; avoid intervention until a checkpoint lands"

    parts: List[str] = []
    if snap.checkpoint_count > 0 and snap.last_checkpoint_step > 0:
        parts.append(f"resume from {snap.last_checkpoint_stage} step {snap.last_checkpoint_step}")
    elif snap.pref_step > 0 or snap.sft_step > 0 or snap.pref_pairs > 0:
        parts.append("partial progress is visible in logs, but no durable checkpoint was detected")
    else:
        parts.append("no resume checkpoint detected; expect a cold restart")

    if snap.save_every_steps is not None and snap.save_every_steps > 0 and snap.status in {"running", "stalled", "stopped"}:
        parts.append(f"checkpoint cadence {snap.save_every_steps} steps")
    if snap.err_signal == "error":
        parts.append("fix the ERR failure before relaunch")
    elif snap.status == "stalled":
        parts.append("capture logs before restarting")
    elif snap.pid_file is not None and not snap.pid_alive:
        parts.append("stale pid marker present")

    return "Recovery: " + " | ".join(parts[:4])


def _build_run_rescue_plan(snap: RunSnapshot) -> str:
    if snap.status == "running" and snap.pid_alive:
        if snap.checkpoint_count > 0 and snap.last_checkpoint_step > 0:
            return (
                "Rescue Plan: keep the live run untouched; if it dies, resume from "
                f"{snap.last_checkpoint_stage} step {snap.last_checkpoint_step}"
            )
        return "Rescue Plan: keep the live run untouched; wait for the first durable checkpoint before intervening"

    steps: List[str] = []
    if snap.err_signal == "error":
        steps.append("review ERR tail first")
    elif snap.status == "stalled":
        steps.append("capture OUT/ERR tails before restart")

    if snap.pid_file is not None and not snap.pid_alive:
        steps.append("clear stale pid marker")

    if snap.checkpoint_count > 0 and snap.last_checkpoint_step > 0:
        steps.append(f"resume from {snap.last_checkpoint_stage} step {snap.last_checkpoint_step}")
    elif snap.pref_step > 0 or snap.sft_step > 0 or snap.pref_pairs > 0:
        steps.append("expect replay from the last durable save; log-only progress is newer than checkpoint state")
    else:
        steps.append("cold restart required")

    launch = str(snap.launch_command or snap.launch_hint or "").strip()
    if launch:
        steps.append("relaunch with the saved launch command")
    else:
        steps.append("reconstruct the launch command before restarting")

    return "Rescue Plan: " + " | ".join(steps[:4])


def _build_run_watch_summary(snap: RunSnapshot) -> str:
    notes: List[str] = []
    device = _runtime_device_value(snap)
    data_counts = _summary_counts(snap.data_summary)
    data_fracs = _summary_fractions(snap.data_summary)
    eval_fracs = _summary_fractions(snap.eval_summary)
    distill_counts = _summary_counts(snap.distill_summary)
    distill_fracs = _summary_fractions(snap.distill_summary)
    pref_counts = _summary_counts(snap.pref_mining_summary)
    pref_fracs = _summary_fractions(snap.pref_mining_summary)

    if snap.err_signal == "error":
        return (
            f"Watch: hard failure in ERR log ({snap.err_summary})"
            if snap.err_summary and snap.err_summary != "-"
            else "Watch: hard failure in ERR log"
        )
    if snap.status == "stalled":
        notes.append(f"no fresh log progress for {snap.stale_minutes:.1f}m")
    elif snap.status == "stopped" and snap.stage != "done":
        notes.append("run stopped before completion")
    elif snap.status == "unknown":
        notes.append("run state is unresolved")

    if snap.status == "running" and not snap.pid_alive and snap.stale_minutes >= 2.0:
        notes.append("marked running but no live pid is attached")

    if device == "cpu" and snap.cpu_percent is not None and snap.cpu_percent >= 80.0:
        notes.append("CPU-bound backend")
    elif device in {"cuda", "dml", "npu"} and snap.cpu_percent is not None and snap.cpu_percent <= 15.0 and snap.stale_minutes >= 2.0:
        notes.append(f"{device} run looks mostly idle from the host side")

    keep_ratio = _safe_ratio(data_counts.get("kept"), data_counts.get("raw"))
    if snap.stage == "data" and keep_ratio is not None and keep_ratio < 0.45:
        notes.append(f"dataset filters are only keeping {keep_ratio:.0%} of raw pairs")

    synthetic_ratio = None
    if "synthetic" in data_fracs:
        synthetic_ratio = _safe_ratio(*data_fracs["synthetic"])
    if snap.stage == "data" and synthetic_ratio is not None and synthetic_ratio > 0.25:
        notes.append(f"synthetic share is elevated at {synthetic_ratio:.0%}")

    distill_visited = distill_fracs.get("visited", (None, None))[0]
    distill_total = distill_fracs.get("visited", (None, None))[1]
    distill_generated = distill_counts.get("generated")
    distill_yield = _safe_ratio(distill_generated, distill_visited)
    if snap.stage == "distill" and distill_visited is not None and distill_visited >= 50 and distill_yield is not None and distill_yield < 0.15:
        notes.append(f"teacher yield is low at {distill_yield:.0%} of visited prompts")
    distill_rate = _parse_rate_per_second(snap.stage_rate_label if snap.stage == "distill" else snap.distill_summary)
    if snap.stage == "distill" and distill_rate is not None and distill_rate < 0.5:
        notes.append(f"teacher throughput is slow at {distill_rate:.2f}/s")
    if snap.stage == "distill" and distill_total is not None and distill_visited is not None and distill_total > 0:
        distill_progress = _safe_ratio(distill_visited, distill_total)
        if distill_progress is not None and distill_progress < 0.1:
            notes.append("teacher distillation is still in very early progress")

    pref_accepted = pref_fracs.get("accepted", (None, None))[0]
    pref_visited = pref_fracs.get("visited", (None, None))[0]
    pref_yield = _safe_ratio(pref_accepted, pref_visited)
    if snap.stage == "preference_mining" and pref_visited is not None and pref_visited >= 200 and pref_yield is not None and pref_yield < 0.1:
        notes.append(f"mining yield is low at {pref_yield:.0%} accepted per visited prompt")
    if snap.stage == "preference_mining":
        generation_failures = pref_counts.get("generation_failures")
        if generation_failures is not None and generation_failures > 0:
            notes.append(f"mining already logged {generation_failures} generation failures")

    eval_keep_ratio = _safe_ratio(*(eval_fracs["kept"]) if "kept" in eval_fracs else (None, None))
    if snap.stage == "eval" and eval_keep_ratio is not None and eval_keep_ratio < 0.5:
        notes.append(f"eval filter kept only {eval_keep_ratio:.0%} of raw eval pairs")

    if snap.checkpoint_eta_seconds is not None and snap.checkpoint_eta_seconds >= 3.0 * 3600.0:
        notes.append(f"next checkpoint is still {_fmt_eta(snap.checkpoint_eta_seconds)} away")

    if snap.status == "running" and snap.stage_eta_seconds is not None and snap.stage_eta_seconds >= 6.0 * 3600.0:
        notes.append(f"{snap.stage} still has a long stage ETA ({_fmt_eta(snap.stage_eta_seconds)})")

    if snap.balance_loss is not None:
        if snap.balance_loss >= 0.25:
            notes.append(f"routing balance is elevated at {snap.balance_loss:.3f}")
        elif snap.balance_loss <= 0.12 and snap.stage in {"stage1", "stage2"}:
            notes.append(f"routing balance is healthy at {snap.balance_loss:.3f}")

    if not notes:
        if snap.status == "finished" or snap.stage == "done":
            return "Watch: finished cleanly; benchmark and archive the run outputs"
        if snap.status == "running":
            return f"Watch: healthy {device} run; next attention point is {snap.stage or 'active'}"
        return "Watch: no immediate operator issues detected"
    return "Watch: " + " | ".join(notes[:4])


def _run_attention_priority(snap: RunSnapshot) -> Tuple[int, float]:
    if snap.err_signal == "error":
        return (5, float(snap.stale_minutes))
    if snap.status == "stalled":
        return (4, float(snap.stale_minutes))
    if snap.status == "stopped" and snap.stage != "done":
        return (3, float(snap.stale_minutes))
    if snap.err_signal == "warn":
        return (2, float(snap.stale_minutes))
    if snap.status == "running":
        return (1, float(snap.stale_minutes))
    return (0, float(snap.stale_minutes))


def _summarize_backend_mix(snapshots: Sequence[RunSnapshot]) -> str:
    counts: Dict[str, int] = {}
    for snap in snapshots:
        backend = _runtime_device_value(snap)
        counts[backend] = counts.get(backend, 0) + 1
    if not counts:
        return "Backend Mix: -"
    parts = [f"{name} {count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:6]]
    return "Backend Mix: " + " | ".join(parts)


def _summarize_fleet_watchlist(snapshots: Sequence[RunSnapshot], limit: int = 3) -> str:
    if not snapshots:
        return "Fleet Watch: -"
    ranked = sorted(snapshots, key=lambda snap: _run_attention_priority(snap), reverse=True)
    items: List[str] = []
    for snap in ranked:
        priority, _ = _run_attention_priority(snap)
        if priority <= 0 and items:
            break
        watch = _build_run_watch_summary(snap)
        trimmed = watch.replace("Watch: ", "", 1)
        items.append(f"{snap.run_name} -> {trimmed}")
        if len(items) >= max(1, int(limit)):
            break
    if not items:
        return "Fleet Watch: no operator watch items"
    return "Fleet Watch: " + " | ".join(items)


def _summarize_fleet_tempo(snapshots: Sequence[RunSnapshot]) -> str:
    active = [snap for snap in snapshots if snap.status in {"running", "stalled"}]
    if not active:
        return "Tempo Board: -"

    parts: List[str] = []

    fastest = [snap for snap in active if snap.step_rate_per_hour is not None]
    if fastest:
        lead = max(fastest, key=lambda snap: float(snap.step_rate_per_hour or 0.0))
        parts.append(f"fastest trainer {lead.run_name} {float(lead.step_rate_per_hour or 0.0):.1f} steps/h")

    checkpoint_rows = [
        (snap, float(snap.checkpoint_eta_seconds))
        for snap in active
        if snap.checkpoint_eta_seconds is not None
    ]
    if checkpoint_rows:
        next_ckpt_snap, next_ckpt_eta = min(checkpoint_rows, key=lambda item: item[1])
        parts.append(f"next checkpoint {next_ckpt_snap.run_name} {_fmt_eta(next_ckpt_eta)}")

    freshest = min(active, key=lambda snap: float(snap.stale_minutes))
    parts.append(f"freshest log {freshest.run_name} {freshest.stale_minutes:.1f}m stale")

    hottest = [snap for snap in active if snap.cpu_percent is not None]
    if hottest:
        hot = max(hottest, key=lambda snap: float(snap.cpu_percent or 0.0))
        parts.append(f"highest host CPU {hot.run_name} {float(hot.cpu_percent or 0.0):.0f}%")

    return "Tempo Board: " + " | ".join(parts[:4])


def _summarize_fleet_checkpoint_posture(snapshots: Sequence[RunSnapshot]) -> str:
    if not snapshots:
        return "Checkpoint Posture: -"

    safeguarded = 0
    fragile_active = 0
    resumable_down = 0
    cold_restart = 0

    for snap in snapshots:
        has_checkpoint = snap.checkpoint_count > 0 and snap.last_checkpoint_step > 0
        active = snap.status in {"running", "stalled"}
        down = snap.status in {"stopped", "unknown"} or (not snap.pid_alive and snap.status != "finished")
        if active and has_checkpoint:
            safeguarded += 1
        elif active and not has_checkpoint:
            fragile_active += 1
        elif down and has_checkpoint:
            resumable_down += 1
        elif down and snap.stage != "done":
            cold_restart += 1

    return (
        "Checkpoint Posture: "
        f"safeguarded {safeguarded} | fragile active {fragile_active} | "
        f"resumable down {resumable_down} | cold restart {cold_restart}"
    )


def _summarize_fleet_spotlight(snapshots: Sequence[RunSnapshot]) -> str:
    if not snapshots:
        return "Spotlight: -"

    active = [snap for snap in snapshots if snap.status in {"running", "stalled"}]
    if not active:
        finished = [snap for snap in snapshots if snap.status == "finished" or snap.stage == "done"]
        if finished:
            latest = max(finished, key=lambda snap: snap.out_last_write_ts)
            return f"Spotlight: latest finished run is {latest.run_name}"
        return "Spotlight: no active runs"

    lead_progress_snap = max(
        active,
        key=lambda snap: (
            -1.0 if _compute_display_progress_percent(snap) is None else _compute_display_progress_percent(snap),
            snap.out_last_write_ts,
        ),
    )
    lead_progress = _compute_display_progress_percent(lead_progress_snap)

    eta_rows = [(snap, snap.eta_seconds if snap.eta_seconds is not None else snap.stage_eta_seconds) for snap in active]
    eta_rows = [(snap, eta) for snap, eta in eta_rows if eta is not None]
    fastest_eta_text = "-"
    if eta_rows:
        fastest_snap, fastest_eta = min(eta_rows, key=lambda item: float(item[1]))
        fastest_eta_text = f"{fastest_snap.run_name} {_fmt_eta(float(fastest_eta))}"

    attention_snap = max(active, key=lambda snap: (_run_attention_priority(snap), snap.out_last_write_ts))
    attention_text = attention_snap.run_name
    if attention_snap.status == "stalled":
        attention_text += " stalled"
    elif attention_snap.err_signal == "error":
        attention_text += " error"
    elif attention_snap.err_signal == "warn":
        attention_text += " warn"
    else:
        attention_text += f" {attention_snap.stage}"

    progress_text = "-" if lead_progress is None else f"{lead_progress:.1f}%"
    return (
        f"Spotlight: lead progress {lead_progress_snap.run_name} {progress_text} | "
        f"fastest ETA {fastest_eta_text} | "
        f"top attention {attention_text}"
    )


def _build_selected_vs_fleet_summary(selected: RunSnapshot, snapshots: Sequence[RunSnapshot]) -> str:
    peers = [snap for snap in snapshots if snap.run_name != selected.run_name]
    if not peers:
        return "Compare: no peer runs in the current fleet view"

    selected_progress = _compute_display_progress_percent(selected)
    peer_progresses = [value for value in (_compute_display_progress_percent(snap) for snap in peers) if value is not None]
    selected_eta = selected.eta_seconds if selected.eta_seconds is not None else selected.stage_eta_seconds
    peer_etas = [
        value
        for value in (
            snap.eta_seconds if snap.eta_seconds is not None else snap.stage_eta_seconds
            for snap in peers
        )
        if value is not None
    ]
    selected_rate = selected.step_rate_per_hour
    peer_rates = [snap.step_rate_per_hour for snap in peers if snap.step_rate_per_hour is not None]

    parts: List[str] = []
    if selected_progress is not None and peer_progresses:
        avg_progress = sum(peer_progresses) / float(len(peer_progresses))
        delta = selected_progress - avg_progress
        direction = "ahead of" if delta >= 0 else "behind"
        parts.append(f"progress {abs(delta):.1f} pts {direction} fleet avg")

    if selected_eta is not None and peer_etas:
        median_eta = sorted(peer_etas)[len(peer_etas) // 2]
        if float(selected_eta) <= float(median_eta):
            parts.append(f"ETA faster than fleet median by {_fmt_eta(float(median_eta) - float(selected_eta))}")
        else:
            parts.append(f"ETA slower than fleet median by {_fmt_eta(float(selected_eta) - float(median_eta))}")

    if selected_rate is not None and peer_rates:
        avg_rate = sum(peer_rates) / float(len(peer_rates))
        if avg_rate > 0:
            ratio = float(selected_rate) / float(avg_rate)
            parts.append(f"throughput {ratio:.2f}x fleet avg")

    selected_backend = _runtime_device_value(selected)
    peer_same_backend = [snap for snap in peers if _runtime_device_value(snap) == selected_backend]
    if peer_same_backend:
        parts.append(f"{selected_backend} peers {len(peer_same_backend)}")
    else:
        parts.append(f"unique backend in current view: {selected_backend}")

    same_stage = [snap for snap in peers if str(snap.stage or "").strip().lower() == str(selected.stage or "").strip().lower()]
    if same_stage:
        parts.append(f"stage peers {len(same_stage)}")

    if not parts:
        return "Compare: not enough comparable fleet telemetry yet"
    return "Compare: " + " | ".join(parts[:4])


def _summarize_stage_mix(snapshots: Sequence[RunSnapshot]) -> str:
    counts: Dict[str, int] = {}
    for snap in snapshots:
        stage = str(snap.stage or "unknown").strip().lower() or "unknown"
        counts[stage] = counts.get(stage, 0) + 1
    if not counts:
        return "Stage Mix: -"
    parts = [f"{stage} {count}" for stage, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    return "Stage Mix: " + " | ".join(parts)


def _summarize_issue_runs(snapshots: Sequence[RunSnapshot], limit: int = 3) -> str:
    issues = [snap for snap in snapshots if snap.status in {"stalled", "stopped", "unknown"} or snap.err_signal in {"error", "warn"}]
    if not issues:
        return "Issue Radar: no active blockers"
    issues.sort(
        key=lambda snap: (
            0 if snap.err_signal == "error" else 1,
            0 if snap.status == "stalled" else 1,
            -float(snap.stale_minutes),
            snap.run_name,
        )
    )
    parts = [
        f"{snap.run_name} [{snap.status}/{snap.stage}{' ' + snap.err_signal if snap.err_signal != 'ok' else ''}]"
        for snap in issues[: max(1, int(limit))]
    ]
    more = len(issues) - len(parts)
    if more > 0:
        parts.append(f"+{more} more")
    return "Issue Radar: " + " | ".join(parts)


def _resolve_canvas_size(
    width: int,
    height: int,
    default_width: int,
    default_height: int,
    min_width: int = 120,
    min_height: int = 48,
) -> Tuple[int, int]:
    try:
        w = int(width)
    except Exception:
        w = 0
    try:
        h = int(height)
    except Exception:
        h = 0
    if w < int(min_width):
        w = int(default_width)
    if h < int(min_height):
        h = int(default_height)
    return max(int(min_width), w), max(int(min_height), h)


def _history_window_seconds(window_key: str) -> float:
    key = str(window_key or "").strip().lower()
    return {
        "10m": 10.0 * 60.0,
        "60m": 60.0 * 60.0,
        "6h": 6.0 * 3600.0,
    }.get(key, 60.0 * 60.0)


def _phase_breakdown_rows(snap: RunSnapshot) -> List[Tuple[str, float, float, bool]]:
    rows: List[Tuple[str, float, float, bool]] = []
    for phase, weight in _phase_weight_plan(snap):
        frac = _phase_completion(snap, phase)
        rows.append(
            (
                phase,
                float(weight),
                0.0 if frac is None else max(0.0, min(1.0, float(frac))),
                str(snap.stage or "").strip().lower() == str(phase or "").strip().lower(),
            )
        )
    return rows


def _runtime_device_value(snap: RunSnapshot) -> str:
    runtime = str(snap.runtime_summary or "").strip().lower()
    match = re.search(r"\bresolved=([a-z0-9_:+.-]+)", runtime)
    if match:
        return match.group(1)
    match = re.search(r"\bdevice=([a-z0-9_:+.-]+)", runtime)
    if match:
        value = match.group(1)
        if value == "privateuseone:0":
            return "dml"
        return value
    return "-"


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
        self.only_pinned_var = tk.BooleanVar(value=False)
        self.filter_status_var = tk.StringVar(value="all")
        self.filter_backend_var = tk.StringVar(value="all")
        self.search_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.selected_progress_var = tk.StringVar(value="Progress: -")
        self.selected_eta_var = tk.StringVar(value="ETA: -")
        self.selected_ckpt_eta_var = tk.StringVar(value="Next Ckpt: -")
        self.selected_rate_var = tk.StringVar(value="Rate: -")
        self.selected_eta_confidence_var = tk.StringVar(value="ETA Confidence: -")
        self.selected_updated_var = tk.StringVar(value="Updated: -")
        self.selected_focus_var = tk.StringVar(value="Focus: -")
        self.selected_watch_var = tk.StringVar(value="Watch: -")
        self.selected_runtime_var = tk.StringVar(value="Runtime: -")
        self.selected_stability_var = tk.StringVar(value="Stability Lab: -")
        self.selected_failure_var = tk.StringVar(value="Failure Lens: -")
        self.selected_recovery_var = tk.StringVar(value="Recovery: -")
        self.selected_rescue_var = tk.StringVar(value="Rescue Plan: -")
        self.selected_compare_var = tk.StringVar(value="Compare: -")
        self.v46_monitor_var = tk.StringVar(value="V46 Status: -")
        self.v46_mod_var = tk.StringVar(value="V46 MoD: -")
        self.v46_got_var = tk.StringVar(value="V46 GoT: -")
        self.v46_ccot_var = tk.StringVar(value="V46 C-CoT: -")
        self.fleet_progress_var = tk.StringVar(value="Fleet Progress: -")
        self.fleet_eta_var = tk.StringVar(value="Fleet ETA: -")
        self.fleet_stage_var = tk.StringVar(value="Stage Mix: -")
        self.fleet_issue_var = tk.StringVar(value="Issue Radar: -")
        self.fleet_backend_var = tk.StringVar(value="Backend Mix: -")
        self.fleet_watch_var = tk.StringVar(value="Fleet Watch: -")
        self.fleet_spotlight_var = tk.StringVar(value="Spotlight: -")
        self.fleet_tempo_var = tk.StringVar(value="Tempo Board: -")
        self.fleet_checkpoint_var = tk.StringVar(value="Checkpoint Posture: -")
        self.fleet_research_var = tk.StringVar(value="Research Radar: -")
        self.research_summary_var = tk.StringVar(value="Research: -")
        self.research_best_var = tk.StringVar(value="Best: -")
        self.research_latest_var = tk.StringVar(value="Latest: -")
        self.research_focus_var = tk.StringVar(value="Top regression: -")
        self.research_prompt_var = tk.StringVar(value="Prompt: -")
        self.research_prediction_var = tk.StringVar(value="Tuned: - | Ref: -")
        self.trend_metric_var = tk.StringVar(value="progress")
        self.trend_window_var = tk.StringVar(value="60m")
        self.current_snapshots: Dict[str, RunSnapshot] = {}
        self.current_research_results: List[ResearchResult] = []
        self.progress_history: Dict[str, List[Tuple[float, float]]] = {}
        self.display_progress_history: Dict[str, List[Tuple[float, float]]] = {}
        self.loss_history: Dict[str, List[Tuple[float, float]]] = {}
        self.lr_history: Dict[str, List[Tuple[float, float]]] = {}
        self.rdrop_history: Dict[str, List[Tuple[float, float]]] = {}
        self.wpo_std_history: Dict[str, List[Tuple[float, float]]] = {}
        self.rate_history: Dict[str, List[Tuple[float, float]]] = {}
        self.cpu_history: Dict[str, List[Tuple[float, float]]] = {}
        self.ram_history: Dict[str, List[Tuple[float, float]]] = {}
        self.stale_history: Dict[str, List[Tuple[float, float]]] = {}
        self.sort_col = "updated"
        self.sort_reverse = True
        self.gpu_var = tk.StringVar(value="GPU: scanning...")
        self._last_error_alert_ts = 0.0
        self._notification_enabled_var = tk.BooleanVar(value=True)
        self.pinned_runs: set[str] = set()

        self._build_ui()
        self._load_settings()
        self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_refresh()
        self._bind_keyboard_shortcuts()
        self._start_gpu_polling()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        
        bg_color = "#1e1e1e"
        fg_color = "#d4d4d4"
        panel_bg = "#252526"
        input_bg = "#3c3c3c"
        select_bg = "#094771"
        border_color = "#454545"
        
        self.root.configure(bg=bg_color)
        
        style.configure(".", background=bg_color, foreground=fg_color, fieldbackground=input_bg, insertcolor=fg_color, bordercolor=border_color, lightcolor=border_color, darkcolor=border_color)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TButton", background=panel_bg, foreground=fg_color, bordercolor=border_color)
        style.map("TButton", background=[("active", "#333333")])
        style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        style.map("TCheckbutton", background=[("active", bg_color)])
        
        style.configure("Treeview", background=panel_bg, foreground=fg_color, fieldbackground=panel_bg, bordercolor=border_color)
        style.map("Treeview", background=[("selected", select_bg)], foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading", background="#333333", foreground=fg_color, bordercolor=border_color)
        style.map("Treeview.Heading", background=[("active", "#3e3e42")])
        
        top = ttk.Frame(self.root, padding=0)
        top.pack(fill=tk.X)

        # ── Branded header banner ──
        header = tk.Canvas(top, height=48, bg="#1a1a2e", highlightthickness=0)
        header.pack(fill=tk.X)
        header.update_idletasks()
        hw = max(header.winfo_width(), 1520)
        # Draw gradient
        for i in range(hw):
            r = int(26 + (16 * i / hw))
            g = int(26 + (40 * i / hw))
            b = int(46 + (60 * i / hw))
            color = f"#{r:02x}{g:02x}{b:02x}"
            header.create_line(i, 0, i, 48, fill=color)
        header.create_text(20, 24, anchor="w", text="SUPERMIX TRAINING MONITOR",
                           fill="#00d4ff", font=("Segoe UI", 16, "bold"))
        header.create_text(hw - 20, 24, anchor="e", text="v29 | Research Workflow",
                           fill="#6a9fb5", font=("Segoe UI", 10))

        # ── Controls row ──
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Workspace").pack(side=tk.LEFT)
        self.root_entry = ttk.Entry(controls, width=50)
        self.root_entry.insert(0, str(self.root_dir))
        self.root_entry.pack(side=tk.LEFT, padx=(8, 10))

        ttk.Label(controls, text="Stall mins").pack(side=tk.LEFT)
        self.stale_entry = ttk.Entry(controls, width=7)
        self.stale_entry.insert(0, str(int(self.stale_minutes)))
        self.stale_entry.pack(side=tk.LEFT, padx=(8, 10))

        ttk.Label(controls, text="Refresh s").pack(side=tk.LEFT)
        self.refresh_entry = ttk.Entry(controls, width=7)
        self.refresh_entry.insert(0, str(int(self.refresh_seconds)))
        self.refresh_entry.pack(side=tk.LEFT, padx=(8, 10))

        ttk.Checkbutton(controls, text="Auto", variable=self.auto_refresh_var).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Checkbutton(controls, text="Alerts", variable=self._notification_enabled_var).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Refresh (F5)", command=self.refresh).pack(side=tk.LEFT)

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

        ttk.Label(filter_row, text="Backend").pack(side=tk.LEFT)
        self.backend_combo = ttk.Combobox(
            filter_row,
            width=8,
            state="readonly",
            values=("all", "cpu", "cuda", "dml", "npu"),
            textvariable=self.filter_backend_var,
        )
        self.backend_combo.pack(side=tk.LEFT, padx=(8, 12))
        self.backend_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh())

        ttk.Checkbutton(filter_row, text="Only Active", variable=self.only_active_var, command=self.refresh).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )
        ttk.Checkbutton(filter_row, text="Only Issues", variable=self.only_issues_var, command=self.refresh).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )
        ttk.Checkbutton(filter_row, text="Only Pinned", variable=self.only_pinned_var, command=self.refresh).pack(
            side=tk.LEFT,
            padx=(0, 12),
        )
        ttk.Label(filter_row, text="Search").pack(side=tk.LEFT)
        self.search_entry = ttk.Entry(filter_row, width=34, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, padx=(8, 12))
        self.search_entry.bind("<Return>", lambda _e: self.refresh())
        ttk.Button(filter_row, text="Clear Filters", command=self._clear_filters).pack(side=tk.LEFT, padx=(0, 16))


        ttk.Button(filter_row, text="Pin/Unpin Run", command=self._toggle_selected_pin).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Open OUT Log", command=self._open_selected_out_log).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Open ERR Log", command=self._open_selected_err_log).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Open Run Dir", command=self._open_selected_run_dir).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Copy CMD/Launch", command=self._copy_selected_command).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Copy Watch", command=self._copy_selected_watch).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(filter_row, text="Copy Rescue", command=self._copy_selected_rescue).pack(side=tk.LEFT, padx=(0, 8))
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
        ttk.Label(summary, textvariable=self.selected_eta_confidence_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(summary, textvariable=self.selected_updated_var).pack(side=tk.LEFT)

        fleet = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        fleet.pack(fill=tk.X)
        self.fleet_progress_bar = ttk.Progressbar(fleet, mode="determinate", maximum=100, length=420)
        self.fleet_progress_bar.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(fleet, textvariable=self.fleet_progress_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(fleet, textvariable=self.fleet_eta_var).pack(side=tk.LEFT, padx=(0, 20))

        fleet_notes = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        fleet_notes.pack(fill=tk.X)
        ttk.Label(fleet_notes, textvariable=self.selected_focus_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w")
        ttk.Label(fleet_notes, textvariable=self.selected_watch_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.selected_runtime_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.selected_stability_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.selected_failure_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.selected_recovery_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.selected_rescue_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.selected_compare_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_spotlight_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(6, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_tempo_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_checkpoint_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_research_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_stage_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(6, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_backend_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_issue_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))
        ttk.Label(fleet_notes, textvariable=self.fleet_watch_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(4, 0))

        # ── V46 Frontier Architecture Panel ──
        v46_panel = ttk.LabelFrame(self.root, text="V46 FRONTIER ARCHITECTURE TELEMETRY", padding=(10, 5, 10, 5))
        v46_panel.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(v46_panel, textvariable=self.v46_monitor_var, font=("Segoe UI", 9, "bold"), foreground="#00d4ff").pack(anchor="w")
        ttk.Label(v46_panel, textvariable=self.v46_mod_var).pack(anchor="w")
        ttk.Label(v46_panel, textvariable=self.v46_got_var).pack(anchor="w")
        ttk.Label(v46_panel, textvariable=self.v46_ccot_var).pack(anchor="w")

        # ── GPU Stats Row ──
        research = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        research.pack(fill=tk.X)
        research_head = ttk.Frame(research)
        research_head.pack(fill=tk.X)
        ttk.Label(research_head, text="Research Board").pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(research_head, text="Open results.tsv", command=self._open_research_results).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(research_head, text="Open selected benchmark", command=self._open_selected_research_benchmark).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(research_head, text="Open selected samples", command=self._open_selected_research_samples).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(research_head, text="Open selected output", command=self._open_selected_research_output_dir).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(research_head, text="Open latest benchmark", command=self._open_latest_research_benchmark).pack(side=tk.LEFT)
        ttk.Label(research, textvariable=self.research_summary_var, justify=tk.LEFT).pack(anchor="w", pady=(6, 0))
        ttk.Label(research, textvariable=self.research_best_var, justify=tk.LEFT).pack(anchor="w", pady=(2, 0))
        ttk.Label(research, textvariable=self.research_latest_var, justify=tk.LEFT).pack(anchor="w", pady=(2, 6))
        ttk.Label(research, textvariable=self.research_focus_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(0, 2))
        ttk.Label(research, textvariable=self.research_prompt_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(0, 2))
        ttk.Label(research, textvariable=self.research_prediction_var, justify=tk.LEFT, wraplength=1480).pack(anchor="w", pady=(0, 6))

        research_table = ttk.Frame(research)
        research_table.pack(fill=tk.X)
        research_cols = ("run", "status", "f1", "char", "gen", "desc")
        self.research_tree = ttk.Treeview(research_table, columns=research_cols, show="headings", height=5)
        for col, width in (
            ("run", 230),
            ("status", 90),
            ("f1", 80),
            ("char", 80),
            ("gen", 90),
            ("desc", 520),
        ):
            heading = {"f1": "F1Δ", "char": "CHARΔ", "gen": "GENΔ"}.get(col, col.upper())
            self.research_tree.heading(col, text=heading)
            self.research_tree.column(col, width=width, anchor=tk.CENTER)
        self.research_tree.column("run", anchor=tk.W)
        self.research_tree.column("desc", anchor=tk.W)
        self.research_tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.research_tree.bind("<<TreeviewSelect>>", lambda _e: self._update_research_focus())
        self.research_tree.bind("<Double-1>", lambda _e: self._open_selected_research_benchmark())
        research_scroll = ttk.Scrollbar(research_table, orient=tk.VERTICAL, command=self.research_tree.yview)
        self.research_tree.configure(yscrollcommand=research_scroll.set)
        research_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        gpu_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        gpu_frame.pack(fill=tk.X)
        self.gpu_canvas = tk.Canvas(
            gpu_frame,
            width=620,
            height=84,
            background=panel_bg,
            highlightthickness=1,
            highlightbackground=border_color,
        )
        self.gpu_canvas.pack(fill=tk.X, expand=True)

        trend = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        trend.pack(fill=tk.X)
        trend_controls = ttk.Frame(trend)
        trend_controls.pack(fill=tk.X)
        ttk.Label(trend_controls, text="Selected Run Trend:").pack(side=tk.LEFT, padx=(0, 8))
        self.trend_metric_combo = ttk.Combobox(
            trend_controls,
            width=12,
            state="readonly",
            values=("progress", "loss", "lr", "rdrop", "wpo_std", "rate", "cpu", "ram", "stale"),
            textvariable=self.trend_metric_var,
        )
        self.trend_metric_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.trend_metric_combo.bind("<<ComboboxSelected>>", lambda _e: self._draw_selected_trend(self._selected_run_name()))
        ttk.Label(trend_controls, text="Window").pack(side=tk.LEFT, padx=(0, 6))
        self.trend_window_combo = ttk.Combobox(
            trend_controls,
            width=6,
            state="readonly",
            values=("10m", "60m", "6h"),
            textvariable=self.trend_window_var,
        )
        self.trend_window_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.trend_window_combo.bind("<<ComboboxSelected>>", lambda _e: self._draw_selected_trend(self._selected_run_name()))

        self.trend_canvas = tk.Canvas(
            trend,
            width=620,
            height=110,
            background=panel_bg,
            highlightthickness=1,
            highlightbackground=border_color,
        )
        self.trend_canvas.pack(fill=tk.X, expand=True, pady=(6, 0))
        self.trend_canvas.bind("<Configure>", lambda _e: self._draw_selected_trend(self._selected_run_name()))

        phase = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        phase.pack(fill=tk.X)
        ttk.Label(phase, text="Phase Breakdown:").pack(side=tk.LEFT, padx=(0, 8))
        self.phase_canvas = tk.Canvas(
            phase,
            width=620,
            height=44,
            background=panel_bg,
            highlightthickness=1,
            highlightbackground=border_color,
        )
        self.phase_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.phase_canvas.bind("<Configure>", lambda _e: self._draw_phase_breakdown(self._selected_snapshot()))

        table_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = (
            "pin",
            "run",
            "status",
            "stage",
            "telemetry",
            "device",
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
            "cpu_ram",
        )
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=16)
        for col, width in (
            ("pin", 42),
            ("run", 260),
            ("status", 90),
            ("stage", 130),
            ("telemetry", 140),
            ("device", 90),
            ("sft", 80),
            ("pref", 80),
            ("pairs", 120),
            ("loss", 90),
            ("lr", 95),
            ("prog", 90),
            ("eta", 130),
            ("eta_conf", 90),
            ("ckpt_eta", 130),
            ("rate", 95),
            ("err", 85),
            ("stale", 75),
            ("updated", 165),
            ("pid", 80),
            ("cpu_ram", 100),
        ):
            heading = {"pairs": "WORK", "pin": "PIN"}.get(col, col.upper())
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
        self.detail_text = tk.Text(detail, wrap=tk.NONE, height=22, font=("Consolas", 10), bg=panel_bg, fg=fg_color, insertbackground=fg_color, highlightthickness=0, borderwidth=1, relief="solid")
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll = ttk.Scrollbar(detail, orient=tk.VERTICAL, command=self.detail_text.yview)
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(side=tk.LEFT)

        self.tree.tag_configure("running", background="#0e4020", foreground="#d4d4d4")
        self.tree.tag_configure("stalled", background="#5b1b15", foreground="#d4d4d4")
        self.tree.tag_configure("finished", background="#153655", foreground="#d4d4d4")
        self.tree.tag_configure("stopped", background="#4a3e14", foreground="#d4d4d4")
        self.tree.tag_configure("err_error", foreground="#f14c4c")
        self.tree.tag_configure("err_warn", foreground="#cca700")
        self.tree.tag_configure("telemetry_state", foreground="#8fb7ff")
        self.research_tree.tag_configure("keep", background="#0e4020", foreground="#d4d4d4")
        self.research_tree.tag_configure("discard", background="#4a3e14", foreground="#d4d4d4")
        self.research_tree.tag_configure("crash", background="#5b1b15", foreground="#d4d4d4")

    def _bind_keyboard_shortcuts(self) -> None:
        self.root.bind("<F5>", lambda _e: self.refresh())
        self.root.bind("<Control-f>", lambda _e: self._focus_search())
        self.root.bind("<Control-l>", lambda _e: self._open_selected_out_log())
        self.root.bind("<Control-e>", lambda _e: self._open_selected_err_log())
        self.root.bind("<Control-d>", lambda _e: self._open_selected_run_dir())
        self.root.bind("<Control-n>", lambda _e: self._select_next_issue())
        self.root.bind("<Control-p>", lambda _e: self._toggle_selected_pin())

    def _focus_search(self) -> None:
        self.search_entry.focus_set()
        self.search_entry.select_range(0, tk.END)

    def _start_gpu_polling(self) -> None:
        def _poll():
            while True:
                try:
                    gpus = _query_gpu_stats()
                    windows_summary = _query_windows_accelerators()
                    self._draw_accelerator_stats(gpus, windows_summary)
                except Exception:
                    pass
                time.sleep(3.0)
        t = threading.Thread(target=_poll, daemon=True)
        t.start()

    def _draw_accelerator_stats(self, gpus: List[Dict[str, str]], windows_summary: Dict[str, Any]) -> None:
        def _update():
            self.gpu_canvas.delete("all")
            backend = windows_summary.get("backend", {}) if isinstance(windows_summary, dict) else {}
            backend_text = (
                f"Torch {backend.get('torch', 'unavailable')} | active={backend.get('resolved', 'unknown')} "
                f"| CUDA {backend.get('cuda', 'no')} | DML pkg {backend.get('dml', 'no')} "
                f"| torch_npu {backend.get('npu', 'no')} | ORT QNN {backend.get('qnn', 'no')}"
            )
            self.gpu_canvas.create_text(
                10,
                12,
                anchor="w",
                text=backend_text,
                fill="#9da5b4",
                font=("Consolas", 9),
            )

            y_mid = 34
            bar_w = 116
            bar_h = 12
            rows_drawn = 1

            def _draw_row(label: str, util: float, detail: str) -> None:
                nonlocal y_mid, rows_drawn
                fill_color = "#00cc66" if util < 70 else ("#ffaa00" if util < 90 else "#ff4444")
                self.gpu_canvas.create_rectangle(
                    10,
                    y_mid - bar_h // 2,
                    10 + bar_w,
                    y_mid + bar_h // 2,
                    fill="#2d2d30",
                    outline="#454545",
                )
                fill_w = int(bar_w * max(0.0, min(100.0, util)) / 100.0)
                if fill_w > 0:
                    self.gpu_canvas.create_rectangle(
                        10,
                        y_mid - bar_h // 2,
                        10 + fill_w,
                        y_mid + bar_h // 2,
                        fill=fill_color,
                        outline="",
                    )
                self.gpu_canvas.create_text(
                    10 + bar_w + 8,
                    y_mid,
                    anchor="w",
                    text=f"{label} {util:.0f}% | {detail}",
                    fill="#d4d4d4",
                    font=("Consolas", 9),
                )
                y_mid += 20
                rows_drawn += 1

            if gpus:
                for gpu in gpus:
                    try:
                        util = float(gpu.get("util", "0") or 0)
                        mem_used = float(gpu.get("mem_used", "0") or 0)
                        mem_total = max(1.0, float(gpu.get("mem_total", "1") or 1))
                        temp = gpu.get("temp", "?")
                        power = gpu.get("power", "?")
                        mem_pct = mem_used / mem_total * 100.0
                        _draw_row(
                            label=f"GPU{gpu.get('index', '?')}",
                            util=util,
                            detail=(
                                f"{gpu.get('name', 'NVIDIA')} | VRAM {mem_used:.0f}/{mem_total:.0f}MB ({mem_pct:.0f}%) "
                                f"| {temp}C | {power}W"
                            ),
                        )
                    except Exception:
                        continue
            else:
                gpu_rows = windows_summary.get("gpus", []) if isinstance(windows_summary, dict) else []
                if gpu_rows:
                    for idx, row in enumerate(gpu_rows):
                        util = float(row.get("util", 0.0) or 0.0)
                        detail = (
                            f"{row.get('name', f'GPU {idx}')} | compute {float(row.get('compute', 0.0) or 0.0):.0f}% "
                            f"| 3d {float(row.get('graphics', 0.0) or 0.0):.0f}% "
                            f"| video {float(row.get('video', 0.0) or 0.0):.0f}% "
                            f"| shared {float(row.get('shared_gb', 0.0) or 0.0):.2f}G"
                        )
                        committed_gb = float(row.get("committed_gb", 0.0) or 0.0)
                        if committed_gb > 0.0:
                            detail += f" / commit {committed_gb:.2f}G"
                        _draw_row(label=f"GPU{idx}", util=util, detail=detail)
                else:
                    self.gpu_canvas.create_text(
                        10,
                        y_mid,
                        anchor="w",
                        text="GPU telemetry unavailable (nvidia-smi absent and Windows counters unavailable)",
                        fill="#777777",
                        font=("Consolas", 9),
                    )
                    y_mid += 20
                    rows_drawn += 1

            npu_rows = windows_summary.get("npus", []) if isinstance(windows_summary, dict) else []
            for idx, row in enumerate(npu_rows):
                manufacturer = str(row.get("manufacturer", "")).strip()
                status = str(row.get("status", "Unknown")).strip()
                detail = f"{row.get('name', f'NPU {idx}')}"
                if manufacturer:
                    detail += f" | {manufacturer}"
                detail += f" | status {status} | live activity counter unavailable"
                _draw_row(label=f"NPU{idx}", util=0.0, detail=detail)

            target_height = max(44, 14 + rows_drawn * 20)
            self.gpu_canvas.configure(height=target_height)
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def _check_and_alert(self, snapshots: Sequence[RunSnapshot]) -> None:
        if not self._notification_enabled_var.get():
            return
        now = time.time()
        if now - self._last_error_alert_ts < 30.0:
            return
        for snap in snapshots:
            if snap.status == "stalled" or snap.err_signal == "error":
                self._last_error_alert_ts = now
                try:
                    threading.Thread(
                        target=lambda: winsound.MessageBeep(winsound.MB_ICONEXCLAMATION),
                        daemon=True
                    ).start()
                except Exception:
                    pass
                return

    def _schedule_refresh(self) -> None:
        self.root.after(int(self.refresh_seconds * 1000), self._tick)

    def _tick(self) -> None:
        if self.auto_refresh_var.get():
            self.refresh()
        self._schedule_refresh()

    def _clear_filters(self) -> None:
        self.filter_status_var.set("all")
        self.filter_backend_var.set("all")
        self.only_active_var.set(False)
        self.only_issues_var.set(False)
        self.only_pinned_var.set(False)
        self.search_var.set("")
        self.refresh()

    def _selected_snapshot(self) -> Optional[RunSnapshot]:
        run_name = self._selected_run_name()
        if not run_name:
            return None
        return self.current_snapshots.get(run_name)

    def _is_pinned_run(self, run_name: str) -> bool:
        return str(run_name or "") in self.pinned_runs

    def _toggle_selected_pin(self) -> None:
        snap = self._selected_snapshot()
        if snap is None:
            return
        run_name = str(snap.run_name)
        if run_name in self.pinned_runs:
            self.pinned_runs.remove(run_name)
            self.status_var.set(f"Unpinned run: {run_name}")
        else:
            self.pinned_runs.add(run_name)
            self.status_var.set(f"Pinned run: {run_name}")
        self._save_settings()
        self.refresh()

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

    def _open_research_results(self) -> None:
        self._open_path(self.root_dir / "research" / "results.tsv")

    def _selected_research_result(self) -> Optional[ResearchResult]:
        selection = self.research_tree.selection()
        if not selection:
            return None
        iid = str(selection[0])
        if "::" not in iid:
            return None
        run_tag = iid.split("::", 1)[1]
        for row in self.current_research_results:
            if row.run_tag == run_tag:
                return row
        return None

    def _open_selected_research_benchmark(self) -> None:
        row = self._selected_research_result()
        if row is None or not row.benchmark_json:
            return
        benchmark_path = Path(row.benchmark_json)
        if benchmark_path.exists():
            self._open_path(benchmark_path)

    def _open_selected_research_samples(self) -> None:
        row = self._selected_research_result()
        if row is None or not row.benchmark_json:
            return
        insight = _load_research_failure_insight(Path(row.benchmark_json))
        if not insight.sample_comparison_jsonl:
            return
        sample_path = Path(insight.sample_comparison_jsonl)
        if sample_path.exists():
            self._open_path(sample_path)

    def _open_selected_research_output_dir(self) -> None:
        row = self._selected_research_result()
        if row is None or not row.output_dir:
            return
        output_path = Path(row.output_dir)
        if output_path.exists():
            self._open_path(output_path)

    def _open_latest_research_benchmark(self) -> None:
        for row in self.current_research_results:
            benchmark_path = Path(row.benchmark_json) if row.benchmark_json else None
            if benchmark_path is not None and benchmark_path.exists():
                self._open_path(benchmark_path)
                return

    def _selected_or_latest_research_result(self) -> Optional[ResearchResult]:
        selected = self._selected_research_result()
        if selected is not None:
            return selected
        if self.current_research_results:
            return self.current_research_results[0]
        return None

    def _update_research_focus(self) -> None:
        row = self._selected_or_latest_research_result()
        if row is None or not row.benchmark_json:
            self.research_focus_var.set("Top regression: -")
            self.research_prompt_var.set("Prompt: -")
            self.research_prediction_var.set("Tuned: - | Ref: -")
            return

        insight = _load_research_failure_insight(Path(row.benchmark_json))
        self.research_focus_var.set(insight.summary_line)
        self.research_prompt_var.set(insight.prompt_line)
        self.research_prediction_var.set(insight.prediction_line)

    def _update_research_board(self, results: Sequence[ResearchResult]) -> None:
        selected = self.research_tree.selection()
        selected_iid = str(selected[0]) if selected else ""
        self.current_research_results = list(results)
        summary, best_line, latest_line = _summarize_research_results(results)
        self.research_summary_var.set(summary)
        self.research_best_var.set(best_line)
        self.research_latest_var.set(latest_line)

        for iid in self.research_tree.get_children():
            self.research_tree.delete(iid)
        for row in list(results)[:8]:
            iid = f"research::{row.run_tag}"
            self.research_tree.insert(
                "",
                tk.END,
                iid=iid,
                values=(
                    row.run_tag,
                    row.status,
                    _format_signed(row.token_f1_delta),
                    _format_signed(row.char_similarity_delta),
                    f"{_format_signed(row.avg_gen_seconds_delta, 2)}s",
                    row.description,
                ),
                tags=(row.status,),
            )
        if selected_iid and self.research_tree.exists(selected_iid):
            self.research_tree.selection_set(selected_iid)
        elif results:
            self.research_tree.selection_set(f"research::{results[0].run_tag}")
        self._update_research_focus()

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
            backend = str(raw.get("filter_backend", "all")).strip().lower()
            if backend in {"all", "cpu", "cuda", "dml", "npu"}:
                self.filter_backend_var.set(backend)
            self.only_active_var.set(bool(raw.get("only_active", False)))
            self.only_issues_var.set(bool(raw.get("only_issues", False)))
            self.only_pinned_var.set(bool(raw.get("only_pinned", False)))
            self.search_var.set(str(raw.get("search", "")))
            self.auto_refresh_var.set(bool(raw.get("auto_refresh", True)))
            pinned_raw = raw.get("pinned_runs", [])
            if isinstance(pinned_raw, list):
                self.pinned_runs = {str(item) for item in pinned_raw if str(item).strip()}
            trend_metric = str(raw.get("trend_metric", "progress")).strip().lower()
            if trend_metric in {"progress", "loss", "lr", "rdrop", "wpo_std", "rate", "cpu", "ram", "stale"}:
                self.trend_metric_var.set(trend_metric)
            trend_window = str(raw.get("trend_window", "60m")).strip().lower()
            if trend_window in {"10m", "60m", "6h"}:
                self.trend_window_var.set(trend_window)

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
                "filter_backend": str(self.filter_backend_var.get()),
                "only_active": bool(self.only_active_var.get()),
                "only_issues": bool(self.only_issues_var.get()),
                "only_pinned": bool(self.only_pinned_var.get()),
                "search": str(self.search_var.get()),
                "auto_refresh": bool(self.auto_refresh_var.get()),
                "trend_metric": str(self.trend_metric_var.get()),
                "trend_window": str(self.trend_window_var.get()),
                "sort_col": self.sort_col,
                "sort_reverse": bool(self.sort_reverse),
                "pinned_runs": sorted(self.pinned_runs),
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
            "pinned": self._is_pinned_run(snap.run_name),
            "run": snap.run_name,
            "status": snap.status,
            "stage": snap.stage,
            "backend": _runtime_device_value(snap),
            "pid": snap.pid,
            "pid_source": snap.pid_source,
            "sft_step": snap.sft_step,
            "sft_target_steps": snap.sft_target_steps,
            "pref_step": snap.pref_step,
            "pref_target_steps": snap.pref_target_steps,
            "has_distill_stage": snap.has_distill_stage,
            "has_pref_mining_stage": snap.has_pref_mining_stage,
            "pref_pairs": snap.pref_pairs,
            "loss": snap.loss,
            "lr": snap.lr,
            "rdrop": snap.rdrop,
            "wpo_std": snap.wpo_std,
            "beta": snap.beta,
            "margin": snap.margin,
            "pref_objective": snap.pref_objective,
            "pref_reference_pairs": snap.pref_reference_pairs,
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
            "runtime_summary": snap.runtime_summary,
            "runtime_headline": _build_runtime_headline(snap),
            "watch_summary": _build_run_watch_summary(snap),
            "stability_lab": _build_run_stability_lab(
                snap,
                self.loss_history.get(snap.run_name, []),
                self.rate_history.get(snap.run_name, []),
                self.cpu_history.get(snap.run_name, []),
                self.stale_history.get(snap.run_name, []),
            ),
            "failure_lens": _failure_lens(snap),
            "recovery_outlook": _build_recovery_outlook(snap),
            "rescue_plan": _build_run_rescue_plan(snap),
            "state_file": str(snap.state_file) if snap.state_file is not None else "",
            "balance_loss": snap.balance_loss,
            "stage_source": snap.stage_source,
            "adapter_summary": snap.adapter_summary,
            "source_balance_summary": snap.source_balance_summary,
            "objective_summary": snap.objective_summary,
            "data_summary": snap.data_summary,
            "eval_summary": snap.eval_summary,
            "v46_mod_summary": snap.v46_mod_summary,
            "v46_got_summary": snap.v46_got_summary,
            "v46_ccot_summary": snap.v46_ccot_summary,
            "sft_filter_summary": snap.sft_filter_summary,
            "distill_config_summary": snap.distill_config_summary,
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
            self.fleet_spotlight_var.set(_summarize_fleet_spotlight(all_snapshots))
            self.fleet_tempo_var.set("Tempo Board: -")
            self.fleet_checkpoint_var.set("Checkpoint Posture: -")
            self.fleet_research_var.set("Research Radar: -")
            self.fleet_stage_var.set("Stage Mix: -")
            self.fleet_issue_var.set("Issue Radar: -")
            self.fleet_backend_var.set("Backend Mix: -")
            self.fleet_watch_var.set("Fleet Watch: -")
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
        self.fleet_spotlight_var.set(_summarize_fleet_spotlight(all_snapshots))
        self.fleet_tempo_var.set(_summarize_fleet_tempo(all_snapshots))
        self.fleet_checkpoint_var.set(_summarize_fleet_checkpoint_posture(all_snapshots))
        self.fleet_research_var.set(
            _summarize_fleet_research_radar(all_snapshots, self.loss_history, self.rate_history)
        )
        self.fleet_stage_var.set(_summarize_stage_mix(all_snapshots))
        self.fleet_backend_var.set(_summarize_backend_mix(all_snapshots))
        self.fleet_issue_var.set(_summarize_issue_runs(all_snapshots))
        self.fleet_watch_var.set(_summarize_fleet_watchlist(all_snapshots))

    def _draw_selected_trend(self, run_name: Optional[str]) -> None:
        self.trend_canvas.delete("all")
        if not run_name:
            self.trend_canvas.create_text(10, 55, anchor="w", text="No run selected", fill="#777777")
            return

        metric = str(self.trend_metric_var.get() or "progress").strip().lower()
        line_color = "#007acc"
        fill_color = "#0a2d4d"
        clamp_zero = False
        fixed_min: Optional[float] = None
        fixed_max: Optional[float] = None

        if metric == "loss":
            history = self.loss_history.get(run_name, [])
            label = "loss"
            formatter = lambda v: f"{v:.4f}"
            line_color = "#d18616"
            fill_color = "#3b2507"
            clamp_zero = True
        elif metric == "lr":
            history = self.lr_history.get(run_name, [])
            label = "lr"
            formatter = lambda v: f"{v:.3g}"
            line_color = "#4ec9b0"
            fill_color = "#0b2f28"
            clamp_zero = True
        elif metric == "rdrop":
            history = self.rdrop_history.get(run_name, [])
            label = "rdrop"
            formatter = lambda v: f"{v:.4f}"
            line_color = "#c586c0"
            fill_color = "#311630"
            clamp_zero = True
        elif metric == "wpo_std":
            history = self.wpo_std_history.get(run_name, [])
            label = "wpo_std"
            formatter = lambda v: f"{v:.4f}"
            line_color = "#d7ba7d"
            fill_color = "#3b2e13"
            clamp_zero = True
        elif metric == "rate":
            history = self.rate_history.get(run_name, [])
            label = "steps/h"
            formatter = lambda v: f"{v:.2f}"
            line_color = "#89d185"
            fill_color = "#163620"
            clamp_zero = True
        elif metric == "cpu":
            history = self.cpu_history.get(run_name, [])
            label = "cpu"
            formatter = lambda v: f"{v:.0f}%"
            line_color = "#f14c4c"
            fill_color = "#3b1212"
            fixed_min = 0.0
            fixed_max = 100.0
        elif metric == "ram":
            history = self.ram_history.get(run_name, [])
            label = "ram"
            formatter = lambda v: f"{v:.2f} GB"
            line_color = "#cca700"
            fill_color = "#3b3209"
            clamp_zero = True
        elif metric == "stale":
            history = self.stale_history.get(run_name, [])
            label = "stale"
            formatter = lambda v: f"{v:.1f}m"
            line_color = "#c586c0"
            fill_color = "#311630"
            clamp_zero = True
        else:
            history = self.display_progress_history.get(run_name, [])
            label = "progress"
            formatter = lambda v: f"{v:.1f}%"
            fixed_min = 0.0
            fixed_max = 100.0

        if len(history) < 2:
            self.trend_canvas.create_text(10, 55, anchor="w", text="Collecting trend data...", fill="#777777")
            return

        w, h = _resolve_canvas_size(
            self.trend_canvas.winfo_width(),
            self.trend_canvas.winfo_height(),
            default_width=620,
            default_height=110,
            min_width=220,
            min_height=72,
        )
        pad = 10
        now = time.time()
        window_key = str(self.trend_window_var.get() or "60m")
        start_ts = now - _history_window_seconds(window_key)
        points = [(t, v) for (t, v) in history if t >= start_ts]
        if len(points) < 2:
            points = history[-2:]
        t_min = points[0][0]
        t_max = points[-1][0]
        v_min = min(v for _, v in points) if fixed_min is None else float(fixed_min)
        v_max = max(v for _, v in points) if fixed_max is None else float(fixed_max)
        if clamp_zero:
            v_min = min(0.0, v_min)
        if t_max - t_min < 1e-6:
            t_max = t_min + 1.0
        if v_max - v_min < 1e-6:
            v_max = v_min + 1.0

        for frac in (0.2, 0.4, 0.6, 0.8):
            y = pad + (h - 2 * pad) * frac
            self.trend_canvas.create_line(pad, y, w - pad, y, fill="#3e3e42")
        self.trend_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline="#454545")
        xy: List[float] = []
        for t, v in points:
            x = pad + (w - 2 * pad) * ((t - t_min) / (t_max - t_min))
            y = (h - pad) - (h - 2 * pad) * ((v - v_min) / (v_max - v_min))
            xy.extend([x, y])

        if len(xy) >= 4:
            area_xy = [xy[0], h - pad] + xy + [xy[-2], h - pad]
            self.trend_canvas.create_polygon(*area_xy, fill=fill_color, outline="")
            self.trend_canvas.create_line(*xy, fill=line_color, width=2, smooth=False)

        self.trend_canvas.create_text(
            pad + 2,
            pad + 2,
            anchor="nw",
            text=f"{window_key} window | {len(points)} pts",
            fill="#9da5b4",
        )
        self.trend_canvas.create_text(
            w - pad - 4,
            pad + 2,
            anchor="ne",
            text=f"{label}: {formatter(points[-1][1])} | min {formatter(v_min)} max {formatter(v_max)}",
            fill="#d4d4d4",
        )

    def _draw_phase_breakdown(self, snap: Optional[RunSnapshot]) -> None:
        self.phase_canvas.delete("all")
        if snap is None:
            self.phase_canvas.create_text(10, 22, anchor="w", text="No run selected", fill="#777777")
            return

        rows = _phase_breakdown_rows(snap)
        if not rows:
            self.phase_canvas.create_text(10, 22, anchor="w", text="No phase data yet", fill="#777777")
            return

        w, h = _resolve_canvas_size(
            self.phase_canvas.winfo_width(),
            self.phase_canvas.winfo_height(),
            default_width=620,
            default_height=44,
            min_width=220,
            min_height=36,
        )
        pad = 8
        total_weight = sum(weight for _phase, weight, _frac, _active in rows)
        if total_weight <= 0:
            self.phase_canvas.create_text(10, 22, anchor="w", text="No phase weights available", fill="#777777")
            return

        phase_colors = {
            "data": "#007acc",
            "distill": "#4ec9b0",
            "sft_setup": "#8a8a8a",
            "sft": "#89d185",
            "preference_mining": "#cca700",
            "preference": "#d18616",
        }

        x = float(pad)
        usable_w = max(1.0, float(w - 2 * pad))
        y0 = float(pad)
        y1 = float(h - pad)

        for phase, weight, frac, active in rows:
            seg_w = usable_w * (float(weight) / float(total_weight))
            base_x1 = x + seg_w
            color = phase_colors.get(phase, "#569cd6")
            self.phase_canvas.create_rectangle(x, y0, base_x1, y1, fill="#1f1f1f", outline="#454545")
            fill_x1 = x + max(0.0, min(seg_w, seg_w * float(frac)))
            if fill_x1 > x:
                self.phase_canvas.create_rectangle(x, y0, fill_x1, y1, fill=color, outline="")
            if active:
                self.phase_canvas.create_rectangle(x, y0, base_x1, y1, outline="#ffffff", width=2)

            label = phase.replace("_", " ")
            pct_txt = f"{int(round(frac * 100.0))}%"
            if seg_w >= 70:
                self.phase_canvas.create_text(
                    x + seg_w / 2.0,
                    (y0 + y1) / 2.0,
                    text=f"{label} {pct_txt}",
                    fill="#d4d4d4",
                )
            x = base_x1

        self.phase_canvas.create_text(
            w - pad,
            3,
            anchor="ne",
            text=f"Stage: {snap.stage}",
            fill="#9da5b4",
        )

    def _append_history_point(
        self,
        store: Dict[str, List[Tuple[float, float]]],
        run_name: str,
        ts: float,
        value: float,
        keep_seconds: float = 21600.0,
        keep_points: int = 1500,
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
        if snap.status == "stalled" or snap.stale_minutes >= 6.0:
            return "low"
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
            confidence = "high"
        elif cv < 0.45:
            confidence = "medium"
        else:
            confidence = "low"

        if snap.stale_minutes >= 3.0:
            if confidence == "high":
                confidence = "medium"
            elif confidence == "medium":
                confidence = "low"
        return confidence

    def _apply_filters(self, snapshots: Sequence[RunSnapshot]) -> List[RunSnapshot]:
        out: List[RunSnapshot] = []
        status_filter = str(self.filter_status_var.get() or "all").strip().lower()
        backend_filter = str(self.filter_backend_var.get() or "all").strip().lower()
        search = str(self.search_var.get() or "").strip().lower()
        only_active = bool(self.only_active_var.get())
        only_issues = bool(self.only_issues_var.get())
        only_pinned = bool(self.only_pinned_var.get())
        for snap in snapshots:
            if status_filter != "all" and snap.status != status_filter:
                continue
            if backend_filter != "all" and _runtime_device_value(snap) != backend_filter:
                continue
            if only_active and snap.status not in {"running", "stalled"}:
                continue
            if only_issues and not self._is_issue_snapshot(snap):
                continue
            if only_pinned and not self._is_pinned_run(snap.run_name):
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
                        (snap.runtime_summary or "").lower(),
                        (snap.adapter_summary or "").lower(),
                        (snap.source_balance_summary or "").lower(),
                        (snap.objective_summary or "").lower(),
                        (snap.pref_objective or "").lower(),
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
        if col == "pin":
            return 1 if self._is_pinned_run(snap.run_name) else 0
        if col == "run":
            return snap.run_name.lower()
        if col == "status":
            return snap.status
        if col == "stage":
            return snap.stage
        if col == "telemetry":
            balance = -1.0 if snap.balance_loss is None else snap.balance_loss
            return (1 if snap.stage_source == "state" else 0, balance)
        if col == "device":
            return _runtime_device_value(snap)
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
        if col == "cpu_ram":
            return -1.0 if snap.cpu_percent is None else snap.cpu_percent
        return snap.run_name.lower()

    def _sort_by_column(self, col: str) -> None:
        if self.sort_col == col:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_col = col
            self.sort_reverse = True if col in {"pin", "updated", "prog", "rate", "loss", "sft", "pref", "eta_conf", "cpu_ram", "telemetry"} else False
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
            if snap.rdrop is not None:
                self._append_history_point(
                    store=self.rdrop_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.rdrop),
                )
            if snap.wpo_std is not None:
                self._append_history_point(
                    store=self.wpo_std_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.wpo_std),
                )
            if snap.cpu_percent is not None:
                self._append_history_point(
                    store=self.cpu_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.cpu_percent),
                )
            if snap.ram_gb is not None:
                self._append_history_point(
                    store=self.ram_history,
                    run_name=snap.run_name,
                    ts=now,
                    value=float(snap.ram_gb),
                )
            self._append_history_point(
                store=self.stale_history,
                run_name=snap.run_name,
                ts=now,
                value=float(snap.stale_minutes),
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

    def _copy_selected_watch(self) -> None:
        snap = self._selected_snapshot()
        if snap is None:
            return
        payload = "\n".join(
            [
                _build_run_watch_summary(snap),
                _build_runtime_headline(snap),
                _build_run_stability_lab(
                    snap,
                    self.loss_history.get(snap.run_name, []),
                    self.rate_history.get(snap.run_name, []),
                    self.cpu_history.get(snap.run_name, []),
                    self.stale_history.get(snap.run_name, []),
                ),
                _failure_lens(snap),
                _build_recovery_outlook(snap),
                f"Focus: {_build_run_recommendation(snap)}",
            ]
        ).strip()
        if not payload:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(payload)
            self.status_var.set(f"Copied watch summary for {snap.run_name}")
        except Exception:
            pass

    def _copy_selected_rescue(self) -> None:
        snap = self._selected_snapshot()
        if snap is None:
            return
        payload = "\n".join(
            [
                _build_recovery_outlook(snap),
                _build_run_rescue_plan(snap),
                f"Launch: {snap.launch_command or snap.launch_hint or '-'}",
            ]
        ).strip()
        if not payload:
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(payload)
            self.status_var.set(f"Copied rescue plan for {snap.run_name}")
        except Exception:
            pass

        all_snapshots = collect_run_snapshots(self.root_dir, stale_minutes_threshold=self.stale_minutes)
        self._update_research_board(_load_research_results(self.root_dir))
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
            updated = f"{_fmt_age_minutes(snap.stale_minutes)} | {_fmt_ts(snap.out_last_write_ts)}"
            telemetry_txt = "state" if snap.stage_source == "state" else "log"
            if snap.balance_loss is not None:
                telemetry_txt = f"{telemetry_txt} | bal={snap.balance_loss:.3f}"

            sft_txt = str(snap.sft_step)
            if snap.sft_target_steps is not None and snap.sft_target_steps > 0:
                sft_txt = f"{snap.sft_step}/{snap.sft_target_steps}"
            pref_txt = str(snap.pref_step)
            if snap.pref_target_steps is not None and snap.pref_target_steps > 0:
                pref_txt = f"{snap.pref_step}/{snap.pref_target_steps}"
            device_txt = _runtime_device_value(snap)

            row_tags = [snap.status]
            if snap.err_signal == "error":
                row_tags.append("err_error")
            elif snap.err_signal == "warn":
                row_tags.append("err_warn")
            if snap.stage_source == "state":
                row_tags.append("telemetry_state")
                
            cpu_ram_txt = "-"
            if snap.cpu_percent is not None and snap.ram_gb is not None:
                cpu_ram_txt = f"{snap.cpu_percent:.0f}% | {snap.ram_gb:.1f}G"

            self.tree.insert(
                "",
                tk.END,
                iid=snap.run_name,
                values=(
                    "★" if self._is_pinned_run(snap.run_name) else "",
                    snap.run_name,
                    snap.status,
                    snap.stage,
                    telemetry_txt,
                    device_txt,
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
                    cpu_ram_txt,
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
            self.selected_updated_var.set("Updated: -")
            self.selected_focus_var.set("Focus: -")
            self.selected_watch_var.set("Watch: -")
            self.selected_runtime_var.set("Runtime: -")
            self.selected_stability_var.set("Stability Lab: -")
            self.selected_failure_var.set("Failure Lens: -")
            self.selected_recovery_var.set("Recovery: -")
            self.selected_rescue_var.set("Rescue Plan: -")
            self.selected_compare_var.set("Compare: -")
            self._draw_phase_breakdown(None)

        self._update_fleet_summary(all_snapshots)
        self._check_and_alert(all_snapshots)

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
        self._draw_phase_breakdown(self._selected_snapshot())

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
            self.selected_updated_var.set("Updated: -")
            self.selected_focus_var.set("Focus: -")
            self.selected_watch_var.set("Watch: -")
            self.selected_runtime_var.set("Runtime: -")
            self.selected_stability_var.set("Stability Lab: -")
            self.selected_failure_var.set("Failure Lens: -")
            self.selected_recovery_var.set("Recovery: -")
            self.selected_rescue_var.set("Rescue Plan: -")
            self.selected_compare_var.set("Compare: -")
            self.v46_monitor_var.set("V46 Status: -")
            self.v46_mod_var.set("V46 MoD: -")
            self.v46_got_var.set("V46 GoT: -")
            self.v46_ccot_var.set("V46 C-CoT: -")
            self._draw_selected_trend(None)
            self._draw_phase_breakdown(None)
            return
        snap = self.current_snapshots.get(run_name)
        if snap is None:
            self.selected_progress_var.set("Progress: -")
            self.selected_eta_var.set("ETA: -")
            self.selected_ckpt_eta_var.set("Next Ckpt: -")
            self.selected_rate_var.set("Rate: -")
            self.selected_eta_confidence_var.set("ETA Confidence: -")
            self.selected_updated_var.set("Updated: -")
            self.selected_focus_var.set("Focus: -")
            self.selected_watch_var.set("Watch: -")
            self.selected_runtime_var.set("Runtime: -")
            self.selected_stability_var.set("Stability Lab: -")
            self.selected_failure_var.set("Failure Lens: -")
            self.selected_recovery_var.set("Recovery: -")
            self.selected_rescue_var.set("Rescue Plan: -")
            self.selected_compare_var.set("Compare: -")
            self.v46_monitor_var.set("V46 Status: -")
            self.v46_mod_var.set("V46 MoD: -")
            self.v46_got_var.set("V46 GoT: -")
            self.v46_ccot_var.set("V46 C-CoT: -")
            self._draw_selected_trend(None)
            self._draw_phase_breakdown(None)
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
        self.selected_updated_var.set(f"Updated: {_fmt_age_minutes(snap.stale_minutes)} ago")
        recommendation = _build_run_recommendation(snap)
        self.selected_focus_var.set(f"Focus: {recommendation}")
        watch_summary = _build_run_watch_summary(snap)
        self.selected_watch_var.set(watch_summary)
        runtime_headline = _build_runtime_headline(snap)
        self.selected_runtime_var.set(runtime_headline)
        stability_lab = _build_run_stability_lab(
            snap,
            self.loss_history.get(snap.run_name, []),
            self.rate_history.get(snap.run_name, []),
            self.cpu_history.get(snap.run_name, []),
            self.stale_history.get(snap.run_name, []),
        )
        self.selected_stability_var.set(stability_lab)
        failure_lens = _failure_lens(snap)
        self.selected_failure_var.set(failure_lens)
        recovery_outlook = _build_recovery_outlook(snap)
        self.selected_recovery_var.set(recovery_outlook)
        rescue_plan = _build_run_rescue_plan(snap)
        self.selected_rescue_var.set(rescue_plan)
        compare_summary = _build_selected_vs_fleet_summary(snap, list(self.current_snapshots.values()))
        self.selected_compare_var.set(compare_summary)

        # Update V46 Frontier Telemetry
        is_v46 = "v46" in snap.run_name.lower() or "v46" in snap.stage.lower()
        if is_v46:
            self.v46_monitor_var.set("V46 Status: ACTIVE FRONTIER LOGIC")
            self.v46_mod_var.set(f"V46 MoD: {snap.v46_mod_summary}")
            self.v46_got_var.set(f"V46 GoT: {snap.v46_got_summary}")
            self.v46_ccot_var.set(f"V46 C-CoT: {snap.v46_ccot_summary}")
        else:
            self.v46_monitor_var.set("V46 Status: INACTIVE (Legacy Model)")
            self.v46_mod_var.set("V46 MoD: -")
            self.v46_got_var.set("V46 GoT: -")
            self.v46_ccot_var.set("V46 C-CoT: -")

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
            f"pinned: {'yes' if self._is_pinned_run(snap.run_name) else 'no'}",
            f"status: {snap.status}",
            f"updated_age: {_fmt_age_minutes(snap.stale_minutes)} ago",
            f"out_last: {out_last}",
            f"err_last: {err_last}",
            f"health: {snap.health_summary}",
            f"focus: {recommendation}",
            f"watch: {watch_summary.replace('Watch: ', '', 1)}",
            f"runtime: {runtime_headline.replace('Runtime: ', '', 1)}",
            f"stability_lab: {stability_lab.replace('Stability Lab: ', '', 1)}",
            f"failure_lens: {failure_lens.replace('Failure Lens: ', '', 1)}",
            f"recovery: {recovery_outlook.replace('Recovery: ', '', 1)}",
            f"rescue_plan: {rescue_plan.replace('Rescue Plan: ', '', 1)}",
            f"compare: {compare_summary.replace('Compare: ', '', 1)}",
            f"stage: {snap.stage}",
            f"pid: {snap.pid if snap.pid is not None else '-'} (alive={snap.pid_alive}, source={snap.pid_source or '-'})",
            f"sft_step: {snap.sft_step} / {snap.sft_target_steps if snap.sft_target_steps is not None else '-'}",
            f"pref_step: {snap.pref_step} / {snap.pref_target_steps if snap.pref_target_steps is not None else '-'}",
            f"pref_pairs: {snap.pref_pairs}",
            f"loss: {snap.loss if snap.loss is not None else '-'}",
            f"lr: {snap.lr if snap.lr is not None else '-'}",
            f"rdrop: {snap.rdrop if snap.rdrop is not None else '-'}",
            f"wpo_std: {snap.wpo_std if snap.wpo_std is not None else '-'}",
            f"beta: {snap.beta if snap.beta is not None else '-'}",
            f"margin: {snap.margin if snap.margin is not None else '-'}",
            f"pref_objective: {snap.pref_objective}",
            f"pref_reference_pairs: {snap.pref_reference_pairs if snap.pref_reference_pairs is not None else '-'}",
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
            f"V46_MOD: {snap.v46_mod_summary}",
            f"V46_GOT: {snap.v46_got_summary}",
            f"V46_CCOT: {snap.v46_ccot_summary}",
            f"step_rate_per_hour: {snap.step_rate_per_hour if snap.step_rate_per_hour is not None else '-'}",
            f"stage_rate: {snap.stage_rate_label}",
            f"stage_source: {snap.stage_source}",
            f"state_file: {snap.state_file if snap.state_file is not None else '-'}",
            f"balance_loss: {snap.balance_loss if snap.balance_loss is not None else '-'}",
            f"runtime_summary: {snap.runtime_summary}",
            f"adapter_summary: {snap.adapter_summary}",
            f"source_balance_summary: {snap.source_balance_summary}",
            f"objective_summary: {snap.objective_summary}",
            f"data_summary: {snap.data_summary}",
            f"eval_summary: {snap.eval_summary}",
            f"sft_filter_summary: {snap.sft_filter_summary}",
            f"distill_config_summary: {snap.distill_config_summary}",
            f"distill_summary: {snap.distill_summary}",
            f"pref_mining_summary: {snap.pref_mining_summary}",
            f"pref_selection_summary: {snap.pref_selection_summary}",
            f"trend_metric: {self.trend_metric_var.get()}",
            f"stale_minutes: {snap.stale_minutes:.2f}",
            f"err_signal: {snap.err_signal}",
            f"err_summary: {snap.err_summary}",
            f"balance_loss: {snap.balance_loss if snap.balance_loss is not None else '-'}",
            f"stage_source: {snap.stage_source}",
            f"out_log: {snap.out_log}",
            f"err_log: {snap.err_log if snap.err_log is not None else '-'}",
            f"pid_file: {snap.pid_file if snap.pid_file is not None else '-'}",
            f"state_file: {snap.state_file if snap.state_file is not None else '-'}",
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
        
        self.detail_text.tag_configure("error", foreground="#f14c4c")
        self.detail_text.tag_configure("warn", foreground="#cca700")
        self.detail_text.tag_configure("ok", foreground="#89d185")
        self.detail_text.tag_configure("sys", foreground="#569cd6")
        
        for i, line in enumerate(lines, start=1):
            self.detail_text.insert(tk.END, line + "\n")
            low = line.lower()
            if "error" in low or "traceback" in low or "exception" in low or "failed" in low:
                self.detail_text.tag_add("error", f"{i}.0", f"{i}.end")
            elif "warn" in low:
                self.detail_text.tag_add("warn", f"{i}.0", f"{i}.end")
            elif "ok" in low or "complete:" in low or "progress:" in low or "[data]" in low or "[sft]" in low:
                self.detail_text.tag_add("ok", f"{i}.0", f"{i}.end")
            elif ":" in line and not line.startswith(" ") and not line.startswith("["):
                self.detail_text.tag_add("sys", f"{i}.0", f"{i}.end")

        self.detail_text.delete("end-1c", tk.END)
        if self.auto_refresh_var.get():
            self.detail_text.see(tk.END)
        self._draw_selected_trend(run_name)
        self._draw_phase_breakdown(snap)


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
