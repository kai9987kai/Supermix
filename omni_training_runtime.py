from __future__ import annotations

import math
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch


def _parse_boolish(value: object) -> bool:
    cooked = str(value or "").strip().lower()
    return cooked in {"1", "true", "yes", "on"}


@dataclass
class TrainingRuntime:
    requested_device: str
    resolved_device: str
    device_type: str
    amp_mode: str
    amp_enabled: bool
    amp_dtype_name: str
    scaler_enabled: bool
    compile_requested: bool
    compile_enabled: bool
    compile_mode: str
    compile_error: Optional[str]
    grad_accum_steps: int
    ema_decay: float
    warmup_steps: int
    warmup_ratio: float
    min_lr_scale: float
    git_commit: Optional[str]
    effective_batch_size: int

    @property
    def device(self) -> torch.device:
        return torch.device(self.resolved_device)

    @property
    def amp_dtype(self) -> Optional[torch.dtype]:
        name = str(self.amp_dtype_name or "").lower()
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        return None

    def to_payload(self) -> Dict[str, Any]:
        return dict(asdict(self))


def _resolve_device(requested_device: str) -> torch.device:
    requested = str(requested_device or "auto").strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(getattr(mps, "is_available", lambda: False)()):
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except Exception:
        return torch.device("cpu")


def _resolve_amp_dtype(device: torch.device, amp_dtype: str) -> tuple[bool, str]:
    requested = str(amp_dtype or "auto").strip().lower()
    if device.type == "cuda":
        if requested in {"auto", ""}:
            if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                return True, "bfloat16"
            return True, "float16"
        if requested in {"bf16", "bfloat16"}:
            return True, "bfloat16"
        if requested in {"fp16", "float16"}:
            return True, "float16"
        if requested in {"off", "false", "0", "none"}:
            return False, "none"
        return True, requested
    if device.type == "cpu":
        if requested in {"bf16", "bfloat16"}:
            return True, "bfloat16"
        return False, "none"
    if requested in {"off", "false", "0", "none"}:
        return False, "none"
    return False, "none"


def detect_git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        cooked = str(result.stdout or "").strip()
        return cooked or None
    except Exception:
        return None


def resolve_training_runtime(
    *,
    repo_root: Path,
    requested_device: str = "auto",
    amp_mode: str = "auto",
    amp_dtype: str = "auto",
    compile_requested: bool = False,
    compile_mode: str = "reduce-overhead",
    grad_accum_steps: int = 1,
    ema_decay: float = 0.999,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.05,
    min_lr_scale: float = 0.05,
    batch_size: int = 1,
) -> TrainingRuntime:
    device = _resolve_device(requested_device)
    amp_requested = str(amp_mode or "auto").strip().lower()
    dtype_enabled, dtype_name = _resolve_amp_dtype(device, amp_dtype)
    if amp_requested in {"off", "false", "0", "none"}:
        amp_enabled = False
    elif amp_requested in {"on", "true", "1"}:
        amp_enabled = dtype_enabled
    else:
        amp_enabled = dtype_enabled
    compile_enabled = bool(compile_requested and hasattr(torch, "compile") and device.type in {"cpu", "cuda"})
    return TrainingRuntime(
        requested_device=str(requested_device or "auto"),
        resolved_device=str(device),
        device_type=device.type,
        amp_mode=amp_requested or "auto",
        amp_enabled=bool(amp_enabled),
        amp_dtype_name=dtype_name,
        scaler_enabled=bool(amp_enabled and device.type == "cuda"),
        compile_requested=bool(compile_requested),
        compile_enabled=compile_enabled,
        compile_mode=str(compile_mode or "reduce-overhead"),
        compile_error=None,
        grad_accum_steps=max(1, int(grad_accum_steps)),
        ema_decay=max(0.0, float(ema_decay)),
        warmup_steps=max(0, int(warmup_steps)),
        warmup_ratio=max(0.0, float(warmup_ratio)),
        min_lr_scale=max(0.0, min(1.0, float(min_lr_scale))),
        git_commit=detect_git_commit(repo_root),
        effective_batch_size=max(1, int(batch_size)) * max(1, int(grad_accum_steps)),
    )


def maybe_compile_model(model: torch.nn.Module, runtime: TrainingRuntime) -> tuple[torch.nn.Module, TrainingRuntime]:
    if not runtime.compile_enabled:
        return model, runtime
    try:
        compiled = torch.compile(model, mode=runtime.compile_mode)
        return compiled, runtime
    except Exception as exc:
        runtime.compile_enabled = False
        runtime.compile_error = f"{type(exc).__name__}: {exc}"
        return model, runtime


def create_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_ratio: float = 0.05,
    min_lr_scale: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(1, int(total_steps))
    warmup = max(0, int(warmup_steps))
    if warmup <= 0 and float(warmup_ratio) > 0.0:
        warmup = min(total - 1 if total > 1 else 0, int(round(total * float(warmup_ratio))))
    floor = max(0.0, min(1.0, float(min_lr_scale)))

    def _lr_lambda(step_index: int) -> float:
        step = max(0, int(step_index))
        if warmup > 0 and step < warmup:
            return max(1.0 / max(1, warmup), float(step + 1) / float(max(1, warmup)))
        if total <= warmup:
            return 1.0
        progress = float(step - warmup) / float(max(1, total - warmup))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = max(0.0, min(0.999999, float(decay)))
        self.shadow = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {key: value.detach().cpu().clone() for key, value in state_dict.items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        for key, value in current.items():
            target = value.detach().cpu()
            shadow_value = self.shadow.get(key)
            if shadow_value is None:
                self.shadow[key] = target.clone()
                continue
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(target, alpha=1.0 - self.decay)
            else:
                self.shadow[key] = target.clone()

    @contextmanager
    def apply_to(self, model: torch.nn.Module) -> Iterator[None]:
        original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(original, strict=True)
