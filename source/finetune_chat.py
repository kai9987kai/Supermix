"""Fine-tuning script for ChampionNet retrieval-style chat models.

This module implements the complete fine-tuning pipeline for Supermix_27,
including data loading, model construction, training with preference
optimization (SimPO/RePO), EMA weight averaging, cosine LR scheduling,
and checkpoint management.

Usage:
    python finetune_chat.py --data conversation_data.jsonl \
        --weights champion_model.pth --model_size smarter_expert \
        --epochs 8 --batch_size 32

Supported model sizes:
    base, large, xlarge, xxlarge, xxxlarge, ultralarge, megalarge,
    ultra_expert, hierarchical_expert, deep_expert, expert_choice,
    smarter_expert
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from chat_pipeline import (
    MODEL_CLASSES,
    assign_labels,
    build_bucket_metadata,
    build_training_tensors,
    load_conversation_examples,
    summarize_label_stats,
)
from model_variants import (
    build_model,
    detect_large_head_expansion_dim,
    detect_model_size_from_state_dict,
    detect_xlarge_aux_expansion_dim,
    detect_xxlarge_third_expansion_dim,
    detect_xxxlarge_fourth_expansion_dim,
    detect_ultralarge_fifth_expansion_dim,
    detect_megalarge_sixth_expansion_dim,
    load_weights_for_model,
)
from device_utils import configure_torch_runtime, resolve_device
from run import safe_load_state_dict


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all backends.

    Args:
        seed: Integer seed value for random, torch, and CUDA RNGs.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_indices(n: int, val_split: float, seed: int, labels: torch.Tensor, split_mode: str):
    """Split dataset indices into train and validation sets.

    Supports stratified splitting (preserving class distribution) and
    random splitting. Falls back to random if stratification is not
    possible (e.g., single-element classes).

    Args:
        n: Total number of samples.
        val_split: Fraction of data to use for validation (0.0–1.0).
        seed: Random seed for reproducible splits.
        labels: Tensor of class labels for stratification.
        split_mode: Either 'stratified' or 'random'.

    Returns:
        Tuple of (train_indices, val_indices) as lists of ints.
    """
    if split_mode == "stratified" and labels.numel() == n and n > 1:
        rng = random.Random(seed)
        grouped: Dict[int, List[int]] = defaultdict(list)
        for i, label in enumerate(labels.tolist()):
            grouped[int(label)].append(i)

        train_idx: List[int] = []
        val_idx: List[int] = []
        for _, cls_idx in sorted(grouped.items()):
            rng.shuffle(cls_idx)
            cls_n = len(cls_idx)
            cls_val_n = int(round(cls_n * val_split))
            if val_split > 0 and cls_n > 1:
                cls_val_n = max(1, cls_val_n)
            cls_val_n = min(cls_val_n, max(0, cls_n - 1))
            val_idx.extend(cls_idx[:cls_val_n])
            train_idx.extend(cls_idx[cls_val_n:])

        if train_idx:
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            return train_idx, val_idx

    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    val_n = int(round(n * val_split))
    val_n = min(max(val_n, 1 if n > 10 else 0), max(0, n - 1))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if not train_idx:
        train_idx = idx
        val_idx = []
    return train_idx, val_idx


def _evaluate(model: nn.Module, loader: DataLoader, device: Any):
    """Evaluate model on a DataLoader, returning loss and accuracy.

    Args:
        model: The neural network model to evaluate.
        loader: DataLoader yielding (input, label) batches.
        device: Torch device to run evaluation on.

    Returns:
        Dict with 'loss' and 'acc' keys, or None if the loader is empty.
    """
    if len(loader) == 0:
        return None

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if xb.dtype != torch.float32:
                xb = xb.float()
            yb = yb.to(device)
            logits = model(xb).squeeze(1)  # (B,10)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item()) * yb.shape[0]
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])

    if total == 0:
        return None
    return {"loss": total_loss / total, "acc": correct / total}


def _make_train_loader(train_x: torch.Tensor, train_y: torch.Tensor, batch_size: int, balanced_sampler: bool):
    """Create a training DataLoader with optional class-balanced sampling.

    When balanced_sampler is True, uses inverse-frequency weighted sampling
    to mitigate class imbalance in the training set.

    Args:
        train_x: Input feature tensor of shape (N, 1, D).
        train_y: Label tensor of shape (N,).
        batch_size: Number of samples per batch.
        balanced_sampler: If True, use WeightedRandomSampler.

    Returns:
        A PyTorch DataLoader ready for training iteration.
    """
    dataset = TensorDataset(train_x, train_y)
    if not balanced_sampler:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class_counts = torch.bincount(train_y, minlength=MODEL_CLASSES).float().clamp_min(1.0)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_y].double()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=int(train_y.shape[0]),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _sample_negative_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hard_negative_ratio: float = 0.5,
    num_negatives: int = 1,
) -> torch.Tensor:
    """Sample negative class labels for preference loss computation.

    Generates a mix of hard negatives (highest-scoring wrong classes) and
    random negatives, controlled by hard_negative_ratio.

    Args:
        logits: Model output logits of shape (B, C).
        labels: Ground-truth labels of shape (B,).
        hard_negative_ratio: Fraction of negatives that should be hard
            (0.0 = all random, 1.0 = all hard).
        num_negatives: Number of negative classes to sample per example.

    Returns:
        Tensor of shape (B, num_negatives) with sampled negative class indices.
    """
    # Uniformly sample labels different from the ground-truth class.
    num_negatives = max(1, int(num_negatives))
    num_classes = logits.shape[-1]
    labels_2d = labels.unsqueeze(1)
    neg = torch.randint(
        0,
        num_classes - 1,
        size=(labels.shape[0], num_negatives),
        device=labels.device,
    )
    random_neg = neg + (neg >= labels_2d).long()

    if hard_negative_ratio <= 0:
        return random_neg

    masked = logits.detach().clone()
    masked.scatter_(1, labels_2d, float("-inf"))
    hard_k = max(1, min(num_negatives, num_classes - 1))
    hard_neg = torch.topk(masked, k=hard_k, dim=1).indices
    if hard_k < num_negatives:
        extra = torch.randint(
            0,
            num_classes - 1,
            size=(labels.shape[0], num_negatives - hard_k),
            device=labels.device,
        )
        extra = extra + (extra >= labels_2d).long()
        hard_neg = torch.cat([hard_neg, extra], dim=1)

    if hard_negative_ratio >= 1:
        return hard_neg

    selector = torch.rand((labels.shape[0], num_negatives), device=labels.device) < hard_negative_ratio
    return torch.where(selector, hard_neg, random_neg)


def _preference_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    hard_negative_ratio: float = 0.5,
    adaptive_weighting: bool = False,
    objective: str = "sigmoid",
    group_size: int = 1,
    group_estimator: str = "pairwise_mean",
    margin: float = 0.2,
) -> torch.Tensor:
    """Compute pairwise preference loss to push gold class above negatives.

    Implements SimPO-style sigmoid or RePO-style ReLU margin objectives.
    Supports group preference estimation (EPO) and confidence-weighted
    adaptive preference terms for robustness to noisy labels.

    Args:
        logits: Model output logits of shape (B, C).
        labels: Ground-truth labels of shape (B,).
        beta: Temperature scale for sigmoid objective.
        hard_negative_ratio: Fraction of hard negatives (0–1).
        adaptive_weighting: If True, weight terms by model confidence.
        objective: 'sigmoid' (SimPO) or 'repo_relu' (RePO margin).
        group_size: Number of negative classes per example.
        group_estimator: 'pairwise_mean' or 'epo' for group reduction.
        margin: Margin for RePO-style objective.

    Returns:
        Scalar loss tensor (mean over batch).
    """
    pos = logits.gather(1, labels.unsqueeze(1))
    neg_labels = _sample_negative_labels(
        logits,
        labels,
        hard_negative_ratio=hard_negative_ratio,
        num_negatives=group_size,
    )
    neg = logits.gather(1, neg_labels)

    if objective == "repo_relu":
        # RePO-style max-margin objective (beta-free by design).
        per_example = F.relu(float(margin) - (pos - neg)).mean(dim=1)
    elif objective == "sigmoid":
        scores = torch.sigmoid(beta * (pos - neg))
        if group_estimator == "epo":
            # EPO-style group preference estimate via expected pairwise win-probability.
            per_example = -torch.log(torch.clamp(scores.mean(dim=1), min=1e-8))
        else:
            per_example = -torch.log(torch.clamp(scores, min=1e-8)).mean(dim=1)
    else:
        raise ValueError(f"Unknown preference objective: {objective}")

    terms = per_example
    if adaptive_weighting:
        # Confidence-weighted preference terms approximate robust preference learning.
        with torch.no_grad():
            conf = torch.softmax(logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
            weights = torch.clamp(conf, min=0.20, max=1.00)
        terms = terms * weights
    return terms.mean()


def _set_trainable_params(model: nn.Module, train_all: bool):
    """Configure which model parameters are trainable.

    By default, only the classifier head (layers 10–11) is trained.
    When train_all is True, all parameters are unfrozen.

    Args:
        model: The model whose parameters will be configured.
        train_all: If True, train all parameters; otherwise head-only.
    """
    if train_all:
        for p in model.parameters():
            p.requires_grad = True
        return

    # Default: adapt only the classifier head + final norm for stability.
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("layers.10.") or name.startswith("layers.11."):
            p.requires_grad = True


def _count_trainable(model: nn.Module):
    """Count trainable and total parameters in a model.

    Args:
        model: The model to count parameters for.

    Returns:
        Tuple of (trainable_params, total_params).
    """
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _make_cosine_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Create a cosine annealing LR scheduler with linear warmup.

    The learning rate linearly increases from 0 to base_lr over warmup_steps,
    then decays following a cosine curve to 0 over the remaining steps.

    Args:
        optimizer: The optimizer to schedule.
        total_steps: Total number of training steps.
        warmup_steps: Number of linear warmup steps.

    Returns:
        A PyTorch LambdaLR scheduler.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _pref_warmup_scale(epoch_idx: int, batch_idx: int, batches_per_epoch: int, warmup_epochs: float) -> float:
    """Compute the preference loss warmup scale factor.

    Linearly ramps from 0.0 to 1.0 over warmup_epochs, allowing the
    model to stabilize on cross-entropy before preference terms kick in.

    Args:
        epoch_idx: Current epoch index (0-based).
        batch_idx: Current batch index within the epoch.
        batches_per_epoch: Total batches per epoch.
        warmup_epochs: Number of epochs over which to ramp.

    Returns:
        Scale factor in [0.0, 1.0].
    """
    if warmup_epochs <= 0:
        return 1.0
    progress_epochs = float(epoch_idx) + float(batch_idx + 1) / float(max(1, batches_per_epoch))
    return min(1.0, progress_epochs / max(1e-6, float(warmup_epochs)))


def _clone_state_dict_tensors(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Deep-clone all tensors in a state dict (for EMA snapshots)."""
    return {k: v.detach().clone() for k, v in state_dict.items()}


def _init_ema_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Initialize EMA shadow weights from the current model state."""
    return _clone_state_dict_tensors(model.state_dict())


def _format_seconds(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string (e.g., '3m42s')."""
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


@torch.no_grad()
def _update_ema_state(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    """Update EMA shadow weights with the current model's parameters.

    Applies exponential moving average: ema = decay * ema + (1-decay) * current.
    Integer buffers are copied directly.

    Args:
        ema_state: Dictionary of EMA shadow tensors.
        model: Current model whose weights to blend in.
        decay: EMA decay factor (typically 0.999).
    """
    current = model.state_dict()
    d = float(max(0.0, min(0.99999, decay)))
    for name, value in current.items():
        ema_value = ema_state[name]
        if ema_value.is_floating_point():
            ema_value.mul_(d).add_(value.detach(), alpha=(1.0 - d))
        else:
            ema_state[name] = value.detach().clone()


def _load_examples_from_manifest(manifest_path: str) -> Tuple[List, List[str]]:
    """Load training examples from a multi-shard manifest JSON.

    The manifest can be a dict with a 'shards' key or a list of shard
    entries. Each shard entry is either a string path or a dict with
    a 'path' key.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        Tuple of (all_examples, shard_paths).

    Raises:
        ValueError: If the manifest contains no valid shard paths.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    shard_paths: List[str] = []
    if isinstance(payload, dict):
        shards = payload.get("shards", [])
        if isinstance(shards, list):
            for item in shards:
                if isinstance(item, dict) and item.get("path"):
                    shard_paths.append(str(item["path"]))
                elif isinstance(item, str):
                    shard_paths.append(str(item))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("path"):
                shard_paths.append(str(item["path"]))
            elif isinstance(item, str):
                shard_paths.append(str(item))

    if not shard_paths:
        raise ValueError("Manifest must contain a non-empty `shards` list with `path` entries.")

    examples = []
    for path in shard_paths:
        shard_examples = load_conversation_examples(path)
        examples.extend(shard_examples)
        print(f"Loaded {len(shard_examples)} examples from shard: {path}")
    return examples, shard_paths


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Fine-tune ChampionNet for retrieval-style chat.")
    ap.add_argument("--data", required=True, help="Path to conversation JSONL.")
    ap.add_argument(
        "--data_manifest",
        default=None,
        help="Optional JSON manifest of dataset shards; overrides --data when provided.",
    )
    ap.add_argument("--weights", default="champion_model.pth", help="Base model checkpoint.")
    ap.add_argument("--output", default="champion_model_chat_ft.pth", help="Output fine-tuned checkpoint.")
    ap.add_argument("--meta", default="chat_model_meta.json", help="Output metadata JSON.")
    ap.add_argument(
        "--model_size",
        choices=["base", "large", "xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge", "ultra_expert", "hierarchical_expert", "deep_expert", "expert_choice", "smarter_expert", "thought_expert", "recursive_expert"],
        default="base",
        help="Model variant to train.",
    )
    ap.add_argument("--expansion_dim", type=int, default=None, help="Hidden size for large model head adapter.")
    ap.add_argument(
        "--extra_expansion_dim",
        type=int,
        default=None,
        help="Aux adapter hidden size for xlarge model head.",
    )
    ap.add_argument(
        "--third_expansion_dim",
        type=int,
        default=None,
        help="Third adapter hidden size for xxlarge model head.",
    )
    ap.add_argument(
        "--fourth_expansion_dim",
        type=int,
        default=None,
        help="Fourth adapter hidden size for xxxlarge model head.",
    )
    ap.add_argument(
        "--fifth_expansion_dim",
        type=int,
        default=None,
        help="Fifth adapter hidden size for ultralarge model head.",
    )
    ap.add_argument(
        "--sixth_expansion_dim",
        type=int,
        default=None,
        help="Sixth adapter hidden size for megalarge model head.",
    )
    ap.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout used in large model adapter branch.")
    ap.add_argument(
        "--feature_mode",
        choices=["legacy", "context_v2", "context_v3", "context_v4", "context_v5", "context_mix_v1", "context_mix_v2_mm"],
        default="context_v2",
        help="Input feature encoding mode for context understanding.",
    )
    ap.add_argument(
        "--feature_storage_dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Store precomputed features in this dtype to reduce RAM (batches are cast to float32 for training).",
    )
    ap.add_argument("--device", default="auto", help="Training device (auto/cpu/cuda/npu/xpu/dml/mps).")
    ap.add_argument(
        "--device_preference",
        default="cuda,npu,xpu,dml,mps,cpu",
        help="Priority order used when --device auto (supports cuda/npu/xpu/dml/mps/cpu).",
    )
    ap.add_argument(
        "--torch_num_threads",
        type=int,
        default=0,
        help="PyTorch intra-op CPU threads (0=auto/all cores).",
    )
    ap.add_argument(
        "--torch_interop_threads",
        type=int,
        default=0,
        help="PyTorch inter-op CPU threads (0=auto).",
    )
    ap.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="torch float32 matmul precision mode when supported.",
    )
    ap.add_argument("--disable_tf32", action="store_true", help="Disable TF32 on supported CUDA devices.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for larger effective batch size.",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument(
        "--split_mode",
        choices=["random", "stratified"],
        default="stratified",
        help="How to create train/validation split.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_all", action="store_true", help="Train the full network instead of only final layers.")
    ap.add_argument(
        "--balanced_sampler",
        action="store_true",
        help="Use inverse-frequency weighted sampling to mitigate class imbalance.",
    )
    ap.add_argument(
        "--pref_weight",
        type=float,
        default=0.15,
        help="Weight of pairwise preference loss (inspired by SimPO/ORPO style objectives).",
    )
    ap.add_argument(
        "--pref_beta",
        type=float,
        default=2.0,
        help="Scale factor for pairwise preference loss.",
    )
    ap.add_argument(
        "--hard_negative_ratio",
        type=float,
        default=0.6,
        help="Fraction of hard negatives used in preference loss (0=random, 1=hard only).",
    )
    ap.add_argument(
        "--pref_objective",
        choices=["sigmoid", "repo_relu"],
        default="sigmoid",
        help="Preference objective: sigmoid pairwise (SimPO-like) or RePO-style ReLU margin.",
    )
    ap.add_argument(
        "--pref_group_size",
        type=int,
        default=1,
        help="How many negative classes to sample per example for preference estimation.",
    )
    ap.add_argument(
        "--pref_group_estimator",
        choices=["pairwise_mean", "epo"],
        default="pairwise_mean",
        help="Group reduction for sigmoid objective. `epo` applies an expectation-style estimator.",
    )
    ap.add_argument(
        "--pref_margin",
        type=float,
        default=0.2,
        help="Margin used by RePO-style ReLU objective.",
    )
    ap.add_argument(
        "--adaptive_pref_weighting",
        action="store_true",
        help="Confidence-weight preference terms for noisy-data robustness.",
    )
    ap.add_argument(
        "--pref_warmup_epochs",
        type=float,
        default=1.0,
        help="Linearly ramp preference-loss weight from 0 to full value over this many epochs.",
    )
    ap.add_argument(
        "--lr_schedule",
        choices=["none", "cosine"],
        default="cosine",
        help="Learning-rate schedule.",
    )
    ap.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for cosine schedule. 0 auto-sets to 6%% of total steps.",
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Stop after this many epochs without val-loss improvement. 0 disables.",
    )
    ap.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum val-loss improvement to reset patience.",
    )
    ap.add_argument(
        "--max_candidates_per_bucket",
        type=int,
        default=64,
        help="Max stored responses per bucket for chat retrieval.",
    )
    ap.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for shadow weights; <=0 disables EMA.",
    )
    ap.add_argument(
        "--disable_ema_eval",
        action="store_true",
        help="Disable EMA-based evaluation/saving even when EMA is enabled.",
    )
    ap.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm. <=0 disables clipping.",
    )
    ap.add_argument(
        "--log_interval_steps",
        type=int,
        default=0,
        help="Print running train metrics every N optimizer steps within each epoch (0 disables).",
    )
    ap.add_argument(
        "--epoch_checkpoint_dir",
        default="",
        help="Optional directory to save per-epoch checkpoints (useful for long runs).",
    )
    ap.add_argument(
        "--aux_loss_weight",
        type=float,
        default=0.01,
        help="Weight of the auxiliary load-balancing loss for MoE models.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    resolved_data_sources: List[str]
    if args.data_manifest:
        examples, resolved_data_sources = _load_examples_from_manifest(args.data_manifest)
    else:
        examples = load_conversation_examples(args.data)
        resolved_data_sources = [str(args.data)]
    num_examples_loaded = len(examples)
    labels, label_mode = assign_labels(examples, seed=args.seed)
    x, y = build_training_tensors(examples, labels, feature_mode=args.feature_mode)
    if args.feature_storage_dtype == "float16":
        x = x.half()
    fast_meta_mode = int(args.max_candidates_per_bucket) <= 0
    if fast_meta_mode:
        # Huge runs do not need per-bucket response exemplars; free raw examples early.
        examples = []
    if y.min().item() < 0 or y.max().item() >= MODEL_CLASSES:
        raise ValueError("Labels must be within [0, 9].")

    train_idx, val_idx = _split_indices(
        n=x.shape[0],
        val_split=args.val_split,
        seed=args.seed,
        labels=y,
        split_mode=args.split_mode,
    )
    train_x = x[train_idx]
    train_y = y[train_idx]
    val_x = x[val_idx] if val_idx else x.new_zeros((0, 1, x.shape[-1]))
    val_y = y[val_idx] if val_idx else y.new_zeros((0,), dtype=torch.long)

    train_loader = _make_train_loader(
        train_x=train_x,
        train_y=train_y,
        batch_size=args.batch_size,
        balanced_sampler=args.balanced_sampler,
    )
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)

    state_dict = safe_load_state_dict(args.weights)
    ckpt_model_size = detect_model_size_from_state_dict(state_dict)

    resolved_expansion_dim = args.expansion_dim
    resolved_extra_expansion_dim = args.extra_expansion_dim
    resolved_third_expansion_dim = args.third_expansion_dim
    resolved_fourth_expansion_dim = args.fourth_expansion_dim
    resolved_fifth_expansion_dim = args.fifth_expansion_dim

    if args.model_size == "large":
        if resolved_expansion_dim is None:
            if ckpt_model_size in {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_expansion_dim = detect_large_head_expansion_dim(state_dict, default=512)
            else:
                resolved_expansion_dim = 512
    elif args.model_size == "xlarge":
        if resolved_expansion_dim is None:
            if ckpt_model_size in {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_expansion_dim = detect_large_head_expansion_dim(state_dict, default=768)
            else:
                resolved_expansion_dim = 768
        if resolved_extra_expansion_dim is None:
            if ckpt_model_size in {"xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_extra_expansion_dim = detect_xlarge_aux_expansion_dim(
                    state_dict,
                    default=max(1024, int(resolved_expansion_dim) * 2),
                )
            else:
                resolved_extra_expansion_dim = max(1024, int(resolved_expansion_dim) * 2)
    elif args.model_size == "xxlarge":
        if resolved_expansion_dim is None:
            if ckpt_model_size in {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_expansion_dim = detect_large_head_expansion_dim(state_dict, default=1024)
            else:
                resolved_expansion_dim = 1024
        if resolved_extra_expansion_dim is None:
            if ckpt_model_size in {"xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_extra_expansion_dim = detect_xlarge_aux_expansion_dim(
                    state_dict,
                    default=max(2048, int(resolved_expansion_dim) * 2),
                )
            else:
                resolved_extra_expansion_dim = max(2048, int(resolved_expansion_dim) * 2)
        if resolved_third_expansion_dim is None:
            if ckpt_model_size in {"xxlarge", "xxxlarge", "ultralarge"}:
                resolved_third_expansion_dim = detect_xxlarge_third_expansion_dim(
                    state_dict,
                    default=max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_third_expansion_dim = max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim))
    elif args.model_size == "xxxlarge":
        if resolved_expansion_dim is None:
            if ckpt_model_size in {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_expansion_dim = detect_large_head_expansion_dim(state_dict, default=1024)
            else:
                resolved_expansion_dim = 1024
        if resolved_extra_expansion_dim is None:
            if ckpt_model_size in {"xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_extra_expansion_dim = detect_xlarge_aux_expansion_dim(
                    state_dict,
                    default=max(2048, int(resolved_expansion_dim) * 2),
                )
            else:
                resolved_extra_expansion_dim = max(2048, int(resolved_expansion_dim) * 2)
        if resolved_third_expansion_dim is None:
            if ckpt_model_size in {"xxlarge", "xxxlarge", "ultralarge"}:
                resolved_third_expansion_dim = detect_xxlarge_third_expansion_dim(
                    state_dict,
                    default=max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_third_expansion_dim = max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim))
        if resolved_fourth_expansion_dim is None:
            if ckpt_model_size in {"xxxlarge", "ultralarge"}:
                resolved_fourth_expansion_dim = detect_xxxlarge_fourth_expansion_dim(
                    state_dict,
                    default=max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_fourth_expansion_dim = max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim))
    elif args.model_size == "ultralarge":
        if resolved_expansion_dim is None:
            if ckpt_model_size in {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_expansion_dim = detect_large_head_expansion_dim(state_dict, default=1024)
            else:
                resolved_expansion_dim = 1024
        if resolved_extra_expansion_dim is None:
            if ckpt_model_size in {"xlarge", "xxlarge", "xxxlarge", "ultralarge"}:
                resolved_extra_expansion_dim = detect_xlarge_aux_expansion_dim(
                    state_dict,
                    default=max(2048, int(resolved_expansion_dim) * 2),
                )
            else:
                resolved_extra_expansion_dim = max(2048, int(resolved_expansion_dim) * 2)
        if resolved_third_expansion_dim is None:
            if ckpt_model_size in {"xxlarge", "xxxlarge", "ultralarge"}:
                resolved_third_expansion_dim = detect_xxlarge_third_expansion_dim(
                    state_dict,
                    default=max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_third_expansion_dim = max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim))
        if resolved_fourth_expansion_dim is None:
            if ckpt_model_size in {"xxxlarge", "ultralarge"}:
                resolved_fourth_expansion_dim = detect_xxxlarge_fourth_expansion_dim(
                    state_dict,
                    default=max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_fourth_expansion_dim = max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim))
        if resolved_fifth_expansion_dim is None:
            if ckpt_model_size == "ultralarge":
                resolved_fifth_expansion_dim = detect_ultralarge_fifth_expansion_dim(
                    state_dict,
                    default=max(6144, int(resolved_fourth_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_fifth_expansion_dim = max(6144, int(resolved_fourth_expansion_dim) + int(resolved_expansion_dim))
    elif args.model_size == "megalarge":
        if resolved_expansion_dim is None:
            if ckpt_model_size in {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"}:
                resolved_expansion_dim = detect_large_head_expansion_dim(state_dict, default=1024)
            else:
                resolved_expansion_dim = 1024
        if resolved_extra_expansion_dim is None:
            if ckpt_model_size in {"xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"}:
                resolved_extra_expansion_dim = detect_xlarge_aux_expansion_dim(
                    state_dict,
                    default=max(2048, int(resolved_expansion_dim) * 2),
                )
            else:
                resolved_extra_expansion_dim = max(2048, int(resolved_expansion_dim) * 2)
        if resolved_third_expansion_dim is None:
            if ckpt_model_size in {"xxlarge", "xxxlarge", "ultralarge", "megalarge"}:
                resolved_third_expansion_dim = detect_xxlarge_third_expansion_dim(
                    state_dict,
                    default=max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_third_expansion_dim = max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim))
        if resolved_fourth_expansion_dim is None:
            if ckpt_model_size in {"xxxlarge", "ultralarge", "megalarge"}:
                resolved_fourth_expansion_dim = detect_xxxlarge_fourth_expansion_dim(
                    state_dict,
                    default=max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_fourth_expansion_dim = max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim))
        if resolved_fifth_expansion_dim is None:
            if ckpt_model_size in {"ultralarge", "megalarge"}:
                resolved_fifth_expansion_dim = detect_ultralarge_fifth_expansion_dim(
                    state_dict,
                    default=max(6144, int(resolved_fourth_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_fifth_expansion_dim = max(6144, int(resolved_fourth_expansion_dim) + int(resolved_expansion_dim))
        if resolved_sixth_expansion_dim is None:
            if ckpt_model_size == "megalarge":
                resolved_sixth_expansion_dim = detect_megalarge_sixth_expansion_dim(
                    state_dict,
                    default=max(8192, int(resolved_fifth_expansion_dim) + int(resolved_expansion_dim)),
                )
            else:
                resolved_sixth_expansion_dim = max(8192, int(resolved_fifth_expansion_dim) + int(resolved_expansion_dim))
    if resolved_expansion_dim is None:
        resolved_expansion_dim = 512
    if resolved_extra_expansion_dim is None:
        resolved_extra_expansion_dim = max(1024, int(resolved_expansion_dim) * 2)
    if resolved_third_expansion_dim is None:
        resolved_third_expansion_dim = max(3072, int(resolved_extra_expansion_dim) + int(resolved_expansion_dim))
    if resolved_fourth_expansion_dim is None:
        resolved_fourth_expansion_dim = max(4096, int(resolved_third_expansion_dim) + int(resolved_expansion_dim))
    if resolved_fifth_expansion_dim is None:
        resolved_fifth_expansion_dim = max(6144, int(resolved_fourth_expansion_dim) + int(resolved_expansion_dim))
    resolved_sixth_expansion_dim = args.sixth_expansion_dim
    if resolved_sixth_expansion_dim is None:
        resolved_sixth_expansion_dim = max(8192, int(resolved_fifth_expansion_dim) + int(resolved_expansion_dim))

    model = build_model(
        model_size=args.model_size,
        expansion_dim=resolved_expansion_dim,
        dropout=args.adapter_dropout,
        extra_expansion_dim=resolved_extra_expansion_dim,
        third_expansion_dim=resolved_third_expansion_dim,
        fourth_expansion_dim=resolved_fourth_expansion_dim,
        fifth_expansion_dim=resolved_fifth_expansion_dim,
        sixth_expansion_dim=resolved_sixth_expansion_dim,
    ).to(device)
    missing, unexpected = load_weights_for_model(model, state_dict, model_size=args.model_size)
    
    # Filter out expected missing/unexpected keys for the new SmarterClassifierHead
    missing = [k for k in missing if not k.startswith("layers.10.")]
    unexpected = [k for k in unexpected if not k.startswith("layers.10.")]
    
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing}, Unexpected={unexpected}")

    _set_trainable_params(model, train_all=args.train_all)
    trainable, total = _count_trainable(model)
    if trainable == 0:
        raise RuntimeError("No trainable parameters selected.")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    steps_per_epoch = max(1, math.ceil(max(1, len(train_loader)) / grad_accum_steps))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(round(0.06 * total_steps))
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = _make_cosine_warmup_scheduler(optim, total_steps=total_steps, warmup_steps=warmup_steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=max(0.0, min(0.3, float(args.label_smoothing))))
    use_ema = args.ema_decay > 0
    ema_eval = (not bool(args.disable_ema_eval)) and use_ema
    ema_state = _init_ema_state(model) if use_ema else None

    print(f"Examples: {num_examples_loaded}")
    print(f"Device: {device_info.get('resolved', args.device)}")
    print(f"Device preference: {args.device_preference}")
    print(f"Torch threads: intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    print(f"Label mode: {label_mode}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Feature storage dtype: {args.feature_storage_dtype}")
    print(f"Label distribution: {summarize_label_stats(y)}")
    print(f"Train/val split: {len(train_idx)}/{len(val_idx)}")
    if len(val_idx) > 0:
        print(f"Validation labels: {summarize_label_stats(val_y)}")
    print(f"Model size: {args.model_size}")
    print(f"Expansion dim: {resolved_expansion_dim}")
    if args.model_size == "xlarge":
        print(f"Aux expansion dim: {resolved_extra_expansion_dim}")
    if args.model_size == "xxlarge":
        print(f"Aux expansion dim: {resolved_extra_expansion_dim}")
        print(f"Third expansion dim: {resolved_third_expansion_dim}")
    if args.model_size == "xxxlarge":
        print(f"Aux expansion dim: {resolved_extra_expansion_dim}")
        print(f"Third expansion dim: {resolved_third_expansion_dim}")
        print(f"Fourth expansion dim: {resolved_fourth_expansion_dim}")
    if args.model_size == "ultralarge":
        print(f"Aux expansion dim: {resolved_extra_expansion_dim}")
        print(f"Third expansion dim: {resolved_third_expansion_dim}")
        print(f"Fourth expansion dim: {resolved_fourth_expansion_dim}")
        print(f"Fifth expansion dim: {resolved_fifth_expansion_dim}")
    if args.model_size == "megalarge":
        print(f"Aux expansion dim: {resolved_extra_expansion_dim}")
        print(f"Third expansion dim: {resolved_third_expansion_dim}")
        print(f"Fourth expansion dim: {resolved_fourth_expansion_dim}")
        print(f"Fifth expansion dim: {resolved_fifth_expansion_dim}")
        print(f"Sixth expansion dim: {resolved_sixth_expansion_dim}")
    print(f"Checkpoint model size: {ckpt_model_size}")
    print(f"Grad accumulation steps: {grad_accum_steps}")
    print(f"Balanced sampler: {args.balanced_sampler}")
    print(f"Preference loss: weight={args.pref_weight} beta={args.pref_beta}")
    print(
        "Preference config: "
        f"objective={args.pref_objective} group_size={args.pref_group_size} "
        f"group_estimator={args.pref_group_estimator} margin={args.pref_margin}"
    )
    print(f"Hard negative ratio: {args.hard_negative_ratio}")
    print(f"Preference warmup epochs: {args.pref_warmup_epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Adaptive pref weighting: {args.adaptive_pref_weighting}")
    print(f"Split mode: {args.split_mode}")
    print(f"LR schedule: {args.lr_schedule}")
    if scheduler is not None:
        print(f"Warmup steps: {warmup_steps}/{total_steps}")
    print(f"EMA: enabled={use_ema} decay={float(args.ema_decay):.6f} eval={ema_eval}")
    print(f"Early stop: patience={args.early_stop_patience} min_delta={args.early_stop_min_delta}")
    print(f"Trainable params: {trainable}/{total}")
    if int(args.log_interval_steps) > 0:
        print(f"In-epoch logging: every {int(args.log_interval_steps)} optimizer steps")
    if str(args.epoch_checkpoint_dir).strip():
        print(f"Epoch checkpoints: {args.epoch_checkpoint_dir}")

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []
    global_optimizer_step = 0
    log_interval_steps = max(0, int(args.log_interval_steps))
    epoch_ckpt_dir = str(args.epoch_checkpoint_dir).strip()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_pref_loss = 0.0
        running_pref_weight = 0.0
        running_total = 0
        running_correct = 0
        batches_per_epoch = max(1, len(train_loader))
        epoch_optimizer_steps = 0
        optim.zero_grad(set_to_none=True)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            if xb.dtype != torch.float32:
                xb = xb.float()
            yb = yb.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                logits = model(xb).squeeze(1)  # (B,10)
                ce_loss = loss_fn(logits, yb)
                pref_loss = _preference_loss(
                    logits,
                    yb,
                    beta=args.pref_beta,
                    hard_negative_ratio=args.hard_negative_ratio,
                    adaptive_weighting=args.adaptive_pref_weighting,
                    objective=args.pref_objective,
                    group_size=args.pref_group_size,
                    group_estimator=args.pref_group_estimator,
                    margin=args.pref_margin,
                )
                pref_scale = _pref_warmup_scale(
                    epoch_idx=epoch - 1,
                    batch_idx=batch_idx,
                    batches_per_epoch=batches_per_epoch,
                    warmup_epochs=args.pref_warmup_epochs,
                )
                effective_pref_weight = args.pref_weight * pref_scale
                
                # Auxiliary load-balancing loss for Hierarchical MoE
                aux_loss = torch.tensor(0.0, device=device)
                head = model.layers[10]
                if hasattr(head, "_aux_loss"):
                    aux_loss = head._aux_loss
                
                loss = ce_loss + (effective_pref_weight * pref_loss) + (args.aux_loss_weight * aux_loss)
                scaled_loss = loss / float(grad_accum_steps)
                
            scaler.scale(scaled_loss).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                if float(args.grad_clip_norm) > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
                scaler.step(optim)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema_state is not None:
                    _update_ema_state(ema_state, model, decay=float(args.ema_decay))
                optim.zero_grad(set_to_none=True)
                epoch_optimizer_steps += 1
                global_optimizer_step += 1

            batch = yb.shape[0]
            running_loss += float(loss.item()) * batch
            running_ce_loss += float(ce_loss.item()) * batch
            running_pref_loss += float(pref_loss.item()) * batch
            running_pref_weight += float(effective_pref_weight) * batch
            pred = torch.argmax(logits, dim=-1)
            running_correct += int((pred == yb).sum().item())
            running_total += int(batch)

            if should_step and log_interval_steps > 0 and (
                (epoch_optimizer_steps % log_interval_steps == 0) or ((batch_idx + 1) == len(train_loader))
            ):
                elapsed = max(1e-6, time.time() - epoch_t0)
                progress = float(batch_idx + 1) / float(max(1, batches_per_epoch))
                eta = (elapsed / max(progress, 1e-6)) * max(0.0, 1.0 - progress)
                avg_loss_so_far = running_loss / max(1, running_total)
                avg_ce_so_far = running_ce_loss / max(1, running_total)
                avg_pref_so_far = running_pref_loss / max(1, running_total)
                avg_acc_so_far = running_correct / max(1, running_total)
                avg_pref_w_so_far = running_pref_weight / max(1, running_total)
                print(
                    f"[epoch {epoch:02d}] step {epoch_optimizer_steps}/{steps_per_epoch} "
                    f"(global {global_optimizer_step}/{total_steps}) "
                    f"batch {batch_idx + 1}/{batches_per_epoch} "
                    f"loss={avg_loss_so_far:.4f} ce={avg_ce_so_far:.4f} pref={avg_pref_so_far:.4f} "
                    f"aux={float(aux_loss.item()):.4f} "
                    f"pref_w={avg_pref_w_so_far:.4f} acc={avg_acc_so_far:.4f} "
                    f"lr={float(optim.param_groups[0]['lr']):.6g} "
                    f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
                )

        train_loss = running_loss / max(1, running_total)
        train_ce_loss = running_ce_loss / max(1, running_total)
        train_pref_loss = running_pref_loss / max(1, running_total)
        train_pref_weight = running_pref_weight / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        lr_now = float(optim.param_groups[0]["lr"])

        backup_state = None
        if ema_eval and ema_state is not None:
            backup_state = _clone_state_dict_tensors(model.state_dict())
            model.load_state_dict(ema_state, strict=True)

        eval_stats = _evaluate(model, val_loader, device)
        hist_row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_ce_loss": float(train_ce_loss),
            "train_pref_loss": float(train_pref_loss),
            "train_pref_weight": float(train_pref_weight),
            "train_acc": float(train_acc),
            "lr": float(lr_now),
        }
        if eval_stats is None:
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
                f"(ce={train_ce_loss:.4f}, pref={train_pref_loss:.4f}, pref_w={train_pref_weight:.4f}) "
                f"train_acc={train_acc:.4f} lr={lr_now:.6g}"
            )
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            history.append(hist_row)
        else:
            hist_row["val_loss"] = float(eval_stats["loss"])
            hist_row["val_acc"] = float(eval_stats["acc"])
            print(
                "Epoch "
                f"{epoch:02d} | train_loss={train_loss:.4f} "
                f"(ce={train_ce_loss:.4f}, pref={train_pref_loss:.4f}, pref_w={train_pref_weight:.4f}) "
                f"train_acc={train_acc:.4f} val_loss={eval_stats['loss']:.4f} val_acc={eval_stats['acc']:.4f} "
                f"lr={lr_now:.6g}"
            )
            history.append(hist_row)
            improved = eval_stats["loss"] < (best_val_loss - float(args.early_stop_min_delta))
            if improved:
                best_val_loss = eval_stats["loss"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                    print(
                        f"Early stopping at epoch {epoch:02d} "
                        f"(best epoch {best_epoch:02d}, best val_loss={best_val_loss:.4f})."
                    )
                    if backup_state is not None:
                        model.load_state_dict(backup_state, strict=True)
                    break

        if backup_state is not None:
            model.load_state_dict(backup_state, strict=True)

        if epoch_ckpt_dir:
            ckpt_dir = Path(epoch_ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_prefix = Path(args.output).stem
            epoch_weights_path = ckpt_dir / f"{ckpt_prefix}.epoch{epoch:02d}.pth"
            epoch_meta_path = ckpt_dir / f"{ckpt_prefix}.epoch{epoch:02d}.json"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, epoch_weights_path)
            epoch_meta_payload = {
                "epoch": int(epoch),
                "output_base": str(args.output),
                "weights_path": str(epoch_weights_path),
                "best_epoch_so_far": int(best_epoch),
                "best_val_loss_so_far": None if best_val_loss == float("inf") else float(best_val_loss),
                "history_tail": history[-1] if history else None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(epoch_meta_path, "w", encoding="utf-8") as f:
                json.dump(epoch_meta_payload, f, indent=2)
            print(f"Saved epoch checkpoint: {epoch_weights_path}")

    if best_state is None:
        if ema_state is not None:
            best_state = {k: v.detach().cpu() for k, v in ema_state.items()}
        else:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, output_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_weights": str(Path(args.weights).name),
        "fine_tuned_weights": str(output_path.name),
        "training_data": str(Path(args.data)),
        "training_data_sources": resolved_data_sources,
        "training_data_manifest": str(args.data_manifest) if args.data_manifest else None,
        "feature_mode": args.feature_mode,
        "feature_storage_dtype": args.feature_storage_dtype,
        "device": device_info.get("resolved", str(args.device)),
        "device_preference": args.device_preference,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "matmul_precision": args.matmul_precision,
        "tf32_enabled": not bool(args.disable_tf32),
        "model_size": args.model_size,
        "expansion_dim": int(resolved_expansion_dim),
        "extra_expansion_dim": int(resolved_extra_expansion_dim) if args.model_size in {"xlarge", "xxlarge", "xxxlarge", "ultralarge"} else None,
        "third_expansion_dim": int(resolved_third_expansion_dim) if args.model_size in {"xxlarge", "xxxlarge", "ultralarge"} else None,
        "fourth_expansion_dim": int(resolved_fourth_expansion_dim) if args.model_size in {"xxxlarge", "ultralarge"} else None,
        "fifth_expansion_dim": int(resolved_fifth_expansion_dim) if args.model_size == "ultralarge" else None,
        "adapter_dropout": float(args.adapter_dropout),
        "grad_accum_steps": int(grad_accum_steps),
        "balanced_sampler": bool(args.balanced_sampler),
        "pref_weight": float(args.pref_weight),
        "pref_beta": float(args.pref_beta),
        "hard_negative_ratio": float(args.hard_negative_ratio),
        "pref_objective": args.pref_objective,
        "pref_group_size": int(args.pref_group_size),
        "pref_group_estimator": args.pref_group_estimator,
        "pref_margin": float(args.pref_margin),
        "label_smoothing": float(args.label_smoothing),
        "adaptive_pref_weighting": bool(args.adaptive_pref_weighting),
        "pref_warmup_epochs": float(args.pref_warmup_epochs),
        "split_mode": args.split_mode,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "ema_decay": float(args.ema_decay),
        "ema_eval": bool(ema_eval),
        "grad_clip_norm": float(args.grad_clip_norm),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "best_epoch": int(best_epoch),
        "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
        "label_mode": label_mode,
        "num_examples": int(num_examples_loaded),
        "num_classes": MODEL_CLASSES,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "history": history,
    }
    metadata.update(
        build_bucket_metadata(
            examples if not fast_meta_mode else [],
            labels,
            max_candidates_per_bucket=args.max_candidates_per_bucket,
            feature_mode=args.feature_mode,
        )
    )
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved fine-tuned weights: {output_path}")
    print(f"Saved chat metadata: {meta_path}")


if __name__ == "__main__":
    main()
