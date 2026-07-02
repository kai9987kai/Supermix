param(
  [string]$BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
  [string]$SourceRunDir = "artifacts\qwen_supermix_enhanced_v28_clean_eval_robust_ipo",
  [string]$OutputDir = "artifacts\qwen_supermix_enhanced_checkpoint_model_one",
  [string]$SourceCheckpointDir = "",
  [string]$Device = "auto",
  [int]$PreferenceSteps = 12,
  [int]$PreferencePairs = 96,
  [int]$BenchmarkEvalLimit = 32
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"

function Resolve-CheckpointMeta([string]$RunDir, [string]$CheckpointDir) {
  $adapterDir = $null
  if ($CheckpointDir) {
    $raw = $CheckpointDir
    if (-not [System.IO.Path]::IsPathRooted($raw)) {
      $raw = Join-Path $repoRoot $raw
    }
    if (-not (Test-Path $raw)) {
      throw "Checkpoint path not found: $CheckpointDir"
    }
    $resolved = (Resolve-Path $raw).Path
    if ((Split-Path $resolved -Leaf) -ieq "adapter") {
      $adapterDir = $resolved
    }
    else {
      $candidateAdapter = Join-Path $resolved "adapter"
      if (-not (Test-Path $candidateAdapter)) {
        throw "Checkpoint path does not contain an adapter directory: $resolved"
      }
      $adapterDir = (Resolve-Path $candidateAdapter).Path
    }
  }
  else {
    $latestFile = Join-Path $RunDir "latest_adapter_checkpoint.txt"
    if (Test-Path $latestFile) {
      $checkpointRaw = (Get-Content $latestFile -Raw).Trim()
      if ($checkpointRaw) {
        $candidatePaths = @($checkpointRaw)
        if (-not [System.IO.Path]::IsPathRooted($checkpointRaw)) {
          $candidatePaths += (Join-Path $repoRoot $checkpointRaw)
          $candidatePaths += (Join-Path $RunDir $checkpointRaw)
        }
        foreach ($candidate in $candidatePaths) {
          if (Test-Path $candidate) {
            $adapterDir = (Resolve-Path $candidate).Path
            break
          }
        }
      }
    }
    if (-not $adapterDir) {
      $latestMeta = Get-ChildItem -Path (Join-Path $RunDir "checkpoints") -Recurse -Filter checkpoint_meta.json -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
      if (-not $latestMeta) {
        throw "No checkpoint metadata found under $RunDir"
      }
      $adapterDir = Join-Path (Split-Path $latestMeta.FullName -Parent) "adapter"
      if (-not (Test-Path $adapterDir)) {
        throw "Latest checkpoint metadata did not have an adapter folder: $($latestMeta.FullName)"
      }
      $adapterDir = (Resolve-Path $adapterDir).Path
    }
  }

  $metaPath = Join-Path (Split-Path $adapterDir -Parent) "checkpoint_meta.json"
  if (-not (Test-Path $metaPath)) {
    throw "Checkpoint metadata missing for adapter: $adapterDir"
  }
  $meta = Get-Content $metaPath -Raw | ConvertFrom-Json
  if (-not $meta) {
    throw "Could not parse checkpoint metadata at $metaPath"
  }
  return [pscustomobject]@{
    AdapterDir = $adapterDir
    MetaPath = (Resolve-Path $metaPath).Path
    Stage = [string]$meta.stage
    SftSteps = if ($null -eq $meta.sft_steps) { 0 } else { [int]$meta.sft_steps }
    PreferenceSteps = if ($null -eq $meta.preference_steps) { 0 } else { [int]$meta.preference_steps }
    SftLossMean = if ($null -eq $meta.sft_loss_mean) { 0.0 } else { [double]$meta.sft_loss_mean }
    PreferenceLossMean = if ($null -eq $meta.preference_loss_mean) { 0.0 } else { [double]$meta.preference_loss_mean }
  }
}

function Copy-RunCacheArtifacts([string]$FromRunDir, [string]$ToRunDir) {
  $cacheFiles = @(
    "prepared_data_cache_meta.json",
    "prepared_train_pairs.jsonl",
    "prepared_eval_pairs.jsonl",
    "teacher_distill_pairs.jsonl"
  )
  foreach ($name in $cacheFiles) {
    $src = Join-Path $FromRunDir $name
    if (Test-Path $src) {
      Copy-Item -Force $src (Join-Path $ToRunDir $name)
    }
  }
}

$resolvedSourceRunDir = $SourceRunDir
if (-not [System.IO.Path]::IsPathRooted($resolvedSourceRunDir)) {
  $resolvedSourceRunDir = Join-Path $repoRoot $resolvedSourceRunDir
}
$resolvedSourceRunDir = (Resolve-Path $resolvedSourceRunDir).Path

$checkpoint = Resolve-CheckpointMeta -RunDir $resolvedSourceRunDir -CheckpointDir $SourceCheckpointDir
if ($checkpoint.Stage -and $checkpoint.Stage.ToLowerInvariant() -ne "sft") {
  throw "Checkpoint Model One expects an SFT checkpoint. Resolved stage was '$($checkpoint.Stage)'."
}

$resolvedOutputDir = $OutputDir
if (-not [System.IO.Path]::IsPathRooted($resolvedOutputDir)) {
  $resolvedOutputDir = Join-Path $repoRoot $resolvedOutputDir
}
if (Test-Path $resolvedOutputDir) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $backupDir = "${resolvedOutputDir}_backup_$stamp"
  Move-Item -Path $resolvedOutputDir -Destination $backupDir
  Write-Host "[checkpoint-model] moved existing output to $backupDir"
}
New-Item -ItemType Directory -Path $resolvedOutputDir -Force | Out-Null
Copy-RunCacheArtifacts -FromRunDir $resolvedSourceRunDir -ToRunDir $resolvedOutputDir

$logicalCpu = [Math]::Max(1, [Environment]::ProcessorCount)
$interopCpu = [Math]::Max(1, [Math]::Min(4, [Math]::Floor($logicalCpu / 2)))

$argsList = @(
  "source\qwen_supermix_pipeline.py",
  "--data", "datasets\conversation_data.quality_anchor_v2.jsonl", "datasets\conversation_data.coding_knowledge_2026_02_19.jsonl", "datasets\conversation_data.world_events_2026_02_19.jsonl", "datasets\conversation_data.supermix_plus_v27_500k.jsonl", "datasets\conversation_data.mega_reasoning_creative_v25_75582.jsonl", "datasets\conversation_data.mega_creative_250k_v2.jsonl",
  "--base_model", $BaseModel,
  "--output_dir", $resolvedOutputDir,
  "--max_records", "600000",
  "--max_source_fraction", "0.52",
  "--max_synthetic_fraction", "0.06",
  "--max_prompt_signature_count", "4",
  "--data_log_every_records", "2000",
  "--prompt_signature_cap_exempt_sources", "conversation_data.quality_anchor_v2.jsonl,conversation_data.mega_reasoning_creative_v25_75582.jsonl",
  "--eval_size", "500",
  "--eval_min_quality_score", "1.05",
  "--eval_drop_synthetic_prompts",
  "--benchmark_eval_limit", "$BenchmarkEvalLimit",
  "--max_length", "448",
  "--batch_size", "1",
  "--grad_accum_steps", "16",
  "--epochs", "1",
  "--max_steps", "6200",
  "--lr", "1.0e-5",
  "--sft_lr_schedule", "cosine_restarts",
  "--sft_lr_restart_period", "620",
  "--sft_warmup_steps", "30",
  "--sft_min_lr_ratio", "0.22",
  "--sft_max_grad_norm", "0.9",
  "--train_log_every_steps", "1",
  "--save_every_steps", "4",
  "--skip_sft",
  "--weight_decay", "0.02",
  "--lora_r", "32",
  "--lora_alpha", "64",
  "--lora_dropout", "0.03",
  "--use_rslora",
  "--use_dora",
  "--lora_init", "pissa_niter_4",
  "--lora_plus_ratio", "16",
  "--neftune_noise_alpha", "5.0",
  "--sft_weight_mode", "quality",
  "--sft_min_weight", "0.62",
  "--sft_max_weight", "1.88",
  "--sft_synthetic_prompt_weight", "0.62",
  "--sft_teacher_source_weight", "0.92",
  "--sft_quality_anchor_boost", "1.14",
  "--sft_coding_boost", "1.24",
  "--sft_events_boost", "1.08",
  "--sft_reasoning_boost", "1.28",
  "--sft_prompt_skill_boost", "1.17",
  "--sft_conversation_boost", "1.24",
  "--sft_creativity_boost", "1.16",
  "--sft_rdrop_alpha", "0.05",
  "--sft_followup_paraphrase_aug", "1",
  "--sft_followup_paraphrase_weight", "0.68",
  "--sft_min_quality_score", "0.98",
  "--sft_quality_filter_exempt_sources", "conversation_data.quality_anchor_v2.jsonl,conversation_data.world_events_2026_02_19.jsonl",
  "--sft_drop_synthetic_prompts",
  "--sft_auto_balance_sources",
  "--sft_source_balance_strength", "0.66",
  "--sft_source_balance_max_scale", "1.95",
  "--preference_objective", "ipo",
  "--preference_steps", "$PreferenceSteps",
  "--preference_rescore_every", "4",
  "--preference_pairs", "$PreferencePairs",
  "--preference_candidate_count", "4",
  "--preference_reject_similarity_min", "0.16",
  "--preference_beta", "1.9",
  "--preference_beta_end", "2.4",
  "--preference_margin", "0.00",
  "--preference_margin_end", "0.00",
  "--preference_label_smoothing", "0.03",
  "--preference_sft_weight", "0.32",
  "--preference_length_weight", "0.08",
  "--preference_hardness_gamma", "1.10",
  "--preference_robust_alpha", "0.24",
  "--preference_robust_eta", "0.06",
  "--preference_robust_clip", "2.0",
  "--preference_wpo_alpha", "0.28",
  "--preference_wpo_clip", "2.2",
  "--preference_reference_anchor_weight", "0.03",
  "--preference_reference_anchor_batch_size", "2",
  "--preference_short_reject_boost", "0.75",
  "--preference_long_reject_boost", "0.25",
  "--preference_min_chosen_quality", "0.92",
  "--preference_min_chosen_words", "8",
  "--preference_min_quality_gap", "0.05",
  "--preference_allow_template_prompts",
  "--preference_max_pairs_per_user", "2",
  "--preference_max_pairs_per_source", "120",
  "--preference_mining_mode", "auto",
  "--preference_mining_progress_every", "12",
  "--preference_mining_max_seconds", "240",
  "--preference_mining_max_attempt_factor", "8",
  "--preference_coding_focus_boost", "1.24",
  "--preference_reasoning_focus_boost", "1.26",
  "--preference_counterfactual_rejects_per_prompt", "2",
  "--preference_selection_strategy", "innovation_mix",
  "--preference_selection_keep_ratio", "0.60",
  "--preference_selection_min_keep", "32",
  "--preference_selection_max_keep", "64",
  "--preference_selection_hardness_target", "0.44",
  "--preference_selection_hardness_bandwidth", "0.24",
  "--preference_lr", "1.4e-5",
  "--preference_lr_schedule", "cosine",
  "--preference_warmup_steps", "2",
  "--preference_min_lr_ratio", "0.35",
  "--preference_max_grad_norm", "0.9",
  "--preference_max_new_tokens", "80",
  "--preference_prompt_max_tokens", "320",
  "--supermix_distill_ratio", "0.14",
  "--supermix_distill_max", "8000",
  "--supermix_distill_best_of", "3",
  "--supermix_distill_log_every", "40",
  "--supermix_distill_max_seconds", "12000",
  "--supermix_distill_min_quality", "0.93",
  "--supermix_distill_min_gain", "0.18",
  "--seed", "48",
  "--device", $Device,
  "--device_preference", "dml,cuda,npu,xpu,mps,cpu",
  "--model_dtype", "auto",
  "--gradient_checkpointing",
  "--torch_num_threads", "$logicalCpu",
  "--torch_interop_threads", "$interopCpu",
  "--init_adapter_dir", $checkpoint.AdapterDir,
  "--init_adapter_match_lora",
  "--resume_sft_steps", "$($checkpoint.SftSteps)",
  "--resume_sft_loss_mean", "$($checkpoint.SftLossMean)"
)

Write-Host "[checkpoint-model] source checkpoint: $($checkpoint.AdapterDir)"
Write-Host "[checkpoint-model] source sft steps: $($checkpoint.SftSteps)"
Write-Host "[checkpoint-model] output dir: $resolvedOutputDir"

python -u @argsList
if ($LASTEXITCODE -ne 0) {
  throw "Checkpoint Model One pipeline failed with exit code $LASTEXITCODE"
}

$pointerPath = Join-Path $repoRoot ".gui_default_adapter.txt"
$adapterPath = Join-Path $resolvedOutputDir "adapter"
if (-not (Test-Path $adapterPath)) {
  throw "Checkpoint Model One completed without a final adapter at $adapterPath"
}
$repoUri = [System.Uri]((Resolve-Path $repoRoot).Path + [System.IO.Path]::DirectorySeparatorChar)
$adapterUri = [System.Uri]((Resolve-Path $adapterPath).Path)
$relativeAdapterPath = [System.Uri]::UnescapeDataString($repoUri.MakeRelativeUri($adapterUri).ToString()).Replace('/', '\')
[System.IO.File]::WriteAllText(
  $pointerPath,
  $relativeAdapterPath,
  [System.Text.UTF8Encoding]::new($false)
)
Write-Host "[checkpoint-model] GUI default adapter -> $relativeAdapterPath"
