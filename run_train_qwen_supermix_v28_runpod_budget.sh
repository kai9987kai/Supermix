#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
ENV_PATH_DEFAULT="${PERSIST_ROOT}/runpod_budget.env"
ENV_PATH="${ENV_PATH:-${ENV_PATH_DEFAULT}}"

if [[ -f "${ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_PATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_HOME="${HF_HOME:-${PERSIST_ROOT}/hf_cache}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget}"
RESUME_WARM_START_DIR="${RESUME_WARM_START_DIR:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v26_full}"
STATE_PATH="${STATE_PATH:-${PERSIST_ROOT}/last_training_launch_runpod_budget.json}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PERSIST_ROOT}/base_models/qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775}"
BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_DIR}}"
DEVICE="${DEVICE:-cuda}"
TRAIN_PROFILE="${TRAIN_PROFILE:-cloud_plus}"
ALLOW_BUNDLED_WARM_START_FALLBACK="${ALLOW_BUNDLED_WARM_START_FALLBACK:-0}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-40}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-16}"
OUT_LOG="${OUT_LOG:-${LOG_DIR}/train_$(basename "${OUTPUT_DIR}").out.log}"
REPO_WARM_START_DIR="${REPO_WARM_START_DIR:-${REPO_ROOT}/artifacts/qwen_supermix_enhanced_v26_full}"
REPO_BUNDLED_WARM_START_DIR="${REPO_BUNDLED_WARM_START_DIR:-${REPO_ROOT}/dist/SupermixQwenDesktopV26/_internal/bundled_latest_artifact}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "$(dirname "${STATE_PATH}")" "$(dirname "${BASE_MODEL}")"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED="1"
export PYTHONHASHSEED="48"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM="false"
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_DESC="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
  echo "[runpod-train] GPU: ${GPU_DESC}"
  GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  if [[ -n "${GPU_MEMORY_MB}" ]]; then
    MIN_GPU_MEMORY_MB=$(( MIN_GPU_MEMORY_GB * 1024 - 256 ))
    if (( GPU_MEMORY_MB < MIN_GPU_MEMORY_MB )); then
      echo "[runpod-train] warning: detected VRAM is below the recommended ${MIN_GPU_MEMORY_GB} GB floor for this recipe" >&2
    fi
  fi
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "[runpod-train] base model directory not found: ${BASE_MODEL}" >&2
  echo "[runpod-train] run bash source/setup_runpod_budget_workspace.sh first" >&2
  exit 1
fi

get_latest_adapter_checkpoint() {
  local run_dir="$1"
  local latest_file checkpoint_dir meta_path

  if [[ ! -d "${run_dir}" ]]; then
    return 1
  fi
  if [[ -f "${run_dir}/adapter/adapter_config.json" && -f "${run_dir}/adapter/adapter_model.safetensors" ]]; then
    printf '%s\n' "${run_dir}/adapter"
    return 0
  fi
  if [[ -f "${run_dir}/adapter_config.json" && -f "${run_dir}/adapter_model.safetensors" ]]; then
    printf '%s\n' "${run_dir}"
    return 0
  fi
  latest_file="${run_dir}/latest_adapter_checkpoint.txt"
  if [[ -f "${latest_file}" ]]; then
    checkpoint_dir="$(<"${latest_file}")"
    checkpoint_dir="${checkpoint_dir#"${checkpoint_dir%%[![:space:]]*}"}"
    checkpoint_dir="${checkpoint_dir%"${checkpoint_dir##*[![:space:]]}"}"
    if [[ -n "${checkpoint_dir}" ]]; then
      if [[ -d "${checkpoint_dir}" ]]; then
        printf '%s\n' "${checkpoint_dir}"
        return 0
      fi
      if [[ -d "${REPO_ROOT}/${checkpoint_dir}" ]]; then
        printf '%s\n' "${REPO_ROOT}/${checkpoint_dir}"
        return 0
      fi
      if [[ -d "${run_dir}/${checkpoint_dir}" ]]; then
        printf '%s\n' "${run_dir}/${checkpoint_dir}"
        return 0
      fi
    fi
  fi
  while IFS= read -r meta_path; do
    checkpoint_dir="$(dirname "${meta_path}")/adapter"
    if [[ -d "${checkpoint_dir}" ]]; then
      printf '%s\n' "${checkpoint_dir}"
      return 0
    fi
  done < <(find "${run_dir}/checkpoints" -type f -name checkpoint_meta.json -print 2>/dev/null | sort -r)
  return 1
}

seed_persistent_warm_start() {
  local persistent_checkpoint repo_checkpoint

  if persistent_checkpoint="$(get_latest_adapter_checkpoint "${RESUME_WARM_START_DIR}")"; then
    printf '%s\n' "${persistent_checkpoint}"
    return 0
  fi
  if repo_checkpoint="$(get_latest_adapter_checkpoint "${REPO_WARM_START_DIR}")"; then
    echo "[runpod-train] seeding persistent warm-start artifact from ${REPO_WARM_START_DIR}" >&2
    rm -rf "${RESUME_WARM_START_DIR}"
    cp -a "${REPO_WARM_START_DIR}" "${RESUME_WARM_START_DIR}"
    get_latest_adapter_checkpoint "${RESUME_WARM_START_DIR}"
    return 0
  fi
  return 1
}

bundled_fallback_checkpoint() {
  get_latest_adapter_checkpoint "${REPO_BUNDLED_WARM_START_DIR}"
}

LOGICAL_CPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
if [[ "${LOGICAL_CPU}" -lt 1 ]]; then
  LOGICAL_CPU=1
fi
INTEROP_CPU=$(( LOGICAL_CPU / 2 ))
if [[ "${INTEROP_CPU}" -lt 1 ]]; then
  INTEROP_CPU=1
fi
if [[ "${INTEROP_CPU}" -gt 4 ]]; then
  INTEROP_CPU=4
fi

DATA_FILES=(
  "datasets/conversation_data.quality_anchor_v2.jsonl"
  "datasets/conversation_data.coding_knowledge_2026_02_19.jsonl"
  "datasets/conversation_data.world_events_2026_02_19.jsonl"
  "datasets/conversation_data.supermix_plus_v27_500k.jsonl"
  "datasets/conversation_data.mega_reasoning_creative_v25_75582.jsonl"
  "datasets/conversation_data.mega_creative_250k_v2.jsonl"
)

EXTRA_ARGS=()
case "${TRAIN_PROFILE}" in
  strict_parity)
    EXTRA_ARGS+=(--strict_determinism --disable_tf32 --matmul_precision highest)
    ;;
  cloud_plus)
    EXTRA_ARGS+=(--eval_split_mode auto --sft_true_packing --sft_packing_max_samples_per_row 2)
    ;;
  *)
    echo "[runpod-train] unsupported TRAIN_PROFILE=${TRAIN_PROFILE}" >&2
    exit 1
    ;;
esac

if [[ -n "${SUPERMIX_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=( ${SUPERMIX_EXTRA_ARGS} )
fi

TRAIN_CMD=(
  "${PYTHON_BIN}"
  -u
  source/qwen_supermix_pipeline.py
  --data
  "${DATA_FILES[@]}"
  --base_model "${BASE_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --max_records 600000
  --max_source_fraction 0.52
  --max_synthetic_fraction 0.06
  --max_prompt_signature_count 4
  --data_log_every_records 2000
  --prompt_signature_cap_exempt_sources "conversation_data.quality_anchor_v2.jsonl,conversation_data.mega_reasoning_creative_v25_75582.jsonl"
  --eval_size 500
  --eval_min_quality_score 1.05
  --eval_drop_synthetic_prompts
  --max_length 448
  --batch_size 1
  --grad_accum_steps 16
  --epochs 6
  --max_steps 6200
  --lr 1.0e-5
  --sft_lr_schedule cosine_restarts
  --sft_lr_restart_period 620
  --sft_warmup_steps 30
  --sft_min_lr_ratio 0.22
  --sft_max_grad_norm 0.9
  --sft_focal_gamma 1.35
  --sft_eval_every_steps 240
  --sft_early_stop_patience 5
  --sft_curriculum_quality_ramp 0.22
  --sft_grad_noise_eta 0.01
  --train_log_every_steps 1
  --save_every_steps "${SAVE_EVERY_STEPS}"
  --weight_decay 0.02
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.03
  --use_rslora
  --use_dora
  --lora_init pissa_niter_4
  --lora_plus_ratio 16
  --neftune_noise_alpha 5.0
  --sft_weight_mode quality
  --sft_min_weight 0.62
  --sft_max_weight 1.88
  --sft_synthetic_prompt_weight 0.62
  --sft_teacher_source_weight 0.92
  --sft_quality_anchor_boost 1.14
  --sft_coding_boost 1.24
  --sft_events_boost 1.08
  --sft_reasoning_boost 1.28
  --sft_prompt_skill_boost 1.17
  --sft_conversation_boost 1.24
  --sft_creativity_boost 1.16
  --sft_knowledge_density_boost 1.22
  --sft_rdrop_alpha 0.05
  --sft_length_bucketed_batches
  --sft_length_bucket_window_mult 24
  --sft_followup_paraphrase_aug 1
  --sft_followup_paraphrase_weight 0.68
  --sft_min_quality_score 0.98
  --sft_quality_filter_exempt_sources "conversation_data.quality_anchor_v2.jsonl,conversation_data.world_events_2026_02_19.jsonl"
  --sft_drop_synthetic_prompts
  --sft_auto_balance_sources
  --sft_source_balance_strength 0.66
  --sft_source_balance_max_scale 1.95
  --preference_objective ipo
  --preference_steps 1500
  --preference_rescore_every 25
  --preference_pairs 34000
  --preference_candidate_count 8
  --preference_reject_similarity_min 0.16
  --preference_beta 1.9
  --preference_beta_end 3.6
  --preference_margin 0.00
  --preference_margin_end 0.00
  --preference_label_smoothing 0.03
  --preference_sft_weight 0.32
  --preference_length_weight 0.08
  --preference_hardness_gamma 1.15
  --preference_robust_alpha 0.30
  --preference_robust_eta 0.08
  --preference_robust_clip 2.5
  --preference_wpo_alpha 0.35
  --preference_wpo_clip 2.5
  --preference_reference_anchor_weight 0.04
  --preference_reference_anchor_batch_size 2
  --preference_short_reject_boost 0.75
  --preference_long_reject_boost 0.25
  --preference_min_chosen_quality 0.92
  --preference_min_chosen_words 8
  --preference_min_quality_gap 0.05
  --preference_allow_template_prompts
  --preference_max_pairs_per_user 2
  --preference_max_pairs_per_source 360
  --preference_mining_mode auto
  --preference_mining_progress_every 30
  --preference_mining_max_seconds 4500
  --preference_mining_max_attempt_factor 20
  --preference_coding_focus_boost 1.30
  --preference_reasoning_focus_boost 1.32
  --preference_counterfactual_rejects_per_prompt 4
  --preference_selection_strategy innovation_mix
  --preference_selection_keep_ratio 0.62
  --preference_selection_min_keep 1800
  --preference_selection_max_keep 2400
  --preference_selection_hardness_target 0.46
  --preference_selection_hardness_bandwidth 0.22
  --preference_length_bucketed_batches
  --preference_length_bucket_window_mult 24
  --preference_lr 1.4e-5
  --preference_lr_schedule cosine
  --preference_warmup_steps 18
  --preference_min_lr_ratio 0.30
  --preference_max_grad_norm 0.9
  --preference_max_new_tokens 112
  --preference_prompt_max_tokens 352
  --supermix_distill_ratio 0.14
  --supermix_distill_max 8000
  --supermix_distill_best_of 3
  --supermix_distill_log_every 40
  --supermix_distill_max_seconds 12000
  --supermix_distill_min_quality 0.93
  --supermix_distill_min_gain 0.18
  --supermix_distill_density_bias 0.20
  --seed 48
  --device "${DEVICE}"
  --device_preference cuda,npu,xpu,mps,cpu,dml
  --model_dtype auto
  --gradient_checkpointing
  --torch_num_threads "${LOGICAL_CPU}"
  --torch_interop_threads "${INTEROP_CPU}"
  --skip_benchmark
)

if get_latest_adapter_checkpoint "${OUTPUT_DIR}" >/dev/null; then
  echo "[runpod-train] resuming from latest checkpoint in ${OUTPUT_DIR}"
  TRAIN_CMD+=(--resume_from_latest_checkpoint)
else
  if WARM_START_CHECKPOINT="$(seed_persistent_warm_start)"; then
    echo "[runpod-train] warm-starting from ${WARM_START_CHECKPOINT}"
    TRAIN_CMD+=(--init_adapter_dir "${WARM_START_CHECKPOINT}" --init_adapter_match_lora)
  elif [[ "${ALLOW_BUNDLED_WARM_START_FALLBACK}" == "1" ]] && BUNDLED_FALLBACK="$(bundled_fallback_checkpoint)"; then
    echo "[runpod-train] using bundled warm-start fallback from ${BUNDLED_FALLBACK}"
    TRAIN_CMD+=(--init_adapter_dir "${BUNDLED_FALLBACK}" --init_adapter_match_lora)
  else
    echo "[runpod-train] exact v26_full warm-start adapter was not found." >&2
    echo "[runpod-train] run the setup script again or set ALLOW_BUNDLED_WARM_START_FALLBACK=1 for the packaged fallback." >&2
    exit 1
  fi
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  echo "[runpod-train] applying extra args: ${EXTRA_ARGS[*]}"
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

export SUPERMIX_STATE_PATH="${STATE_PATH}"
export SUPERMIX_STATE_WORKSPACE_DIR="${REPO_ROOT}"
export SUPERMIX_STATE_OUTPUT_DIR="${OUTPUT_DIR}"
export SUPERMIX_STATE_RESUME_WARM_START_DIR="${RESUME_WARM_START_DIR}"
export SUPERMIX_STATE_BASE_MODEL="${BASE_MODEL}"
export SUPERMIX_STATE_DEVICE="${DEVICE}"
export SUPERMIX_STATE_OUT_LOG="${OUT_LOG}"

"${PYTHON_BIN}" - "${TRAIN_CMD[@]}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

state = {
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "workspace_dir": os.environ["SUPERMIX_STATE_WORKSPACE_DIR"],
    "output_dir": os.environ["SUPERMIX_STATE_OUTPUT_DIR"],
    "resume_warm_start_dir": os.environ["SUPERMIX_STATE_RESUME_WARM_START_DIR"],
    "base_model": os.environ["SUPERMIX_STATE_BASE_MODEL"],
    "device": os.environ["SUPERMIX_STATE_DEVICE"],
    "out_log": os.environ["SUPERMIX_STATE_OUT_LOG"],
    "command": sys.argv[1:],
}
with open(os.environ["SUPERMIX_STATE_PATH"], "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2)
print(json.dumps(state, indent=2))
PY

echo "[runpod-train] command preview:"
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'
echo "[runpod-train] streaming to ${OUT_LOG}"

cd "${REPO_ROOT}"
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${OUT_LOG}"
