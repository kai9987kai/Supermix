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
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PERSIST_ROOT}/base_models/qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775}"
BASE_MODEL="${BASE_MODEL:-${BASE_MODEL_DIR}}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${ANCHOR_OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v30_anchor_refresh_20260326}"
BASELINE_ADAPTER="${BASELINE_ADAPTER:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget/adapter}"
OUT_LOG="${ANCHOR_OUT_LOG:-${LOG_DIR}/train_$(basename "${OUTPUT_DIR}").out.log}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED="1"
export PYTHONHASHSEED="48"
export TOKENIZERS_PARALLELISM="false"
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "[anchor-refresh] base model directory not found: ${BASE_MODEL}" >&2
  exit 1
fi

if [[ ! -d "${BASELINE_ADAPTER}" ]]; then
  echo "[anchor-refresh] baseline adapter not found: ${BASELINE_ADAPTER}" >&2
  exit 1
fi

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

DATA_FILE="datasets/conversation_data.delta_anchor_mix_2026_03_26.jsonl"
if [[ ! -f "${REPO_ROOT}/${DATA_FILE}" ]]; then
  echo "[anchor-refresh] dataset file not found: ${REPO_ROOT}/${DATA_FILE}" >&2
  exit 1
fi

TRAIN_CMD=(
  "${PYTHON_BIN}"
  -u
  source/qwen_supermix_pipeline.py
  --data "${DATA_FILE}"
  --base_model "${BASE_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --max_records 600
  --max_source_fraction 1.0
  --max_synthetic_fraction 0.0
  --max_prompt_signature_count 24
  --data_log_every_records 100
  --prompt_signature_cap_exempt_sources "conversation_data.delta_anchor_mix_2026_03_26.jsonl"
  --eval_size 48
  --eval_min_quality_score 0.0
  --max_length 512
  --batch_size 1
  --grad_accum_steps 8
  --epochs 1
  --max_steps 48
  --lr 1.5e-6
  --sft_lr_schedule cosine
  --sft_warmup_steps 6
  --sft_min_lr_ratio 0.4
  --sft_max_grad_norm 0.7
  --train_log_every_steps 1
  --save_every_steps 16
  --weight_decay 0.01
  --lora_r 32
  --lora_alpha 64
  --lora_dropout 0.03
  --use_rslora
  --use_dora
  --lora_init pissa_niter_4
  --lora_plus_ratio 16
  --neftune_noise_alpha 0.0
  --sft_weight_mode quality
  --sft_min_weight 1.0
  --sft_max_weight 1.15
  --sft_coding_boost 1.08
  --sft_events_boost 1.12
  --sft_reasoning_boost 1.08
  --sft_prompt_skill_boost 1.05
  --sft_knowledge_density_boost 1.10
  --sft_followup_paraphrase_aug 0
  --sft_rdrop_alpha 0.0
  --sft_min_quality_score 0.0
  --sft_quality_filter_exempt_sources "conversation_data.delta_anchor_mix_2026_03_26.jsonl"
  --sft_eval_every_steps 16
  --sft_early_stop_patience 2
  --preference_objective none
  --preference_steps 0
  --preference_pairs 0
  --supermix_distill_ratio 0.0
  --supermix_distill_max 0
  --seed 48
  --device "${DEVICE}"
  --device_preference cuda,npu,xpu,mps,cpu,dml
  --model_dtype auto
  --gradient_checkpointing
  --torch_num_threads "${LOGICAL_CPU}"
  --torch_interop_threads "${INTEROP_CPU}"
  --benchmark_eval_limit 48
)

LATEST_PTR="${OUTPUT_DIR}/latest_adapter_checkpoint.txt"
if [[ -f "${LATEST_PTR}" ]]; then
  echo "[anchor-refresh] resuming existing anchor refresh run from ${OUTPUT_DIR}"
  TRAIN_CMD+=(--resume_from_latest_checkpoint)
else
  echo "[anchor-refresh] warm-starting from baseline adapter ${BASELINE_ADAPTER}"
  TRAIN_CMD+=(--init_adapter_dir "${BASELINE_ADAPTER}" --init_adapter_match_lora)
fi

printf '[anchor-refresh] command preview:\n %q' "${TRAIN_CMD[@]}"
printf '\n'
printf '[anchor-refresh] streaming to %s\n' "${OUT_LOG}"

"${TRAIN_CMD[@]}" 2>&1 | tee "${OUT_LOG}"
