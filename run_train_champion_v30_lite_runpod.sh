#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${LITE_OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326}"
TRAIN_DATA="${LITE_TRAIN_DATA:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v30_anchor_refresh_20260326/prepared_train_pairs.jsonl}"
EVAL_DATA="${LITE_EVAL_DATA:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v30_anchor_refresh_20260326/prepared_eval_pairs.jsonl}"
BASE_WEIGHTS="${LITE_BASE_WEIGHTS:-${REPO_ROOT}/source/champion_model_chat_v28_expert.pth}"
BASE_META="${LITE_BASE_META:-${REPO_ROOT}/source/chat_model_meta_v28_expert.json}"
MODEL_OUT="${MODEL_OUT:-${OUTPUT_DIR}/champion_model_chat_v30_lite_student.pth}"
META_OUT="${META_OUT:-${OUTPUT_DIR}/chat_model_meta_v30_lite_student.json}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train_champion_v30_lite_student.out.log}"
BENCH_LOG="${BENCH_LOG:-${LOG_DIR}/benchmark_champion_v30_lite_student.out.log}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  echo "[lite] missing train data: ${TRAIN_DATA}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_DATA}" ]]; then
  echo "[lite] missing eval data: ${EVAL_DATA}" >&2
  exit 1
fi
if [[ ! -f "${BASE_WEIGHTS}" ]]; then
  echo "[lite] missing base weights: ${BASE_WEIGHTS}" >&2
  exit 1
fi
if [[ ! -f "${BASE_META}" ]]; then
  echo "[lite] missing base metadata: ${BASE_META}" >&2
  exit 1
fi

TRAIN_CMD=(
  python3
  source/finetune_chat.py
  --data "${TRAIN_DATA}"
  --weights "${BASE_WEIGHTS}"
  --output "${MODEL_OUT}"
  --meta "${META_OUT}"
  --model_size ultra_expert
  --feature_mode context_v2
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --epochs 6
  --batch_size 128
  --grad_accum_steps 1
  --lr 8e-5
  --weight_decay 0.01
  --label_smoothing 0.04
  --val_split 0.12
  --split_mode stratified
  --seed 48
  --balanced_sampler
  --pref_weight 0.18
  --pref_beta 2.2
  --hard_negative_ratio 0.65
  --pref_objective sigmoid
  --pref_group_size 4
  --pref_group_estimator epo
  --adaptive_pref_weighting
  --pref_warmup_epochs 1.0
  --lr_schedule cosine
  --warmup_steps 20
  --early_stop_patience 2
  --max_candidates_per_bucket 160
  --ema_decay 0.999
  --log_interval_steps 2
  --epoch_checkpoint_dir "${OUTPUT_DIR}/checkpoints"
)

printf '[lite] train command:\n %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"

BENCH_CMD=(
  python3
  source/benchmark.py
  --weights_a "${BASE_WEIGHTS}"
  --meta_a "${BASE_META}"
  --weights_b "${MODEL_OUT}"
  --meta_b "${META_OUT}"
  --data "${EVAL_DATA}"
  --device cuda
)

printf '[lite] benchmark command:\n %q' "${BENCH_CMD[@]}"
printf '\n'
"${BENCH_CMD[@]}" 2>&1 | tee "${BENCH_LOG}"
