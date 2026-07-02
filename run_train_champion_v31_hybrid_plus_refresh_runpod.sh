#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/champion_v31_hybrid_plus_refresh_20260326}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PERSIST_ROOT}/base_models/qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775}"
PROMPT_DATA="${PROMPT_DATA:-${REPO_ROOT}/datasets/conversation_data.delta_anchor_mix_2026_03_26.jsonl}"
NEW_DATA="${NEW_DATA:-${REPO_ROOT}/datasets/conversation_data.delta_official_refresh_2026_03_26.jsonl}"
TUNED_ADAPTER_DIR="${TUNED_ADAPTER_DIR:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget/adapter}"
LITE_WEIGHTS="${LITE_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326/champion_model_chat_v30_lite_student.pth}"
LITE_META="${LITE_META:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326/chat_model_meta_v30_lite_student.json}"
DISTILL_DATA="${DISTILL_DATA:-${OUTPUT_DIR}/hybrid_distill_train.jsonl}"
DISTILL_SUMMARY="${DISTILL_SUMMARY:-${OUTPUT_DIR}/hybrid_distill_summary.json}"
DISTILL_AUDIT="${DISTILL_AUDIT:-${OUTPUT_DIR}/hybrid_distill_audit.jsonl}"
STAGE1_OUT="${STAGE1_OUT:-${OUTPUT_DIR}/champion_model_chat_v31_hybrid_student_stage1.pth}"
STAGE1_META="${STAGE1_META:-${OUTPUT_DIR}/chat_model_meta_v31_hybrid_student_stage1.json}"
FINAL_OUT="${FINAL_OUT:-${OUTPUT_DIR}/champion_model_chat_v31_hybrid_plus_refresh.pth}"
FINAL_META="${FINAL_META:-${OUTPUT_DIR}/chat_model_meta_v31_hybrid_plus_refresh.json}"
BUILD_LOG="${BUILD_LOG:-${LOG_DIR}/build_v31_hybrid_distill.out.log}"
STAGE1_LOG="${STAGE1_LOG:-${LOG_DIR}/train_champion_v31_stage1.out.log}"
STAGE2_LOG="${STAGE2_LOG:-${LOG_DIR}/train_champion_v31_stage2_refresh.out.log}"
BENCH_STAGE1_LOG="${BENCH_STAGE1_LOG:-${LOG_DIR}/benchmark_champion_v31_stage1_vs_lite_refresh.out.log}"
BENCH_FINAL_LOG="${BENCH_FINAL_LOG:-${LOG_DIR}/benchmark_champion_v31_final_vs_lite_refresh.out.log}"
BENCH_REFRESH_LOG="${BENCH_REFRESH_LOG:-${LOG_DIR}/benchmark_champion_v31_final_vs_stage1_refresh.out.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_DIR}/v31_hybrid_training_summary.json}"
FINAL_ZIP="${FINAL_ZIP:-${PERSIST_ROOT}/champion_v31_hybrid_plus_refresh_final_model_20260326.zip}"
BUNDLE_ZIP="${BUNDLE_ZIP:-${PERSIST_ROOT}/champion_v31_hybrid_plus_refresh_bundle_20260326.zip}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

for required in \
  "${BASE_MODEL_DIR}" \
  "${PROMPT_DATA}" \
  "${NEW_DATA}" \
  "${TUNED_ADAPTER_DIR}" \
  "${LITE_WEIGHTS}" \
  "${LITE_META}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[v31] missing required path: ${required}" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"

BUILD_CMD=(
  python3
  source/build_v31_hybrid_distill_dataset.py
  --prompt_data "${PROMPT_DATA}"
  --output_data "${DISTILL_DATA}"
  --summary_out "${DISTILL_SUMMARY}"
  --audit_out "${DISTILL_AUDIT}"
  --base_model "${BASE_MODEL_DIR}"
  --adapter_dir "${TUNED_ADAPTER_DIR}"
  --lite_weights "${LITE_WEIGHTS}"
  --lite_meta "${LITE_META}"
  --sample_size 240
  --exclude_source_prefix official_refresh_
  --seed 48
  --device cuda
  --max_new_tokens 120
  --log_every 16
  --min_teacher_score 0.18
)

printf '[v31] build distill command:\n %q' "${BUILD_CMD[@]}"
printf '\n'
"${BUILD_CMD[@]}" 2>&1 | tee "${BUILD_LOG}"

STAGE1_CMD=(
  python3
  source/finetune_chat.py
  --data "${DISTILL_DATA}"
  --weights "${LITE_WEIGHTS}"
  --output "${STAGE1_OUT}"
  --meta "${STAGE1_META}"
  --model_size smarter_expert
  --feature_mode context_v2
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 8
  --batch_size 96
  --grad_accum_steps 1
  --lr 6e-5
  --weight_decay 0.015
  --label_smoothing 0.03
  --val_split 0.14
  --split_mode stratified
  --seed 48
  --balanced_sampler
  --pref_weight 0.14
  --pref_beta 2.0
  --hard_negative_ratio 0.60
  --pref_objective sigmoid
  --pref_group_size 4
  --pref_group_estimator epo
  --adaptive_pref_weighting
  --pref_warmup_epochs 1.0
  --lr_schedule cosine
  --warmup_steps 16
  --early_stop_patience 3
  --max_candidates_per_bucket 192
  --ema_decay 0.999
  --log_interval_steps 2
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage1_checkpoints"
)

printf '[v31] stage1 train command:\n %q' "${STAGE1_CMD[@]}"
printf '\n'
"${STAGE1_CMD[@]}" 2>&1 | tee "${STAGE1_LOG}"

BENCH_STAGE1_CMD=(
  python3
  source/benchmark.py
  --weights_a "${LITE_WEIGHTS}"
  --meta_a "${LITE_META}"
  --weights_b "${STAGE1_OUT}"
  --meta_b "${STAGE1_META}"
  --data "${NEW_DATA}"
  --device cuda
)

printf '[v31] stage1 benchmark command:\n %q' "${BENCH_STAGE1_CMD[@]}"
printf '\n'
"${BENCH_STAGE1_CMD[@]}" 2>&1 | tee "${BENCH_STAGE1_LOG}"

STAGE2_CMD=(
  python3
  source/finetune_chat.py
  --data "${NEW_DATA}"
  --weights "${STAGE1_OUT}"
  --output "${FINAL_OUT}"
  --meta "${FINAL_META}"
  --model_size smarter_expert
  --feature_mode context_v2
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 14
  --batch_size 32
  --grad_accum_steps 1
  --lr 2.5e-5
  --weight_decay 0.01
  --label_smoothing 0.02
  --val_split 0.22
  --split_mode stratified
  --seed 48
  --balanced_sampler
  --pref_weight 0.08
  --pref_beta 1.6
  --hard_negative_ratio 0.35
  --pref_objective sigmoid
  --pref_group_size 2
  --pref_group_estimator pairwise_mean
  --adaptive_pref_weighting
  --pref_warmup_epochs 0.6
  --lr_schedule cosine
  --warmup_steps 4
  --early_stop_patience 4
  --max_candidates_per_bucket 96
  --ema_decay 0.999
  --log_interval_steps 1
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage2_checkpoints"
)

printf '[v31] stage2 refresh command:\n %q' "${STAGE2_CMD[@]}"
printf '\n'
"${STAGE2_CMD[@]}" 2>&1 | tee "${STAGE2_LOG}"

BENCH_REFRESH_CMD=(
  python3
  source/benchmark.py
  --weights_a "${STAGE1_OUT}"
  --meta_a "${STAGE1_META}"
  --weights_b "${FINAL_OUT}"
  --meta_b "${FINAL_META}"
  --data "${NEW_DATA}"
  --device cuda
)

printf '[v31] refresh benchmark command:\n %q' "${BENCH_REFRESH_CMD[@]}"
printf '\n'
"${BENCH_REFRESH_CMD[@]}" 2>&1 | tee "${BENCH_REFRESH_LOG}"

BENCH_FINAL_CMD=(
  python3
  source/benchmark.py
  --weights_a "${LITE_WEIGHTS}"
  --meta_a "${LITE_META}"
  --weights_b "${FINAL_OUT}"
  --meta_b "${FINAL_META}"
  --data "${NEW_DATA}"
  --device cuda
)

printf '[v31] final benchmark command:\n %q' "${BENCH_FINAL_CMD[@]}"
printf '\n'
"${BENCH_FINAL_CMD[@]}" 2>&1 | tee "${BENCH_FINAL_LOG}"

python3 - \
  "${SUMMARY_JSON}" \
  "${DISTILL_SUMMARY}" \
  "${STAGE1_META}" \
  "${FINAL_META}" \
  "${LITE_WEIGHTS}" \
  "${STAGE1_OUT}" \
  "${FINAL_OUT}" \
  "${BENCH_STAGE1_LOG}" \
  "${BENCH_REFRESH_LOG}" \
  "${BENCH_FINAL_LOG}" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
distill_summary_path = Path(sys.argv[2])
stage1_meta_path = Path(sys.argv[3])
final_meta_path = Path(sys.argv[4])
lite_weights_path = Path(sys.argv[5])
stage1_weights_path = Path(sys.argv[6])
final_weights_path = Path(sys.argv[7])
bench_stage1_path = Path(sys.argv[8])
bench_refresh_path = Path(sys.argv[9])
bench_final_path = Path(sys.argv[10])

accuracy_re = re.compile(r"Accuracy:\s+([0-9.]+)\s+\((\d+)/(\d+)\)")

def parse_benchmark(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = accuracy_re.findall(text)
    rows = []
    for idx, match in enumerate(matches):
        acc, correct, total = match
        rows.append(
            {
                "slot": idx,
                "accuracy": float(acc),
                "correct": int(correct),
                "total": int(total),
            }
        )
    return rows

distill_summary = json.loads(distill_summary_path.read_text(encoding="utf-8"))
stage1_meta = json.loads(stage1_meta_path.read_text(encoding="utf-8"))
final_meta = json.loads(final_meta_path.read_text(encoding="utf-8"))

summary = {
    "created_at": final_meta.get("created_at"),
    "note": (
        "Official refresh data was used for the second-stage fine-tune, "
        "so benchmark_on_refresh is post-train confirmation rather than a strict holdout."
    ),
    "student_model_size": final_meta.get("model_size"),
    "feature_mode": final_meta.get("feature_mode"),
    "distill_dataset": distill_summary,
    "weights_bytes": {
        "lite_start": lite_weights_path.stat().st_size,
        "stage1": stage1_weights_path.stat().st_size,
        "final": final_weights_path.stat().st_size,
    },
    "stage1_meta": {
        "best_epoch": stage1_meta.get("best_epoch"),
        "best_val_loss": stage1_meta.get("best_val_loss"),
        "num_examples": stage1_meta.get("num_examples"),
        "trainable_parameters": stage1_meta.get("trainable_parameters"),
        "total_parameters": stage1_meta.get("total_parameters"),
    },
    "final_meta": {
        "best_epoch": final_meta.get("best_epoch"),
        "best_val_loss": final_meta.get("best_val_loss"),
        "num_examples": final_meta.get("num_examples"),
        "trainable_parameters": final_meta.get("trainable_parameters"),
        "total_parameters": final_meta.get("total_parameters"),
    },
    "benchmarks": {
        "lite_vs_stage1_on_refresh": parse_benchmark(bench_stage1_path),
        "stage1_vs_final_on_refresh": parse_benchmark(bench_refresh_path),
        "lite_vs_final_on_refresh": parse_benchmark(bench_final_path),
    },
}

summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

python3 - \
  "${FINAL_ZIP}" \
  "${BUNDLE_ZIP}" \
  "${FINAL_OUT}" \
  "${FINAL_META}" \
  "${STAGE1_OUT}" \
  "${STAGE1_META}" \
  "${DISTILL_DATA}" \
  "${DISTILL_SUMMARY}" \
  "${DISTILL_AUDIT}" \
  "${SUMMARY_JSON}" \
  "${BUILD_LOG}" \
  "${STAGE1_LOG}" \
  "${STAGE2_LOG}" \
  "${BENCH_STAGE1_LOG}" \
  "${BENCH_REFRESH_LOG}" \
  "${BENCH_FINAL_LOG}" \
  "${REPO_ROOT}/source/build_v31_hybrid_distill_dataset.py" \
  "${REPO_ROOT}/source/run_train_champion_v31_hybrid_plus_refresh_runpod.sh" <<'PY'
import sys
import zipfile
from pathlib import Path

final_zip = Path(sys.argv[1])
bundle_zip = Path(sys.argv[2])
files = [Path(arg) for arg in sys.argv[3:]]

for target in (final_zip, bundle_zip):
    if target.exists():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files[:2]:
        zf.write(path, arcname=path.name)

with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        zf.write(path, arcname=path.name)
PY

echo "[v31] final model zip: ${FINAL_ZIP}"
echo "[v31] bundle zip: ${BUNDLE_ZIP}"
echo "[v31] summary: ${SUMMARY_JSON}"
