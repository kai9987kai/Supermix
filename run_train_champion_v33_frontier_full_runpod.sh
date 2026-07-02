#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/champion_v33_frontier_full_20260326}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PERSIST_ROOT}/base_models/qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775}"

PROMPT_DATA_1="${PROMPT_DATA_1:-${REPO_ROOT}/datasets/conversation_data.delta_anchor_mix_2026_03_26.jsonl}"
PROMPT_DATA_2="${PROMPT_DATA_2:-${REPO_ROOT}/datasets/conversation_data.delta_official_refresh_2026_03_26.jsonl}"
PROMPT_DATA_3="${PROMPT_DATA_3:-${REPO_ROOT}/datasets/conversation_data.coding_knowledge_2026_02_19.jsonl}"
PROMPT_DATA_4="${PROMPT_DATA_4:-${REPO_ROOT}/datasets/conversation_data.world_events_2026_02_19.jsonl}"
PROMPT_DATA_5="${PROMPT_DATA_5:-${REPO_ROOT}/datasets/conversation_data.quality_anchor_v2.jsonl}"
PROMPT_DATA_6="${PROMPT_DATA_6:-${REPO_ROOT}/datasets/conversation_data.science_essentials_smoke.jsonl}"
PROMPT_DATA_7="${PROMPT_DATA_7:-${REPO_ROOT}/datasets/conversation_data.science_novel_examples_smoke.jsonl}"
PROMPT_DATA_8="${PROMPT_DATA_8:-${REPO_ROOT}/datasets/conversation_data.hybrid_v6_live_knowledge.jsonl}"
PROMPT_DATA_9="${PROMPT_DATA_9:-${REPO_ROOT}/datasets/conversation_data.mega_reasoning_creative_v25_75582.jsonl}"
PROMPT_DATA_10="${PROMPT_DATA_10:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget/prepared_train_pairs.jsonl}"
PROMPT_DATA_11="${PROMPT_DATA_11:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v29_delta_official_refresh_20260326/prepared_train_pairs.jsonl}"
PROMPT_DATA_12="${PROMPT_DATA_12:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v30_anchor_refresh_20260326/prepared_train_pairs.jsonl}"

EVAL_DATA_1="${EVAL_DATA_1:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget/prepared_eval_pairs.jsonl}"
EVAL_DATA_2="${EVAL_DATA_2:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v29_delta_official_refresh_20260326/prepared_eval_pairs.jsonl}"
EVAL_DATA_3="${EVAL_DATA_3:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v30_anchor_refresh_20260326/prepared_eval_pairs.jsonl}"

QWEN_V28_ADAPTER="${QWEN_V28_ADAPTER:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget/adapter}"

CHAMPION_V28_WEIGHTS="${CHAMPION_V28_WEIGHTS:-${REPO_ROOT}/source/champion_model_chat_v28_expert.pth}"
CHAMPION_V28_META="${CHAMPION_V28_META:-${REPO_ROOT}/source/chat_model_meta_v28_expert.json}"
LITE_WEIGHTS="${LITE_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326/champion_model_chat_v30_lite_student.pth}"
LITE_META="${LITE_META:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326/chat_model_meta_v30_lite_student.json}"
HYBRID_WEIGHTS="${HYBRID_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v31_hybrid_plus_refresh_20260326/champion_model_chat_v31_hybrid_plus_refresh.pth}"
HYBRID_META="${HYBRID_META:-${PERSIST_ROOT}/artifacts/champion_v31_hybrid_plus_refresh_20260326/chat_model_meta_v31_hybrid_plus_refresh.json}"
V32_WEIGHTS="${V32_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v32_omnifuse_20260326/champion_model_chat_v32_omnifuse_final.pth}"
V32_META="${V32_META:-${PERSIST_ROOT}/artifacts/champion_v32_omnifuse_20260326/chat_model_meta_v32_omnifuse_final.json}"

DISTILL_DATA="${DISTILL_DATA:-${OUTPUT_DIR}/v33_frontier_distill_train.jsonl}"
STAGE2_TRAIN_DATA="${STAGE2_TRAIN_DATA:-${OUTPUT_DIR}/v33_frontier_stage2_train_mix.jsonl}"
STAGE2_EVAL_DATA="${STAGE2_EVAL_DATA:-${OUTPUT_DIR}/v33_frontier_stage2_eval_mix.jsonl}"
STAGE3_POLISH_DATA="${STAGE3_POLISH_DATA:-${OUTPUT_DIR}/v33_frontier_stage3_polish.jsonl}"
DISTILL_SUMMARY="${DISTILL_SUMMARY:-${OUTPUT_DIR}/v33_frontier_dataset_summary.json}"
DISTILL_AUDIT="${DISTILL_AUDIT:-${OUTPUT_DIR}/v33_frontier_dataset_audit.jsonl}"

STAGE1_OUT="${STAGE1_OUT:-${OUTPUT_DIR}/champion_model_chat_v33_frontier_stage1.pth}"
STAGE1_META="${STAGE1_META:-${OUTPUT_DIR}/chat_model_meta_v33_frontier_stage1.json}"
STAGE2_OUT="${STAGE2_OUT:-${OUTPUT_DIR}/champion_model_chat_v33_frontier_stage2.pth}"
STAGE2_META="${STAGE2_META:-${OUTPUT_DIR}/chat_model_meta_v33_frontier_stage2.json}"
FINAL_OUT="${FINAL_OUT:-${OUTPUT_DIR}/champion_model_chat_v33_frontier_full_final.pth}"
FINAL_META="${FINAL_META:-${OUTPUT_DIR}/chat_model_meta_v33_frontier_full_final.json}"

BUILD_LOG="${BUILD_LOG:-${LOG_DIR}/build_v33_frontier.out.log}"
STAGE1_LOG="${STAGE1_LOG:-${LOG_DIR}/train_champion_v33_frontier_stage1.out.log}"
STAGE2_LOG="${STAGE2_LOG:-${LOG_DIR}/train_champion_v33_frontier_stage2.out.log}"
STAGE3_LOG="${STAGE3_LOG:-${LOG_DIR}/train_champion_v33_frontier_stage3.out.log}"
BENCH_STAGE1_LOG="${BENCH_STAGE1_LOG:-${LOG_DIR}/benchmark_champion_v33_stage1_vs_v32.out.log}"
BENCH_STAGE2_LOG="${BENCH_STAGE2_LOG:-${LOG_DIR}/benchmark_champion_v33_stage2_vs_stage1.out.log}"
BENCH_STAGE3_LOG="${BENCH_STAGE3_LOG:-${LOG_DIR}/benchmark_champion_v33_stage3_vs_stage2.out.log}"
BENCH_FINAL_LOG="${BENCH_FINAL_LOG:-${LOG_DIR}/benchmark_champion_v33_final_vs_v32.out.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_DIR}/v33_frontier_training_summary.json}"
FINAL_ZIP="${FINAL_ZIP:-${PERSIST_ROOT}/champion_v33_frontier_full_model_20260326.zip}"
BUNDLE_ZIP="${BUNDLE_ZIP:-${PERSIST_ROOT}/champion_v33_frontier_full_bundle_20260326.zip}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

for required in \
  "${BASE_MODEL_DIR}" \
  "${PROMPT_DATA_1}" \
  "${PROMPT_DATA_2}" \
  "${PROMPT_DATA_3}" \
  "${PROMPT_DATA_4}" \
  "${PROMPT_DATA_5}" \
  "${PROMPT_DATA_6}" \
  "${PROMPT_DATA_7}" \
  "${PROMPT_DATA_8}" \
  "${PROMPT_DATA_9}" \
  "${PROMPT_DATA_10}" \
  "${PROMPT_DATA_11}" \
  "${PROMPT_DATA_12}" \
  "${EVAL_DATA_1}" \
  "${EVAL_DATA_2}" \
  "${EVAL_DATA_3}" \
  "${QWEN_V28_ADAPTER}" \
  "${LITE_WEIGHTS}" \
  "${LITE_META}" \
  "${HYBRID_WEIGHTS}" \
  "${HYBRID_META}" \
  "${V32_WEIGHTS}" \
  "${V32_META}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[v33] missing required path: ${required}" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"

BUILD_CMD=(
  python3
  source/build_v33_frontier_dataset.py
  --prompt_data
  "${PROMPT_DATA_1}"
  "${PROMPT_DATA_2}"
  "${PROMPT_DATA_3}"
  "${PROMPT_DATA_4}"
  "${PROMPT_DATA_5}"
  "${PROMPT_DATA_6}"
  "${PROMPT_DATA_7}"
  "${PROMPT_DATA_8}"
  "${PROMPT_DATA_9}"
  "${PROMPT_DATA_10}"
  "${PROMPT_DATA_11}"
  "${PROMPT_DATA_12}"
  --eval_data
  "${EVAL_DATA_1}"
  "${EVAL_DATA_2}"
  "${EVAL_DATA_3}"
  --distill_output "${DISTILL_DATA}"
  --stage2_train_output "${STAGE2_TRAIN_DATA}"
  --stage2_eval_output "${STAGE2_EVAL_DATA}"
  --stage3_polish_output "${STAGE3_POLISH_DATA}"
  --summary_out "${DISTILL_SUMMARY}"
  --audit_out "${DISTILL_AUDIT}"
  --base_model "${BASE_MODEL_DIR}"
  --qwen_v28_adapter "${QWEN_V28_ADAPTER}"
  --v32_weights "${V32_WEIGHTS}"
  --v32_meta "${V32_META}"
  --hybrid_weights "${HYBRID_WEIGHTS}"
  --hybrid_meta "${HYBRID_META}"
  --lite_weights "${LITE_WEIGHTS}"
  --lite_meta "${LITE_META}"
  --champion_v28_weights "${CHAMPION_V28_WEIGHTS}"
  --champion_v28_meta "${CHAMPION_V28_META}"
  --sample_size 960
  --stage2_reference_cap 2600
  --eval_fraction 0.14
  --eval_cap 192
  --paper_eval_fraction 0.20
  --paper_eval_cap 24
  --polish_limit 320
  --seed 63
  --device cuda
  --max_new_tokens 112
  --log_every 32
  --min_teacher_score 0.14
)

printf '[v33] build command:\n %q' "${BUILD_CMD[@]}"
printf '\n'
if [[ "${RESUME_FROM_STAGE1:-0}" != "1" ]]; then
  "${BUILD_CMD[@]}" 2>&1 | tee "${BUILD_LOG}"
else
  echo "[v33] skipping build because RESUME_FROM_STAGE1=1"
fi

STAGE1_CMD=(
  python3
  source/finetune_chat.py
  --data "${DISTILL_DATA}"
  --weights "${V32_WEIGHTS}"
  --output "${STAGE1_OUT}"
  --meta "${STAGE1_META}"
  --model_size frontier_expert
  --feature_mode context_mix_v4
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 12
  --batch_size 96
  --grad_accum_steps 1
  --lr 3.8e-5
  --weight_decay 0.014
  --label_smoothing 0.02
  --val_split 0.10
  --split_mode stratified
  --seed 63
  --balanced_sampler
  --pref_weight 0.11
  --pref_beta 1.8
  --hard_negative_ratio 0.45
  --pref_objective sigmoid
  --pref_group_size 4
  --pref_group_estimator epo
  --adaptive_pref_weighting
  --pref_warmup_epochs 1.0
  --lr_schedule cosine
  --warmup_steps 20
  --early_stop_patience 3
  --max_candidates_per_bucket 192
  --ema_decay 0.999
  --log_interval_steps 2
  --aux_loss_weight 0.018
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage1_checkpoints"
)

printf '[v33] stage1 train command:\n %q' "${STAGE1_CMD[@]}"
printf '\n'
if [[ "${RESUME_FROM_STAGE1:-0}" != "1" ]]; then
  "${STAGE1_CMD[@]}" 2>&1 | tee "${STAGE1_LOG}"
else
  echo "[v33] skipping stage1 train because RESUME_FROM_STAGE1=1"
fi

BENCH_STAGE1_CMD=(
  python3
  source/benchmark.py
  --weights_a "${V32_WEIGHTS}"
  --meta_a "${V32_META}"
  --weights_b "${STAGE1_OUT}"
  --meta_b "${STAGE1_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v33] stage1 benchmark command:\n %q' "${BENCH_STAGE1_CMD[@]}"
printf '\n'
"${BENCH_STAGE1_CMD[@]}" 2>&1 | tee "${BENCH_STAGE1_LOG}"

STAGE2_CMD=(
  python3
  source/finetune_chat.py
  --data "${STAGE2_TRAIN_DATA}"
  --weights "${STAGE1_OUT}"
  --output "${STAGE2_OUT}"
  --meta "${STAGE2_META}"
  --model_size frontier_expert
  --feature_mode context_mix_v4
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 14
  --batch_size 96
  --grad_accum_steps 1
  --lr 2.2e-5
  --weight_decay 0.011
  --label_smoothing 0.018
  --val_split 0.12
  --split_mode stratified
  --seed 63
  --balanced_sampler
  --pref_weight 0.08
  --pref_beta 1.6
  --hard_negative_ratio 0.35
  --pref_objective sigmoid
  --pref_group_size 3
  --pref_group_estimator epo
  --adaptive_pref_weighting
  --pref_warmup_epochs 0.8
  --lr_schedule cosine
  --warmup_steps 12
  --early_stop_patience 4
  --max_candidates_per_bucket 192
  --ema_decay 0.999
  --log_interval_steps 2
  --aux_loss_weight 0.014
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage2_checkpoints"
)

printf '[v33] stage2 train command:\n %q' "${STAGE2_CMD[@]}"
printf '\n'
"${STAGE2_CMD[@]}" 2>&1 | tee "${STAGE2_LOG}"

BENCH_STAGE2_CMD=(
  python3
  source/benchmark.py
  --weights_a "${STAGE1_OUT}"
  --meta_a "${STAGE1_META}"
  --weights_b "${STAGE2_OUT}"
  --meta_b "${STAGE2_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v33] stage2 benchmark command:\n %q' "${BENCH_STAGE2_CMD[@]}"
printf '\n'
"${BENCH_STAGE2_CMD[@]}" 2>&1 | tee "${BENCH_STAGE2_LOG}"

STAGE3_CMD=(
  python3
  source/finetune_chat.py
  --data "${STAGE3_POLISH_DATA}"
  --weights "${STAGE2_OUT}"
  --output "${FINAL_OUT}"
  --meta "${FINAL_META}"
  --model_size frontier_expert
  --feature_mode context_mix_v4
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 6
  --batch_size 64
  --grad_accum_steps 1
  --lr 1.2e-5
  --weight_decay 0.008
  --label_smoothing 0.015
  --val_split 0.10
  --split_mode stratified
  --seed 63
  --balanced_sampler
  --pref_weight 0.04
  --pref_beta 1.35
  --hard_negative_ratio 0.20
  --pref_objective sigmoid
  --pref_group_size 2
  --pref_group_estimator pairwise_mean
  --adaptive_pref_weighting
  --pref_warmup_epochs 0.4
  --lr_schedule cosine
  --warmup_steps 6
  --early_stop_patience 2
  --max_candidates_per_bucket 160
  --ema_decay 0.999
  --log_interval_steps 2
  --aux_loss_weight 0.010
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage3_checkpoints"
)

printf '[v33] stage3 train command:\n %q' "${STAGE3_CMD[@]}"
printf '\n'
"${STAGE3_CMD[@]}" 2>&1 | tee "${STAGE3_LOG}"

BENCH_STAGE3_CMD=(
  python3
  source/benchmark.py
  --weights_a "${STAGE2_OUT}"
  --meta_a "${STAGE2_META}"
  --weights_b "${FINAL_OUT}"
  --meta_b "${FINAL_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v33] stage3 benchmark command:\n %q' "${BENCH_STAGE3_CMD[@]}"
printf '\n'
"${BENCH_STAGE3_CMD[@]}" 2>&1 | tee "${BENCH_STAGE3_LOG}"

BENCH_FINAL_CMD=(
  python3
  source/benchmark.py
  --weights_a "${V32_WEIGHTS}"
  --meta_a "${V32_META}"
  --weights_b "${FINAL_OUT}"
  --meta_b "${FINAL_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v33] final benchmark command:\n %q' "${BENCH_FINAL_CMD[@]}"
printf '\n'
"${BENCH_FINAL_CMD[@]}" 2>&1 | tee "${BENCH_FINAL_LOG}"

python3 - \
  "${SUMMARY_JSON}" \
  "${DISTILL_SUMMARY}" \
  "${STAGE1_META}" \
  "${STAGE2_META}" \
  "${FINAL_META}" \
  "${V32_WEIGHTS}" \
  "${STAGE1_OUT}" \
  "${STAGE2_OUT}" \
  "${FINAL_OUT}" \
  "${BENCH_STAGE1_LOG}" \
  "${BENCH_STAGE2_LOG}" \
  "${BENCH_STAGE3_LOG}" \
  "${BENCH_FINAL_LOG}" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
distill_summary_path = Path(sys.argv[2])
stage1_meta_path = Path(sys.argv[3])
stage2_meta_path = Path(sys.argv[4])
final_meta_path = Path(sys.argv[5])
start_weights_path = Path(sys.argv[6])
stage1_weights_path = Path(sys.argv[7])
stage2_weights_path = Path(sys.argv[8])
final_weights_path = Path(sys.argv[9])
bench_stage1_path = Path(sys.argv[10])
bench_stage2_path = Path(sys.argv[11])
bench_stage3_path = Path(sys.argv[12])
bench_final_path = Path(sys.argv[13])

accuracy_re = re.compile(r"Accuracy:\s+([0-9.]+)\s+\((\d+)/(\d+)\)")

def parse_benchmark(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for idx, match in enumerate(accuracy_re.findall(text)):
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
stage2_meta = json.loads(stage2_meta_path.read_text(encoding="utf-8"))
final_meta = json.loads(final_meta_path.read_text(encoding="utf-8"))

summary = {
    "created_at": final_meta.get("created_at"),
    "student_model_size": final_meta.get("model_size"),
    "feature_mode": final_meta.get("feature_mode"),
    "distill_dataset": distill_summary,
    "weights_bytes": {
        "start_v32": start_weights_path.stat().st_size,
        "stage1": stage1_weights_path.stat().st_size,
        "stage2": stage2_weights_path.stat().st_size,
        "final": final_weights_path.stat().st_size,
    },
    "stage1_meta": {
        "best_epoch": stage1_meta.get("best_epoch"),
        "best_val_loss": stage1_meta.get("best_val_loss"),
        "num_examples": stage1_meta.get("num_examples"),
        "trainable_parameters": stage1_meta.get("trainable_parameters"),
        "total_parameters": stage1_meta.get("total_parameters"),
    },
    "stage2_meta": {
        "best_epoch": stage2_meta.get("best_epoch"),
        "best_val_loss": stage2_meta.get("best_val_loss"),
        "num_examples": stage2_meta.get("num_examples"),
        "trainable_parameters": stage2_meta.get("trainable_parameters"),
        "total_parameters": stage2_meta.get("total_parameters"),
    },
    "final_meta": {
        "best_epoch": final_meta.get("best_epoch"),
        "best_val_loss": final_meta.get("best_val_loss"),
        "num_examples": final_meta.get("num_examples"),
        "trainable_parameters": final_meta.get("trainable_parameters"),
        "total_parameters": final_meta.get("total_parameters"),
    },
    "benchmarks": {
        "v32_vs_stage1_on_eval": parse_benchmark(bench_stage1_path),
        "stage1_vs_stage2_on_eval": parse_benchmark(bench_stage2_path),
        "stage2_vs_stage3_on_eval": parse_benchmark(bench_stage3_path),
        "v32_vs_final_on_eval": parse_benchmark(bench_final_path),
    },
    "note": (
        "The v33 frontier run combines latest-paper synthesis rows with a staged distill, full-mix refinement, "
        "and concise polish pass. It is a real new checkpoint, not a router over prior models."
    ),
}

summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

python3 - \
  "${FINAL_ZIP}" \
  "${BUNDLE_ZIP}" \
  "${FINAL_OUT}" \
  "${FINAL_META}" \
  "${SUMMARY_JSON}" \
  "${STAGE2_OUT}" \
  "${STAGE2_META}" \
  "${STAGE1_OUT}" \
  "${STAGE1_META}" \
  "${DISTILL_DATA}" \
  "${STAGE2_TRAIN_DATA}" \
  "${STAGE2_EVAL_DATA}" \
  "${STAGE3_POLISH_DATA}" \
  "${DISTILL_SUMMARY}" \
  "${DISTILL_AUDIT}" \
  "${BUILD_LOG}" \
  "${STAGE1_LOG}" \
  "${STAGE2_LOG}" \
  "${STAGE3_LOG}" \
  "${BENCH_STAGE1_LOG}" \
  "${BENCH_STAGE2_LOG}" \
  "${BENCH_STAGE3_LOG}" \
  "${BENCH_FINAL_LOG}" \
  "${REPO_ROOT}/source/build_v33_frontier_dataset.py" \
  "${REPO_ROOT}/source/model_frontier_v33.py" \
  "${REPO_ROOT}/source/model_variants.py" \
  "${REPO_ROOT}/source/chat_pipeline.py" \
  "${REPO_ROOT}/source/benchmark.py" \
  "${REPO_ROOT}/source/finetune_chat.py" \
  "${REPO_ROOT}/source/run_train_champion_v33_frontier_full_runpod.sh" <<'PY'
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
    for path in files[:3]:
        if path.exists():
            zf.write(path, arcname=path.name)

with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        if path.exists():
            zf.write(path, arcname=path.name)
PY

echo "[v33] final model zip: ${FINAL_ZIP}"
echo "[v33] bundle zip: ${BUNDLE_ZIP}"
echo "[v33] summary: ${SUMMARY_JSON}"
