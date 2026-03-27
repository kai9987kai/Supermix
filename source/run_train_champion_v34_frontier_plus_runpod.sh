#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/champion_v34_frontier_plus_20260326}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PERSIST_ROOT}/base_models/qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775}"

PROMPT_DATA_1="${PROMPT_DATA_1:-${REPO_ROOT}/datasets/conversation_data.delta_anchor_mix_2026_03_26.jsonl}"
PROMPT_DATA_2="${PROMPT_DATA_2:-${REPO_ROOT}/datasets/conversation_data.delta_official_refresh_2026_03_26.jsonl}"
PROMPT_DATA_3="${PROMPT_DATA_3:-${REPO_ROOT}/datasets/conversation_data.coding_knowledge_2026_02_19.jsonl}"
PROMPT_DATA_4="${PROMPT_DATA_4:-${REPO_ROOT}/datasets/conversation_data.world_events_2026_02_19.jsonl}"
PROMPT_DATA_5="${PROMPT_DATA_5:-${REPO_ROOT}/datasets/conversation_data.quality_anchor_v2.jsonl}"
PROMPT_DATA_6="${PROMPT_DATA_6:-${REPO_ROOT}/datasets/conversation_data.mega_creative_250k_v2.jsonl}"
PROMPT_DATA_7="${PROMPT_DATA_7:-${REPO_ROOT}/datasets/conversation_data.mega_reasoning_creative_v25_75582.jsonl}"
PROMPT_DATA_8="${PROMPT_DATA_8:-${REPO_ROOT}/datasets/conversation_data.supermix_plus_v27_500k.jsonl}"
PROMPT_DATA_9="${PROMPT_DATA_9:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget/prepared_train_pairs.jsonl}"
PROMPT_DATA_10="${PROMPT_DATA_10:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v29_delta_official_refresh_20260326/prepared_train_pairs.jsonl}"
PROMPT_DATA_11="${PROMPT_DATA_11:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v30_anchor_refresh_20260326/prepared_train_pairs.jsonl}"

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
V33_STAGE1_WEIGHTS="${V33_STAGE1_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v33_frontier_full_20260326/champion_model_chat_v33_frontier_stage1.pth}"
V33_STAGE1_META="${V33_STAGE1_META:-${PERSIST_ROOT}/artifacts/champion_v33_frontier_full_20260326/chat_model_meta_v33_frontier_stage1.json}"
V33_STAGE2_WEIGHTS="${V33_STAGE2_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v33_frontier_full_20260326/champion_model_chat_v33_frontier_stage2.pth}"
V33_STAGE2_META="${V33_STAGE2_META:-${PERSIST_ROOT}/artifacts/champion_v33_frontier_full_20260326/chat_model_meta_v33_frontier_stage2.json}"

START_WEIGHTS="${START_WEIGHTS:-${V33_STAGE2_WEIGHTS}}"
START_META="${START_META:-${V33_STAGE2_META}}"
BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-${V32_WEIGHTS}}"
BASELINE_META="${BASELINE_META:-${V32_META}}"

DISTILL_DATA="${DISTILL_DATA:-${OUTPUT_DIR}/v34_frontier_plus_distill_train.jsonl}"
STAGE2_TRAIN_DATA="${STAGE2_TRAIN_DATA:-${OUTPUT_DIR}/v34_frontier_plus_stage2_train_mix.jsonl}"
STAGE2_EVAL_DATA="${STAGE2_EVAL_DATA:-${OUTPUT_DIR}/v34_frontier_plus_stage2_eval_mix.jsonl}"
STAGE3_POLISH_DATA="${STAGE3_POLISH_DATA:-${OUTPUT_DIR}/v34_frontier_plus_stage3_polish.jsonl}"
DISTILL_SUMMARY="${DISTILL_SUMMARY:-${OUTPUT_DIR}/v34_frontier_plus_dataset_summary.json}"
DISTILL_AUDIT="${DISTILL_AUDIT:-${OUTPUT_DIR}/v34_frontier_plus_dataset_audit.jsonl}"

STAGE1_OUT="${STAGE1_OUT:-${OUTPUT_DIR}/champion_model_chat_v34_frontier_plus_stage1.pth}"
STAGE1_META="${STAGE1_META:-${OUTPUT_DIR}/chat_model_meta_v34_frontier_plus_stage1.json}"
STAGE2_OUT="${STAGE2_OUT:-${OUTPUT_DIR}/champion_model_chat_v34_frontier_plus_stage2.pth}"
STAGE2_META="${STAGE2_META:-${OUTPUT_DIR}/chat_model_meta_v34_frontier_plus_stage2.json}"
STAGE3_OUT="${STAGE3_OUT:-${OUTPUT_DIR}/champion_model_chat_v34_frontier_plus_stage3.pth}"
STAGE3_META="${STAGE3_META:-${OUTPUT_DIR}/chat_model_meta_v34_frontier_plus_stage3.json}"
CHOSEN_JSON="${CHOSEN_JSON:-${OUTPUT_DIR}/v34_frontier_plus_chosen_checkpoint.json}"

BUILD_LOG="${BUILD_LOG:-${LOG_DIR}/build_v34_frontier_plus.out.log}"
STAGE1_LOG="${STAGE1_LOG:-${LOG_DIR}/train_champion_v34_frontier_plus_stage1.out.log}"
STAGE2_LOG="${STAGE2_LOG:-${LOG_DIR}/train_champion_v34_frontier_plus_stage2.out.log}"
STAGE3_LOG="${STAGE3_LOG:-${LOG_DIR}/train_champion_v34_frontier_plus_stage3.out.log}"
BENCH_STAGE1_LOG="${BENCH_STAGE1_LOG:-${LOG_DIR}/benchmark_champion_v34_stage1_vs_v32.out.log}"
BENCH_STAGE2_LOG="${BENCH_STAGE2_LOG:-${LOG_DIR}/benchmark_champion_v34_stage2_vs_stage1.out.log}"
BENCH_STAGE3_LOG="${BENCH_STAGE3_LOG:-${LOG_DIR}/benchmark_champion_v34_stage3_vs_stage2.out.log}"
BENCH_CHOSEN_LOG="${BENCH_CHOSEN_LOG:-${LOG_DIR}/benchmark_champion_v34_chosen_vs_v32.out.log}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_DIR}/v34_frontier_plus_training_summary.json}"
FINAL_ZIP="${FINAL_ZIP:-${PERSIST_ROOT}/champion_v34_frontier_plus_full_model_20260326.zip}"
BUNDLE_ZIP="${BUNDLE_ZIP:-${PERSIST_ROOT}/champion_v34_frontier_plus_full_bundle_20260326.zip}"

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
  "${EVAL_DATA_1}" \
  "${EVAL_DATA_2}" \
  "${EVAL_DATA_3}" \
  "${QWEN_V28_ADAPTER}" \
  "${LITE_WEIGHTS}" \
  "${LITE_META}" \
  "${HYBRID_WEIGHTS}" \
  "${HYBRID_META}" \
  "${V32_WEIGHTS}" \
  "${V32_META}" \
  "${V33_STAGE1_WEIGHTS}" \
  "${V33_STAGE1_META}" \
  "${V33_STAGE2_WEIGHTS}" \
  "${V33_STAGE2_META}" \
  "${START_WEIGHTS}" \
  "${START_META}" \
  "${BASELINE_WEIGHTS}" \
  "${BASELINE_META}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[v34] missing required path: ${required}" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED="1"
export TOKENIZERS_PARALLELISM="false"

BUILD_CMD=(
  python3
  source/build_v34_frontier_plus_dataset.py
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
  --v33_stage1_weights "${V33_STAGE1_WEIGHTS}"
  --v33_stage1_meta "${V33_STAGE1_META}"
  --v33_stage2_weights "${V33_STAGE2_WEIGHTS}"
  --v33_stage2_meta "${V33_STAGE2_META}"
  --hybrid_weights "${HYBRID_WEIGHTS}"
  --hybrid_meta "${HYBRID_META}"
  --lite_weights "${LITE_WEIGHTS}"
  --lite_meta "${LITE_META}"
  --champion_v28_weights "${CHAMPION_V28_WEIGHTS}"
  --champion_v28_meta "${CHAMPION_V28_META}"
  --sample_size 1152
  --stage2_reference_cap 4200
  --eval_fraction 0.15
  --eval_cap 320
  --paper_eval_fraction 0.22
  --paper_eval_cap 36
  --polish_limit 448
  --self_repair_limit 448
  --seed 77
  --device cuda
  --max_new_tokens 120
  --log_every 32
  --min_teacher_score 0.16
)

printf '[v34] build command:\n %q' "${BUILD_CMD[@]}"
printf '\n'
"${BUILD_CMD[@]}" 2>&1 | tee "${BUILD_LOG}"

STAGE1_CMD=(
  python3
  source/finetune_chat.py
  --data "${DISTILL_DATA}"
  --weights "${START_WEIGHTS}"
  --output "${STAGE1_OUT}"
  --meta "${STAGE1_META}"
  --model_size frontier_expert
  --feature_mode context_mix_v4
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 9
  --batch_size 96
  --grad_accum_steps 1
  --lr 2.9e-5
  --weight_decay 0.013
  --label_smoothing 0.018
  --val_split 0.10
  --split_mode stratified
  --seed 77
  --balanced_sampler
  --pref_weight 0.10
  --pref_beta 1.7
  --hard_negative_ratio 0.40
  --pref_objective sigmoid
  --pref_group_size 4
  --pref_group_estimator epo
  --adaptive_pref_weighting
  --pref_warmup_epochs 0.9
  --lr_schedule cosine
  --warmup_steps 18
  --early_stop_patience 3
  --max_candidates_per_bucket 224
  --ema_decay 0.999
  --log_interval_steps 2
  --aux_loss_weight 0.016
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage1_checkpoints"
)

printf '[v34] stage1 train command:\n %q' "${STAGE1_CMD[@]}"
printf '\n'
"${STAGE1_CMD[@]}" 2>&1 | tee "${STAGE1_LOG}"

BENCH_STAGE1_CMD=(
  python3
  source/benchmark.py
  --weights_a "${BASELINE_WEIGHTS}"
  --meta_a "${BASELINE_META}"
  --weights_b "${STAGE1_OUT}"
  --meta_b "${STAGE1_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v34] stage1 benchmark command:\n %q' "${BENCH_STAGE1_CMD[@]}"
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
  --epochs 12
  --batch_size 96
  --grad_accum_steps 1
  --lr 1.8e-5
  --weight_decay 0.010
  --label_smoothing 0.016
  --val_split 0.12
  --split_mode stratified
  --seed 77
  --balanced_sampler
  --pref_weight 0.08
  --pref_beta 1.5
  --hard_negative_ratio 0.32
  --pref_objective sigmoid
  --pref_group_size 3
  --pref_group_estimator epo
  --adaptive_pref_weighting
  --pref_warmup_epochs 0.7
  --lr_schedule cosine
  --warmup_steps 12
  --early_stop_patience 4
  --max_candidates_per_bucket 224
  --ema_decay 0.999
  --log_interval_steps 2
  --aux_loss_weight 0.012
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage2_checkpoints"
)

printf '[v34] stage2 train command:\n %q' "${STAGE2_CMD[@]}"
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

printf '[v34] stage2 benchmark command:\n %q' "${BENCH_STAGE2_CMD[@]}"
printf '\n'
"${BENCH_STAGE2_CMD[@]}" 2>&1 | tee "${BENCH_STAGE2_LOG}"

STAGE3_CMD=(
  python3
  source/finetune_chat.py
  --data "${STAGE3_POLISH_DATA}"
  --weights "${STAGE2_OUT}"
  --output "${STAGE3_OUT}"
  --meta "${STAGE3_META}"
  --model_size frontier_expert
  --feature_mode context_mix_v4
  --device cuda
  --device_preference cuda,npu,xpu,dml,mps,cpu
  --train_all
  --epochs 4
  --batch_size 80
  --grad_accum_steps 1
  --lr 8.5e-6
  --weight_decay 0.008
  --label_smoothing 0.014
  --val_split 0.10
  --split_mode stratified
  --seed 77
  --balanced_sampler
  --pref_weight 0.03
  --pref_beta 1.25
  --hard_negative_ratio 0.16
  --pref_objective sigmoid
  --pref_group_size 2
  --pref_group_estimator pairwise_mean
  --adaptive_pref_weighting
  --pref_warmup_epochs 0.3
  --lr_schedule cosine
  --warmup_steps 4
  --early_stop_patience 2
  --max_candidates_per_bucket 176
  --ema_decay 0.999
  --log_interval_steps 2
  --aux_loss_weight 0.009
  --epoch_checkpoint_dir "${OUTPUT_DIR}/stage3_checkpoints"
)

printf '[v34] stage3 train command:\n %q' "${STAGE3_CMD[@]}"
printf '\n'
"${STAGE3_CMD[@]}" 2>&1 | tee "${STAGE3_LOG}"

BENCH_STAGE3_CMD=(
  python3
  source/benchmark.py
  --weights_a "${STAGE2_OUT}"
  --meta_a "${STAGE2_META}"
  --weights_b "${STAGE3_OUT}"
  --meta_b "${STAGE3_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v34] stage3 benchmark command:\n %q' "${BENCH_STAGE3_CMD[@]}"
printf '\n'
"${BENCH_STAGE3_CMD[@]}" 2>&1 | tee "${BENCH_STAGE3_LOG}"

mapfile -t CHOSEN_INFO < <(
  python3 - \
    "${CHOSEN_JSON}" \
    "${STAGE1_OUT}" \
    "${STAGE1_META}" \
    "${STAGE2_OUT}" \
    "${STAGE2_META}" \
    "${STAGE3_OUT}" \
    "${STAGE3_META}" \
    "${BENCH_STAGE1_LOG}" \
    "${BENCH_STAGE2_LOG}" \
    "${BENCH_STAGE3_LOG}" <<'PY'
import json
import re
import sys
from pathlib import Path

chosen_json = Path(sys.argv[1])
stage1_out = Path(sys.argv[2])
stage1_meta = Path(sys.argv[3])
stage2_out = Path(sys.argv[4])
stage2_meta = Path(sys.argv[5])
stage3_out = Path(sys.argv[6])
stage3_meta = Path(sys.argv[7])
bench_stage1 = Path(sys.argv[8])
bench_stage2 = Path(sys.argv[9])
bench_stage3 = Path(sys.argv[10])

accuracy_re = re.compile(r"Accuracy:\s+([0-9.]+)\s+\((\d+)/(\d+)\)")

def parse_benchmark(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [
        {"slot": idx, "accuracy": float(acc), "correct": int(correct), "total": int(total)}
        for idx, (acc, correct, total) in enumerate(accuracy_re.findall(text))
    ]

stage1_rows = parse_benchmark(bench_stage1)
stage2_rows = parse_benchmark(bench_stage2)
stage3_rows = parse_benchmark(bench_stage3)

scores = {
    "stage1": stage1_rows[1]["accuracy"] if len(stage1_rows) > 1 else float("-inf"),
    "stage2": stage2_rows[1]["accuracy"] if len(stage2_rows) > 1 else float("-inf"),
    "stage3": stage3_rows[1]["accuracy"] if len(stage3_rows) > 1 else float("-inf"),
}
priority = {"stage2": 3, "stage3": 2, "stage1": 1}
chosen_stage = max(scores, key=lambda key: (scores[key], priority[key]))

stage_map = {
    "stage1": {"weights": stage1_out, "meta": stage1_meta},
    "stage2": {"weights": stage2_out, "meta": stage2_meta},
    "stage3": {"weights": stage3_out, "meta": stage3_meta},
}
payload = {
    "chosen_stage": chosen_stage,
    "chosen_accuracy": scores[chosen_stage],
    "scores": scores,
    "paths": {
        name: {"weights": str(info["weights"]), "meta": str(info["meta"])}
        for name, info in stage_map.items()
    },
}
chosen_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(chosen_stage)
print(stage_map[chosen_stage]["weights"])
print(stage_map[chosen_stage]["meta"])
PY
)

CHOSEN_STAGE="${CHOSEN_INFO[0]}"
CHOSEN_WEIGHTS="${CHOSEN_INFO[1]}"
CHOSEN_META="${CHOSEN_INFO[2]}"

echo "[v34] chosen final checkpoint: ${CHOSEN_STAGE}"
echo "[v34] chosen final weights: ${CHOSEN_WEIGHTS}"

BENCH_CHOSEN_CMD=(
  python3
  source/benchmark.py
  --weights_a "${BASELINE_WEIGHTS}"
  --meta_a "${BASELINE_META}"
  --weights_b "${CHOSEN_WEIGHTS}"
  --meta_b "${CHOSEN_META}"
  --data "${STAGE2_EVAL_DATA}"
  --device cuda
)

printf '[v34] chosen benchmark command:\n %q' "${BENCH_CHOSEN_CMD[@]}"
printf '\n'
"${BENCH_CHOSEN_CMD[@]}" 2>&1 | tee "${BENCH_CHOSEN_LOG}"

python3 - \
  "${SUMMARY_JSON}" \
  "${DISTILL_SUMMARY}" \
  "${CHOSEN_JSON}" \
  "${STAGE1_META}" \
  "${STAGE2_META}" \
  "${STAGE3_META}" \
  "${START_WEIGHTS}" \
  "${BASELINE_WEIGHTS}" \
  "${STAGE1_OUT}" \
  "${STAGE2_OUT}" \
  "${STAGE3_OUT}" \
  "${BENCH_STAGE1_LOG}" \
  "${BENCH_STAGE2_LOG}" \
  "${BENCH_STAGE3_LOG}" \
  "${BENCH_CHOSEN_LOG}" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
distill_summary_path = Path(sys.argv[2])
chosen_json_path = Path(sys.argv[3])
stage1_meta_path = Path(sys.argv[4])
stage2_meta_path = Path(sys.argv[5])
stage3_meta_path = Path(sys.argv[6])
start_weights_path = Path(sys.argv[7])
baseline_weights_path = Path(sys.argv[8])
stage1_weights_path = Path(sys.argv[9])
stage2_weights_path = Path(sys.argv[10])
stage3_weights_path = Path(sys.argv[11])
bench_stage1_path = Path(sys.argv[12])
bench_stage2_path = Path(sys.argv[13])
bench_stage3_path = Path(sys.argv[14])
bench_chosen_path = Path(sys.argv[15])

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
chosen_payload = json.loads(chosen_json_path.read_text(encoding="utf-8"))
stage1_meta = json.loads(stage1_meta_path.read_text(encoding="utf-8"))
stage2_meta = json.loads(stage2_meta_path.read_text(encoding="utf-8"))
stage3_meta = json.loads(stage3_meta_path.read_text(encoding="utf-8"))

summary = {
    "created_at": stage3_meta.get("created_at") or stage2_meta.get("created_at") or stage1_meta.get("created_at"),
    "student_model_size": stage2_meta.get("model_size") or stage1_meta.get("model_size"),
    "feature_mode": stage2_meta.get("feature_mode") or stage1_meta.get("feature_mode"),
    "distill_dataset": distill_summary,
    "weights_bytes": {
        "baseline_v32": baseline_weights_path.stat().st_size,
        "start_v33_stage2": start_weights_path.stat().st_size,
        "stage1": stage1_weights_path.stat().st_size,
        "stage2": stage2_weights_path.stat().st_size,
        "stage3": stage3_weights_path.stat().st_size,
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
    "stage3_meta": {
        "best_epoch": stage3_meta.get("best_epoch"),
        "best_val_loss": stage3_meta.get("best_val_loss"),
        "num_examples": stage3_meta.get("num_examples"),
        "trainable_parameters": stage3_meta.get("trainable_parameters"),
        "total_parameters": stage3_meta.get("total_parameters"),
    },
    "benchmarks": {
        "baseline_vs_stage1_on_eval": parse_benchmark(bench_stage1_path),
        "stage1_vs_stage2_on_eval": parse_benchmark(bench_stage2_path),
        "stage2_vs_stage3_on_eval": parse_benchmark(bench_stage3_path),
        "baseline_vs_chosen_on_eval": parse_benchmark(bench_chosen_path),
    },
    "chosen_final_checkpoint": chosen_payload.get("chosen_stage"),
    "chosen_final_accuracy": chosen_payload.get("chosen_accuracy"),
    "note": (
        "The v34 frontier+ run starts from the stronger v33 stage2 checkpoint, expands the real dataset mix, "
        "uses newer paper synthesis rows, and auto-selects the best checkpoint instead of assuming the last stage wins."
    ),
}

summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

python3 - \
  "${FINAL_ZIP}" \
  "${BUNDLE_ZIP}" \
  "${CHOSEN_WEIGHTS}" \
  "${CHOSEN_META}" \
  "${SUMMARY_JSON}" \
  "${CHOSEN_JSON}" \
  "${STAGE1_OUT}" \
  "${STAGE1_META}" \
  "${STAGE2_OUT}" \
  "${STAGE2_META}" \
  "${STAGE3_OUT}" \
  "${STAGE3_META}" \
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
  "${BENCH_CHOSEN_LOG}" \
  "${REPO_ROOT}/source/build_v33_frontier_dataset.py" \
  "${REPO_ROOT}/source/build_v34_frontier_plus_dataset.py" \
  "${REPO_ROOT}/source/model_frontier_v33.py" \
  "${REPO_ROOT}/source/model_variants.py" \
  "${REPO_ROOT}/source/chat_pipeline.py" \
  "${REPO_ROOT}/source/benchmark.py" \
  "${REPO_ROOT}/source/finetune_chat.py" \
  "${REPO_ROOT}/source/run_train_champion_v34_frontier_plus_runpod.sh" <<'PY'
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
    seen = set()
    for path in files[:4]:
        key = str(path.resolve()) if path.exists() else ""
        if path.exists() and key not in seen:
            seen.add(key)
            zf.write(path, arcname=path.name)

with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    seen = set()
    for path in files:
        key = str(path.resolve()) if path.exists() else ""
        if path.exists() and key not in seen:
            seen.add(key)
            zf.write(path, arcname=path.name)
PY

echo "[v34] final model zip: ${FINAL_ZIP}"
echo "[v34] bundle zip: ${BUNDLE_ZIP}"
echo "[v34] summary: ${SUMMARY_JSON}"
