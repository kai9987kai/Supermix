#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/champion_v37_native_image_lite_20260327}"

TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v36_native_image_20260327/champion_model_chat_v36_native_image_single_checkpoint.pth}"
TEACHER_META="${TEACHER_META:-${PERSIST_ROOT}/artifacts/champion_v36_native_image_20260327/chat_model_meta_v36_native_image_single_checkpoint.json}"
PAIR_DATA="${PAIR_DATA:-${PERSIST_ROOT}/artifacts/champion_v36_native_image_20260327/v36_native_image_pairs.jsonl}"
STUDENT_BASE_WEIGHTS="${STUDENT_BASE_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326/champion_model_chat_v30_lite_student.pth}"
STUDENT_BASE_META="${STUDENT_BASE_META:-${PERSIST_ROOT}/artifacts/champion_v30_lite_student_20260326/chat_model_meta_v30_lite_student.json}"

TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/distill_native_image_lite_v37.out.log}"

OUTPUT_WEIGHTS="${OUTPUT_WEIGHTS:-${OUTPUT_DIR}/champion_model_chat_v37_native_image_lite_single_checkpoint.pth}"
OUTPUT_META="${OUTPUT_META:-${OUTPUT_DIR}/chat_model_meta_v37_native_image_lite_single_checkpoint.json}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUTPUT_DIR}/v37_native_image_lite_training_summary.json}"
SAMPLE_OUT="${SAMPLE_OUT:-${OUTPUT_DIR}/sample_native_image_lite_water_cycle.png}"

FINAL_ZIP="${FINAL_ZIP:-${PERSIST_ROOT}/champion_v37_native_image_lite_single_checkpoint_model_20260327.zip}"
BUNDLE_ZIP="${BUNDLE_ZIP:-${PERSIST_ROOT}/champion_v37_native_image_lite_single_checkpoint_bundle_20260327.zip}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

for required in "${TEACHER_WEIGHTS}" "${TEACHER_META}" "${PAIR_DATA}" "${STUDENT_BASE_WEIGHTS}" "${STUDENT_BASE_META}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[v37-lite] missing required path: ${required}" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED="1"

python3 source/distill_native_image_lite_v37.py \
  --pairs "${PAIR_DATA}" \
  --teacher_weights "${TEACHER_WEIGHTS}" \
  --teacher_meta "${TEACHER_META}" \
  --student_base_weights "${STUDENT_BASE_WEIGHTS}" \
  --student_base_meta "${STUDENT_BASE_META}" \
  --output_weights "${OUTPUT_WEIGHTS}" \
  --output_meta "${OUTPUT_META}" \
  --summary_out "${SUMMARY_OUT}" \
  --sample_prompt "simple water cycle icon with blue water, white cloud, rain, and a yellow sun" \
  --sample_output "${SAMPLE_OUT}" \
  --image_size 64 \
  --feature_mode context_mix_v4 \
  --epochs 56 \
  --batch_size 32 \
  --lr 0.0008 \
  --weight_decay 0.00001 \
  --val_fraction 0.15 \
  --seed 37 \
  --device cuda \
  --freeze_text_backbone \
  --unfreeze_text_layers 4 \
  --gt_image_weight 1.0 \
  --teacher_image_weight 0.45 \
  --teacher_text_weight 0.08 2>&1 | tee "${TRAIN_LOG}"

python3 - <<PY
import zipfile
from pathlib import Path

final_zip = Path(r"""${FINAL_ZIP}""")
bundle_zip = Path(r"""${BUNDLE_ZIP}""")
output_dir = Path(r"""${OUTPUT_DIR}""")

model_files = [
    output_dir / "champion_model_chat_v37_native_image_lite_single_checkpoint.pth",
    output_dir / "chat_model_meta_v37_native_image_lite_single_checkpoint.json",
    output_dir / "v37_native_image_lite_training_summary.json",
    output_dir / "sample_native_image_lite_water_cycle.png",
]
bundle_files = model_files + [
    Path(r"""${PERSIST_ROOT}/artifacts/champion_v36_native_image_20260327/v36_native_image_pairs.jsonl"""),
    Path(r"""${REPO_ROOT}/source/model_native_image_lite_v37.py"""),
    Path(r"""${REPO_ROOT}/source/distill_native_image_lite_v37.py"""),
    Path(r"""${REPO_ROOT}/source/native_image_infer_v37_lite.py"""),
    Path(r"""${REPO_ROOT}/source/run_distill_native_image_lite_v37_runpod.sh"""),
]

for target, files in ((final_zip, model_files), (bundle_zip, bundle_files)):
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen = set()
        for path in files:
            path = Path(path)
            if not path.exists():
                continue
            arcname = path.name if path.parent == output_dir else f"source/{path.name}"
            if arcname in seen:
                continue
            zf.write(path, arcname=arcname)
            seen.add(arcname)
    print(f"[v37-lite] wrote {target}")
PY

echo "[v37-lite] weights=${OUTPUT_WEIGHTS}"
echo "[v37-lite] meta=${OUTPUT_META}"
echo "[v37-lite] final_zip=${FINAL_ZIP}"
echo "[v37-lite] bundle_zip=${BUNDLE_ZIP}"
