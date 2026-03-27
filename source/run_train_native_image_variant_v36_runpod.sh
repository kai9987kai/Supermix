#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/champion_v36_native_image_20260327}"

BASE_WEIGHTS="${BASE_WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v35_collective_allteachers_20260326/champion_model_chat_v35_collective_allteachers_stage2.pth}"
BASE_META="${BASE_META:-${PERSIST_ROOT}/artifacts/champion_v35_collective_allteachers_20260326/chat_model_meta_v35_collective_allteachers_stage2.json}"

PAIR_DATA="${PAIR_DATA:-${OUTPUT_DIR}/v36_native_image_pairs.jsonl}"
IMAGES_DIR="${IMAGES_DIR:-${OUTPUT_DIR}/images}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train_native_image_variant_v36.out.log}"
BUILD_LOG="${BUILD_LOG:-${LOG_DIR}/build_native_image_pairs_v36.out.log}"

OUTPUT_WEIGHTS="${OUTPUT_WEIGHTS:-${OUTPUT_DIR}/champion_model_chat_v36_native_image_single_checkpoint.pth}"
OUTPUT_META="${OUTPUT_META:-${OUTPUT_DIR}/chat_model_meta_v36_native_image_single_checkpoint.json}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUTPUT_DIR}/v36_native_image_training_summary.json}"
SAMPLE_OUT="${SAMPLE_OUT:-${OUTPUT_DIR}/sample_native_image_water_cycle.png}"

FINAL_ZIP="${FINAL_ZIP:-${PERSIST_ROOT}/champion_v36_native_image_single_checkpoint_model_20260327.zip}"
BUNDLE_ZIP="${BUNDLE_ZIP:-${PERSIST_ROOT}/champion_v36_native_image_single_checkpoint_bundle_20260327.zip}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${IMAGES_DIR}"

for required in "${BASE_WEIGHTS}" "${BASE_META}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[v36-native] missing required path: ${required}" >&2
    exit 1
  fi
done

export PYTHONUNBUFFERED="1"

python3 source/build_native_image_pairs_v36.py \
  --output "${PAIR_DATA}" \
  --images_dir "${IMAGES_DIR}" \
  --image_size 64 \
  --seed 36 2>&1 | tee "${BUILD_LOG}"

python3 source/train_native_image_variant_v36.py \
  --pairs "${PAIR_DATA}" \
  --base_weights "${BASE_WEIGHTS}" \
  --base_meta "${BASE_META}" \
  --output_weights "${OUTPUT_WEIGHTS}" \
  --output_meta "${OUTPUT_META}" \
  --summary_out "${SUMMARY_OUT}" \
  --sample_prompt "simple water cycle icon with blue water, white cloud, rain, and a yellow sun" \
  --sample_output "${SAMPLE_OUT}" \
  --image_size 64 \
  --feature_mode context_mix_v4 \
  --epochs 48 \
  --batch_size 32 \
  --lr 0.0012 \
  --weight_decay 0.0 \
  --val_fraction 0.15 \
  --seed 36 \
  --device cuda \
  --freeze_text_backbone 2>&1 | tee "${TRAIN_LOG}"

python3 - <<PY
import json
import zipfile
from pathlib import Path

final_zip = Path(r"""${FINAL_ZIP}""")
bundle_zip = Path(r"""${BUNDLE_ZIP}""")
output_dir = Path(r"""${OUTPUT_DIR}""")

model_files = [
    output_dir / "champion_model_chat_v36_native_image_single_checkpoint.pth",
    output_dir / "chat_model_meta_v36_native_image_single_checkpoint.json",
    output_dir / "v36_native_image_training_summary.json",
    output_dir / "sample_native_image_water_cycle.png",
]
bundle_files = model_files + [
    output_dir / "v36_native_image_pairs.jsonl",
    Path(r"""${REPO_ROOT}/source/model_frontier_native_image_v36.py"""),
    Path(r"""${REPO_ROOT}/source/build_native_image_pairs_v36.py"""),
    Path(r"""${REPO_ROOT}/source/train_native_image_variant_v36.py"""),
    Path(r"""${REPO_ROOT}/source/native_image_infer_v36.py"""),
    Path(r"""${REPO_ROOT}/source/run_train_native_image_variant_v36_runpod.sh"""),
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
    print(f"[v36-native] wrote {target}")
PY

echo "[v36-native] weights=${OUTPUT_WEIGHTS}"
echo "[v36-native] meta=${OUTPUT_META}"
echo "[v36-native] final_zip=${FINAL_ZIP}"
echo "[v36-native] bundle_zip=${BUNDLE_ZIP}"
