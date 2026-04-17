#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
WEIGHTS="${WEIGHTS:-${PERSIST_ROOT}/artifacts/champion_v31_hybrid_plus_refresh_20260326/champion_model_chat_v31_hybrid_plus_refresh.pth}"
META="${META:-${PERSIST_ROOT}/artifacts/champion_v31_hybrid_plus_refresh_20260326/chat_model_meta_v31_hybrid_plus_refresh.json}"
IMAGE_MODEL="${IMAGE_MODEL:-stabilityai/sdxl-turbo}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/generated_images_v31_variant}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "[image-variant] missing weights: ${WEIGHTS}" >&2
  exit 1
fi
if [[ ! -f "${META}" ]]; then
  echo "[image-variant] missing metadata: ${META}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if ! python3 - <<'PY' >/dev/null 2>&1
import importlib
for name in ("flask", "PIL", "diffusers", "transformers", "accelerate", "safetensors"):
    importlib.import_module(name)
PY
then
  python3 -m pip install -r source/requirements_runtime_interface.txt
fi

exec python3 source/chat_image_variant_app.py \
  --weights "${WEIGHTS}" \
  --meta "${META}" \
  --autoload \
  --host "${HOST}" \
  --port "${PORT}" \
  --image_model "${IMAGE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
