#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-benchmark_all_models_common_plus_20260327}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}.out.log}"
SAMPLE_PER_BENCHMARK="${SAMPLE_PER_BENCHMARK:-50}"
LOG_EVERY="${LOG_EVERY:-25}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
export PYTHONUNBUFFERED="1"

python3 - <<'PY'
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("datasets") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
PY

python3 source/benchmark_all_models_common.py \
  --persist_root "${PERSIST_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda \
  --sample_per_benchmark "${SAMPLE_PER_BENCHMARK}" \
  --seed 20260327 \
  --log_every "${LOG_EVERY}" \
  --include_qwen_base 2>&1 | tee "${LOG_FILE}"

echo "[bench] output_dir=${OUTPUT_DIR}"
echo "[bench] log_file=${LOG_FILE}"
