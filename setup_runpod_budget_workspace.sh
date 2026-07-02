#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/kai9987kai/Supermix_29.git}"
BRANCH="${BRANCH:-main}"
RUNPOD_ROOT="${RUNPOD_ROOT:-/workspace/supermix_runpod_budget}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${RUNPOD_ROOT}/Supermix_29}"
PERSIST_ROOT="${PERSIST_ROOT:-${RUNPOD_ROOT}/persistent}"
HF_HOME="${HF_HOME:-${PERSIST_ROOT}/hf_cache}"
LOG_DIR="${LOG_DIR:-${PERSIST_ROOT}/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v28_cloud_plus_runpod_budget}"
RESUME_WARM_START_DIR="${RESUME_WARM_START_DIR:-${PERSIST_ROOT}/artifacts/qwen_supermix_enhanced_v26_full}"
STATE_PATH="${STATE_PATH:-${PERSIST_ROOT}/last_training_launch_runpod_budget.json}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PERSIST_ROOT}/base_models/qwen2_5_0_5b_instruct_${BASE_MODEL_REVISION}}"
ENV_PATH="${ENV_PATH:-${PERSIST_ROOT}/runpod_budget.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PROFILE="${TRAIN_PROFILE:-cloud_plus}"
ALLOW_BUNDLED_WARM_START_FALLBACK="${ALLOW_BUNDLED_WARM_START_FALLBACK:-0}"
WORKSPACE_RESET="${WORKSPACE_RESET:-0}"
MIN_GPU_MEMORY_GB="${MIN_GPU_MEMORY_GB:-16}"
REPO_WARM_START_DIR_REL="artifacts/qwen_supermix_enhanced_v26_full"
REPO_BUNDLED_WARM_START_DIR_REL="dist/SupermixQwenDesktopV26/_internal/bundled_latest_artifact"
REPO_RUNTIME_DIR_REL="runtime_python"

PINNED_PACKAGES=(
  "torch==2.4.1"
  "transformers==5.2.0"
  "peft==0.18.1"
  "accelerate==1.12.0"
  "safetensors==0.7.0"
  "matplotlib==3.10.8"
  "pillow==12.1.0"
  "nltk==3.9.2"
  "tokenizers==0.22.2"
  "huggingface_hub==1.7.2"
  "sentencepiece"
  "tqdm"
)

mkdir -p \
  "$RUNPOD_ROOT" \
  "$PERSIST_ROOT" \
  "$HF_HOME" \
  "$LOG_DIR" \
  "$OUTPUT_DIR" \
  "$RESUME_WARM_START_DIR" \
  "$(dirname "$STATE_PATH")" \
  "$(dirname "$BASE_MODEL_DIR")"

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_DESC="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1)"
  GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]')"
  echo "[runpod-setup] detected GPU: ${GPU_DESC}"
  if [[ -n "${GPU_MEMORY_MB}" ]]; then
    MIN_GPU_MEMORY_MB=$(( MIN_GPU_MEMORY_GB * 1024 - 256 ))
    if (( GPU_MEMORY_MB < MIN_GPU_MEMORY_MB )); then
      echo "[runpod-setup] warning: this recipe expects a CUDA GPU with roughly ${MIN_GPU_MEMORY_GB} GB or more"
    fi
  fi
else
  echo "[runpod-setup] warning: nvidia-smi was not found; CUDA validation will happen during the Python checks"
fi

if [[ "${WORKSPACE_RESET}" == "1" && -d "${WORKSPACE_DIR}" ]]; then
  echo "[runpod-setup] removing existing workspace at ${WORKSPACE_DIR}"
  rm -rf "${WORKSPACE_DIR}"
fi

if [[ ! -d "${WORKSPACE_DIR}/.git" ]]; then
  echo "[runpod-setup] cloning ${REPO_URL}#${BRANCH} into ${WORKSPACE_DIR}"
  git clone --filter=blob:none --sparse --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${WORKSPACE_DIR}"
else
  echo "[runpod-setup] reusing existing workspace at ${WORKSPACE_DIR}"
fi

git -C "${WORKSPACE_DIR}" sparse-checkout init --cone >/dev/null 2>&1 || true
git -C "${WORKSPACE_DIR}" sparse-checkout set \
  source \
  datasets \
  "${REPO_RUNTIME_DIR_REL}" \
  "${REPO_WARM_START_DIR_REL}" \
  "${REPO_BUNDLED_WARM_START_DIR_REL}"

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "[runpod-setup] installing git-lfs"
  apt-get update
  apt-get install -y git-lfs
fi

git -C "${WORKSPACE_DIR}" lfs install
git -C "${WORKSPACE_DIR}" lfs pull \
  --include="datasets/**,${REPO_RUNTIME_DIR_REL}/**,${REPO_WARM_START_DIR_REL}/**,${REPO_BUNDLED_WARM_START_DIR_REL}/**"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTHONUNBUFFERED="1"
export PYTHONHASHSEED="48"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM="false"
export BASE_MODEL_REPO
export BASE_MODEL_REVISION
export BASE_MODEL_DIR
unset HF_HUB_OFFLINE || true
unset TRANSFORMERS_OFFLINE || true

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -r "${WORKSPACE_DIR}/source/requirements_train_build.txt"
"${PYTHON_BIN}" -m pip install "${PINNED_PACKAGES[@]}"

"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["BASE_MODEL_REPO"]
revision = os.environ["BASE_MODEL_REVISION"]
local_dir = os.environ["BASE_MODEL_DIR"]

print(f"[runpod-setup] downloading base model {repo_id}@{revision} -> {local_dir}")
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print("[runpod-setup] base model ready")
PY

"${PYTHON_BIN}" - <<'PY'
import os
import torch

print("[runpod-setup] torch:", torch.__version__)
print("[runpod-setup] CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available after dependency installation.")
print("[runpod-setup] GPU:", torch.cuda.get_device_name(0))
PY

REPO_WARM_START_DIR="${WORKSPACE_DIR}/${REPO_WARM_START_DIR_REL}"
if [[ ! -e "${RESUME_WARM_START_DIR}/adapter" && -d "${REPO_WARM_START_DIR}" ]]; then
  echo "[runpod-setup] seeding persistent warm-start artifact from ${REPO_WARM_START_DIR}"
  rm -rf "${RESUME_WARM_START_DIR}"
  cp -a "${REPO_WARM_START_DIR}" "${RESUME_WARM_START_DIR}"
fi

mkdir -p "$(dirname "${ENV_PATH}")"
: > "${ENV_PATH}"
write_env_var() {
  local key="$1"
  local value="$2"
  printf 'export %s=%q\n' "${key}" "${value}" >> "${ENV_PATH}"
}

write_env_var REPO_URL "${REPO_URL}"
write_env_var BRANCH "${BRANCH}"
write_env_var RUNPOD_ROOT "${RUNPOD_ROOT}"
write_env_var WORKSPACE_DIR "${WORKSPACE_DIR}"
write_env_var PERSIST_ROOT "${PERSIST_ROOT}"
write_env_var HF_HOME "${HF_HOME}"
write_env_var LOG_DIR "${LOG_DIR}"
write_env_var OUTPUT_DIR "${OUTPUT_DIR}"
write_env_var RESUME_WARM_START_DIR "${RESUME_WARM_START_DIR}"
write_env_var STATE_PATH "${STATE_PATH}"
write_env_var BASE_MODEL_REPO "${BASE_MODEL_REPO}"
write_env_var BASE_MODEL_REVISION "${BASE_MODEL_REVISION}"
write_env_var BASE_MODEL_DIR "${BASE_MODEL_DIR}"
write_env_var ENV_PATH "${ENV_PATH}"
write_env_var PYTHON_BIN "${PYTHON_BIN}"
write_env_var TRAIN_PROFILE "${TRAIN_PROFILE}"
write_env_var ALLOW_BUNDLED_WARM_START_FALLBACK "${ALLOW_BUNDLED_WARM_START_FALLBACK}"

echo "[runpod-setup] workspace ready"
echo "[runpod-setup] workspace: ${WORKSPACE_DIR}"
echo "[runpod-setup] persistent env: ${ENV_PATH}"
echo "[runpod-setup] next:"
echo "  cd ${WORKSPACE_DIR}"
echo "  bash source/run_train_qwen_supermix_v28_runpod_budget.sh"
