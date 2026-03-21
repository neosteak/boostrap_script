#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 22B I2V — Full Setup for Vast AI (ComfyUI template)
#
#  Usage: bash install_ltx23_i2v.sh
#
#  This script installs everything needed for LTX 2.3 Image-to-Video
#  generation on a Vast AI instance running the ComfyUI template.
#
#  Pipeline: Image → LTX 2.3 22B (fp8, 2-stage + distilled LoRA)
#            → Latent Upscale 2x → 1280x720 25fps → SaveVideo
#
#  After running this script, load the workflow via:
#    Workflow → Browse Templates → Video → "LTX-2.3 I2V"
#    OR import the official workflow JSON.
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"
COMFY_BRANCH="${COMFY_BRANCH:-master}"

# ── Model sources ──

LTX23_REPO="Lightricks/LTX-2.3"
LTX23_FP8_REPO="Lightricks/LTX-2.3-fp8"
LTX2_TEXT_REPO="Comfy-Org/ltx-2"

CHECKPOINT_FILE="ltx-2.3-22b-dev-fp8.safetensors"
DISTILLED_LORA_FILE="ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSCALER_FILE="ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
TEXT_ENCODER_FILE="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"

# ── Locate ComfyUI ──

find_comfyui_dir() {
  if [[ -n "${COMFYUI_DIR:-}" && -f "${COMFYUI_DIR}/main.py" ]]; then
    echo "${COMFYUI_DIR}"
    return 0
  fi

  local pid
  pid="$(pgrep -f "main.py.*18188" | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    local cwd
    cwd="$(readlink -f "/proc/${pid}/cwd" || true)"
    if [[ -n "${cwd}" && -f "${cwd}/main.py" ]]; then
      echo "${cwd}"
      return 0
    fi
  fi

  local candidates=(
    "/workspace/ComfyUI"
    "/opt/ComfyUI"
    "/workspace/comfyui"
    "/root/ComfyUI"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/main.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

restart_comfy_services() {
  if ! command -v supervisorctl >/dev/null 2>&1; then
    return 0
  fi
  local services
  services="$(supervisorctl status | awk 'tolower($1) ~ /comfy/ {print $1}' || true)"
  if [[ -z "${services}" ]]; then
    return 0
  fi
  local service
  for service in ${services}; do
    supervisorctl restart "${service}" || true
  done
}

# ── Activate venv ──

echo "[ltx23] Activating /venv/main if present"
if [[ -f "/venv/main/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ltx23] Missing Python at ${PYTHON_BIN}"
  exit 1
fi

COMFYUI_DIR="$(find_comfyui_dir)" || {
  echo "[ltx23] Unable to locate ComfyUI. Set COMFYUI_DIR and retry."
  exit 1
}

echo "[ltx23] Using ComfyUI directory: ${COMFYUI_DIR}"

# ── Update ComfyUI core ──

echo "[ltx23] Updating ComfyUI core"
git -C "${COMFYUI_DIR}" fetch --all --tags
git -C "${COMFYUI_DIR}" checkout "${COMFY_BRANCH}"
git -C "${COMFYUI_DIR}" pull --ff-only

"${PIP_BIN}" install -r "${COMFYUI_DIR}/requirements.txt"
"${PIP_BIN}" install -U \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs" \
  "huggingface_hub[hf_transfer]"

# ── Create model directories ──

mkdir -p \
  "${COMFYUI_DIR}/models/checkpoints" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/latent_upscale_models" \
  "${COMFYUI_DIR}/custom_nodes"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# ── Download LTX 2.3 models ──

echo "[ltx23] Downloading LTX 2.3 22B models (this may take a while)..."

"${PYTHON_BIN}" - <<PY
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge_cache(repo_id: str) -> None:
    repo_cache = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    if repo_cache.exists():
        shutil.rmtree(repo_cache, ignore_errors=True)
    for extra in (hf_home / "hub" / "tmp", Path("/workspace/.cache/huggingface/xet")):
        if extra.exists():
            shutil.rmtree(extra, ignore_errors=True)

jobs = [
    ("${LTX23_FP8_REPO}",   "${CHECKPOINT_FILE}",        Path("${COMFYUI_DIR}/models/checkpoints/${CHECKPOINT_FILE}")),
    ("${LTX23_REPO}",       "${DISTILLED_LORA_FILE}",    Path("${COMFYUI_DIR}/models/loras/${DISTILLED_LORA_FILE}")),
    ("${LTX23_REPO}",       "${SPATIAL_UPSCALER_FILE}",  Path("${COMFYUI_DIR}/models/latent_upscale_models/${SPATIAL_UPSCALER_FILE}")),
    ("${LTX2_TEXT_REPO}",   "${TEXT_ENCODER_FILE}",       Path("${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors")),
]

for repo_id, filename, destination in jobs:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[ltx23] already present: {destination}")
        purge_cache(repo_id)
        continue
    print(f"[ltx23] downloading: {filename} ...")
    source = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    shutil.copy2(source, destination)
    print(f"[ltx23] installed: {destination}")
    purge_cache(repo_id)
PY

# ── Install custom nodes ──

echo "[ltx23] Installing custom nodes..."

install_or_update_node() {
  local name="$1"
  local repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[ltx23] updating: ${name}"
    git -C "${dest}" pull --ff-only || true
  else
    echo "[ltx23] installing: ${name}"
    git clone "${repo}" "${dest}"
  fi

  if [[ -f "${dest}/requirements.txt" ]]; then
    "${PIP_BIN}" install -r "${dest}/requirements.txt" 2>/dev/null || true
  fi
}

install_or_update_node "ComfyUI-LTXVideo"           "https://github.com/Lightricks/ComfyUI-LTXVideo"
install_or_update_node "ComfyUI-VideoHelperSuite"    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"

# ── Download official I2V workflow ──

WORKFLOW_DIR="${COMFYUI_DIR}/user/default/workflows"
WORKFLOW_DEST="${WORKFLOW_DIR}/ltx23_i2v_official.json"
WORKFLOW_URL="https://raw.githubusercontent.com/Comfy-Org/workflow_templates/refs/heads/main/templates/video_ltx2_3_i2v.json"
mkdir -p "${WORKFLOW_DIR}"
if [[ -f "${WORKFLOW_DEST}" ]]; then
  echo "[ltx23] workflow already present: ${WORKFLOW_DEST}"
else
  echo "[ltx23] downloading official I2V workflow..."
  wget -q --show-progress -O "${WORKFLOW_DEST}" "${WORKFLOW_URL}" || true
  echo "[ltx23] installed: ${WORKFLOW_DEST}"
fi

# ── Restart ComfyUI ──

restart_comfy_services

cat <<EOF

============================================================
[ltx23] LTX 2.3 22B I2V environment is ready!
============================================================

Models installed:
  - checkpoints/ltx-2.3-22b-dev-fp8.safetensors
  - loras/ltx-2.3-22b-distilled-lora-384.safetensors
  - latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
  - text_encoders/gemma_3_12B_it_fp4_mixed.safetensors

Custom nodes:
  - ComfyUI-LTXVideo (Lightricks official)
  - ComfyUI-VideoHelperSuite

Workflow:
  Loaded at: ${WORKFLOW_DEST}
  OR use: Workflow > Browse Templates > Video > "LTX-2.3 I2V"

Pipeline (2-stage):
  Stage 1: Image → LTX 2.3 (fp8 + distilled LoRA) → low-res sample
  Stage 2: Latent Upscale 2x → high-res refine → 1280x720 25fps
  Output: H.264 MP4

Settings:
  - Sampler: euler_ancestral_cfg_pp (stage 1), euler_cfg_pp (stage 2)
  - CFG: 1.0
  - Frames: 121 (~4.8s at 25fps)
  - Resolution: 1280x720
  - I2V denoise: 0.7

EOF
