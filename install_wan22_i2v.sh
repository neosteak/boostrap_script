#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — Full Setup for Vast AI (ComfyUI template)
#
#  Usage: bash install_wan22_i2v.sh
#
#  This script installs everything needed for Wan 2.2 Image-to-Video
#  generation with upscale + frame interpolation on a Vast AI instance
#  running the ComfyUI template.
#
#  Pipeline: Image → Wan 2.2 14B I2V (fp8, 3-pass + LightX2V LoRA) → RIFE 2x → 4xPurePhoto → 1080p 32fps
#
#  After running this script, load the workflow from:
#    Workflow → Browse Templates → Video → "Wan2.2 14B I2V"
#  or import a custom workflow JSON.
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"
COMFY_BRANCH="${COMFY_BRANCH:-master}"

# ── Model sources (Comfy-Org official repackaged, HuggingFace) ──

WAN22_REPO="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
HIGH_NOISE_FILE="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
LOW_NOISE_FILE="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
TEXT_ENCODER_FILE="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE_FILE="split_files/vae/wan_2.1_vae.safetensors"

# LightX2V acceleration LoRAs (3-KSampler speed workflow)
LORA_HIGH_FILE="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LORA_LOW_FILE="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# Upscale model
UPSCALE_URL="https://huggingface.co/AshtakaOOf/safetensored-upscalers/resolve/main/span/4xPurePhoto-span.safetensors"

# RIFE frame interpolation (auto-download is broken upstream, manual required)
RIFE_URL="https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth"

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

echo "[wan22] Activating /venv/main if present"
if [[ -f "/venv/main/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[wan22] Missing Python at ${PYTHON_BIN}"
  exit 1
fi

COMFYUI_DIR="$(find_comfyui_dir)" || {
  echo "[wan22] Unable to locate ComfyUI. Set COMFYUI_DIR and retry."
  exit 1
}

echo "[wan22] Using ComfyUI directory: ${COMFYUI_DIR}"

# ── Update ComfyUI core ──

echo "[wan22] Updating ComfyUI core"
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
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/upscale_models" \
  "${COMFYUI_DIR}/custom_nodes"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# ── Download Wan 2.2 models ──

echo "[wan22] Downloading Wan 2.2 14B I2V models (this may take a while)..."

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
    ("${WAN22_REPO}", "${HIGH_NOISE_FILE}",    Path("${COMFYUI_DIR}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")),
    ("${WAN22_REPO}", "${LOW_NOISE_FILE}",     Path("${COMFYUI_DIR}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")),
    ("${WAN22_REPO}", "${TEXT_ENCODER_FILE}",   Path("${COMFYUI_DIR}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")),
    ("${WAN22_REPO}", "${VAE_FILE}",            Path("${COMFYUI_DIR}/models/vae/wan_2.1_vae.safetensors")),
    ("${WAN22_REPO}", "${LORA_HIGH_FILE}",     Path("${COMFYUI_DIR}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")),
    ("${WAN22_REPO}", "${LORA_LOW_FILE}",      Path("${COMFYUI_DIR}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")),
]

for repo_id, filename, destination in jobs:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[wan22] already present: {destination}")
        purge_cache(repo_id)
        continue
    print(f"[wan22] downloading: {filename} ...")
    source = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    shutil.copy2(source, destination)
    print(f"[wan22] installed: {destination}")
    purge_cache(repo_id)
PY

# ── Download upscale model ──

UPSCALE_DEST="${COMFYUI_DIR}/models/upscale_models/4xPurePhoto-span.safetensors"
if [[ -f "${UPSCALE_DEST}" && -s "${UPSCALE_DEST}" ]]; then
  echo "[wan22] already present: ${UPSCALE_DEST}"
else
  echo "[wan22] downloading 4xPurePhoto-span upscale model..."
  wget -q --show-progress -O "${UPSCALE_DEST}" "${UPSCALE_URL}"
  echo "[wan22] installed: ${UPSCALE_DEST}"
fi

# ── Install custom nodes ──

echo "[wan22] Installing custom nodes..."

install_or_update_node() {
  local name="$1"
  local repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[wan22] updating: ${name}"
    git -C "${dest}" pull --ff-only || true
  else
    echo "[wan22] installing: ${name}"
    git clone "${repo}" "${dest}"
  fi

  if [[ -f "${dest}/requirements.txt" ]]; then
    "${PIP_BIN}" install -r "${dest}/requirements.txt" 2>/dev/null || true
  fi
}

install_or_update_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
install_or_update_node "ComfyUI-VideoHelperSuite"    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_or_update_node "ComfyUI-KJNodes"             "https://github.com/kijai/ComfyUI-KJNodes"
install_or_update_node "ComfyUI_essentials"          "https://github.com/cubiq/ComfyUI_essentials"

# ── Download RIFE model (auto-download is broken upstream) ──

RIFE_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"
RIFE_DEST="${RIFE_DIR}/rife49.pth"
mkdir -p "${RIFE_DIR}"
if [[ -f "${RIFE_DEST}" && -s "${RIFE_DEST}" ]]; then
  echo "[wan22] already present: ${RIFE_DEST}"
else
  echo "[wan22] downloading RIFE 4.9 model..."
  wget -q --show-progress -O "${RIFE_DEST}" "${RIFE_URL}"
  echo "[wan22] installed: ${RIFE_DEST}"
fi

# ── Restart ComfyUI ──

restart_comfy_services

cat <<EOF

============================================================
[wan22] Wan 2.2 14B I2V environment is ready!
============================================================

Models installed:
  - diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  - diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  - text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  - vae/wan_2.1_vae.safetensors
  - loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
  - loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
  - upscale_models/4xPurePhoto-span.safetensors

Custom nodes:
  - ComfyUI-Frame-Interpolation (RIFE VFI)
  - ComfyUI-VideoHelperSuite
  - ComfyUI-KJNodes (CFGZeroStarAndInit)
  - ComfyUI_essentials

RIFE model:
  - ckpts/rife/rife49.pth

Workflow (3-pass + LightX2V LoRA):
  Import wan22_i2v_workflow_api.json into ComfyUI
  Pipeline: Image → 3-pass KSampler (14 steps) → RIFE 2x (32fps) → 4xPurePhoto → 1080p
  Pass 1: high_noise (no LoRA, CFG 3, steps 0-2) — anchors motion
  Pass 2: high_noise + LightX2V LoRA (CFG 1, steps 2-8) — accelerated generation
  Pass 3: low_noise + LightX2V LoRA (CFG 1, steps 8-14) — detail refinement

EOF
