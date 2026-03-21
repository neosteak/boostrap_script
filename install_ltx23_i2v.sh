#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 22B I2V — Full Setup for Vast AI (ComfyUI template)
#
#  Usage: bash install_ltx23_i2v.sh
#
#  Pipeline: Image → LTX 2.3 22B (fp8 transformer + separate VAE, triple-stage)
#            → Latent 2x → Latent 4x → decode → 30fps SaveVideo
#
#  Uses the CivitAI "Best Quality" triple-stage workflow (19 positive reviews)
#  with separate transformer/VAE loading for lower VRAM usage.
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"
COMFY_BRANCH="${COMFY_BRANCH:-master}"

# ── Model sources ──

LTX23_REPO="Lightricks/LTX-2.3"
KIJAI_REPO="Kijai/LTX2.3_comfy"
COMFY_LTX2_REPO="Comfy-Org/ltx-2"

# Transformer only (fp8, from Kijai repack) — loaded via CheckpointLoaderSimple
TRANSFORMER_FILE="diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"

# Separate VAEs (bf16) — loaded via VAELoaderKJ / LTXVAudioVAELoader
VIDEO_VAE_URL="https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3_video_vae_bf16.safetensors"
AUDIO_VAE_URL="https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3_audio_vae_bf16.safetensors"

# Text encoder (fp8) — better prompt quality than fp4
TEXT_PROJ_URL="https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3_text_projection_bf16.safetensors"

# Distilled LoRA + spatial upscaler
DISTILLED_LORA_FILE="ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSCALER_FILE="ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

# Text encoder
TEXT_ENCODER_FILE="split_files/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors"

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
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/latent_upscale_models" \
  "${COMFYUI_DIR}/custom_nodes"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# ── Download LTX 2.3 models (HuggingFace) ──

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
    ("${KIJAI_REPO}",         "${TRANSFORMER_FILE}",       Path("${COMFYUI_DIR}/models/diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors")),
    ("${LTX23_REPO}",        "${DISTILLED_LORA_FILE}",    Path("${COMFYUI_DIR}/models/loras/${DISTILLED_LORA_FILE}")),
    ("${LTX23_REPO}",        "${SPATIAL_UPSCALER_FILE}",  Path("${COMFYUI_DIR}/models/latent_upscale_models/${SPATIAL_UPSCALER_FILE}")),
    ("${COMFY_LTX2_REPO}",   "${TEXT_ENCODER_FILE}",      Path("${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors")),
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

# ── Download separate VAE + text projection (wget, direct URLs) ──

download_if_missing() {
  local dest="$1"
  local url="$2"
  local name
  name="$(basename "${dest}")"
  if [[ -f "${dest}" && -s "${dest}" ]]; then
    echo "[ltx23] already present: ${dest}"
  else
    echo "[ltx23] downloading ${name}..."
    wget -q --show-progress -O "${dest}" "${url}"
    echo "[ltx23] installed: ${dest}"
  fi
}

download_if_missing "${COMFYUI_DIR}/models/vae/LTX23_video_vae_bf16.safetensors" "${VIDEO_VAE_URL}"
download_if_missing "${COMFYUI_DIR}/models/vae/LTX23_audio_vae_bf16.safetensors" "${AUDIO_VAE_URL}"
download_if_missing "${COMFYUI_DIR}/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors" "${TEXT_PROJ_URL}"

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
install_or_update_node "ComfyUI-KJNodes"             "https://github.com/kijai/ComfyUI-KJNodes"

# ── Copy triple-stage I2V workflow ──

WORKFLOW_DIR="${COMFYUI_DIR}/user/default/workflows"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${WORKFLOW_DIR}"

# CivitAI "Best Quality" workflow (already adapted for our models)
if [[ -f "${SCRIPT_DIR}/civitai_workflow/LTX2.3-I2V-Best.json" ]]; then
  cp "${SCRIPT_DIR}/civitai_workflow/LTX2.3-I2V-Best.json" "${WORKFLOW_DIR}/ltx23_i2v_best.json"
  echo "[ltx23] workflow installed: ${WORKFLOW_DIR}/ltx23_i2v_best.json"
elif [[ -f "${SCRIPT_DIR}/ltx23_triple_stage_workflow.json" ]]; then
  cp "${SCRIPT_DIR}/ltx23_triple_stage_workflow.json" "${WORKFLOW_DIR}/ltx23_i2v_triple_stage.json"
  echo "[ltx23] workflow installed: ${WORKFLOW_DIR}/ltx23_i2v_triple_stage.json"
else
  echo "[ltx23] WARNING: no workflow JSON found next to install script"
fi

# ── Restart ComfyUI ──

restart_comfy_services

cat <<EOF

============================================================
[ltx23] LTX 2.3 22B I2V environment is ready!
============================================================

Models installed:
  Transformer:
    - diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors
  VAE (separate, bf16):
    - vae/LTX23_video_vae_bf16.safetensors
    - vae/LTX23_audio_vae_bf16.safetensors
  Text encoder:
    - text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors
    - text_encoders/ltx-2.3_text_projection_bf16.safetensors
  LoRA + Upscaler:
    - loras/ltx-2.3-22b-distilled-lora-384.safetensors
    - latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors

Custom nodes:
  - ComfyUI-LTXVideo (Lightricks official)
  - ComfyUI-VideoHelperSuite
  - ComfyUI-KJNodes (VAELoaderKJ)

Workflow:
  CivitAI "Best Quality" triple-stage I2V (19 positive reviews, 0 negative)
  Import: ltx23_i2v_best.json

Pipeline (3-stage):
  Stage 1: Image → LTX 2.3 fp8 + distilled LoRA (0.5) → low-res (224x320, 9 steps)
  Stage 2: Latent Upscale 2x → refine (4 steps)
  Stage 3: Latent Upscale 2x → refine (4 steps)
  Output: 30fps H.264 MP4

EOF
