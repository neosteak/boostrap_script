#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 22B I2V — Full Setup for Vast AI (ComfyUI template)
#
#  Usage: bash install_ltx23_i2v.sh
#
#  Pipeline: Image → LTX 2.3 22B fp8 (triple-stage + distilled LoRA)
#            → Latent 2x → Latent 4x → decode → 30fps MP4
#
#  Uses the CivitAI "Best Quality" triple-stage I2V workflow.
#  All model paths match exactly what the workflow JSON expects.
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"
COMFY_BRANCH="${COMFY_BRANCH:-master}"

# ── Locate ComfyUI ──

find_comfyui_dir() {
  if [[ -n "${COMFYUI_DIR:-}" && -f "${COMFYUI_DIR}/main.py" ]]; then
    echo "${COMFYUI_DIR}"; return 0
  fi
  local pid
  pid="$(pgrep -f "main.py.*18188" | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    local cwd
    cwd="$(readlink -f "/proc/${pid}/cwd" || true)"
    if [[ -n "${cwd}" && -f "${cwd}/main.py" ]]; then
      echo "${cwd}"; return 0
    fi
  fi
  for candidate in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
    if [[ -f "${candidate}/main.py" ]]; then
      echo "${candidate}"; return 0
    fi
  done
  return 1
}

restart_comfy_services() {
  if ! command -v supervisorctl >/dev/null 2>&1; then return 0; fi
  local services
  services="$(supervisorctl status | awk 'tolower($1) ~ /comfy/ {print $1}' || true)"
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

# ── Create model directories (matching CivitAI workflow paths exactly) ──

mkdir -p \
  "${COMFYUI_DIR}/models/checkpoints" \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/loras/ltx2" \
  "${COMFYUI_DIR}/models/text_encoders/gemma3" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/latent_upscale_models" \
  "${COMFYUI_DIR}/custom_nodes"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

# ── Download models ──

echo "[ltx23] Downloading LTX 2.3 22B models..."

"${PYTHON_BIN}" - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

COMFYUI = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge_cache(repo_id):
    repo_cache = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    if repo_cache.exists():
        shutil.rmtree(repo_cache, ignore_errors=True)
    for extra in (hf_home / "hub" / "tmp", Path("/workspace/.cache/huggingface/xet")):
        if extra.exists():
            shutil.rmtree(extra, ignore_errors=True)

def dl(repo, filename, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ltx23] already present: {dest}")
        purge_cache(repo)
        return
    print(f"[ltx23] downloading: {filename} ...")
    src = hf_hub_download(repo_id=repo, filename=filename, token=token)
    shutil.copy2(src, dest)
    print(f"[ltx23] installed: {dest}")
    purge_cache(repo)

# Transformer (fp8, from Kijai) → checkpoints/ (for CheckpointLoaderSimple)
dl("Kijai/LTX2.3_comfy",
   "diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
   f"{COMFYUI}/models/checkpoints/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors")

# Distilled LoRA → loras/ltx2/ (workflow references "ltx2/...")
dl("Lightricks/LTX-2.3",
   "ltx-2.3-22b-distilled-lora-384.safetensors",
   f"{COMFYUI}/models/loras/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors")

# Spatial upscaler → latent_upscale_models/
dl("Lightricks/LTX-2.3",
   "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
   f"{COMFYUI}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# Text encoder (fp4, renamed to match workflow) → text_encoders/gemma3/
dl("Comfy-Org/ltx-2",
   "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
   f"{COMFYUI}/models/text_encoders/gemma3/gemma_3_12B_it_fp8_e4m3fn.safetensors")

# Text projection → text_encoders/
dl("Kijai/LTX2.3_comfy",
   "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
   f"{COMFYUI}/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors")

# Video VAE → vae/
dl("Kijai/LTX2.3_comfy",
   "vae/LTX23_video_vae_bf16.safetensors",
   f"{COMFYUI}/models/vae/LTX23_video_vae_bf16.safetensors")

# Audio VAE → vae/
dl("Kijai/LTX2.3_comfy",
   "vae/LTX23_audio_vae_bf16.safetensors",
   f"{COMFYUI}/models/vae/LTX23_audio_vae_bf16.safetensors")
PY

# ── Symlink transformer into diffusion_models/ too (some nodes look there) ──

TRANSFORMER_SRC="${COMFYUI_DIR}/models/checkpoints/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"
TRANSFORMER_LINK="${COMFYUI_DIR}/models/diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors"
if [[ -f "${TRANSFORMER_SRC}" && ! -e "${TRANSFORMER_LINK}" ]]; then
  ln -s "${TRANSFORMER_SRC}" "${TRANSFORMER_LINK}"
  echo "[ltx23] symlinked transformer → diffusion_models/"
fi

# ── Install custom nodes ──

echo "[ltx23] Installing custom nodes..."

install_or_update_node() {
  local name="$1" repo="$2"
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

install_or_update_node "ComfyUI-LTXVideo"        "https://github.com/Lightricks/ComfyUI-LTXVideo"
install_or_update_node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_or_update_node "ComfyUI-KJNodes"          "https://github.com/kijai/ComfyUI-KJNodes"

# ── Restart ComfyUI ──

restart_comfy_services

cat <<'EOF'

============================================================
[ltx23] LTX 2.3 22B I2V — Ready!
============================================================

Models:
  checkpoints/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors
  loras/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors
  latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  text_encoders/gemma3/gemma_3_12B_it_fp8_e4m3fn.safetensors
  text_encoders/ltx-2.3_text_projection_bf16.safetensors
  vae/LTX23_video_vae_bf16.safetensors
  vae/LTX23_audio_vae_bf16.safetensors

Custom nodes:
  ComfyUI-LTXVideo | ComfyUI-VideoHelperSuite | ComfyUI-KJNodes

Workflow: import LTX2.3-I2V-Best.json (CivitAI triple-stage)

Pipeline:
  Stage 1: 224x320 latent, 9 steps (euler_ancestral_cfg_pp)
  Stage 2: Latent 2x upscale + 4 steps refine
  Stage 3: Latent 2x upscale + 4 steps refine
  Output: 30fps H.264 MP4

EOF
