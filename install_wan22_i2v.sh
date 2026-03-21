#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — Official Comfy-Org Template — Plug & Play for Vast AI
#
#  Workflow: Built-in template (Workflow → Browse Templates → Video → "Wan2.2 14B I2V")
#  Models: All from Comfy-Org/Wan_2.2_ComfyUI_Repackaged (fp8)
#  Custom nodes: NONE required (all native ComfyUI)
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[wan22] =========================================="
echo "[wan22] Wan 2.2 14B I2V — Official Template Setup"
echo "[wan22] =========================================="

# ── Activate venv ──
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# ── Find ComfyUI ──
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[wan22] ERROR: ComfyUI not found" && exit 1
echo "[wan22] ComfyUI: ${COMFYUI_DIR}"

# ══════════════════════════════════════════════════════════════
# 1. UPDATE COMFYUI + TEMPLATES
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 1/3: Updating ComfyUI..."
git -C "${COMFYUI_DIR}" fetch --all 2>/dev/null || true
git -C "${COMFYUI_DIR}" checkout master 2>/dev/null || true
git -C "${COMFYUI_DIR}" pull --ff-only 2>/dev/null || true

"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet -U \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs" \
  "huggingface_hub[hf_transfer]"

FRONTEND_VER=$("${PIP_BIN}" show comfyui-frontend-package 2>/dev/null | grep Version | awk '{print $2}')
echo "[wan22] Frontend: ${FRONTEND_VER:-unknown}"

# ══════════════════════════════════════════════════════════════
# 2. DOWNLOAD MODELS
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 2/3: Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge():
    for p in [
        hf_home / "hub" / f"models--{REPO.replace('/', '--')}",
        hf_home / "hub" / "tmp",
        Path("/workspace/.cache/huggingface/xet"),
    ]:
        if p.exists(): shutil.rmtree(p, ignore_errors=True)

def dl(hf_path, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[wan22] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(); return
    print(f"[wan22] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=REPO, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    sz = dest.stat().st_size / 1e9
    print(f"[wan22] installed: {dest.name} ({sz:.1f}G)")
    purge()

# Diffusion models (fp8)
dl("split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")

dl("split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")

# LightX2V LoRA (used by official template)
dl("split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
   f"{C}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")

dl("split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
   f"{C}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")

# Text encoder (fp8)
dl("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   f"{C}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# VAE
dl("split_files/vae/wan_2.1_vae.safetensors",
   f"{C}/models/vae/wan_2.1_vae.safetensors")

PYEOF

# ══════════════════════════════════════════════════════════════
# 3. RESTART COMFYUI
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 3/3: Restarting ComfyUI..."

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[wan22] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  Wan 2.2 14B I2V — Official Template — READY
============================================================

Models (all from Comfy-Org, fp8):
  diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
  loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors

Custom nodes: NONE (all native ComfyUI)

Usage:
  1. Open ComfyUI in your browser
  2. Workflow → Browse Templates → Video → "Wan2.2 14B I2V"
  3. Load an image, write a prompt, generate

EOF
