#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V + SeedVR2 Upscale + RIFE Interpolation
#
#  Pipeline: Wan 2.2 I2V (832x480) → SeedVR2 upscale (1920x1080) → RIFE (30fps)
#
#  Models:
#    - Wan 2.2 I2V fp8 (Comfy-Org) — 6 files (~27GB)
#    - SeedVR2 7B fp8 (numz) — dit + vae (~9GB)
#    - RIFE 4.9 — frame interpolation (~100MB)
#
#  Custom nodes:
#    - ComfyUI-SeedVR2_VideoUpscaler (upscale)
#    - ComfyUI-Frame-Interpolation (RIFE)
#    - ComfyUI-VideoHelperSuite (MP4 export)
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[wan22] ======================================================"
echo "[wan22] Wan 2.2 I2V + SeedVR2 + RIFE — Full 1080p Pipeline"
echo "[wan22] ======================================================"

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
# 1. UPDATE COMFYUI
# ══════════════════════════════════════════════════════════════

echo ""
echo "[wan22] Step 1/5: Updating ComfyUI..."
git -C "${COMFYUI_DIR}" fetch --all 2>/dev/null || true
git -C "${COMFYUI_DIR}" checkout master 2>/dev/null || true
git -C "${COMFYUI_DIR}" pull --ff-only 2>/dev/null || true

"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet -U \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs" \
  "huggingface_hub[hf_transfer]"

# ══════════════════════════════════════════════════════════════
# 2. DOWNLOAD WAN 2.2 MODELS (6 files ~27GB)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[wan22] Step 2/5: Downloading Wan 2.2 models..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export COMFYUI_DIR

"${PYTHON_BIN}" - <<'PYEOF' || { echo "[wan22] WARNING: Wan 2.2 model download failed"; exit 1; }
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ["COMFYUI_DIR"]
REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge(repo=REPO):
    for p in [
        hf_home / "hub" / f"models--{repo.replace('/', '--')}",
        hf_home / "hub" / "tmp",
        Path("/workspace/.cache/huggingface/xet"),
    ]:
        if p.exists(): shutil.rmtree(p, ignore_errors=True)

def dl(hf_path, local, repo=REPO):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[wan22] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(repo); return
    print(f"[wan22] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    print(f"[wan22] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge(repo)

# Diffusion models (fp8)
dl("split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")

dl("split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")

# Text encoder (fp8)
dl("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   f"{C}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# VAE
dl("split_files/vae/wan_2.1_vae.safetensors",
   f"{C}/models/vae/wan_2.1_vae.safetensors")

# LoRA — LightX2V 4-step acceleration
dl("split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
   f"{C}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")

dl("split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
   f"{C}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")

PYEOF

# ══════════════════════════════════════════════════════════════
# 3. INSTALL CUSTOM NODES
# ══════════════════════════════════════════════════════════════

echo ""
echo "[wan22] Step 3/5: Installing custom nodes..."

CN="${COMFYUI_DIR}/custom_nodes"

# SeedVR2 Video Upscaler
if [[ -d "${CN}/ComfyUI-SeedVR2_VideoUpscaler" ]]; then
  echo "[wan22] OK: ComfyUI-SeedVR2_VideoUpscaler (exists)"
  git -C "${CN}/ComfyUI-SeedVR2_VideoUpscaler" pull --ff-only 2>/dev/null || true
else
  echo "[wan22] cloning: ComfyUI-SeedVR2_VideoUpscaler ..."
  git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "${CN}/ComfyUI-SeedVR2_VideoUpscaler"
fi
if [[ -f "${CN}/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt" ]]; then
  "${PIP_BIN}" install --quiet -r "${CN}/ComfyUI-SeedVR2_VideoUpscaler/requirements.txt" 2>/dev/null || true
fi

# Frame Interpolation (RIFE)
if [[ -d "${CN}/ComfyUI-Frame-Interpolation" ]]; then
  echo "[wan22] OK: ComfyUI-Frame-Interpolation (exists)"
  git -C "${CN}/ComfyUI-Frame-Interpolation" pull --ff-only 2>/dev/null || true
else
  echo "[wan22] cloning: ComfyUI-Frame-Interpolation ..."
  git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git "${CN}/ComfyUI-Frame-Interpolation"
fi
if [[ -f "${CN}/ComfyUI-Frame-Interpolation/requirements.txt" ]]; then
  "${PIP_BIN}" install --quiet -r "${CN}/ComfyUI-Frame-Interpolation/requirements.txt" 2>/dev/null || true
fi

# VideoHelperSuite (MP4 export)
if [[ -d "${CN}/ComfyUI-VideoHelperSuite" ]]; then
  echo "[wan22] OK: ComfyUI-VideoHelperSuite (exists)"
  git -C "${CN}/ComfyUI-VideoHelperSuite" pull --ff-only 2>/dev/null || true
else
  echo "[wan22] cloning: ComfyUI-VideoHelperSuite ..."
  git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "${CN}/ComfyUI-VideoHelperSuite"
fi
if [[ -f "${CN}/ComfyUI-VideoHelperSuite/requirements.txt" ]]; then
  "${PIP_BIN}" install --quiet -r "${CN}/ComfyUI-VideoHelperSuite/requirements.txt" 2>/dev/null || true
fi

# ══════════════════════════════════════════════════════════════
# 4. DOWNLOAD SEEDVR2 MODELS (~9GB)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[wan22] Step 4/5: Downloading SeedVR2 models..."

mkdir -p "${COMFYUI_DIR}/models/SEEDVR2"

"${PYTHON_BIN}" - <<'PYEOF' || { echo "[wan22] WARNING: SeedVR2 model download failed"; }
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ["COMFYUI_DIR"]
REPO = "numz/SeedVR2_comfyUI"
token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge():
    for p in [
        hf_home / "hub" / f"models--{REPO.replace('/', '--')}",
        hf_home / "hub" / "tmp",
    ]:
        if p.exists(): shutil.rmtree(p, ignore_errors=True)

def dl(filename, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[wan22] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(); return
    print(f"[wan22] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=REPO, filename=filename, token=token)
    shutil.copy2(src, dest)
    print(f"[wan22] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge()

# DIT model — 7B fp8 (best quality for 24GB VRAM)
dl("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
   f"{C}/models/SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors")

# VAE
dl("ema_vae_fp16.safetensors",
   f"{C}/models/SEEDVR2/ema_vae_fp16.safetensors")

PYEOF

# ══════════════════════════════════════════════════════════════
# 5. DOWNLOAD RIFE MODEL
# ══════════════════════════════════════════════════════════════

echo ""
echo "[wan22] Step 5/5: Downloading RIFE model..."

RIFE_DIR="${CN}/ComfyUI-Frame-Interpolation/ckpts/rife"
RIFE_FILE="${RIFE_DIR}/rife49.pth"

mkdir -p "${RIFE_DIR}"

if [[ -s "${RIFE_FILE}" ]]; then
  echo "[wan22] OK: rife49.pth (exists)"
else
  echo "[wan22] downloading: rife49.pth ..."
  # GitHub links are 404 (known bug), use HuggingFace mirror
  wget -q -O "${RIFE_FILE}" \
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth" \
    || curl -sL -o "${RIFE_FILE}" \
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth" \
    || { echo "[wan22] ERROR: Failed to download rife49.pth"; }
  [[ -s "${RIFE_FILE}" ]] && echo "[wan22] installed: rife49.pth" || echo "[wan22] ERROR: rife49.pth is empty"
fi

# ══════════════════════════════════════════════════════════════
# 6. RESTART COMFYUI
# ══════════════════════════════════════════════════════════════

echo ""
echo "[wan22] Restarting ComfyUI..."

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[wan22] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  Wan 2.2 I2V + SeedVR2 + RIFE — 1080p Pipeline READY
============================================================

Wan 2.2 Models (6 files):
  diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors
  loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
  loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

SeedVR2 Models (2 files):
  SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors
  SEEDVR2/ema_vae_fp16.safetensors

Custom Nodes:
  ComfyUI-SeedVR2_VideoUpscaler (upscale to 1080p)
  ComfyUI-Frame-Interpolation   (RIFE 30fps)
  ComfyUI-VideoHelperSuite      (MP4 export)

Pipeline:
  1. Load template: Workflow > Browse Templates > Video > Wan 2.2 14B I2V
  2. Set width: 832, height: 480 (16:9)
  3. Generate I2V clip (~71s with LoRA)
  4. SeedVR2 upscale → 1920x1080
  5. RIFE interpolation → 30fps
  6. Export MP4 via VideoHelperSuite

EOF
