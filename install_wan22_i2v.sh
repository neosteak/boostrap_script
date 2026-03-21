#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — Base Quality (No LoRA, 20 steps)
#
#  Source: Official Comfy-Org template parameters
#  Steps: 20 (10 high + 10 low), CFG 3.5, euler/simple, shift 5.0
#  Models: fp8 from Comfy-Org/Wan_2.2_ComfyUI_Repackaged
#  Custom nodes: ComfyUI-VideoHelperSuite (MP4 output)
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[wan22] =========================================="
echo "[wan22] Wan 2.2 14B I2V — Base Quality Setup"
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
# 1. UPDATE COMFYUI
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 1/2: Updating ComfyUI..."
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
# 2. DOWNLOAD MODELS (4 files only — no LoRA needed)
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 2/2: Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export COMFYUI_DIR

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
    print(f"[wan22] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge()

# High noise diffusion model (fp8)
dl("split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")

# Low noise diffusion model (fp8)
dl("split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")

# Text encoder (fp8)
dl("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   f"{C}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# VAE
dl("split_files/vae/wan_2.1_vae.safetensors",
   f"{C}/models/vae/wan_2.1_vae.safetensors")

PYEOF

# ══════════════════════════════════════════════════════════════
# 3. INSTALL VIDEOHELPER SUITE (MP4 output)
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 3/4: Installing custom nodes..."

VHS_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-VideoHelperSuite"
if [[ -d "${VHS_DIR}/.git" ]]; then
  echo "[wan22] updating: ComfyUI-VideoHelperSuite"
  git -C "${VHS_DIR}" pull --ff-only 2>/dev/null || true
else
  echo "[wan22] installing: ComfyUI-VideoHelperSuite"
  git clone --depth 1 "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" "${VHS_DIR}" 2>/dev/null
fi
[[ -f "${VHS_DIR}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${VHS_DIR}/requirements.txt" 2>/dev/null || true

# Frame interpolation (RIFE for 32fps)
FI_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation"
if [[ -d "${FI_DIR}/.git" ]]; then
  echo "[wan22] updating: ComfyUI-Frame-Interpolation"
  git -C "${FI_DIR}" pull --ff-only 2>/dev/null || true
else
  echo "[wan22] installing: ComfyUI-Frame-Interpolation"
  git clone --depth 1 "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation" "${FI_DIR}" 2>/dev/null
fi
[[ -f "${FI_DIR}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${FI_DIR}/requirements.txt" 2>/dev/null || true

# RIFE model (auto-download broken — all GitHub URLs return 404, use HuggingFace mirrors)
RIFE_DIR="${FI_DIR}/ckpts/rife"
mkdir -p "${RIFE_DIR}"
if [[ ! -s "${RIFE_DIR}/rife49.pth" ]]; then
  echo "[wan22] downloading: rife49.pth ..."
  RIFE_URLS=(
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth"
    "https://huggingface.co/marduk191/rife/resolve/main/rife49.pth"
    "https://huggingface.co/MachineDelusions/RIFE/resolve/main/rife49.pth"
  )
  for url in "${RIFE_URLS[@]}"; do
    echo "[wan22]   trying: ${url}"
    wget -q -O "${RIFE_DIR}/rife49.pth" "${url}" 2>/dev/null && break || rm -f "${RIFE_DIR}/rife49.pth"
  done
  [[ -s "${RIFE_DIR}/rife49.pth" ]] && echo "[wan22] OK: rife49.pth" || echo "[wan22] WARNING: rife49.pth download failed from all mirrors"
fi

# ══════════════════════════════════════════════════════════════
# 4. RESTART COMFYUI
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 4/4: Restarting..."

# ── Restart ComfyUI ──
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[wan22] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  Wan 2.2 14B I2V — Base Quality — READY
============================================================

Models (4 files, all fp8):
  diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors

Custom nodes: NONE (all native ComfyUI)

Official parameters (no LoRA, max quality):
  Steps: 20 (split at 10)
  CFG: 3.5
  Sampler: euler
  Scheduler: simple
  Shift: 5.0
  Resolution: 640x640 (adjustable)
  Frames: 81

Usage:
  1. Import wan22_base_quality.json into ComfyUI
  2. Load an image, write a prompt, generate

EOF
