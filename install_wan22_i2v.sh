#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — RTX 5090 (32GB) — Maximum Quality Pipeline
#
#  FP16 models (no fp8 quantization artifacts)
#  Pipeline: Wan 2.2 I2V (1280x720) → GIMM-VFI → SeedVR2 (1080p) → Export
#
#  Models (~70GB total):
#    - Wan 2.2 I2V fp16 high/low noise (2 × ~28GB)
#    - Text encoder fp8 (~10GB)
#    - VAE (~800MB)
#    - SeedVR2 7B fp8 dit + vae (~9GB)
#
#  Custom nodes:
#    - ComfyUI-KJNodes (CFGZeroStar, ColorMatch, SkipLayerGuidance)
#    - ComfyUI-GIMM-VFI (frame interpolation, better than RIFE)
#    - ComfyUI-FBCNN (artifact removal)
#    - ComfyUI-SeedVR2_VideoUpscaler (diffusion upscale)
#    - ComfyUI-VideoHelperSuite (MP4 export)
#
#  RTX 5090 notes:
#    - Use PyTorch cu130 (CUDA 13.0 for sm_120)
#    - Launch ComfyUI with: --disable-async-offload
#    - sageattn3_blackwell auto-detected by KJNodes
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "================================================================"
echo "  Wan 2.2 I2V — RTX 5090 — Maximum Quality (FP16)"
echo "================================================================"

# ── Activate venv ──
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# ── Find ComfyUI ──
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[ERROR] ComfyUI not found" && exit 1
echo "[info] ComfyUI: ${COMFYUI_DIR}"

# ══════════════════════════════════════════════════════════════
# 1. UPDATE COMFYUI
# ══════════════════════════════════════════════════════════════

echo ""
echo "[1/5] Updating ComfyUI..."
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
# 2. DOWNLOAD WAN 2.2 FP16 MODELS (~67GB)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[2/5] Downloading Wan 2.2 FP16 models (this takes a while)..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export COMFYUI_DIR

"${PYTHON_BIN}" - <<'PYEOF' || { echo "[ERROR] Wan 2.2 model download failed"; exit 1; }
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
        print(f"[OK] {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(repo); return
    print(f"[downloading] {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    print(f"[installed] {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge(repo)

# ── FP16 diffusion models (NOT fp8) ──
dl("split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors")

dl("split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors")

# ── Text encoder (fp8 is fine here, saves space) ──
dl("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   f"{C}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# ── VAE ──
dl("split_files/vae/wan_2.1_vae.safetensors",
   f"{C}/models/vae/wan_2.1_vae.safetensors")

# No LoRA — max quality, 20 steps, no distillation shortcuts

PYEOF

# ══════════════════════════════════════════════════════════════
# 3. INSTALL CUSTOM NODES (6 nodes)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[3/5] Installing custom nodes..."

CN="${COMFYUI_DIR}/custom_nodes"

install_node() {
  local name="$1" url="$2"
  if [[ -d "${CN}/${name}" ]]; then
    echo "[OK] ${name} (updating)"
    git -C "${CN}/${name}" pull --ff-only 2>/dev/null || true
  else
    echo "[cloning] ${name} ..."
    git clone "${url}" "${CN}/${name}"
  fi
  if [[ -f "${CN}/${name}/requirements.txt" ]]; then
    "${PIP_BIN}" install --quiet -r "${CN}/${name}/requirements.txt" 2>/dev/null || true
  fi
}

# KJNodes — CFGZeroStar, ColorMatch, SkipLayerGuidance, WanVideoNAG
install_node "ComfyUI-KJNodes" "https://github.com/kijai/ComfyUI-KJNodes.git"

# GIMM-VFI — frame interpolation (better temporal consistency than RIFE)
install_node "ComfyUI-GIMM-VFI" "https://github.com/kijai/ComfyUI-GIMM-VFI.git"

# FBCNN — JPEG artifact removal
install_node "ComfyUI-FBCNN" "https://github.com/Miosp/ComfyUI-FBCNN.git"

# SeedVR2 — diffusion-based video upscaler
install_node "ComfyUI-SeedVR2_VideoUpscaler" "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"

# VideoHelperSuite — MP4 export
install_node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"

# Frame Interpolation — RIFE backup
install_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"

# ══════════════════════════════════════════════════════════════
# 4. DOWNLOAD SEEDVR2 + RIFE MODELS (~10GB)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[4/5] Downloading SeedVR2 + RIFE models..."

mkdir -p "${COMFYUI_DIR}/models/SEEDVR2"

"${PYTHON_BIN}" - <<'PYEOF' || { echo "[WARNING] SeedVR2 model download failed"; }
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
        print(f"[OK] {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(); return
    print(f"[downloading] {dest.name} ...")
    src = hf_hub_download(repo_id=REPO, filename=filename, token=token)
    shutil.copy2(src, dest)
    print(f"[installed] {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge()

# DIT model — 7B fp8 (upscaler, not the main model)
dl("seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors",
   f"{C}/models/SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors")

# VAE
dl("ema_vae_fp16.safetensors",
   f"{C}/models/SEEDVR2/ema_vae_fp16.safetensors")

PYEOF

# RIFE model (backup interpolator)
RIFE_DIR="${CN}/ComfyUI-Frame-Interpolation/ckpts/rife"
RIFE_FILE="${RIFE_DIR}/rife49.pth"
mkdir -p "${RIFE_DIR}"

if [[ -s "${RIFE_FILE}" ]]; then
  echo "[OK] rife49.pth (exists)"
else
  echo "[downloading] rife49.pth ..."
  wget -q -O "${RIFE_FILE}" \
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth" \
    || curl -sL -o "${RIFE_FILE}" \
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth" \
    || echo "[WARNING] Failed to download rife49.pth"
  [[ -s "${RIFE_FILE}" ]] && echo "[installed] rife49.pth"
fi

# GIMM-VFI model auto-downloads on first use
# FBCNN model auto-downloads on first use

# ══════════════════════════════════════════════════════════════
# 5. RESTART COMFYUI
# ══════════════════════════════════════════════════════════════

echo ""
echo "[5/5] Restarting ComfyUI..."

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[info] ComfyUI restarted"
fi

cat <<'EOF'

================================================================
  Wan 2.2 I2V — RTX 5090 — Maximum Quality — READY
================================================================

  IMPORTANT: Launch ComfyUI with:
    --disable-async-offload

Wan 2.2 Models (FP16, no quantization):
  diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors

SeedVR2 Models:
  SEEDVR2/seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors
  SEEDVR2/ema_vae_fp16.safetensors

Custom Nodes:
  ComfyUI-KJNodes         (CFGZeroStar, ColorMatch, SkipLayerGuidance)
  ComfyUI-GIMM-VFI        (frame interpolation)
  ComfyUI-FBCNN           (artifact removal)
  ComfyUI-SeedVR2_VideoUpscaler  (upscale to 1080p)
  ComfyUI-VideoHelperSuite       (MP4 export)
  ComfyUI-Frame-Interpolation    (RIFE backup)

Pipeline:
  1. Load wan22_i2v_5090_maxquality.json
  2. Set your image + prompt
  3. Generate → CFGZeroStar → 1280x720 @ 16fps
  4. GIMM-VFI → 32fps
  5. SeedVR2 → 1920x1080
  6. ColorMatch + FBCNN → clean
  7. Export MP4 H.264 30fps

EOF
