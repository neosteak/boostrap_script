#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — Official Template + Post-Processing
#
#  Downloads exactly what the built-in ComfyUI template needs:
#    - 2 diffusion models (fp8_scaled, 13.3GB each)
#    - 2 LoRA (4steps acceleration, 1.1GB each)
#    - 1 text encoder (fp8, ~10GB)
#    - 1 VAE (~800MB)
#
#  Post-processing (interpolation + upscale):
#    - ComfyUI-Frame-Interpolation (RIFE v4.7, 16fps→32fps)
#    - 4x-UltraSharp upscaler (~67MB)
#
#  Total: ~40GB + ~67MB
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "================================================================"
echo "  Wan 2.2 I2V — Official Template Setup"
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
# 1. UPDATE COMFYUI + INSTALL HF TRANSFER
# ══════════════════════════════════════════════════════════════

echo ""
echo "[1/4] Updating ComfyUI..."
git -C "${COMFYUI_DIR}" pull --ff-only 2>/dev/null || true
"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet -U "huggingface_hub[hf_transfer]"

# ══════════════════════════════════════════════════════════════
# 2. DOWNLOAD MODELS (~40GB)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[2/4] Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export COMFYUI_DIR

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ["COMFYUI_DIR"]
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

def dl(hf_path, local_dir, filename):
    dest = Path(f"{C}/{local_dir}/{filename}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [OK] {filename} ({dest.stat().st_size/1e9:.1f}G)")
        purge()
        return
    print(f"  [downloading] {filename} ...")
    src = hf_hub_download(repo_id=REPO, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    print(f"  [installed] {filename} ({dest.stat().st_size/1e9:.1f}G)")
    purge()

# ── Diffusion models (fp8_scaled) ──
print("\n-- Diffusion models --")
dl("split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
   "models/diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")

dl("split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
   "models/diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")

# ── LoRA (4steps acceleration) ──
print("\n-- LoRA --")
dl("split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
   "models/loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors")

dl("split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
   "models/loras", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")

# ── Text encoder ──
print("\n-- Text encoder --")
dl("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   "models/text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# ── VAE ──
print("\n-- VAE --")
dl("split_files/vae/wan_2.1_vae.safetensors",
   "models/vae", "wan_2.1_vae.safetensors")

print("\n[done] All 6 models downloaded.")
PYEOF

# ══════════════════════════════════════════════════════════════
# 3. INSTALL CUSTOM NODES + UPSCALER
# ══════════════════════════════════════════════════════════════

echo ""
echo "[3/4] Installing Frame Interpolation + Upscaler..."

# ── ComfyUI-Frame-Interpolation (RIFE) ──
FI_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation"
if [[ -d "${FI_DIR}" ]]; then
  echo "  [OK] ComfyUI-Frame-Interpolation exists, pulling updates..."
  git -C "${FI_DIR}" pull --ff-only 2>/dev/null || true
else
  echo "  [installing] ComfyUI-Frame-Interpolation..."
  git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git "${FI_DIR}"
fi
"${PIP_BIN}" install --quiet -r "${FI_DIR}/requirements.txt" 2>/dev/null || true

# ── RIFE model (rife49.pth) ──
RIFE_DIR="${FI_DIR}/ckpts/rife"
mkdir -p "${RIFE_DIR}"
if [[ -s "${RIFE_DIR}/rife49.pth" ]]; then
  echo "  [OK] rife49.pth"
else
  echo "  [downloading] rife49.pth..."
  wget -q -O "${RIFE_DIR}/rife49.pth" \
    "https://huggingface.co/Fannovel16/ComfyUI-Frame-Interpolation/resolve/main/ckpts/rife/rife49.pth" \
    || echo "  [WARN] rife49.pth download failed — will auto-download on first use"
fi

# ── 4x-UltraSharp upscaler ──
mkdir -p "${COMFYUI_DIR}/models/upscale_models"
if [[ -s "${COMFYUI_DIR}/models/upscale_models/4x-UltraSharp.pth" ]]; then
  echo "  [OK] 4x-UltraSharp.pth"
else
  echo "  [downloading] 4x-UltraSharp.pth (~67MB)..."
  wget -q -O "${COMFYUI_DIR}/models/upscale_models/4x-UltraSharp.pth" \
    "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
fi

# ══════════════════════════════════════════════════════════════
# 4. VERIFY + RESTART
# ══════════════════════════════════════════════════════════════

echo ""
echo "[4/4] Verifying..."

FAIL=0
for f in \
  "${COMFYUI_DIR}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
  "${COMFYUI_DIR}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
  "${COMFYUI_DIR}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" \
  "${COMFYUI_DIR}/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" \
  "${COMFYUI_DIR}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
  "${COMFYUI_DIR}/models/vae/wan_2.1_vae.safetensors" \
  "${COMFYUI_DIR}/models/upscale_models/4x-UltraSharp.pth"; do
  if [[ -s "$f" ]]; then
    echo "  [OK] $(basename "$f")"
  else
    echo "  [MISSING] $(basename "$f")"
    FAIL=1
  fi
done

if [[ $FAIL -eq 1 ]]; then
  echo ""
  echo "[ERROR] Some models are missing. Re-run the script."
  exit 1
fi

# Restart ComfyUI
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[info] ComfyUI restarted"
fi

cat <<'EOF'

================================================================
  Wan 2.2 I2V — Official Template — READY
================================================================

  All models + post-processing installed:
  - 6 Wan 2.2 models (diffusion, LoRA, text encoder, VAE)
  - RIFE frame interpolation (16fps -> 32fps)
  - 4x-UltraSharp upscaler (848x480 -> 1920x1080)

  Load the workflow JSON and click Run.

EOF
