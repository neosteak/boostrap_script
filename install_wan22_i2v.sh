#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — CivitAI 131-Review Workflow — Plug & Play for Vast AI
#
#  Source: https://civitai.com/models/1898210/wan-22-14b-i2v-workflow (v1.1)
#  131 positive reviews, 0 negative
#  Pipeline: GGUF Q4 + Lightning LoRA (4 steps) → RIFE 4x → RealESRGAN 2x
#  No subgraphs — compatible all ComfyUI versions
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[wan22] =========================================="
echo "[wan22] Wan 2.2 14B I2V — CivitAI 131-Review"
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

echo "[wan22] Step 1/4: Updating ComfyUI..."
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
# 2. DOWNLOAD MODELS (8 files)
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 2/4: Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/upscale_models" \
  "${COMFYUI_DIR}/custom_nodes"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export COMFYUI_DIR

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
import urllib.request

C = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge(repo):
    for p in [
        hf_home / "hub" / f"models--{repo.replace('/', '--')}",
        hf_home / "hub" / "tmp",
        Path("/workspace/.cache/huggingface/xet"),
    ]:
        if p.exists(): shutil.rmtree(p, ignore_errors=True)

def dl(repo, hf_path, local):
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

def wget(url, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[wan22] OK: {dest.name}")
        return
    print(f"[wan22] downloading: {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"[wan22] installed: {dest.name}")

# === GGUF Q4 diffusion models ===
dl("QuantStack/Wan2.2-I2V-A14B-GGUF",
   "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf",
   f"{C}/models/diffusion_models/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf")

dl("QuantStack/Wan2.2-I2V-A14B-GGUF",
   "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf",
   f"{C}/models/diffusion_models/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf")

# === Lightning LoRA (Seko V1 for I2V) ===
dl("lightx2v/Wan2.2-Lightning",
   "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
   f"{C}/models/loras/wan2.2_Lightning_lora_high_noise_model.safetensors")

dl("lightx2v/Wan2.2-Lightning",
   "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
   f"{C}/models/loras/wan2.2_Lightning_lora_low_noise_model.safetensors")

# === Text encoder ===
dl("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
   "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   f"{C}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# === VAE ===
dl("Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
   "split_files/vae/wan_2.1_vae.safetensors",
   f"{C}/models/vae/wan_2.1_vae.safetensors")

# === RealESRGAN x2 upscaler ===
wget("https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth",
     f"{C}/models/upscale_models/RealESRGAN_x2.pth")

PYEOF

# ══════════════════════════════════════════════════════════════
# 3. INSTALL CUSTOM NODES + RIFE MODEL
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 3/4: Installing custom nodes..."

install_node() {
  local name="$1" repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -d "${dest}/.git" ]]; then
    echo "[wan22] updating: ${name}"
    git -C "${dest}" pull --ff-only 2>/dev/null || true
  else
    echo "[wan22] installing: ${name}"
    git clone --depth 1 "${repo}" "${dest}" 2>/dev/null
  fi
  [[ -f "${dest}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${dest}/requirements.txt" 2>/dev/null || true
}

install_node "ComfyUI-GGUF"               "https://github.com/city96/ComfyUI-GGUF"
install_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
install_node "ComfyUI-VideoHelperSuite"    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_node "ComfyUI-KJNodes"             "https://github.com/kijai/ComfyUI-KJNodes"
install_node "ComfyUI-bleh"               "https://github.com/blepping/ComfyUI-bleh"
install_node "ComfyUI-Easy-Use"           "https://github.com/yolain/ComfyUI-Easy-Use"
install_node "rgthree-comfy"              "https://github.com/rgthree/rgthree-comfy"

# RIFE model (rife47 — used by this workflow)
RIFE_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"
RIFE_DEST="${RIFE_DIR}/rife47.pth"
mkdir -p "${RIFE_DIR}"
if [[ -f "${RIFE_DEST}" && -s "${RIFE_DEST}" ]]; then
  echo "[wan22] OK: rife47.pth"
else
  echo "[wan22] downloading: rife47.pth ..."
  wget -q -O "${RIFE_DEST}" \
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife47.pth" 2>/dev/null || true
fi

# ══════════════════════════════════════════════════════════════
# 4. RESTART COMFYUI
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 4/4: Restarting ComfyUI..."

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[wan22] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  Wan 2.2 14B I2V — CivitAI 131-Review — READY
============================================================

Models (8 files):
  diffusion_models/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf
  diffusion_models/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf
  loras/wan2.2_Lightning_lora_high_noise_model.safetensors
  loras/wan2.2_Lightning_lora_low_noise_model.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors
  upscale_models/RealESRGAN_x2.pth
  rife/rife47.pth

Custom nodes (7):
  ComfyUI-GGUF, Frame-Interpolation, VideoHelperSuite,
  KJNodes, bleh, Easy-Use, rgthree-comfy

Import "WAN 2.2 14b i2v 1.1.json" into ComfyUI.

EOF
