#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Wan 2.2 14B I2V — Option A (Max Quality) — Plug & Play for Vast AI
#
#  Pipeline: Image → Wan 2.2 14B (fp8, 20 steps, NO LoRA)
#            → RIFE 2x → 4xPurePhoto → lanczos 1920×1080 → H.264 MP4
#
#  Settings: DDIM(H) + Heun(L), beta scheduler, CFG 5/2.5, shift 8
#  Output:   1920×1080, 32fps, ~5 seconds
#  GPU:      RTX 5090 32GB (or any 24GB+ with offloading)
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[wan22] =========================================="
echo "[wan22] Wan 2.2 14B I2V (Option A — Max Quality)"
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
# 2. CREATE DIRECTORIES
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 2/5: Creating directories..."
mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/upscale_models" \
  "${COMFYUI_DIR}/custom_nodes" \
  "${COMFYUI_DIR}/user/default/workflows"

# ══════════════════════════════════════════════════════════════
# 3. DOWNLOAD MODELS
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 3/5: Downloading models..."

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

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

REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Diffusion models (high + low noise, fp8)
dl(REPO, "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")

dl(REPO, "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")

# Text encoder (umt5 fp8)
dl(REPO, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
   f"{C}/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")

# VAE
dl(REPO, "split_files/vae/wan_2.1_vae.safetensors",
   f"{C}/models/vae/wan_2.1_vae.safetensors")

# Upscale model (4xPurePhoto-span)
import urllib.request
upscale_dest = Path(f"{C}/models/upscale_models/4xPurePhoto-span.safetensors")
if upscale_dest.exists() and upscale_dest.stat().st_size > 0:
    print(f"[wan22] OK: {upscale_dest.name}")
else:
    print(f"[wan22] downloading: {upscale_dest.name} ...")
    urllib.request.urlretrieve(
        "https://huggingface.co/AshtakaOOf/safetensored-upscalers/resolve/main/span/4xPurePhoto-span.safetensors",
        str(upscale_dest))
    print(f"[wan22] installed: {upscale_dest.name}")

PYEOF

# ══════════════════════════════════════════════════════════════
# 4. INSTALL CUSTOM NODES + RIFE MODEL
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 4/5: Installing custom nodes..."

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

install_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
install_node "ComfyUI-VideoHelperSuite"    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"

# RIFE model (auto-download is broken upstream)
RIFE_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife"
RIFE_DEST="${RIFE_DIR}/rife49.pth"
mkdir -p "${RIFE_DIR}"
if [[ -f "${RIFE_DEST}" && -s "${RIFE_DEST}" ]]; then
  echo "[wan22] OK: rife49.pth"
else
  echo "[wan22] downloading: rife49.pth ..."
  wget -q --show-progress -O "${RIFE_DEST}" \
    "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/rife49.pth"
fi

# ══════════════════════════════════════════════════════════════
# 5. DOWNLOAD WORKFLOW + CONFIGURE COMFYUI
# ══════════════════════════════════════════════════════════════

echo "[wan22] Step 5/5: Installing workflow..."

# The workflow JSON will be created by the companion script
# For now, just place a marker
cat > "${COMFYUI_DIR}/user/default/workflows/README.txt" <<'WFEOF'
Import the wan22_i2v_optionA.json workflow file into ComfyUI.
Drag & drop the JSON file into the ComfyUI browser interface.
WFEOF

# Configure ComfyUI launch flags for RTX 5090
# Check if extra_model_paths.yaml exists, create if not
EXTRA_ARGS_FILE="${COMFYUI_DIR}/extra_model_paths.yaml"
if [[ ! -f "${EXTRA_ARGS_FILE}" ]]; then
  touch "${EXTRA_ARGS_FILE}"
fi

# Restart ComfyUI with correct flags
if command -v supervisorctl >/dev/null 2>&1; then
  # Check if comfyui service exists and patch launch args
  COMFY_CONF=$(find /etc/supervisor -name "comfyui*" -type f 2>/dev/null | head -1)
  if [[ -n "${COMFY_CONF}" ]] && ! grep -q "disable-async-offload" "${COMFY_CONF}" 2>/dev/null; then
    # Add --disable-async-offload for RTX 5090 compatibility
    sed -i 's|main.py|main.py --disable-async-offload|' "${COMFY_CONF}" 2>/dev/null || true
    echo "[wan22] Added --disable-async-offload to ComfyUI launch config"
  fi
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[wan22] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  Wan 2.2 14B I2V — Option A (Max Quality) — READY
============================================================

Models installed:
  diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors
  upscale_models/4xPurePhoto-span.safetensors

Custom nodes:
  ComfyUI-Frame-Interpolation (RIFE VFI)
  ComfyUI-VideoHelperSuite (VHS_VideoCombine)

RIFE model:
  ckpts/rife/rife49.pth

Pipeline (Option A — 20 steps, NO LoRA):
  Image → WanImageToVideo (768x432, 81 frames)
  High pass: DDIM, beta, 10 steps, CFG 5.0, shift 8
  Low pass:  Heun, beta, 10 steps, CFG 2.5, shift 8
  → VAEDecode → RIFE 2x (32fps)
  → 4xPurePhoto upscale → lanczos 1920x1080
  → H.264 MP4

Import wan22_i2v_optionA.json into ComfyUI to start.

EOF
