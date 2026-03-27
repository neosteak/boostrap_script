#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 Three-Stage Workflow
#  Full dev checkpoint + distilled LoRA + 2x/4x spatial upscale chain
#  Matches ltx2.3-3stage-workflow.json
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[ltx23-3stage] =========================================="
echo "[ltx23-3stage] LTX 2.3 Three-Stage Workflow - Installing"
echo "[ltx23-3stage] =========================================="

# Activate venv when present
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# Find ComfyUI
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[ltx23-3stage] ERROR: ComfyUI not found" && exit 1
echo "[ltx23-3stage] ComfyUI: ${COMFYUI_DIR}"
export COMFYUI_DIR

echo "[ltx23-3stage] Step 1/4: Updating ComfyUI + dependencies..."
cd "${COMFYUI_DIR}"
git fetch --all 2>/dev/null || true
git checkout master 2>/dev/null || true
git reset --hard origin/master 2>/dev/null || true

"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet --upgrade --force-reinstall \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs"
"${PIP_BIN}" install --quiet -U "huggingface_hub[hf_transfer]" 2>/dev/null || \
  "${PIP_BIN}" install --quiet -U "huggingface_hub"

echo "[ltx23-3stage] Step 2/4: Installing custom nodes..."

install_node() {
  local name="$1" repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" pull --ff-only 2>/dev/null || true
  else
    git clone --depth 1 "${repo}" "${dest}" 2>/dev/null
  fi
  [[ -f "${dest}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${dest}/requirements.txt" 2>/dev/null || true
  echo "[ltx23-3stage] OK: ${name}"
}

# Workflow dependencies extracted from ltx2.3-3stage-workflow.json
install_node "ComfyUI-LTXVideo" "https://github.com/Lightricks/ComfyUI-LTXVideo"
install_node "ComfyMath" "https://github.com/evanspearman/ComfyMath"
install_node "ComfyUI_FearnworksNodes" "https://github.com/fearnworks/ComfyUI_FearnworksNodes"

echo "[ltx23-3stage] Step 3/4: Downloading models..."

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

"${PYTHON_BIN}" - <<'PYEOF'
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

C = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))


def purge_tmp() -> None:
    tmp = hf_home / "hub" / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)


def dl_link(repo: str, hf_path: str, local: str) -> None:
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (dest.is_symlink() or dest.stat().st_size > 0):
        print(f"[ltx23-3stage] OK: {dest.name}")
        return
    print(f"[ltx23-3stage] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path)
    if dest.exists():
        dest.unlink()
    os.symlink(src, dest)
    print(f"[ltx23-3stage] installed: {dest.name}")


# Main BF16 checkpoint used by CheckpointLoaderSimple and LTXVAudioVAELoader
dl_link(
    "Lightricks/LTX-2.3",
    "ltx-2.3-22b-dev.safetensors",
    f"{C}/models/checkpoints/ltx-2.3-22b-dev.safetensors",
)

# Full Gemma text encoder used by LTXAVTextEncoderLoader
dl_link(
    "Comfy-Org/ltx-2",
    "split_files/text_encoders/gemma_3_12B_it.safetensors",
    f"{C}/models/text_encoders/gemma_3_12B_it.safetensors",
)

# Recommended upscaler asset from LTX 2.3
dl_link(
    "Lightricks/LTX-2.3",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    f"{C}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
)

# Distilled LoRA path expected by the workflow: models/loras/ltx2/...
dl_link(
    "Lightricks/LTX-2.3",
    "ltx-2.3-22b-distilled-lora-384.safetensors",
    f"{C}/models/loras/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors",
)

# Optional camera motion LoRA path expected by the workflow: models/loras/camera/...
dl_link(
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
    "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    f"{C}/models/loras/camera/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
)

purge_tmp()
PYEOF

echo "[ltx23-3stage] Step 4/4: Finalizing paths..."

# Some ComfyUI loaders look in clip/ for text encoders
mkdir -p "${COMFYUI_DIR}/models/clip"
ln -sf "${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it.safetensors" \
       "${COMFYUI_DIR}/models/clip/gemma_3_12B_it.safetensors" 2>/dev/null || true

# Workflow JSON still references x2-1.0; alias it to the newer x2-1.1 asset
mkdir -p "${COMFYUI_DIR}/models/latent_upscale_models"
ln -sf "${COMFYUI_DIR}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
       "${COMFYUI_DIR}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" 2>/dev/null || true

# Convenience aliases so the assets are also visible at the loras root
mkdir -p "${COMFYUI_DIR}/models/loras"
ln -sf "${COMFYUI_DIR}/models/loras/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors" \
       "${COMFYUI_DIR}/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors" 2>/dev/null || true
ln -sf "${COMFYUI_DIR}/models/loras/camera/ltx-2-19b-lora-camera-control-dolly-out.safetensors" \
       "${COMFYUI_DIR}/models/loras/ltx-2-19b-lora-camera-control-dolly-out.safetensors" 2>/dev/null || true

if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[ltx23-3stage] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  LTX 2.3 Three-Stage Workflow - READY
============================================================

Models:
  checkpoints/
    ltx-2.3-22b-dev.safetensors
  text_encoders/
    gemma_3_12B_it.safetensors
  latent_upscale_models/
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors
    ltx-2.3-spatial-upscaler-x2-1.0.safetensors -> x2-1.1 alias
  loras/ltx2/
    ltx-2.3-22b-distilled-lora-384.safetensors
  loras/camera/
    ltx-2-19b-lora-camera-control-dolly-out.safetensors

Custom Nodes:
  ComfyUI-LTXVideo, ComfyMath, ComfyUI_FearnworksNodes

Notes:
  This matches ltx2.3-3stage-workflow.json as-is.
  The workflow expects the distilled LoRA in loras/ltx2/ and camera LoRA in loras/camera/.

EOF
