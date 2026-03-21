#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 22B I2V — Triple-Stage — Plug & Play for Vast AI
#
#  Workflow: Pastebin triple-stage (Reddit r/StableDiffusion)
#  Pipeline: Image → Stage 1 (224x320, 9 steps) → 2x upscale (4 steps)
#            → 4x upscale (4 steps) → decode → 30fps MP4
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[ltx23] =========================================="
echo "[ltx23] LTX 2.3 Triple-Stage I2V — Installing"
echo "[ltx23] =========================================="

# ── Venv ──
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# ── Find ComfyUI ──
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[ltx23] ERROR: ComfyUI not found" && exit 1
echo "[ltx23] ComfyUI: ${COMFYUI_DIR}"

# ══════════════════════════════════════════════════════════════
# 1. UPDATE COMFYUI + FRONTEND (subgraph support required)
# ══════════════════════════════════════════════════════════════

echo "[ltx23] Step 1/4: Updating ComfyUI + frontend..."
git -C "${COMFYUI_DIR}" fetch --all 2>/dev/null || true
git -C "${COMFYUI_DIR}" pull --ff-only 2>/dev/null || true

"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet --upgrade --force-reinstall \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs"
"${PIP_BIN}" install --quiet -U "huggingface_hub[hf_transfer]"

FRONTEND_VER=$("${PIP_BIN}" show comfyui-frontend-package 2>/dev/null | grep Version | awk '{print $2}')
echo "[ltx23] Frontend version: ${FRONTEND_VER:-unknown}"

# ══════════════════════════════════════════════════════════════
# 2. DOWNLOAD MODELS (exact names matching the workflow)
# ══════════════════════════════════════════════════════════════

echo "[ltx23] Step 2/4: Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/checkpoints" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/latent_upscale_models" \
  "${COMFYUI_DIR}/custom_nodes"

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
        print(f"[ltx23] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(repo); return
    print(f"[ltx23] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    print(f"[ltx23] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge(repo)

# Workflow uses: ltx-2.3-22b-dev-fp8.safetensors (CheckpointLoaderSimple + LTXAVTextEncoderLoader + LTXVAudioVAELoader)
dl("Lightricks/LTX-2.3-fp8",
   "ltx-2.3-22b-dev-fp8.safetensors",
   f"{C}/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors")

# Workflow uses: ltx-2.3-22b-distilled-lora-384.safetensors (LoraLoaderModelOnly, strength 0.5)
dl("Lightricks/LTX-2.3",
   "ltx-2.3-22b-distilled-lora-384.safetensors",
   f"{C}/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors")

# Workflow uses: ltx-2.3-spatial-upscaler-x2-1.0.safetensors (LatentUpscaleModelLoader, used 2x)
dl("Lightricks/LTX-2.3",
   "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
   f"{C}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# Workflow uses: gemma_3_12B_it_fp4_mixed.safetensors (LTXAVTextEncoderLoader)
dl("Comfy-Org/ltx-2",
   "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
   f"{C}/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors")

PYEOF

# ══════════════════════════════════════════════════════════════
# 3. INSTALL CUSTOM NODES
# ══════════════════════════════════════════════════════════════

echo "[ltx23] Step 3/4: Installing custom nodes..."

install_node() {
  local name="$1" repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" pull --ff-only 2>/dev/null || true
  else
    git clone --depth 1 "${repo}" "${dest}" 2>/dev/null
  fi
  [[ -f "${dest}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${dest}/requirements.txt" 2>/dev/null || true
  echo "[ltx23] OK: ${name}"
}

install_node "ComfyUI-LTXVideo"        "https://github.com/Lightricks/ComfyUI-LTXVideo"
install_node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_node "ComfyUI-KJNodes"          "https://github.com/kijai/ComfyUI-KJNodes"

# ══════════════════════════════════════════════════════════════
# 4. DOWNLOAD TRIPLE-STAGE WORKFLOW
# ══════════════════════════════════════════════════════════════

echo "[ltx23] Step 4/4: Downloading workflow..."

WORKFLOW_DIR="${COMFYUI_DIR}/user/default/workflows"
mkdir -p "${WORKFLOW_DIR}"
wget -q -O "${WORKFLOW_DIR}/LTX23_TripleStage_I2V.json" \
  "https://pastebin.com/raw/A5wR4PVG" 2>/dev/null

# Patch workflow: replace full model with fp8 + fp4 text encoder
if [[ -f "${WORKFLOW_DIR}/LTX23_TripleStage_I2V.json" ]]; then
  sed -i \
    -e 's/ltx-2\.3-22b-dev\.safetensors/ltx-2.3-22b-dev-fp8.safetensors/g' \
    -e 's/gemma_3_12B_it\.safetensors/gemma_3_12B_it_fp4_mixed.safetensors/g' \
    "${WORKFLOW_DIR}/LTX23_TripleStage_I2V.json"
  echo "[ltx23] workflow patched for fp8 + fp4 models"
  echo "[ltx23] saved: workflows/LTX23_TripleStage_I2V.json"
else
  echo "[ltx23] WARNING: workflow download failed, import manually from pastebin.com/A5wR4PVG"
fi

# ── Restart ComfyUI ──
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[ltx23] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  LTX 2.3 Triple-Stage I2V — READY
============================================================

Models:
  checkpoints/ltx-2.3-22b-dev-fp8.safetensors          (22G)
  loras/ltx-2.3-22b-distilled-lora-384.safetensors      (7G)
  latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0 (1G)
  text_encoders/gemma_3_12B_it_fp4_mixed.safetensors     (9G)

Nodes: ComfyUI-LTXVideo, VideoHelperSuite, KJNodes

Workflow: workflows/LTX23_TripleStage_I2V.json
  Stage 1: 224x320 → 9 steps (euler_ancestral_cfg_pp)
  Stage 2: 2x upscale → 4 steps (euler_cfg_pp)
  Stage 3: 2x upscale → 4 steps (euler_cfg_pp)
  Output: 30fps MP4

EOF
