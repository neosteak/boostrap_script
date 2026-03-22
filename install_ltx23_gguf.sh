#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 22B GGUF Q8 — I2V — RTX 5090 (32GB VRAM)
#
#  Based on: vantagewithai GGUF workflow (Reddit)
#  Models: Q8_0 GGUF + Kijai VAE/LoRA + MachineDelusions I2V Adapter
#  Pipeline: Image → Stage 1 (768×432, 8 steps) → 2x latent upscale (4 steps)
#            → decode → 24fps MP4
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[ltx23-gguf] =========================================="
echo "[ltx23-gguf] LTX 2.3 GGUF Q8 I2V — Installing"
echo "[ltx23-gguf] =========================================="

# ── Venv ──
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# ── Find ComfyUI ──
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[ltx23-gguf] ERROR: ComfyUI not found" && exit 1
echo "[ltx23-gguf] ComfyUI: ${COMFYUI_DIR}"

# ══════════════════════════════════════════════════════════════
# 1. UPDATE COMFYUI + INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════

echo "[ltx23-gguf] Step 1/4: Updating ComfyUI + dependencies..."
git -C "${COMFYUI_DIR}" fetch --all 2>/dev/null || true
git -C "${COMFYUI_DIR}" checkout master 2>/dev/null || true
git -C "${COMFYUI_DIR}" reset --hard origin/master 2>/dev/null || true

"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet --upgrade --force-reinstall \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs"
"${PIP_BIN}" install --quiet -U "huggingface_hub[hf_transfer]"

# ══════════════════════════════════════════════════════════════
# 2. INSTALL CUSTOM NODES
# ══════════════════════════════════════════════════════════════

echo "[ltx23-gguf] Step 2/4: Installing custom nodes..."

install_node() {
  local name="$1" repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" pull --ff-only 2>/dev/null || true
  else
    git clone --depth 1 "${repo}" "${dest}" 2>/dev/null
  fi
  [[ -f "${dest}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${dest}/requirements.txt" 2>/dev/null || true
  echo "[ltx23-gguf] OK: ${name}"
}

# GGUF loader (UnetLoaderGGUF)
install_node "ComfyUI-GGUF"            "https://github.com/city96/ComfyUI-GGUF"
# LTX Video nodes (MultimodalGuider, GuiderParameters, LTXVScheduler, etc.)
install_node "ComfyUI-LTXVideo"        "https://github.com/Lightricks/ComfyUI-LTXVideo"
# RES4LYF (ClownSampler_Beta — res_2s sampler for HQ pipeline)
install_node "RES4LYF"                 "https://github.com/ClownsharkBatwing/RES4LYF"
# ComfyMath (CM_FloatToInt conversion node)
install_node "ComfyMath"               "https://github.com/evanspearman/ComfyMath"
# KJNodes (VAELoaderKJ, ResizeImagesByLongerEdge, etc.)
install_node "ComfyUI-KJNodes"         "https://github.com/kijai/ComfyUI-KJNodes"
# Frame interpolation (FILM VFI for fps upscale)
install_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"

# ══════════════════════════════════════════════════════════════
# 3. DOWNLOAD MODELS
# ══════════════════════════════════════════════════════════════

echo "[ltx23-gguf] Step 3/4: Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/clip" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras/LTX-2" \
  "${COMFYUI_DIR}/models/latent_upscale_models" \
  "${COMFYUI_DIR}/models/upscale_models"

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
        print(f"[ltx23-gguf] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        purge(repo); return
    print(f"[ltx23-gguf] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    print(f"[ltx23-gguf] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
    purge(repo)

# ── 1. FP8 checkpoint (23 GB) — used by CheckpointLoaderSimple ──
dl("Comfy-Org/ltx-2",
   "split_files/checkpoints/ltx-2.3-22b-dev-fp8.safetensors",
   f"{C}/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors")

# ── 2. Text encoder: comfy_gemma_3_12B_it (full, for LTXAVTextEncoderLoader) ──
dl("Comfy-Org/ltx-2",
   "split_files/text_encoders/comfy_gemma_3_12B_it.safetensors",
   f"{C}/models/text_encoders/comfy_gemma_3_12B_it.safetensors")

# ── 3. Distilled LoRA (official, 1.5 GB) — path: ltxv/ltx2/ ──
os.makedirs(f"{C}/models/loras/ltxv/ltx2", exist_ok=True)
dl("Comfy-Org/ltx-2",
   "split_files/loras/ltx-2.3-22b-distilled-lora-384.safetensors",
   f"{C}/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors")

PYEOF

# ══════════════════════════════════════════════════════════════
# 4. COPY WORKFLOW
# ══════════════════════════════════════════════════════════════

echo "[ltx23-gguf] Step 4/4: Installing workflow..."

WORKFLOW_DIR="${COMFYUI_DIR}/user/default/workflows"
mkdir -p "${WORKFLOW_DIR}"

# Copy the GGUF I2V workflow if available next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/ltx23_gguf_i2v_5090.json" ]]; then
  cp "${SCRIPT_DIR}/ltx23_gguf_i2v_5090.json" "${WORKFLOW_DIR}/LTX23_GGUF_I2V.json"
  echo "[ltx23-gguf] workflow copied: LTX23_GGUF_I2V.json"
fi

# ── Restart ComfyUI ──
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[ltx23-gguf] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  LTX 2.3 FP8 I2V — READY (RTX 5090)
  Official Lightricks HQ Workflow
============================================================

Models:
  checkpoints/ltx-2.3-22b-dev-fp8.safetensors              (23G)
  text_encoders/comfy_gemma_3_12B_it.safetensors            (24G)
  loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384           (1.5G)

Custom Nodes:
  ComfyUI-GGUF, ComfyUI-LTXVideo, RES4LYF, ComfyMath,
  ComfyUI-KJNodes, ComfyUI-Frame-Interpolation

Workflow: Official Lightricks Single Stage Distilled+Full
  Distilled: 8 steps, euler_ancestral_cfg_pp, CFG 1
  Full: 15 steps, res_2s (ClownSampler), MultimodalGuider, CFG 3
  Output: single stage @ 24fps

EOF
