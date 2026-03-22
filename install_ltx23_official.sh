#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 22B — Official ComfyUI Template Setup
#
#  Downloads what the built-in ComfyUI template needs:
#    - 1 checkpoint (FP8, 29.1GB)
#    - 1 LoRA distilled (7.6GB)
#    - 1 spatial upscaler 2x (996MB)
#    - 1 text encoder Gemma 3 FP4 (~6GB)
#
#  Use the built-in template: Menu → Templates → search "LTX-2.3"
#  Available templates: I2V, FLF2V, T2V, IA2V
#
#  Total download: ~44GB
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "================================================================"
echo "  LTX 2.3 — Official Template Setup"
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
# 1. UPDATE COMFYUI (>= 0.16.1 required for LTX 2.3 templates)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[1/3] Updating ComfyUI + packages..."
git -C "${COMFYUI_DIR}" pull --ff-only 2>/dev/null || true
"${PIP_BIN}" install --quiet -r "${COMFYUI_DIR}/requirements.txt" 2>/dev/null
"${PIP_BIN}" install --quiet --upgrade \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs"
"${PIP_BIN}" install --quiet -U "huggingface_hub[hf_transfer]"

# Audio dependencies (needed for LTX 2.3 audio features)
"${PIP_BIN}" install --quiet librosa==0.10.2 soundfile==0.12.1 torchaudio 2>/dev/null || true

FRONTEND_VER=$("${PIP_BIN}" show comfyui-frontend-package 2>/dev/null | grep Version | awk '{print $2}')
echo "[info] Frontend: ${FRONTEND_VER:-unknown}"

# ══════════════════════════════════════════════════════════════
# 2. DOWNLOAD MODELS (~44GB)
# ══════════════════════════════════════════════════════════════

echo ""
echo "[2/3] Downloading models..."

mkdir -p \
  "${COMFYUI_DIR}/models/checkpoints" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/latent_upscale_models"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export COMFYUI_DIR

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ["COMFYUI_DIR"]
token = os.getenv("HF_TOKEN") or None
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge(repo):
    for p in [
        hf_home / "hub" / f"models--{repo.replace('/', '--')}",
        hf_home / "hub" / "tmp",
        Path("/workspace/.cache/huggingface/xet"),
    ]:
        if p.exists(): shutil.rmtree(p, ignore_errors=True)

def dl(repo, hf_path, local_dir, filename):
    dest = Path(f"{C}/{local_dir}/{filename}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [OK] {filename} ({dest.stat().st_size/1e9:.1f}G)")
        purge(repo)
        return
    print(f"  [downloading] {filename} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path, token=token)
    shutil.copy2(src, dest)
    print(f"  [installed] {filename} ({dest.stat().st_size/1e9:.1f}G)")
    purge(repo)

# ── Checkpoint (FP8, 29.1GB) ──
print("\n-- Checkpoint --")
dl("Lightricks/LTX-2.3-fp8",
   "ltx-2.3-22b-dev-fp8.safetensors",
   "models/checkpoints", "ltx-2.3-22b-dev-fp8.safetensors")

# ── Distilled LoRA (7.6GB) ──
print("\n-- Distilled LoRA --")
dl("Lightricks/LTX-2.3",
   "ltx-2.3-22b-distilled-lora-384.safetensors",
   "models/loras", "ltx-2.3-22b-distilled-lora-384.safetensors")

# ── Spatial Upscaler 2x (996MB) ──
print("\n-- Spatial Upscaler --")
dl("Lightricks/LTX-2.3",
   "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
   "models/latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# ── Text Encoder Gemma 3 FP4 (~6GB) ──
print("\n-- Text Encoder (Gemma 3) --")
dl("Comfy-Org/ltx-2",
   "split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
   "models/text_encoders", "gemma_3_12B_it_fp4_mixed.safetensors")

print("\n[done] All 4 models downloaded.")
PYEOF

# ══════════════════════════════════════════════════════════════
# 3. VERIFY + RESTART
# ══════════════════════════════════════════════════════════════

echo ""
echo "[3/3] Verifying..."

FAIL=0
for f in \
  "${COMFYUI_DIR}/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors" \
  "${COMFYUI_DIR}/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors" \
  "${COMFYUI_DIR}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors" \
  "${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"; do
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
  LTX 2.3 — Official Template — READY
================================================================

  Models installed:
  - checkpoints/ltx-2.3-22b-dev-fp8.safetensors       (29GB)
  - loras/ltx-2.3-22b-distilled-lora-384.safetensors   (7.6GB)
  - latent_upscale_models/ltx-2.3-spatial-upscaler-x2  (1GB)
  - text_encoders/gemma_3_12B_it_fp4_mixed.safetensors  (6GB)

  No custom nodes needed — LTX 2.3 is built into ComfyUI >= 0.16.1

  Usage:
  1. Open ComfyUI
  2. Menu → Templates → search "LTX-2.3"
  3. Choose: I2V, FLF2V, T2V, or IA2V
  4. Load your image + write prompt
  5. Click Run

  RTX 5090 notes:
  - Do NOT install xformers (breaks Blackwell)
  - Use FP8 model (not NVFP4)
  - If framing issues: pin ComfyUI commit

EOF
