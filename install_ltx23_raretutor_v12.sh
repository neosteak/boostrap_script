#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 I2V — Raretutor v1.2 Workflow (FP8 Distilled Safetensors)
#  Best quality path: FP8 input-scaled distilled + DualCLIPLoader Safetensors
#  Also downloads GGUF models so both workflow paths work
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[raretutor] =========================================="
echo "[raretutor] LTX 2.3 Raretutor v1.2 — Installing"
echo "[raretutor] =========================================="

# ── Venv ──
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# ── Find ComfyUI ──
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[raretutor] ERROR: ComfyUI not found" && exit 1
echo "[raretutor] ComfyUI: ${COMFYUI_DIR}"
export COMFYUI_DIR

# ══════════════════════════════════════════════════════════════
# 1. UPDATE COMFYUI + INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════

echo "[raretutor] Step 1/5: Updating ComfyUI + dependencies..."
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

# ══════════════════════════════════════════════════════════════
# 2. INSTALL CUSTOM NODES
# ══════════════════════════════════════════════════════════════

echo "[raretutor] Step 2/5: Installing custom nodes..."

install_node() {
  local name="$1" repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" pull --ff-only 2>/dev/null || true
  else
    git clone --depth 1 "${repo}" "${dest}" 2>/dev/null
  fi
  [[ -f "${dest}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${dest}/requirements.txt" 2>/dev/null || true
  echo "[raretutor]   OK: ${name}"
}

# Required by workflow
install_node "ComfyUI-GGUF"              "https://github.com/city96/ComfyUI-GGUF"
install_node "ComfyUI-LTXVideo"          "https://github.com/Lightricks/ComfyUI-LTXVideo"
install_node "ComfyUI-KJNodes"           "https://github.com/kijai/ComfyUI-KJNodes"
install_node "ComfyUI-VideoHelperSuite"  "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
install_node "ComfyMath"                 "https://github.com/evanspearman/ComfyMath"
install_node "ComfyLiterals"             "https://github.com/M1kep/ComfyLiterals"
install_node "ComfyUI-Easy-Use"          "https://github.com/yolain/ComfyUI-Easy-Use"

# ══════════════════════════════════════════════════════════════
# 3. PURGE HF CACHE (free disk before big downloads)
# ══════════════════════════════════════════════════════════════

echo "[raretutor] Step 3/5: Purging HF cache..."
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

HF_TMP="${HF_HOME}/hub/tmp"
[[ -d "${HF_TMP}" ]] && rm -rf "${HF_TMP}" && echo "[raretutor]   Purged HF tmp cache"

# ══════════════════════════════════════════════════════════════
# 4. DOWNLOAD MODELS
# ══════════════════════════════════════════════════════════════

echo "[raretutor] Step 4/5: Downloading models..."

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge_tmp():
    tmp = hf_home / "hub" / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)

def dl(repo, hf_path, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (dest.stat().st_size > 0 or dest.is_symlink()):
        print(f"[raretutor]   OK: {dest.name} (exists)")
        return
    print(f"[raretutor]   downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path)
    if dest.exists(): dest.unlink()
    os.symlink(src, dest)
    print(f"[raretutor]   installed: {dest.name}")

def dl_url(url, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[raretutor]   OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        return
    print(f"[raretutor]   downloading: {dest.name} from URL ...")
    import urllib.request
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[raretutor]   installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")

# ════════════════════════════════════════════
# A. DIFFUSION MODELS
# ════════════════════════════════════════════

# Safetensors path (FP8 distilled — best quality + speed)
# Workflow: CheckpointLoaderSimple → checkpoints/LTXV2/
dl(
    "Kijai/LTX2.3_comfy",
    "diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v2.safetensors",
    f"{C}/models/checkpoints/LTXV2/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v2.safetensors",
)

# GGUF path (Q8_0 distilled — backup path)
# Workflow: UnetLoaderGGUF → diffusion_models/LTXV2/ (or unet/)
dl(
    "unsloth/LTX-2.3-GGUF",
    "distilled/ltx-2.3-22b-distilled-Q8_0.gguf",
    f"{C}/models/diffusion_models/LTXV2/ltx-2.3-22b-distilled-Q8_0.gguf",
)
purge_tmp()

# ════════════════════════════════════════════
# B. TEXT ENCODERS
# ════════════════════════════════════════════

# Safetensors: gemma_3_12B_it_fp8_e4m3fn.safetensors
# (Comfy-Org distributes as fp8_scaled, workflow expects e4m3fn name)
dl(
    "Comfy-Org/ltx-2",
    "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
    f"{C}/models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
)

# Text projection
dl(
    "Kijai/LTX2.3_comfy",
    "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
    f"{C}/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
)

# GGUF: gemma-3-12b-it-Q4_K_M.gguf (backup GGUF path)
dl(
    "lmstudio-community/gemma-3-12b-it-GGUF",
    "gemma-3-12b-it-Q4_K_M.gguf",
    f"{C}/models/text_encoders/gemma-3-12b-it-Q4_K_M.gguf",
)
purge_tmp()

# ════════════════════════════════════════════
# C. VAE
# ════════════════════════════════════════════

# Video VAE
dl(
    "Kijai/LTX2.3_comfy",
    "vae/LTX23_video_vae_bf16.safetensors",
    f"{C}/models/vae/LTX23_video_vae_bf16.safetensors",
)

# Audio VAE
dl(
    "Kijai/LTX2.3_comfy",
    "vae/LTX23_audio_vae_bf16.safetensors",
    f"{C}/models/vae/LTX23_audio_vae_bf16.safetensors",
)

# Tiny VAE (sample preview)
try:
    dl(
        "Kijai/LTX2.3_comfy",
        "vae/taeltx2_3.safetensors",
        f"{C}/models/vae/taeltx2_3.safetensors",
    )
except Exception as e:
    print(f"[raretutor]   fallback tiny VAE: {e}")
    dl_url(
        "https://github.com/madebyollin/taehv/raw/main/safetensors/taeltx2_3.safetensors",
        f"{C}/models/vae/taeltx2_3.safetensors",
    )

# ════════════════════════════════════════════
# D. UPSCALERS
# ════════════════════════════════════════════

dl(
    "Lightricks/LTX-2.3",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    f"{C}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
)
purge_tmp()

# ════════════════════════════════════════════
# E. LORAS
# ════════════════════════════════════════════

# IC-LoRA Detailer
dl(
    "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
    "ltx-2-19b-ic-lora-detailer.safetensors",
    f"{C}/models/loras/LTXV2/ltx-2-19b-ic-lora-detailer.safetensors",
)

# Image2Vid Adapter
dl(
    "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa",
    "LTX-2-Image2Vid-Adapter.safetensors",
    f"{C}/models/loras/LTXV2/LTX-2-Image2Vid-Adapter.safetensors",
)
purge_tmp()

# ════════════════════════════════════════════
# F. CHECKPOINTS (Audio-Video — gated)
# ════════════════════════════════════════════

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    try:
        dl(
            "Lightricks/ltx-av",
            "ltx-av-step-1751000_vocoder_24K.safetensors",
            f"{C}/models/checkpoints/ltx-av-step-1751000_vocoder_24K.safetensors",
        )
    except Exception as e:
        print(f"[raretutor]   WARNING: ltx-av download failed: {e}")
        print(f"[raretutor]   Accept license at https://huggingface.co/Lightricks/ltx-av")
else:
    ckpt = Path(f"{C}/models/checkpoints/ltx-av-step-1751000_vocoder_24K.safetensors")
    if not ckpt.exists():
        print("[raretutor]   WARNING: ltx-av requires HF_TOKEN")
        print("[raretutor]   1. Accept license at https://huggingface.co/Lightricks/ltx-av")
        print("[raretutor]   2. Set HF_TOKEN=hf_xxx and re-run")

purge_tmp()

PYEOF

# ══════════════════════════════════════════════════════════════
# 5. FIX PATHS + SYMLINKS + RESTART
# ══════════════════════════════════════════════════════════════

echo "[raretutor] Step 5/5: Finalizing paths..."

# text_encoders -> clip symlinks (ComfyUI looks in both dirs)
mkdir -p "${COMFYUI_DIR}/models/clip"
for f in gemma_3_12B_it_fp8_scaled.safetensors ltx-2.3_text_projection_bf16.safetensors gemma-3-12b-it-Q4_K_M.gguf; do
  [[ -f "${COMFYUI_DIR}/models/text_encoders/${f}" || -L "${COMFYUI_DIR}/models/text_encoders/${f}" ]] && \
    ln -sf "${COMFYUI_DIR}/models/text_encoders/${f}" \
           "${COMFYUI_DIR}/models/clip/${f}" 2>/dev/null || true
done

# fp8_e4m3fn symlink -> fp8_scaled (workflow expects e4m3fn name)
ln -sf "${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
       "${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it_fp8_e4m3fn.safetensors" 2>/dev/null || true
ln -sf "${COMFYUI_DIR}/models/clip/gemma_3_12B_it_fp8_scaled.safetensors" \
       "${COMFYUI_DIR}/models/clip/gemma_3_12B_it_fp8_e4m3fn.safetensors" 2>/dev/null || true

# GGUF diffusion_models -> unet symlink (UnetLoaderGGUF may look in unet/)
mkdir -p "${COMFYUI_DIR}/models/unet/LTXV2"
ln -sf "${COMFYUI_DIR}/models/diffusion_models/LTXV2/ltx-2.3-22b-distilled-Q8_0.gguf" \
       "${COMFYUI_DIR}/models/unet/LTXV2/ltx-2.3-22b-distilled-Q8_0.gguf" 2>/dev/null || true

# Tiny VAE also accessible from vae_approx/ (some nodes look there)
mkdir -p "${COMFYUI_DIR}/models/vae_approx"
ln -sf "${COMFYUI_DIR}/models/vae/taeltx2_3.safetensors" \
       "${COMFYUI_DIR}/models/vae_approx/taeltx2_3.safetensors" 2>/dev/null || true

# Restart ComfyUI
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[raretutor] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  LTX 2.3 Raretutor v1.2 — READY
============================================================

Safetensors path (recommended):
  checkpoints/LTXV2/
    ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v2    (23G)
  text_encoders/
    gemma_3_12B_it_fp8_scaled (+ e4m3fn symlink)                  (13G)
    ltx-2.3_text_projection_bf16                                   (0.1G)

GGUF path (backup):
  diffusion_models/LTXV2/
    ltx-2.3-22b-distilled-Q8_0.gguf                               (24G)
  text_encoders/
    gemma-3-12b-it-Q4_K_M.gguf                                    (8G)

VAE:
    LTX23_video_vae_bf16                                           (1.4G)
    LTX23_audio_vae_bf16                                           (0.3G)
    taeltx2_3 (tiny preview)                                       (0.01G)

Upscalers:
    ltx-2.3-spatial-upscaler-x2-1.1                                (1G)

LoRAs:
    ltx-2-19b-ic-lora-detailer                                     (4.9G)
    LTX-2-Image2Vid-Adapter                                        (4.9G)

Checkpoints:
    ltx-av-step-1751000_vocoder_24K (requires HF_TOKEN)            (5G)

Custom Nodes:
  ComfyUI-GGUF, ComfyUI-LTXVideo, ComfyUI-KJNodes,
  ComfyUI-VideoHelperSuite, ComfyMath, ComfyLiterals,
  ComfyUI-Easy-Use

NOTE: If ltx-av checkpoint is missing, set HF_TOKEN and re-run.
EOF
