#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  LTX 2.3 I2V — Dev Full-Steps Workflow
#  Kijai FP8 input-scaled + GGUF Q4 distilled + Two-stage upscale
#  RTX 5090 (32GB VRAM)
# ============================================================================

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"

echo "[ltx23-dev] =========================================="
echo "[ltx23-dev] LTX 2.3 Dev Full-Steps — Installing"
echo "[ltx23-dev] =========================================="

# ── Venv ──
[[ -f "/venv/main/bin/activate" ]] && source /venv/main/bin/activate

# ── Find ComfyUI ──
COMFYUI_DIR=""
for d in /workspace/ComfyUI /opt/ComfyUI /workspace/comfyui /root/ComfyUI; do
  [[ -f "${d}/main.py" ]] && COMFYUI_DIR="${d}" && break
done
[[ -z "${COMFYUI_DIR}" ]] && echo "[ltx23-dev] ERROR: ComfyUI not found" && exit 1
echo "[ltx23-dev] ComfyUI: ${COMFYUI_DIR}"
export COMFYUI_DIR

# ══════════════════════════════════════════════════════════════
# 1. UPDATE COMFYUI + INSTALL DEPENDENCIES
# ══════════════════════════════════════════════════════════════

echo "[ltx23-dev] Step 1/4: Updating ComfyUI + dependencies..."
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

echo "[ltx23-dev] Step 2/4: Installing custom nodes..."

install_node() {
  local name="$1" repo="$2"
  local dest="${COMFYUI_DIR}/custom_nodes/${name}"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" pull --ff-only 2>/dev/null || true
  else
    git clone --depth 1 "${repo}" "${dest}" 2>/dev/null
  fi
  [[ -f "${dest}/requirements.txt" ]] && "${PIP_BIN}" install --quiet -r "${dest}/requirements.txt" 2>/dev/null || true
  echo "[ltx23-dev] OK: ${name}"
}

# GGUF loader (UnetLoaderGGUF, DualCLIPLoaderGGUF)
install_node "ComfyUI-GGUF"            "https://github.com/city96/ComfyUI-GGUF"
# LTX Video nodes (MultimodalGuider, GuiderParameters, LTXVScheduler, etc.)
install_node "ComfyUI-LTXVideo"        "https://github.com/Lightricks/ComfyUI-LTXVideo"
# KJNodes (VAELoaderKJ, SimpleCalculatorKJ, ImageResizeKJv2, etc.)
install_node "ComfyUI-KJNodes"         "https://github.com/kijai/ComfyUI-KJNodes"
# VideoHelperSuite (VHS_VideoCombine)
install_node "ComfyUI-VideoHelperSuite" "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
# rgthree nodes (Fast Groups Bypasser, Power Lora Loader)
install_node "rgthree-comfy"           "https://github.com/rgthree/rgthree-comfy"
# ComfyMath (ComfyMathExpression)
install_node "ComfyMath"               "https://github.com/evanspearman/ComfyMath"
# RES4LYF (ClownSampler_Beta for res_2s sampler)
install_node "RES4LYF"                "https://github.com/ClownsharkBatwing/RES4LYF"
# Frame interpolation (FILM VFI)
install_node "ComfyUI-Frame-Interpolation" "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
# easy-use nodes (easy showAnything)
install_node "ComfyUI-Easy-Use"        "https://github.com/yolain/ComfyUI-Easy-Use"

# ══════════════════════════════════════════════════════════════
# 3. DOWNLOAD MODELS
# ══════════════════════════════════════════════════════════════

echo "[ltx23-dev] Step 3/4: Downloading models..."

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"

"${PYTHON_BIN}" - <<'PYEOF'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

C = os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")
hf_home = Path(os.getenv("HF_HOME", "/workspace/.hf_home"))

def purge(repo):
    for p in [
        hf_home / "hub" / f"models--{repo.replace('/', '--')}",
        hf_home / "hub" / "tmp",
    ]:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

def dl(repo, hf_path, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ltx23-dev] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        return
    print(f"[ltx23-dev] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path)
    try:
        shutil.move(src, dest)
    except Exception:
        shutil.copy2(src, dest)
        try:
            Path(src).unlink(missing_ok=True)
        except Exception:
            pass
    print(f"[ltx23-dev] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")

def dl_link(repo, hf_path, local):
    """Symlink instead of copy to save disk space"""
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (dest.stat().st_size > 0 or dest.is_symlink()):
        print(f"[ltx23-dev] OK: {dest.name} (symlinked)")
        return
    print(f"[ltx23-dev] downloading: {dest.name} ...")
    src = hf_hub_download(repo_id=repo, filename=hf_path)
    if dest.exists(): dest.unlink()
    os.symlink(src, dest)
    print(f"[ltx23-dev] installed: {dest.name} (symlinked)")

def dl_url(url, local):
    dest = Path(local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ltx23-dev] OK: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")
        return
    print(f"[ltx23-dev] downloading: {dest.name} from direct URL ...")
    import urllib.request
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"[ltx23-dev] installed: {dest.name} ({dest.stat().st_size/1e9:.1f}G)")

# ── 1. Kijai FP8 scaled dev model (23 GB) ──
dl("Kijai/LTX2.3_comfy",
   "diffusion_models/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors",
   f"{C}/models/diffusion_models/LTXVideo/v2/ltx-2.3-22b-dev_transformer_only_fp8_scaled.safetensors")

# ── 2. GGUF Q4 distilled model (16 GB) ──
dl_link("quantstack/LTX-2.3-GGUF",
   "LTX-2.3-distilled/LTX-2.3-distilled-Q4_K_S.gguf",
   f"{C}/models/diffusion_models/LTXvideo/LTX-2/quantstack/LTX-2.3-distilled-Q4_K_S.gguf")

# ── 3. Text encoder: Gemma 3 12B FP8 scaled ──
dl_link("Comfy-Org/ltx-2",
   "split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
   f"{C}/models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors")

# ── 4. Text projection (Kijai) ──
dl("Kijai/LTX2.3_comfy",
   "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
   f"{C}/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors")

# ── 5. Gemma GGUF Q2_K (for GGUF branch) ──
dl_link("Diangle/gemma-3-12b-it-Q2_K-GGUF",
   "gemma-3-12b-it-q2_k.gguf",
   f"{C}/models/text_encoders/gemma-3-12b-it-Q2_K.gguf")

# ── 6. Video VAE (Kijai) ──
dl("Kijai/LTX2.3_comfy",
   "vae/LTX23_video_vae_bf16.safetensors",
   f"{C}/models/vae/LTX23_video_vae_bf16_KJ.safetensors")

# ── 7. Audio VAE (Kijai) ──
dl("Kijai/LTX2.3_comfy",
   "vae/LTX23_audio_vae_bf16.safetensors",
   f"{C}/models/vae/LTX23_audio_vae_bf16_KJ.safetensors")

# ── 8. Tiny VAE (preview) ──
try:
    dl("Kijai/LTX2.3_comfy",
       "vae/taeltx2_3.safetensors",
       f"{C}/models/vae_approx/taeltx2_3.safetensors")
except Exception as e:
    print(f"[ltx23-dev] fallback tiny VAE download after HF failure: {e}")
    dl_url(
        "https://github.com/madebyollin/taehv/raw/main/safetensors/taeltx2_3.safetensors",
        f"{C}/models/vae_approx/taeltx2_3.safetensors",
    )

# Safe to purge because every Kijai file above is copied, not symlinked.
purge("Kijai/LTX2.3_comfy")

# ── 9. Spatial upscaler x2 ──
dl("Lightricks/LTX-2.3",
   "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
   f"{C}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

# ── 10. Temporal upscaler x2 ──
dl("Lightricks/LTX-2.3",
   "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
   f"{C}/models/latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors")

# ── 11. Distilled LoRA ──
dl("Lightricks/LTX-2.3",
   "ltx-2.3-22b-distilled-lora-384.safetensors",
   f"{C}/models/loras/LTX/LTX-2/ltx-2.3-22b-distilled-lora-384.safetensors")

# Safe to purge because every Lightricks file above is copied, not symlinked.
purge("Lightricks/LTX-2.3")

PYEOF

# ══════════════════════════════════════════════════════════════
# 4. FIX PATHS + RESTART
# ══════════════════════════════════════════════════════════════

echo "[ltx23-dev] Step 4/4: Finalizing..."

# Create symlinks for text_encoders -> clip (ComfyUI looks in both)
mkdir -p "${COMFYUI_DIR}/models/clip"
ln -sf "${COMFYUI_DIR}/models/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
       "${COMFYUI_DIR}/models/clip/gemma_3_12B_it_fp8_scaled.safetensors" 2>/dev/null || true
ln -sf "${COMFYUI_DIR}/models/text_encoders/ltx-2.3_text_projection_bf16.safetensors" \
       "${COMFYUI_DIR}/models/clip/ltx-2.3_text_projection_bf16.safetensors" 2>/dev/null || true
ln -sf "${COMFYUI_DIR}/models/text_encoders/gemma-3-12b-it-Q2_K.gguf" \
       "${COMFYUI_DIR}/models/clip/gemma-3-12b-it-Q2_K.gguf" 2>/dev/null || true

# Fix Tiny VAE path — VAELoader looks in vae/ not vae_approx/
mkdir -p "${COMFYUI_DIR}/models/vae/vae_approx"
ln -sf "${COMFYUI_DIR}/models/vae_approx/taeltx2_3.safetensors" \
       "${COMFYUI_DIR}/models/vae/vae_approx/taeltx2_3.safetensors" 2>/dev/null || true

# Restart ComfyUI
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart comfyui 2>/dev/null || true
  echo "[ltx23-dev] ComfyUI restarted"
fi

cat <<'EOF'

============================================================
  LTX 2.3 Dev Full-Steps I2V — READY (RTX 5090)
============================================================

Models:
  diffusion_models/LTXVideo/v2/
    ltx-2.3-22b-dev_transformer_only_fp8_scaled       (23G, Kijai input-scaled)
  diffusion_models/LTXvideo/LTX-2/quantstack/
    LTX-2.3-distilled-Q4_K_S.gguf                     (6.5G)
  text_encoders/
    gemma_3_12B_it_fp8_scaled.safetensors              (12G)
    ltx-2.3_text_projection_bf16.safetensors           (0.1G)
    gemma-3-12b-it-Q2_K.gguf                           (4.5G)
  vae/
    LTX23_video_vae_bf16_KJ.safetensors                (1.4G)
    LTX23_audio_vae_bf16_KJ.safetensors                (0.3G)
  vae_approx/
    taeltx2_3.safetensors                              (0.01G)
  latent_upscale_models/
    ltx-2.3-spatial-upscaler-x2-1.0                    (1G)
    ltx-2.3-temporal-upscaler-x2-1.0                   (1G)
  loras/LTX/LTX-2/
    ltx-2.3-22b-distilled-lora-384                     (1.5G)

Custom Nodes:
  ComfyUI-GGUF, ComfyUI-LTXVideo, RES4LYF, ComfyUI-KJNodes,
  ComfyUI-VideoHelperSuite, rgthree-comfy, ComfyMath,
  ComfyUI-Frame-Interpolation, ComfyUI-Easy-Use

EOF
