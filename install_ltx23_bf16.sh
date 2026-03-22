#!/bin/bash
set -e
PREFIX="[ltx23-bf16]"
C="/workspace/ComfyUI"

echo "$PREFIX =========================================="
echo "$PREFIX LTX 2.3 BF16 I2V — A100/H100 Install"
echo "$PREFIX =========================================="

# Activate venv
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate
fi

echo "$PREFIX ComfyUI: $C"

# ─── Step 1: Update ComfyUI ───
echo "$PREFIX Step 1/4: Updating ComfyUI + dependencies..."
cd "$C" && git fetch --all 2>/dev/null && git reset --hard origin/master 2>/dev/null || true
pip install --quiet --upgrade pip 2>/dev/null
pip install --quiet -r requirements.txt 2>/dev/null
pip install --quiet huggingface_hub 2>/dev/null

# ─── Step 2: Install custom nodes ───
echo "$PREFIX Step 2/4: Installing custom nodes..."
cd "$C/custom_nodes"

declare -A NODES=(
    ["ComfyUI-LTXVideo"]="https://github.com/Lightricks/ComfyUI-LTXVideo.git"
    ["RES4LYF"]="https://github.com/ClownsharkBatwing/RES4LYF.git"
    ["ComfyMath"]="https://github.com/evanspearman/ComfyMath.git"
    ["ComfyUI-KJNodes"]="https://github.com/kijai/ComfyUI-KJNodes.git"
    ["ComfyUI-VideoHelperSuite"]="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
)

for name in "${!NODES[@]}"; do
    if [ -d "$name" ]; then
        echo "$PREFIX OK: $name"
    else
        git clone --quiet "${NODES[$name]}" "$name" 2>/dev/null && echo "$PREFIX OK: $name" || echo "$PREFIX WARN: $name clone failed"
    fi
    [ -f "$name/requirements.txt" ] && pip install --quiet -r "$name/requirements.txt" 2>/dev/null || true
done

# ─── Step 3: Download models ───
echo "$PREFIX Step 3/4: Downloading models..."

python -c "
import os, sys
from huggingface_hub import hf_hub_download

C = '$C/models'

def dl(repo, path, dest):
    full = os.path.join(C, dest)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    if os.path.exists(full) and os.path.getsize(full) > 1000:
        print(f'  OK: {os.path.basename(dest)}')
        return
    print(f'  Downloading: {os.path.basename(dest)}...')
    try:
        src = hf_hub_download(repo, path)
        os.symlink(src, full)
        print(f'  OK: {os.path.basename(dest)}')
    except Exception as e:
        print(f'  ERROR: {os.path.basename(dest)} - {e}')

# === Main checkpoint BF16 (46GB) ===
dl('Lightricks/LTX-2.3',
   'ltx-2.3-22b-dev.safetensors',
   'checkpoints/ltx-2.3-22b-dev.safetensors')

# === Distilled LoRA ===
dl('Lightricks/LTX-2.3',
   'ltx-2.3-22b-distilled-lora-384.safetensors',
   'loras/ltx-2.3-22b-distilled-lora-384.safetensors')

# === Spatial upscaler 2x ===
dl('Lightricks/LTX-2.3',
   'ltx-2.3-spatial-upscaler-x2-1.0.safetensors',
   'latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors')

# === Temporal upscaler 2x ===
dl('Lightricks/LTX-2.3',
   'ltx-2.3-temporal-upscaler-x2-1.0.safetensors',
   'latent_upscale_models/ltx-2.3-temporal-upscaler-x2-1.0.safetensors')

# === Text encoder Gemma 3 12B (full BF16 for max quality) ===
dl('Comfy-Org/ltx-2',
   'split_files/text_encoders/gemma_3_12B_it.safetensors',
   'text_encoders/comfy_gemma_3_12B_it.safetensors')

# === Text projection ===
dl('Kijai/LTX2.3_comfy',
   'text_encoders/ltx-2.3_text_projection_bf16.safetensors',
   'text_encoders/ltx-2.3_text_projection_bf16.safetensors')

# === Video VAE ===
dl('Kijai/LTX2.3_comfy',
   'vae/LTX23_video_vae_bf16.safetensors',
   'vae/LTX23_video_vae_bf16_KJ.safetensors')

# === Audio VAE ===
dl('Kijai/LTX2.3_comfy',
   'vae/LTX23_audio_vae_bf16.safetensors',
   'vae/LTX23_audio_vae_bf16_KJ.safetensors')

# === Tiny VAE (preview) ===
dl('Kijai/LTX2.3_comfy',
   'vae/taeltx2_3.safetensors',
   'vae_approx/taeltx2_3.safetensors')

print('All models downloaded')
" || echo "$PREFIX Model download encountered issues"

# ─── Step 4: Fix paths and restart ───
echo "$PREFIX Step 4/4: Creating path symlinks..."

# The workflow references loras at ltxv/ltx2/ path
mkdir -p "/workspace/ComfyUI/models/loras/ltxv/ltx2" 2>/dev/null
ln -sf "/workspace/ComfyUI/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors" \
       "/workspace/ComfyUI/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384.safetensors" 2>/dev/null

# Symlink text encoders to clip folder (some nodes look there)
mkdir -p "/workspace/ComfyUI/models/clip" 2>/dev/null
for f in /workspace/ComfyUI/models/text_encoders/*.safetensors; do
    [ -f "$f" ] && ln -sf "$f" "/workspace/ComfyUI/models/clip/$(basename $f)" 2>/dev/null
done

echo "$PREFIX =========================================="
echo "$PREFIX Installation complete!"
echo "$PREFIX GPU: A100/H100 80GB+ recommended"
echo "$PREFIX Model: BF16 full precision (46GB)"
echo "$PREFIX =========================================="

# Remove provisioning lock if present
rm -f /.provisioning 2>/dev/null
