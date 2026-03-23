#!/bin/bash
set -e

LOG_PREFIX="[tts-install]"
C="/workspace/ComfyUI"

echo "$LOG_PREFIX =========================================="
echo "$LOG_PREFIX TTS Test Suite — Qwen3-TTS + Chatterbox"
echo "$LOG_PREFIX =========================================="

# Activate venv
if [ -f /venv/main/bin/activate ]; then
    . /venv/main/bin/activate
fi

# ==========================================
# Step 1: Update ComfyUI
# ==========================================
echo "$LOG_PREFIX Step 1/4: Updating ComfyUI..."
cd "$C"
git fetch --all 2>/dev/null || true
git reset --hard origin/master 2>/dev/null || true
pip install --quiet -r requirements.txt 2>/dev/null || true

# ==========================================
# Step 2: Install custom nodes
# ==========================================
echo "$LOG_PREFIX Step 2/4: Installing custom nodes..."

cd "$C/custom_nodes"

# Qwen3-TTS (flybirdxx)
if [ ! -d "ComfyUI-Qwen-TTS" ]; then
    git clone https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git
    echo "$LOG_PREFIX OK: ComfyUI-Qwen-TTS"
else
    echo "$LOG_PREFIX SKIP: ComfyUI-Qwen-TTS (exists)"
fi

# Chatterbox (filliptm - supports 23 languages, multilingual, turbo)
if [ ! -d "ComfyUI_Fill-ChatterBox" ]; then
    git clone https://github.com/filliptm/ComfyUI_Fill-ChatterBox.git
    echo "$LOG_PREFIX OK: ComfyUI_Fill-ChatterBox"
else
    echo "$LOG_PREFIX SKIP: ComfyUI_Fill-ChatterBox (exists)"
fi

# VideoHelperSuite (for audio preview/save)
if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    echo "$LOG_PREFIX OK: ComfyUI-VideoHelperSuite"
else
    echo "$LOG_PREFIX SKIP: ComfyUI-VideoHelperSuite (exists)"
fi

# ==========================================
# Step 3: Install Python dependencies
# ==========================================
echo "$LOG_PREFIX Step 3/4: Installing Python dependencies..."

# Qwen3-TTS deps
pip install --quiet qwen-tts 2>/dev/null || pip install --quiet torch torchaudio transformers librosa accelerate soundfile 2>/dev/null

# Chatterbox deps
if [ -f "$C/custom_nodes/ComfyUI_Fill-ChatterBox/requirements.txt" ]; then
    pip install --quiet -r "$C/custom_nodes/ComfyUI_Fill-ChatterBox/requirements.txt" 2>/dev/null || true
fi

# Optional speedups
pip install --quiet sage-attn 2>/dev/null || true
pip install --quiet flash-attn --no-build-isolation 2>/dev/null || true

echo "$LOG_PREFIX Dependencies installed"

# ==========================================
# Step 4: Pre-download models
# ==========================================
echo "$LOG_PREFIX Step 4/4: Pre-downloading models..."

# Qwen3-TTS models (auto-download on first use, but we can pre-cache)
python -c "
from huggingface_hub import snapshot_download
import os

qwen_dir = '$C/models/qwen-tts'
os.makedirs(qwen_dir, exist_ok=True)

# VoiceClone model (Base) - for cloning reference voices
print('Downloading Qwen3-TTS-12Hz-1.7B-Base (voice cloning)...')
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir=f'{qwen_dir}/Qwen/Qwen3-TTS-12Hz-1.7B-Base')
print('OK: Qwen3-TTS Base')

# Tokenizer
print('Downloading Qwen3-TTS-Tokenizer-12Hz...')
snapshot_download('Qwen/Qwen3-TTS-Tokenizer-12Hz', local_dir=f'{qwen_dir}/Qwen/Qwen3-TTS-Tokenizer-12Hz')
print('OK: Qwen3-TTS Tokenizer')

# VoiceDesign model - for creating voices from text descriptions
print('Downloading Qwen3-TTS-12Hz-1.7B-VoiceDesign...')
snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign', local_dir=f'{qwen_dir}/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign')
print('OK: Qwen3-TTS VoiceDesign')

print('All Qwen3-TTS models ready')
" 2>&1 || echo "$LOG_PREFIX [WARN] Some Qwen3-TTS models failed to download"

# Chatterbox models auto-download on first use to ComfyUI/models/chatterbox/
echo "$LOG_PREFIX Chatterbox models will auto-download on first use"

# ==========================================
# Done
# ==========================================
echo "$LOG_PREFIX =========================================="
echo "$LOG_PREFIX DONE! Restart ComfyUI and load workflows."
echo "$LOG_PREFIX "
echo "$LOG_PREFIX Available:"
echo "$LOG_PREFIX - Qwen3-TTS: VoiceClone + VoiceDesign nodes"
echo "$LOG_PREFIX - Chatterbox: TTS + Voice Conversion nodes"
echo "$LOG_PREFIX =========================================="

# Remove provisioning flag if exists
rm -f /.provisioning 2>/dev/null || true

# Restart ComfyUI
supervisorctl restart comfyui 2>/dev/null || true
