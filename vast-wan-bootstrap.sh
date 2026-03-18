#!/usr/bin/env bash
set -Eeuo pipefail

# Vast bootstrap for a minimal, native ComfyUI Wan 2.2 I2V setup.
#
# Verified sources on 2026-03-18:
# - Vast ComfyUI Serverless docs:
#   https://docs.vast.ai/documentation/serverless/comfy-ui
# - Vast advanced template setup:
#   https://docs.vast.ai/documentation/templates/advanced-setup
# - Official ComfyUI Wan 2.2 docs:
#   https://docs.comfy.org/tutorials/video/wan/wan2_2
# - Official ComfyUI repackaged Wan 2.2 model repo:
#   https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged
# - Official LightX2V Wan 2.2 Lightning repo:
#   https://huggingface.co/lightx2v/Wan2.2-Lightning
#
# This script intentionally installs the smallest reliable set for a first
# serverless bootstrap:
# - native ComfyUI Wan 2.2 I2V fp16 models
# - native ComfyUI text encoder + VAE
# - official LightX2V I2V Lightning LoRAs
# - no extra custom nodes
#
# Why fp16 here:
# - the official NativeComfy Lightning I2V workflow references these exact
#   filenames:
#   - wan2.2_i2v_high_noise_14B_fp16.safetensors
#   - wan2.2_i2v_low_noise_14B_fp16.safetensors
#   - umt5_xxl_fp16.safetensors
#   - wan_2.1_vae.safetensors
#   - Wan2.2-Lightning/.../high_noise_model.safetensors
#   - Wan2.2-Lightning/.../low_noise_model.safetensors

log() {
  printf '[vast-wan-bootstrap] %s\n' "$*"
}

find_comfy_root() {
  local candidate
  for candidate in \
    "/root/ComfyUI" \
    "/workspace/ComfyUI" \
    "/opt/ComfyUI" \
    "/app/ComfyUI"
  do
    if [[ -d "$candidate" && -d "$candidate/models" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    log "Missing required command: $1"
    exit 1
  }
}

main() {
  require_cmd python3

  local comfy_root
  if ! comfy_root="$(find_comfy_root)"; then
    log "Could not locate ComfyUI root automatically."
    log "Expected one of: /root/ComfyUI, /workspace/ComfyUI, /opt/ComfyUI, /app/ComfyUI"
    exit 1
  fi

  log "Using ComfyUI root: $comfy_root"

  mkdir -p \
    "$comfy_root/models/diffusion_models" \
    "$comfy_root/models/text_encoders" \
    "$comfy_root/models/vae" \
    "$comfy_root/models/loras/Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"

  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN is not set. Proceeding anonymously."
    log "If Hugging Face rate limits or gated access applies, set HF_TOKEN in Vast Account Settings."
  else
    log "HF_TOKEN detected."
  fi

  python3 - <<'PY' "$comfy_root"
import importlib.util
import os
import shutil
import subprocess
import sys

comfy_root = sys.argv[1]

def ensure_hf_hub() -> None:
    if importlib.util.find_spec("huggingface_hub") is not None:
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "huggingface_hub>=0.31.0"]
    )

ensure_hf_hub()

from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN") or None

downloads = [
    {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "dest": os.path.join(comfy_root, "models", "diffusion_models", "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
    },
    {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
        "dest": os.path.join(comfy_root, "models", "diffusion_models", "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
    },
    {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/text_encoders/umt5_xxl_fp16.safetensors",
        "dest": os.path.join(comfy_root, "models", "text_encoders", "umt5_xxl_fp16.safetensors"),
    },
    {
        "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "filename": "split_files/vae/wan_2.1_vae.safetensors",
        "dest": os.path.join(comfy_root, "models", "vae", "wan_2.1_vae.safetensors"),
    },
    {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        "dest": os.path.join(
            comfy_root,
            "models",
            "loras",
            "Wan2.2-Lightning",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
            "high_noise_model.safetensors",
        ),
    },
    {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
        "dest": os.path.join(
            comfy_root,
            "models",
            "loras",
            "Wan2.2-Lightning",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
            "low_noise_model.safetensors",
        ),
    },
    {
        "repo_id": "lightx2v/Wan2.2-Lightning",
        "filename": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-NativeComfy.json",
        "dest": os.path.join(
            comfy_root,
            "models",
            "loras",
            "Wan2.2-Lightning",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1-NativeComfy.json",
        ),
    },
]

for item in downloads:
    dest = item["dest"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
      print(f"[vast-wan-bootstrap] exists, skip: {dest}")
      continue

    print(f"[vast-wan-bootstrap] downloading {item['repo_id']}::{item['filename']}")
    cached = hf_hub_download(
        repo_id=item["repo_id"],
        filename=item["filename"],
        token=token,
        resume_download=True,
    )
    shutil.copy2(cached, dest)
    print(f"[vast-wan-bootstrap] installed: {dest}")
PY

  log "Bootstrap complete."
  log "Installed native ComfyUI Wan 2.2 I2V fp16 base files and official I2V Lightning LoRAs."
  log "The official NativeComfy Lightning workflow JSON was also downloaded next to the LoRAs for reference."
}

main "$@"
