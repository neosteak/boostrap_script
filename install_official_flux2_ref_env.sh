#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
PIP_BIN="${PIP_BIN:-/venv/main/bin/pip}"
COMFY_BRANCH="${COMFY_BRANCH:-master}"
INSTALL_TURBO_LORA="${INSTALL_TURBO_LORA:-1}"
INSTALL_GGUF_STACK="${INSTALL_GGUF_STACK:-1}"
UPDATE_GGUF_NODE="${UPDATE_GGUF_NODE:-1}"

OFFICIAL_REPO="${OFFICIAL_REPO:-Comfy-Org/flux2-dev}"
DIFFUSION_FILE="${DIFFUSION_FILE:-split_files/diffusion_models/flux2_dev_fp8mixed.safetensors}"
TEXT_ENCODER_BF16_FILE="${TEXT_ENCODER_BF16_FILE:-split_files/text_encoders/mistral_3_small_flux2_bf16.safetensors}"
TEXT_ENCODER_FP8_FILE="${TEXT_ENCODER_FP8_FILE:-split_files/text_encoders/mistral_3_small_flux2_fp8.safetensors}"
VAE_FILE="${VAE_FILE:-split_files/vae/flux2-vae.safetensors}"
TURBO_LORA_FILE="${TURBO_LORA_FILE:-split_files/loras/Flux_2-Turbo-LoRA_comfyui.safetensors}"
GGUF_NODE_REPO="${GGUF_NODE_REPO:-https://github.com/city96/ComfyUI-GGUF}"
GGUF_MODEL_REPO="${GGUF_MODEL_REPO:-alb3530/Flux.2-dev-FALAI-Turbo-Merged-GGUF}"
GGUF_MODEL_FILE="${GGUF_MODEL_FILE:-flux2-dev_falai_turbo_merged_Q8_0.GGUF}"

find_comfyui_dir() {
  if [[ -n "${COMFYUI_DIR:-}" && -f "${COMFYUI_DIR}/main.py" ]]; then
    echo "${COMFYUI_DIR}"
    return 0
  fi

  local pid
  pid="$(pgrep -f "main.py.*18188" | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    local cwd
    cwd="$(readlink -f "/proc/${pid}/cwd" || true)"
    if [[ -n "${cwd}" && -f "${cwd}/main.py" ]]; then
      echo "${cwd}"
      return 0
    fi
  fi

  local candidates=(
    "/workspace/ComfyUI"
    "/opt/ComfyUI"
    "/workspace/comfyui"
    "/workspace/ComfyUI_windows_portable/ComfyUI"
    "/root/ComfyUI"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/main.py" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  return 1
}

restart_comfy_services() {
  if ! command -v supervisorctl >/dev/null 2>&1; then
    return 0
  fi

  local services
  services="$(supervisorctl status | awk 'tolower($1) ~ /comfy/ {print $1}' || true)"
  if [[ -z "${services}" ]]; then
    return 0
  fi

  local service
  for service in ${services}; do
    supervisorctl restart "${service}" || true
  done
}

echo "[flux2-ref] Activating /venv/main if present"
if [[ -f "/venv/main/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[flux2-ref] Missing Python at ${PYTHON_BIN}"
  exit 1
fi

COMFYUI_DIR="$(find_comfyui_dir)" || {
  echo "[flux2-ref] Unable to locate ComfyUI. Set COMFYUI_DIR and retry."
  exit 1
}

echo "[flux2-ref] Using ComfyUI directory: ${COMFYUI_DIR}"
echo "[flux2-ref] Updating ComfyUI core and workflow template packages"

git -C "${COMFYUI_DIR}" fetch --all --tags
git -C "${COMFYUI_DIR}" checkout "${COMFY_BRANCH}"
git -C "${COMFYUI_DIR}" pull --ff-only

"${PIP_BIN}" install -r "${COMFYUI_DIR}/requirements.txt"
"${PIP_BIN}" install -U \
  "comfyui-frontend-package" \
  "comfyui-workflow-templates" \
  "comfyui-embedded-docs" \
  "huggingface_hub[hf_transfer]"

if [[ "${UPDATE_GGUF_NODE}" == "1" && -d "${COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF/.git" ]]; then
  echo "[flux2-ref] Updating ComfyUI-GGUF"
  git -C "${COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF" pull --ff-only || true
  if [[ -f "${COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF/requirements.txt" ]]; then
    "${PIP_BIN}" install -r "${COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF/requirements.txt"
  fi
fi

mkdir -p \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/models/loras" \
  "${COMFYUI_DIR}/models/unet" \
  "${COMFYUI_DIR}/custom_nodes"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

"${PIP_BIN}" install -U "gguf"

if [[ "${INSTALL_GGUF_STACK}" == "1" ]]; then
  GGUF_NODE_DIR="${COMFYUI_DIR}/custom_nodes/ComfyUI-GGUF"
  if [[ -d "${GGUF_NODE_DIR}/.git" ]]; then
    if [[ "${UPDATE_GGUF_NODE}" == "1" ]]; then
      echo "[flux2-ref] Updating ComfyUI-GGUF"
      git -C "${GGUF_NODE_DIR}" pull --ff-only || true
    fi
  else
    echo "[flux2-ref] Installing ComfyUI-GGUF"
    git clone "${GGUF_NODE_REPO}" "${GGUF_NODE_DIR}"
  fi

  if [[ -f "${GGUF_NODE_DIR}/requirements.txt" ]]; then
    "${PIP_BIN}" install -r "${GGUF_NODE_DIR}/requirements.txt"
  fi
fi

"${PYTHON_BIN}" - <<PY
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

token = os.getenv("HF_TOKEN") or None
jobs = [
    ("${OFFICIAL_REPO}", "${DIFFUSION_FILE}", Path("${COMFYUI_DIR}") / "models" / "diffusion_models" / "flux2_dev_fp8mixed.safetensors"),
    ("${OFFICIAL_REPO}", "${TEXT_ENCODER_BF16_FILE}", Path("${COMFYUI_DIR}") / "models" / "text_encoders" / "mistral_3_small_flux2_bf16.safetensors"),
    ("${OFFICIAL_REPO}", "${TEXT_ENCODER_FP8_FILE}", Path("${COMFYUI_DIR}") / "models" / "text_encoders" / "mistral_3_small_flux2_fp8.safetensors"),
    ("${OFFICIAL_REPO}", "${VAE_FILE}", Path("${COMFYUI_DIR}") / "models" / "vae" / "flux2-vae.safetensors"),
]

if "${INSTALL_TURBO_LORA}" == "1":
    jobs.append(
        ("${OFFICIAL_REPO}", "${TURBO_LORA_FILE}", Path("${COMFYUI_DIR}") / "models" / "loras" / "Flux_2-Turbo-LoRA_comfyui.safetensors")
    )

if "${INSTALL_GGUF_STACK}" == "1":
    jobs.append(
        ("${GGUF_MODEL_REPO}", "${GGUF_MODEL_FILE}", Path("${COMFYUI_DIR}") / "models" / "unet" / "${GGUF_MODEL_FILE}")
    )

for repo_id, filename, destination in jobs:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[flux2-ref] already present: {destination}")
        continue
    source = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    shutil.copy2(source, destination)
    print(f"[flux2-ref] installed: {destination}")
PY

restart_comfy_services

cat <<EOF

[flux2-ref] Official Flux.2 Dev ref-image environment is ready.

Installed/updated:
- ComfyUI core: ${COMFYUI_DIR}
- Workflow templates package: comfyui-workflow-templates
- diffusion_models/flux2_dev_fp8mixed.safetensors
- text_encoders/mistral_3_small_flux2_bf16.safetensors
- text_encoders/mistral_3_small_flux2_fp8.safetensors
- vae/flux2-vae.safetensors
EOF

if [[ "${INSTALL_TURBO_LORA}" == "1" ]]; then
  echo "- loras/Flux_2-Turbo-LoRA_comfyui.safetensors"
fi

if [[ "${INSTALL_GGUF_STACK}" == "1" ]]; then
  cat <<EOF
- custom_nodes/ComfyUI-GGUF
- models/unet/${GGUF_MODEL_FILE}
EOF
fi

cat <<EOF

Next:
1. Refresh ComfyUI.
2. Check Templates > Flux.2 Dev.
3. Look for the official Multi-image reference workflow.
4. Keep your GGUF workflow untouched; use the official workflow only as the ref-image base.

Useful checks:
- tail -n 120 /var/log/portal/comfyui.log
- python - <<'PY'
import comfyui_workflow_templates
print(comfyui_workflow_templates.__file__)
PY
EOF
