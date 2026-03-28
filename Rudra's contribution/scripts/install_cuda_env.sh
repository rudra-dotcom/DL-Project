#!/usr/bin/env bash
set -euo pipefail

CUDA_WHEEL="${1:-cu121}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
PIP_TIMEOUT="${PIP_TIMEOUT:-300}"
PIP_RETRIES="${PIP_RETRIES:-10}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install \
  --default-timeout "$PIP_TIMEOUT" \
  --retries "$PIP_RETRIES" \
  "torch==${TORCH_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
python -m pip install \
  --default-timeout "$PIP_TIMEOUT" \
  --retries "$PIP_RETRIES" \
  "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
python -m pip install -r requirements.txt

python - <<'PY'
import torch
import timm
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
PY
