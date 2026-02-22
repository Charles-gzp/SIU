#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-siu310}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH."
  exit 1
fi

echo "== Create/activate conda env: ${ENV_NAME} =="
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env ${ENV_NAME} already exists, reusing it."
else
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"

echo "== Upgrade packaging tools =="
python -m pip install --upgrade pip setuptools wheel

echo "== Install PyTorch CUDA 12.1 wheels =="
python -m pip install \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo "== Install minimal inference dependencies =="
python -m pip install \
  transformers==4.37.2 \
  tokenizers==0.15.1 \
  accelerate==0.21.0 \
  peft==0.4.0 \
  sentencepiece \
  requests \
  pillow \
  numpy \
  bitsandbytes \
  huggingface_hub

echo "== Runtime checks =="
python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

echo
echo "Use this before running project commands in each shell:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "  export PYTHONPATH=/data1/LLaVA:\${PYTHONPATH:-}"
