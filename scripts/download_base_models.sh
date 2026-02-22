#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-/data1/LLaVA}"
BASE_MODEL_REPO="${BASE_MODEL_REPO:-liuhaotian/llava-v1.5-7b}"
VISION_REPO="${VISION_REPO:-openai/clip-vit-large-patch14-336}"

BASE_MODEL_DIR="${BASE_DIR}/llava-v1.5-7b"
VISION_DIR="${BASE_DIR}/clip-vit-large-patch14-336"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install huggingface_hub in your conda env."
  exit 1
fi

mkdir -p "${BASE_MODEL_DIR}" "${VISION_DIR}"

echo "== Download base LLaVA model =="
huggingface-cli download "${BASE_MODEL_REPO}" --local-dir "${BASE_MODEL_DIR}"

echo "== Download vision tower =="
huggingface-cli download "${VISION_REPO}" --local-dir "${VISION_DIR}"

echo "== Verify downloaded files =="
test -f "${BASE_MODEL_DIR}/config.json"
test -f "${BASE_MODEL_DIR}/tokenizer_config.json"
test -f "${VISION_DIR}/config.json"
echo "Model download verification passed."
