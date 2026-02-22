#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/data1/LLaVA}"
MODEL_PATH="${2:-/data1/LLaVA/llava-v1.5-7b-lora21mixloss}"
MODEL_BASE="${3:-/data1/LLaVA/llava-v1.5-7b}"
IMAGE_FILE="${4:-/data1/LLaVA/testdon/Trump_Image745.jpg}"
QUERY="${5:-Who is the person in this image?}"
MAX_NEW_TOKENS="${6:-64}"

cd "${REPO_DIR}"
export PYTHONPATH="/data1/LLaVA:${PYTHONPATH:-}"

if [[ ! -f "${IMAGE_FILE}" ]]; then
  echo "ERROR: image file not found: ${IMAGE_FILE}"
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -d "${MODEL_BASE}" ]]; then
  echo "ERROR: model base not found: ${MODEL_BASE}"
  exit 1
fi

echo "== Run single-image smoke test =="
set +e
python llava/eval/run_llava.py \
  --model-path "${MODEL_PATH}" \
  --model-base "${MODEL_BASE}" \
  --image-file "${IMAGE_FILE}" \
  --query "${QUERY}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" 2>&1 | tee /tmp/siu_smoke.log
rc="${PIPESTATUS[0]}"
set -e

if [[ "${rc}" -ne 0 ]]; then
  if grep -qi "out of memory" /tmp/siu_smoke.log; then
    echo "OOM detected, retrying with --max_new_tokens 32"
    python llava/eval/run_llava.py \
      --model-path "${MODEL_PATH}" \
      --model-base "${MODEL_BASE}" \
      --image-file "${IMAGE_FILE}" \
      --query "${QUERY}" \
      --max_new_tokens 32
  else
    echo "Smoke test failed. See /tmp/siu_smoke.log"
    exit "${rc}"
  fi
fi

echo "Smoke test finished."
