#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-/data1/LLaVA}"
MODEL_DIR="${REPO_DIR}/llava-v1.5-7b-lora21mixloss"
ADAPTER_FILE="${MODEL_DIR}/adapter_model.bin"
NON_LORA_FILE="${MODEL_DIR}/non_lora_trainables.bin"

cd "${REPO_DIR}"

echo "== Git LFS pull =="
git lfs install
git lfs pull

echo "== Verify LoRA weight files =="
for f in "${ADAPTER_FILE}" "${NON_LORA_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing file: ${f}"
    exit 1
  fi
  size_bytes="$(stat -c%s "${f}")"
  echo "${f}: ${size_bytes} bytes"
  if (( size_bytes < 1000000 )); then
    echo "ERROR: ${f} is still too small. It may still be an LFS pointer file."
    exit 1
  fi
done

echo "LFS verification passed."
