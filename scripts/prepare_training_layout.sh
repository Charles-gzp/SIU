#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:-$(pwd)}"
TARGET_ROOT="/data1/LLaVA"
TRAIN_DIR="${WORKDIR}/llava/train"
TRAINER_SIU_DIR="${TRAIN_DIR}/trainer for SIU"
TARGET_IMG_DIR="${TARGET_ROOT}/machineunlearning/imgsdon"
SOURCE_IMG="${WORKDIR}/traindon/Trump_Image118.jpg"

echo "== Prepare /data1/LLaVA mapping =="
if [[ ! -d /data1 ]]; then
  mkdir -p /data1 2>/dev/null || true
fi
if [[ -L "${TARGET_ROOT}" || -d "${TARGET_ROOT}" ]]; then
  echo "${TARGET_ROOT} already exists."
  if [[ -L "${TARGET_ROOT}" ]]; then
    echo "Current mapping: ${TARGET_ROOT} -> $(readlink "${TARGET_ROOT}")"
  fi
else
  ln -s "${WORKDIR}" "${TARGET_ROOT}" 2>/dev/null || true
  if [[ -L "${TARGET_ROOT}" ]]; then
    echo "Created symlink: ${TARGET_ROOT} -> ${WORKDIR}"
  else
    echo "ERROR: could not create ${TARGET_ROOT}. Please create it manually, then rerun."
    exit 1
  fi
fi

echo "== Copy SIU trainer files into active path =="
cp "${TRAINER_SIU_DIR}/train.py" "${TRAIN_DIR}/train.py"
cp "${TRAINER_SIU_DIR}/llava_trainer.py" "${TRAIN_DIR}/llava_trainer.py"

echo "== Ensure zero3 config exists at hardcoded path =="
mkdir -p "${TARGET_ROOT}/scripts"
cp "${WORKDIR}/scripts/zero3.json" "${TARGET_ROOT}/scripts/zero3.json"

echo "== Prepare image folder for finetune JSON absolute paths =="
mkdir -p "${TARGET_IMG_DIR}"
cp "${SOURCE_IMG}" "${TARGET_IMG_DIR}/Trump_Image118.jpg"

if [[ "${CHECK_TRAIN_HELP:-0}" == "1" ]]; then
  echo "== Optional training entrypoint sanity check =="
  python "${TARGET_ROOT}/llava/train/train_mem.py" --help >/tmp/siu_train_help.log
  echo "train_mem.py --help succeeded."
else
  echo "Skip train_mem.py --help check (set CHECK_TRAIN_HELP=1 to enable)."
fi
