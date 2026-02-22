#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/train_siu.sh
#   GPU_IDS=1 MAX_STEPS=6 GRAD_ACC=4 OUTPUT_DIR=checkpoints/siu-step6 bash scripts/train_siu.sh
#   GPU_IDS=1,2,3,4 MAX_STEPS=6 GRAD_ACC=1 OUTPUT_DIR=checkpoints/siu-step6-4gpu bash scripts/train_siu.sh
#
# Environment variables:
#   GPU_IDS           Physical GPU ids to use, comma separated (default: 1)
#   MAX_STEPS         Train steps (default: 1)
#   GRAD_ACC          Gradient accumulation steps (default: 1)
#   PER_DEVICE_BS     Per-device train/eval batch size (default: 1)
#   MODEL_MAX_LENGTH  Sequence length (default: 2048)
#   DATALOADER_WORKERS Number of data loader workers (default: 2)
#   OUTPUT_DIR        Output checkpoint directory (default: checkpoints/siu-smoke)
#   DATA_PATH         Training data json path (default: finetunedata/SIU.local.json)
#   IMAGE_FOLDER      Image folder path (default: machineunlearning/imgsdon)
#   MODEL_BASE_PATH   Base model path (default: llava-v1.5-7b)
#   VISION_TOWER_PATH Vision tower path (default: clip-vit-large-patch14-336)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
GPU_IDS="${GPU_IDS:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
GRAD_ACC="${GRAD_ACC:-1}"
PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"

DATA_PATH="${DATA_PATH:-$ROOT/finetunedata/SIU.local.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-$ROOT/machineunlearning/imgsdon}"
MODEL_BASE_PATH="${MODEL_BASE_PATH:-$ROOT/llava-v1.5-7b}"
VISION_TOWER_PATH="${VISION_TOWER_PATH:-$ROOT/clip-vit-large-patch14-336}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/siu-smoke}"

mkdir -p "${ROOT}/machineunlearning/imgsdon"
if [[ -f "${ROOT}/traindon/Trump_Image118.jpg" ]]; then
  cp -f "${ROOT}/traindon/Trump_Image118.jpg" "${ROOT}/machineunlearning/imgsdon/"
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  sed 's#/data1/LLaVA/machineunlearning/imgsdon/##g' \
    "${ROOT}/finetunedata/SIU.json" > "${ROOT}/finetunedata/SIU.local.json"
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
TRAIN_BATCH_SIZE=$((PER_DEVICE_BS * GRAD_ACC * WORLD_SIZE))
DS_TEMPLATE="${ROOT}/scripts/zero3.json"
DS_RUNTIME="${ROOT}/scripts/zero3.runtime.json"

python - "$DS_TEMPLATE" "$DS_RUNTIME" "$GRAD_ACC" "$PER_DEVICE_BS" "$TRAIN_BATCH_SIZE" <<'PY'
import json
import sys

tmpl, out, grad_acc, micro_bs, train_bs = sys.argv[1:]
with open(tmpl, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["gradient_accumulation_steps"] = int(grad_acc)
cfg["train_micro_batch_size_per_gpu"] = int(micro_bs)
cfg["train_batch_size"] = int(train_bs)

with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

export ROOT
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
unset CUDA_VISIBLE_DEVICES || true

echo "ROOT=${ROOT}"
echo "GPU_IDS=${GPU_IDS}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "GRAD_ACC=${GRAD_ACC}"
echo "PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "DATA_PATH=${DATA_PATH}"
echo "IMAGE_FOLDER=${IMAGE_FOLDER}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "DS_RUNTIME=${DS_RUNTIME}"

deepspeed --include "localhost:${GPU_IDS}" "${ROOT}/llava/train/train.py" \
  --deepspeed "${DS_RUNTIME}" \
  --optim adamw_torch \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 2e-5 \
  --model_name_or_path "${MODEL_BASE_PATH}" \
  --version v1 \
  --data_path "${DATA_PATH}" \
  --image_folder "${IMAGE_FOLDER}" \
  --vision_tower "${VISION_TOWER_PATH}" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${MAX_STEPS}" \
  --save_total_limit 1 \
  --learning_rate 3e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --gradient_checkpointing True \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --lazy_preprocess True
