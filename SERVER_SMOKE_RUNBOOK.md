# SIU A800 Reproduction Runbook (Environment + Single-Image Smoke)

This runbook implements the agreed plan for:
- Ubuntu 22.04 server
- A800 GPU
- Conda environment
- Flexible project directory with `/data1/LLaVA` compatibility
- First milestone: one image, one prompt inference smoke test

## 0) Project Layout and New Scripts

Added files:
- `llava/train/train.py` (copied from `llava/train/trainer for SIU/train.py`)
- `llava/train/llava_trainer.py` (copied from `llava/train/trainer for SIU/llava_trainer.py`)
- `scripts/zero3.json`
- `scripts/server_precheck.sh`
- `scripts/setup_conda_env.sh`
- `scripts/pull_lfs_and_verify.sh`
- `scripts/download_base_models.sh`
- `scripts/run_smoke_test.sh`
- `scripts/prepare_training_layout.sh`

## 1) Server Precheck

```bash
cd /path/to/your/repo
chmod +x scripts/*.sh
./scripts/server_precheck.sh
```

Acceptance:
- `nvidia-smi` shows A800
- available disk is recommended >= 300G

## 2) Path Mapping (`/data1/LLaVA -> WORKDIR`)

If your repo is not already at `/data1/LLaVA`, run:

```bash
./scripts/prepare_training_layout.sh "$(pwd)"
```

Acceptance:
- `ls /data1/LLaVA` shows this project
- `llava/train/train.py` and `llava/train/llava_trainer.py` exist
- `/data1/LLaVA/scripts/zero3.json` exists
- `/data1/LLaVA/machineunlearning/imgsdon/Trump_Image118.jpg` exists

## 3) Pull LFS Weights

```bash
./scripts/pull_lfs_and_verify.sh /data1/LLaVA
```

Acceptance:
- `/data1/LLaVA/llava-v1.5-7b-lora21mixloss/adapter_model.bin` is large (not ~104 bytes)
- `/data1/LLaVA/llava-v1.5-7b-lora21mixloss/non_lora_trainables.bin` is large (not ~103 bytes)

## 4) Build Conda Environment

```bash
./scripts/setup_conda_env.sh siu310
```

Then in each shell session:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate siu310
export PYTHONPATH=/data1/LLaVA:${PYTHONPATH:-}
```

Acceptance:
- `torch.cuda.is_available()` is `True`

## 5) Download Base Model and Vision Tower

```bash
./scripts/download_base_models.sh /data1/LLaVA
```

Default repos:
- `liuhaotian/llava-v1.5-7b`
- `openai/clip-vit-large-patch14-336`

Acceptance:
- `/data1/LLaVA/llava-v1.5-7b/config.json` exists
- `/data1/LLaVA/clip-vit-large-patch14-336/config.json` exists

## 6) Run Smoke Test

```bash
./scripts/run_smoke_test.sh \
  /data1/LLaVA \
  /data1/LLaVA/llava-v1.5-7b-lora21mixloss \
  /data1/LLaVA/llava-v1.5-7b \
  /data1/LLaVA/testdon/Trump_Image745.jpg \
  "Who is the person in this image?" \
  64
```

Acceptance:
- no traceback
- model prints one natural language answer

If OOM appears, script auto-retries with `--max_new_tokens 32`.

## 7) Failure Branches

- LoRA load error:
  - rerun `./scripts/pull_lfs_and_verify.sh /data1/LLaVA`
- Vision tower path/model-base error:
  - rerun `./scripts/download_base_models.sh /data1/LLaVA`
- CUDA/Torch mismatch:
  - recreate env with `./scripts/setup_conda_env.sh siu310`
- OOM:
  - reduce max tokens to 32 or lower

## 8) Training-Path Prepared (No Training Started)

The repo is now prepped for later training entrypoint checks:

```bash
python -m pip install deepspeed
CHECK_TRAIN_HELP=1 ./scripts/prepare_training_layout.sh "$(pwd)"
python /data1/LLaVA/llava/train/train_mem.py --help
```

Expected:
- help text prints without import/path errors
