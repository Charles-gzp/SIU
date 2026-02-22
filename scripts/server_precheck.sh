#!/usr/bin/env bash
set -euo pipefail

echo "== SIU server precheck =="
echo "Timestamp: $(date -Iseconds)"
echo

echo "== OS release =="
cat /etc/os-release
echo

echo "== Kernel =="
uname -a
echo

echo "== GPU =="
nvidia-smi
echo

echo "== Disk usage =="
df -h
echo

TARGET_MOUNT="/data1"
if [[ ! -d "${TARGET_MOUNT}" ]]; then
  TARGET_MOUNT="/"
fi

AVAIL_GB="$(df -BG "${TARGET_MOUNT}" | awk 'NR==2 {gsub("G","",$4); print $4}')"
if [[ -n "${AVAIL_GB}" ]]; then
  echo "Available space on ${TARGET_MOUNT}: ${AVAIL_GB}G"
  if (( AVAIL_GB < 300 )); then
    echo "WARNING: available space is below 300G. Download + checkpoints may fail."
  fi
fi

if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "A800"; then
  echo "GPU check passed: A800 detected."
else
  echo "WARNING: A800 was not detected in GPU name list."
fi
