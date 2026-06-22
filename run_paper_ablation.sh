#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_YAML="${DATA_YAML:-datasets/ICML_2026_dataset/split/label.yaml}"
WEIGHTS="${WEIGHTS:-weights/yolov10s.pt}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
PROJECT="${PROJECT:-runs/paper_repro}"

for stage in baseline fdaf fdaf_lsg full; do
  echo "[INFO] Starting paper ablation stage: ${stage}"
  "${PYTHON_BIN}" train_frequency.py \
    --stage "${stage}" \
    --data "${DATA_YAML}" \
    --weights "${WEIGHTS}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --workers "${WORKERS}" \
    --project "${PROJECT}" \
    --name "yolov10s_${stage}"
done
