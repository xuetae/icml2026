#!/usr/bin/env bash
set -euo pipefail

# One-click baseline fine-tuning launcher.
# It uses finetune_yolo_generic.py, which:
# 1) resolves local weights
# 2) auto-downloads supported weights if missing
# 3) starts training with unified arguments

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_withoutfrequency.py"

DATA_YAML="${DATA_YAML:-F:/nus/icml/split_5/label.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-0}"
OPTIMIZER="${OPTIMIZER:-SGD}"
AMP_FLAG="${AMP_FLAG:-0}" # 0 or 1

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[ERROR] Missing training script: ${TRAIN_SCRIPT}"
  exit 1
fi

if [[ ! -f "${DATA_YAML}" ]]; then
  echo "[ERROR] Missing dataset yaml: ${DATA_YAML}"
  exit 1
fi

# Paper-related YOLO baselines that can be directly used in this pipeline.
# - YOLOv8: s/l
# - YOLOv9: c/e (official naming used by current Ultralytics assets)
# - YOLOv10: s/l
MODELS=(
  "yolov7-tiny.pt"
  "yolov7.pt"
  "yolov8s.pt"
  "yolov8l.pt"
  "yolov9s.pt"
  "yolov9l.pt"
  "yolov10s.pt"
  "yolov10l.pt"
)

echo "[INFO] Training script: ${TRAIN_SCRIPT}"
echo "[INFO] Dataset: ${DATA_YAML}"
echo "[INFO] Models: ${MODELS[*]}"

for model in "${MODELS[@]}"; do
  run_name="repro_${model%.pt}"
  cmd=(
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
    --model "${model}"
    --data "${DATA_YAML}"
    --epochs "${EPOCHS}"
    --batch "${BATCH}"
    --imgsz "${IMGSZ}"
    --device "${DEVICE}"
    --workers "${WORKERS}"
    --optimizer "${OPTIMIZER}"
    --name "${run_name}"
  )

  if [[ "${AMP_FLAG}" == "1" ]]; then
    cmd+=(--amp)
  fi

  echo
  echo "[INFO] Starting: ${model} -> run name: ${run_name}"
  "${cmd[@]}"
done

echo
echo "[INFO] All training jobs finished."
