# Paper-aligned reproduction

This runner keeps the data split and training hyperparameters fixed across the
four YOLOv10-S ablation stages reported by the paper:

1. `baseline`: standard YOLOv10-S
2. `fdaf`: frequency branch and cross-domain fusion, without spectral gating
3. `fdaf_lsg`: FDAF with learnable spectral gating
4. `full`: FDAF, gating, and Frequency-Consistency Loss

The paper explicitly reports SGD with momentum `0.937`, weight decay `5e-4`,
and frequency-loss weight `beta=0.05`. Parameters not reported by the paper use
the local YOLO defaults and are exposed as command-line arguments.

## Audit labels

```bash
python audit_dataset.py \
  --labels datasets/ICML_2026_dataset/dataset/labels \
  --nc 33
```

Do not start a reproduction run unless this command exits successfully.

## Single experiment

```bash
python train_frequency.py \
  --stage full \
  --data datasets/ICML_2026_dataset/split/label.yaml \
  --weights weights/yolov10s.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --workers 8
```

## Full ablation

```bash
bash run_paper_ablation.sh
```

All stages must use the exact same dataset split. Do not compare a random local
split directly with the paper's reported test metrics unless the authors'
official split is available.

## Resume

```bash
python train_frequency.py \
  --data datasets/ICML_2026_dataset/split/label.yaml \
  --resume runs/paper_repro/yolov10s_full/weights/last.pt
```

The corrected loss applies `beta * L_freq` outside the YOLO box-loss gain and
only to the one-to-many branch. Baseline runs do not compute frequency loss.
For FDAF stages, pretrained layers after each inserted FDAF block are remapped
to their shifted target indices instead of being silently skipped.

## Test evaluation

```bash
python evaluate_repro.py \
  --stage full \
  --model runs/paper_repro/yolov10s_full/weights/best.pt \
  --data datasets/ICML_2026_dataset/split/label.yaml \
  --split test
```
