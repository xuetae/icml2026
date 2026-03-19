# FEM Training (Short Guide)

This file briefly describes the **frequency-enhanced (FEM)** training code in this bundle.

For non-frequency baselines, see `README_NO_FREQ_FINETUNE.md`.

## What This Code Does

- Builds a YOLOv10 model with FDAF layers in the neck.
- Trains with the privacy dataset at `split/label.yaml`.
- Uses pretrained `yolov10s.pt` and fine-tunes on your dataset.

Main FEM scripts:

- `train_frequency.py` — main FEM training entry.
- `run_training.sh` — one-click shell launcher for FEM training.
- `yolov10s.yaml` — FEM model structure reference (includes FDAF blocks).
- `core_modules/fdaf_block.py` — extracted adaptive spectral gating block (FDAF).
- `core_modules/frequency_consistency_loss.py` — extracted `L_freq` core implementation.
- `ultralytics/nn/modules/block.py` — runtime FDAF implementation used in training.
- `ultralytics/utils/loss.py` — runtime frequency-consistency loss used in training.

## Run

### PowerShell (Windows)

```powershell
python .\train_frequency.py
```

### Bash

```bash
bash run_training.sh
```

## Output

Training outputs are written to `runs/detect/` (weights, logs, curves).
