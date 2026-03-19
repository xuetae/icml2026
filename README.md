# YOLO Privacy Repro Bundle

Self-contained training bundle for privacy detection reproduction.

This folder includes:
- local `ultralytics/` runtime code (trainer, model, losses, FDAF)
- frequency and non-frequency training scripts
- batch launch scripts for PowerShell/Bash

## Setup

Run all commands inside this folder:

```powershell
pip install -r .\requirements.txt
```

## Run

```powershell
# non-frequency single model
python .\train_withoutfrequency.py --model yolov10s.pt --data split/label.yaml --name ft_v10s_no_freq

# non-frequency batch
.\run_repro_train.ps1

# frequency-enhanced (FEM)
python .\train_frequency.py --data split/label.yaml --weights yolov10s.pt
```

## Docs

- `README_FEM.md`
- `README_NO_FREQ_FINETUNE.md`

## Notes

- Dataset yaml default: `split/label.yaml`
- Dataset release: `https://huggingface.co/datasets/zuoenpu/ICML_2026_dataset`
- If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

