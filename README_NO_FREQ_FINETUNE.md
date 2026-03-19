# Non-Frequency Fine-Tuning

Minimal guide for standard YOLO fine-tuning (no FDAF, no frequency loss).

## Scripts

- `train_withoutfrequency.py`: generic training entry (YOLOv7/v8/v9/v10)
- `run_repro_train.ps1`: batch run on Windows PowerShell
- `run_repro_train.sh`: batch run on Bash
- `ultralytics/`: local runtime package used directly by these scripts

## Run

```powershell
# single model
python .\train_withoutfrequency.py --model yolov10s.pt --data split/label.yaml --name ft_v10s_no_freq

# batch (Windows)
.\run_repro_train.ps1
```

```bash
# batch (Bash)
bash run_repro_train.sh
```

## Notes

- Default dataset: `split/label.yaml`
- Outputs: `runs/detect/`