param(
    [string]$PythonBin = "python",
    [string]$DataYaml = "F:/nus/icml/split_5/label.yaml",
    [int]$Epochs = 100,
    [int]$Batch = 16,
    [int]$Imgsz = 640,
    [string]$Device = "0",
    [int]$Workers = 0,
    [string]$Optimizer = "SGD",
    [switch]$Amp
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TrainScript = Join-Path $ScriptDir "train_withoutfrequency.py"

if (-not (Test-Path $TrainScript)) {
    throw "Missing training script: $TrainScript"
}
if (-not (Test-Path $DataYaml)) {
    throw "Missing dataset yaml: $DataYaml"
}

$models = @(
    "yolov7-tiny.pt",
    "yolov7.pt",
    "yolov8s.pt",
    "yolov8l.pt",
    "yolov9s.pt",
    "yolov9l.pt",
    "yolov10s.pt",
    "yolov10l.pt"
)

Write-Host "[INFO] Training script: $TrainScript"
Write-Host "[INFO] Dataset: $DataYaml"
Write-Host "[INFO] Models: $($models -join ', ')"

foreach ($model in $models) {
    $runName = "repro_" + [System.IO.Path]::GetFileNameWithoutExtension($model)
    Write-Host ""
    Write-Host "[INFO] Starting: $model -> run name: $runName"

    $args = @(
        $TrainScript,
        "--model", $model,
        "--data", $DataYaml,
        "--epochs", "$Epochs",
        "--batch", "$Batch",
        "--imgsz", "$Imgsz",
        "--device", "$Device",
        "--workers", "$Workers",
        "--optimizer", $Optimizer,
        "--name", $runName
    )
    if ($Amp.IsPresent) {
        $args += "--amp"
    }

    & $PythonBin $args
}

Write-Host ""
Write-Host "[INFO] All training jobs finished."
