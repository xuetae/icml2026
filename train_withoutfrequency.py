import argparse
from pathlib import Path
from urllib.request import urlretrieve

from ultralytics import YOLO, YOLOv10
from ultralytics.utils.downloads import attempt_download_asset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic YOLO fine-tuning script (v8/v9/v10 and more)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov10s.pt",
        help="Model path or name, e.g. yolov10s.pt, yolov9s.pt, yolov8s.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="F:/nus/icml/split_5/label.yaml",
        help="Dataset yaml path",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="generic_yolo_finetune")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer used in training",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP mixed precision (default: disabled)",
    )
    return parser.parse_args()


def normalize_model_key(model_arg: str) -> str:
    return model_arg.lower().replace("\\", "/").split("/")[-1]


def apply_paper_alias(model_arg: str) -> str:
    """Map paper naming to downloadable names in current toolchain."""
    key = normalize_model_key(model_arg)
    alias = {
        # Paper names -> Ultralytics downloadable names in this env
        "yolov9s.pt": "yolov9c.pt",
        "yolov9l.pt": "yolov9e.pt",
    }
    if key in alias:
        mapped = alias[key]
        print(f"[WARN] Alias mapping: {key} -> {mapped} for reproducible download/training.")
        return mapped
    return model_arg


def maybe_download_known_weights(model_arg: str) -> str:
    key = normalize_model_key(model_arg)
    release_urls = {
        # YOLOv7 official repo release
        "yolov7.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
        "yolov7-tiny.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
        # YOLOv10 official repo release
        "yolov10n.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt",
        "yolov10s.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt",
        "yolov10m.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
        "yolov10b.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt",
        "yolov10l.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt",
        "yolov10x.pt": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt",
    }
    if key not in release_urls:
        return model_arg

    model_path = Path(model_arg)
    if model_path.exists():
        return str(model_path)

    target = Path("weights") / key
    if target.exists():
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading YOLOv10 pretrained weight: {key}")
    urlretrieve(release_urls[key], target)
    print(f"[INFO] Saved to: {target}")
    return str(target)


def resolve_model_spec(model_arg: str) -> str:
    """Resolve model source across local file, URL, and official assets."""
    local_path = Path(model_arg)
    if local_path.exists():
        return str(local_path)

    model_arg = apply_paper_alias(model_arg)
    key = normalize_model_key(model_arg)
    if "yolov10" in key or "yolov7" in key:
        # v10/v7 weights are hosted in their upstream releases.
        return maybe_download_known_weights(model_arg)

    # For v8/v9/other Ultralytics assets, try built-in downloader.
    try:
        resolved = attempt_download_asset(model_arg)
        if Path(resolved).exists():
            return str(resolved)
    except Exception as exc:
        print(f"[WARN] Auto-download attempt failed: {exc}")

    raise FileNotFoundError(
        f"Model not found or not downloadable automatically: {model_arg}\n"
        "Tips:\n"
        "- Use a local path to existing .pt/.yaml\n"
        "- Or use names like yolov7-tiny.pt, yolov7.pt, yolov8s.pt, yolov9c.pt, yolov9e.pt, yolov10s.pt\n"
        "- If paper uses yolov9s/yolov9l, this script maps them to yolov9c/yolov9e\n"
        "- Or pass a direct URL to the weight file"
    )


def select_yolo_class(model_arg: str):
    key = normalize_model_key(model_arg)
    if "yolov10" in key:
        return YOLOv10
    return YOLO


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model_spec = resolve_model_spec(args.model)
    yolo_class = select_yolo_class(model_spec)
    print(f"[INFO] Using class: {yolo_class.__name__}, model: {model_spec}")

    # Keep paper-compatible defaults while allowing generic usage.
    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        momentum=0.937 if args.optimizer == "SGD" else 0.9,
        weight_decay=5e-4,
        amp=args.amp,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
    )

    model = yolo_class(model_spec)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
