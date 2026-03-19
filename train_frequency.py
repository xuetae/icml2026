import argparse
from pathlib import Path

import yaml
from ultralytics import YOLOv10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frequency-enhanced YOLOv10 training (FDAF) with dataset yaml input."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="F:/nus/icml/split_5/label.yaml",
        help="Dataset yaml path",
    )
    parser.add_argument("--weights", type=str, default="yolov10s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="train_custom_yolov10_frequency")
    return parser.parse_args()


def load_dataset_info(data_yaml_path: Path):
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg.get("names")
    nc = data_cfg.get("nc")

    if names is None and nc is None:
        raise ValueError("Dataset yaml must contain either 'names' or 'nc'.")

    if names is not None:
        if isinstance(names, dict):
            names = dict(sorted(names.items(), key=lambda x: int(x[0])))
            nc_from_names = len(names)
        elif isinstance(names, list):
            nc_from_names = len(names)
        else:
            raise TypeError("'names' must be a dict or list in dataset yaml.")

        if nc is None:
            nc = nc_from_names
        elif int(nc) != nc_from_names:
            raise ValueError(f"Inconsistent dataset yaml: nc={nc}, but len(names)={nc_from_names}.")
    else:
        nc = int(nc)
        names = {i: str(i) for i in range(nc)}

    return int(nc), names


def build_frequency_model_config(nc: int):
    return {
        "nc": nc,
        "scales": {"s": [0.33, 0.50, 1024]},
        "backbone": [
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 3, "C2f", [128, True]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 6, "C2f", [256, True]],
            [-1, 1, "SCDown", [512, 3, 2]],
            [-1, 6, "C2f", [512, True]],
            [-1, 1, "SCDown", [1024, 3, 2]],
            [-1, 3, "C2fCIB", [1024, True, True]],
            [-1, 1, "SPPF", [1024, 5]],
            [-1, 1, "PSA", [1024]],
        ],
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 3, "C2f", [512]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "FDAF", [256]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 3, "C2f", [512]],
            [-1, 1, "FDAF", [512]],
            [-1, 1, "SCDown", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 3, "C2fCIB", [1024, True, True]],
            [-1, 1, "FDAF", [1024]],
            [[17, 21, 25], 1, "v10Detect", ["nc"]],
        ],
    }


def main():
    args = parse_args()
    data_yaml_path = Path(args.data)
    nc, _ = load_dataset_info(data_yaml_path)

    model_cfg = build_frequency_model_config(nc)
    model_yaml_path = Path("model_config_frequency.yaml")
    with model_yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(model_cfg, f, allow_unicode=True)

    print(">>> Loading frequency-enhanced YOLOv10 architecture...")
    model = YOLOv10(str(model_yaml_path))

    print(f">>> Loading pretrained weights: {args.weights}")
    model.load(args.weights)

    print(">>> Starting training...")
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        plots=True,
        workers=args.workers,
        amp=False,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
