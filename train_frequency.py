import argparse
from pathlib import Path

import yaml
from ultralytics import YOLOv10


STAGES = ("baseline", "fdaf", "fdaf_lsg", "full")
MODEL_SIZES = ("s", "l")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper-aligned YOLOv10-S/FEM training and ablation runner."
    )
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--weights", default="weights/yolov10s.pt")
    parser.add_argument(
        "--model-size",
        choices=MODEL_SIZES,
        default="s",
        help="YOLOv10 model size to build. Use 'l' with yolov10l.pt for YOLOv10-L experiments.",
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="full",
        help="baseline; FDAF only; FDAF+LSG; or complete FEM",
    )
    parser.add_argument("--resume", help="Resume an interrupted run from last.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/paper_repro")
    parser.add_argument("--name", help="Run name; defaults to yolov10s_<stage>")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--freq-beta", type=float, default=0.05)
    parser.add_argument("--freq-lambda", type=float, default=1.0)
    parser.add_argument("--freq-roi-size", type=int, default=16)
    parser.add_argument("--amp", action="store_true", help="Enable AMP; paper does not specify it")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def load_dataset_info(data_yaml_path: Path):
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    with data_yaml_path.open("r", encoding="utf-8") as file:
        data_cfg = yaml.safe_load(file)

    names = data_cfg.get("names")
    nc = data_cfg.get("nc")
    if names is None and nc is None:
        raise ValueError("Dataset YAML must contain either 'names' or 'nc'.")

    if names is not None:
        if isinstance(names, dict):
            names = dict(sorted(names.items(), key=lambda item: int(item[0])))
        elif not isinstance(names, list):
            raise TypeError("'names' must be a dict or list in dataset YAML.")
        names_count = len(names)
        if nc is not None and int(nc) != names_count:
            raise ValueError(f"Inconsistent dataset YAML: nc={nc}, len(names)={names_count}.")
        nc = names_count

    return int(nc)


def model_family_params(model_size):
    if model_size == "s":
        return {
            "scales": {"s": [0.33, 0.50, 1024]},
            "backbone_cib_args": [1024, True, True],
            "neck_p4_block": "C2f",
            "neck_p4_args": [512],
            "neck_p5_args": [1024, True, True],
        }
    if model_size == "l":
        return {
            "scales": {"l": [1.00, 1.00, 512]},
            "backbone_cib_args": [1024, True],
            "neck_p4_block": "C2fCIB",
            "neck_p4_args": [512, True],
            "neck_p5_args": [1024, True],
        }
    raise ValueError(f"Unsupported model size: {model_size}")


def build_model_config(nc, stage, model_size, freq_beta, freq_lambda, freq_roi_size):
    use_fdaf = stage != "baseline"
    use_gate = stage in {"fdaf_lsg", "full"}
    use_frequency_loss = stage == "full"
    params = model_family_params(model_size)

    backbone = [
        [-1, 1, "Conv", [64, 3, 2]],
        [-1, 1, "Conv", [128, 3, 2]],
        [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]],
        [-1, 6, "C2f", [256, True]],
        [-1, 1, "SCDown", [512, 3, 2]],
        [-1, 6, "C2f", [512, True]],
        [-1, 1, "SCDown", [1024, 3, 2]],
        [-1, 3, "C2fCIB", params["backbone_cib_args"]],
        [-1, 1, "SPPF", [1024, 5]],
        [-1, 1, "PSA", [1024]],
    ]

    if use_fdaf:
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 3, params["neck_p4_block"], params["neck_p4_args"]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "FDAF", [256, use_gate]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 3, params["neck_p4_block"], params["neck_p4_args"]],
            [-1, 1, "FDAF", [512, use_gate]],
            [-1, 1, "SCDown", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 3, "C2fCIB", params["neck_p5_args"]],
            [-1, 1, "FDAF", [1024, use_gate]],
            [[17, 21, 25], 1, "v10Detect", ["nc"]],
        ]
    else:
        head = [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 3, params["neck_p4_block"], params["neck_p4_args"]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 3, "C2f", [256]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 13], 1, "Concat", [1]],
            [-1, 3, params["neck_p4_block"], params["neck_p4_args"]],
            [-1, 1, "SCDown", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 3, "C2fCIB", params["neck_p5_args"]],
            [[16, 19, 22], 1, "v10Detect", ["nc"]],
        ]

    return {
        "nc": nc,
        "scales": params["scales"],
        "model_size": model_size,
        "repro_stage": stage,
        "freq_loss_branch": "one2many" if use_frequency_loss else "none",
        "freq_beta": freq_beta,
        "freq_lambda": freq_lambda,
        "freq_roi_size": freq_roi_size,
        "backbone": backbone,
        "head": head,
    }


def load_pretrained_weights(model, weights_path, stage):
    if stage == "baseline":
        model.load(str(weights_path))
        return

    # FDAF layers are inserted after the standard P3/P4/P5 neck outputs. Remap
    # subsequent layer indices so the remaining pretrained neck/head tensors
    # are not silently discarded by exact-name loading.
    source = YOLOv10(str(weights_path)).model.state_dict()
    target = model.model.state_dict()
    source_to_target_layer = {
        **{index: index for index in range(17)},
        17: 18,
        18: 19,
        19: 20,
        20: 22,
        21: 23,
        22: 24,
        23: 26,
    }

    transferred = {}
    for source_key, value in source.items():
        parts = source_key.split(".", 2)
        if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
            continue
        target_layer = source_to_target_layer.get(int(parts[1]))
        if target_layer is None:
            continue
        target_key = f"model.{target_layer}.{parts[2]}"
        if target_key in target and target[target_key].shape == value.shape:
            transferred[target_key] = value

    incompatible = model.model.load_state_dict(transferred, strict=False)
    print(
        f">>> Transferred {len(transferred)}/{len(target)} compatible tensors "
        "with FDAF layer-index remapping"
    )
    print(f">>> New or class-specific tensors: {len(incompatible.missing_keys)}")


def main():
    args = parse_args()
    if args.resume:
        checkpoint = Path(args.resume)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f">>> Resuming training from {checkpoint}")
        YOLOv10(str(checkpoint)).train(resume=True)
        return

    data_yaml_path = Path(args.data)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    nc = load_dataset_info(data_yaml_path)
    model_cfg = build_model_config(
        nc, args.stage, args.model_size, args.freq_beta, args.freq_lambda, args.freq_roi_size
    )
    generated_dir = Path("generated_configs")
    generated_dir.mkdir(exist_ok=True)
    model_yaml_path = generated_dir / f"yolov10{args.model_size}_{args.stage}.yaml"
    with model_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(model_cfg, file, sort_keys=False)

    print(f">>> Building YOLOv10-{args.model_size.upper()} stage={args.stage}")
    model = YOLOv10(str(model_yaml_path))
    load_pretrained_weights(model, weights_path, args.stage)

    run_name = args.name or f"yolov10{args.model_size}_{args.stage}"
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer="SGD",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        deterministic=True,
        close_mosaic=args.close_mosaic,
        mosaic=1.0,
        mixup=0.0,
        amp=args.amp,
        cache=args.cache,
        cos_lr=args.cos_lr,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=run_name,
        plots=True,
    )


if __name__ == "__main__":
    main()
