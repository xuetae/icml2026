import argparse
import json
from pathlib import Path

from ultralytics import YOLOv10


PAPER_TARGETS = {
    "baseline": {"map": 0.463, "map50": 0.627, "map75": 0.513},
    "fdaf": {"map": 0.485, "map50": 0.642, "map75": 0.528},
    "fdaf_lsg": {"map": 0.509, "map50": 0.658, "map75": 0.539},
    "full": {"map": 0.521, "map50": 0.671, "map75": 0.546},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a reproduction checkpoint.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--stage", choices=PAPER_TARGETS, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/paper_eval")
    parser.add_argument("--name")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    run_name = args.name or f"yolov10s_{args.stage}_{args.split}"
    metrics = YOLOv10(str(model_path)).val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=args.project,
        name=run_name,
    )

    actual = {
        "map": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
    }
    target = PAPER_TARGETS[args.stage]
    report = {
        "stage": args.stage,
        "split": args.split,
        "model": str(model_path),
        "actual": actual,
        "paper_target": target,
        "gap_percentage_points": {
            key: round((actual[key] - target[key]) * 100, 3) for key in actual
        },
    }

    output_dir = Path(args.project) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paper_comparison.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
