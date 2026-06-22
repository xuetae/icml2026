import argparse
import math
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit YOLO detection labels before training.")
    parser.add_argument("--labels", required=True, help="Directory containing YOLO .txt labels")
    parser.add_argument("--nc", type=int, required=True)
    parser.add_argument("--max-examples", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    labels_dir = Path(args.labels)
    if not labels_dir.is_dir():
        raise NotADirectoryError(labels_dir)

    counts = Counter()
    examples = []
    for path in sorted(labels_dir.glob("*.txt")):
        counts["files"] += 1
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            counts["empty_files"] += 1

        seen = set()
        for line_number, line in enumerate(lines, 1):
            counts["rows"] += 1
            parts = line.split()
            problem = None
            if len(parts) != 5:
                problem = f"expected 5 columns, found {len(parts)}"
                counts["bad_columns"] += 1
            else:
                try:
                    cls_value, x, y, width, height = map(float, parts)
                except ValueError:
                    problem = "non-numeric value"
                    counts["non_numeric"] += 1
                else:
                    values = (cls_value, x, y, width, height)
                    if not all(math.isfinite(value) for value in values):
                        problem = "NaN or Inf"
                        counts["non_finite"] += 1
                    elif cls_value != int(cls_value) or not 0 <= int(cls_value) < args.nc:
                        problem = f"class {cls_value} outside [0, {args.nc - 1}]"
                        counts["bad_class"] += 1
                    elif not (0 <= x <= 1 and 0 <= y <= 1):
                        problem = f"center outside image: {(x, y)}"
                        counts["bad_center"] += 1
                    elif not (0 < width <= 1 and 0 < height <= 1):
                        problem = f"invalid size: {(width, height)}"
                        counts["bad_size"] += 1
                    elif (
                        x - width / 2 < -1e-6
                        or y - height / 2 < -1e-6
                        or x + width / 2 > 1 + 1e-6
                        or y + height / 2 > 1 + 1e-6
                    ):
                        problem = "box crosses image boundary"
                        counts["out_of_bounds"] += 1

            normalized_line = " ".join(parts)
            if normalized_line in seen:
                counts["duplicate_rows"] += 1
                problem = problem or "duplicate row"
            seen.add(normalized_line)

            if problem and len(examples) < args.max_examples:
                examples.append(f"{path}:{line_number}: {problem}: {line}")

    error_keys = (
        "bad_columns",
        "non_numeric",
        "non_finite",
        "bad_class",
        "bad_center",
        "bad_size",
        "out_of_bounds",
        "duplicate_rows",
    )
    print("Dataset audit")
    for key in ("files", "rows", "empty_files", *error_keys):
        print(f"{key}: {counts[key]}")

    if examples:
        print("\nExamples")
        print("\n".join(examples))

    error_count = sum(counts[key] for key in error_keys)
    if error_count:
        raise SystemExit(f"Audit failed with {error_count} label issues.")
    print("Audit passed.")


if __name__ == "__main__":
    main()
