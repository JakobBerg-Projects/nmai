"""Train a YOLOv8 detector — either nc=1 (class-agnostic) or nc=356 (multiclass).

Usage:
    # Detection-only (Strategy 2 / Stage 1 of Strategy 3):
    python -m training.train_detector --mode single --model yolov8x.pt --epochs 100

    # Multiclass end-to-end (Strategy 1):
    python -m training.train_detector --mode multi --model yolov8x.pt --epochs 100
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"


def prepare_single_class():
    """Prepare YOLO dataset with all categories mapped to class 0."""
    from prepare_dataset import prepare, ROOT as DS_ROOT

    output_dir = DS_ROOT / "data" / "yolo_single"
    if (output_dir / "images" / "train").exists():
        print(f"Single-class dataset already at {output_dir}")
        return output_dir / "dataset.yaml"

    categories = prepare(output_dir=output_dir, single_class=True)
    _write_single_yaml(output_dir)
    return output_dir / "dataset.yaml"


def _write_single_yaml(output_dir: Path):
    lines = [
        "# Single-class dataset (all products = class 0)",
        f"path: {output_dir}",
        "train: images/train",
        "val: images/val",
        "",
        "nc: 1",
        "names:",
        "  - 'product'",
    ]
    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {yaml_path}")
    return yaml_path


def prepare_multiclass():
    """Prepare YOLO dataset with all 356 categories."""
    from prepare_dataset import prepare, write_yaml, ROOT as DS_ROOT

    yaml_path = DS_ROOT / "data" / "yolo" / "dataset.yaml"
    if yaml_path.exists():
        print(f"Multiclass dataset already at {yaml_path.parent}")
        return yaml_path

    categories = prepare()
    write_yaml(categories)
    return yaml_path


def train(mode: str, model_name: str, epochs: int, imgsz: int):
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if mode == "single":
        data_yaml = prepare_single_class()
        out_name = "detector"
    else:
        data_yaml = prepare_multiclass()
        out_name = "multiclass"

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(ROOT / "runs"),
        name=out_name,
        exist_ok=True,
    )

    # Copy best weights to weights/ for run.py
    best_src = ROOT / "runs" / out_name / "weights" / "best.pt"
    if mode == "single":
        dst = WEIGHTS_DIR / "detector.pt"
    else:
        dst = WEIGHTS_DIR / "best.pt"
    shutil.copy2(best_src, dst)
    print(f"\nSaved weights to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single = nc=1 (detection only), multi = nc=356")
    parser.add_argument("--model", default="yolov8x.pt", help="YOLO model variant")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train(mode=args.mode, model_name=args.model, epochs=args.epochs, imgsz=args.imgsz)
