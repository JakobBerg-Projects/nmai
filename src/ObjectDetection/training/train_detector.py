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

import ultralytics.utils.metrics as _metrics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"


def _map50_fitness(x):
    """Select best checkpoint by mAP@0.5 only (competition metric)."""
    w = [0.0, 0.0, 1.0, 0.0]  # [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


_metrics.fitness = _map50_fitness


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


def _eval_map50(model_path: str, data_yaml: str, imgsz: int) -> float:
    """Evaluate a YOLO model and return mAP@0.5."""
    m = YOLO(model_path)
    results = m.val(data=data_yaml, imgsz=imgsz, verbose=False)
    return float(results.box.map50)


def train(mode: str, model_name: str, epochs: int, imgsz: int, batch: int = 16,
          fp16: bool = False):
    WEIGHTS_DIR.mkdir(exist_ok=True)

    if mode == "single":
        data_yaml = prepare_single_class()
        out_name = "detector"
        dst = WEIGHTS_DIR / "detector.pt"
    else:
        data_yaml = prepare_multiclass()
        out_name = "multiclass"
        dst = WEIGHTS_DIR / "best.pt"

    marker_path = WEIGHTS_DIR / f"{dst.stem}_updated"
    marker_path.unlink(missing_ok=True)

    # Evaluate existing weights to establish baseline
    existing_map50 = 0.0
    if dst.exists():
        print(f"\n=== Evaluating existing {dst.name} ===")
        existing_map50 = _eval_map50(str(dst), str(data_yaml), imgsz)
        print(f"  Existing mAP@0.5: {existing_map50:.4f}")
        print(f"  New model must beat {existing_map50:.4f} to save weights\n")

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(ROOT / "runs"),
        name=out_name,
        exist_ok=True,
    )

    # Evaluate the new best checkpoint
    best_src = ROOT / "runs" / out_name / "weights" / "best.pt"
    new_map50 = _eval_map50(str(best_src), str(data_yaml), imgsz)
    print(f"\nNew model mAP@0.5:      {new_map50:.4f}")
    print(f"Existing model mAP@0.5: {existing_map50:.4f}")

    if new_map50 > existing_map50:
        shutil.copy2(best_src, dst)
        print(f"Improvement! Saved weights to {dst}")

        # Export to ONNX
        best_model = YOLO(str(dst))
        onnx_name = dst.stem + ".onnx"
        best_model.export(format="onnx", imgsz=imgsz, opset=17, half=fp16)
        if fp16:
            print(f"  Exported as FP16 (half precision)")
        onnx_src = dst.with_suffix(".onnx")
        onnx_dst = WEIGHTS_DIR / onnx_name
        if onnx_src != onnx_dst:
            shutil.move(str(onnx_src), str(onnx_dst))
        print(f"Exported ONNX to {onnx_dst}")

        marker_path.write_text(f"{new_map50:.6f}")
    else:
        print(f"No improvement — keeping existing {dst.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "multi"], default="single",
                        help="single = nc=1 (detection only), multi = nc=356")
    parser.add_argument("--model", default="yolov8x.pt", help="YOLO model variant")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fp16", action="store_true",
                        help="Export ONNX in FP16 (halves weight size)")
    args = parser.parse_args()

    train(mode=args.mode, model_name=args.model, epochs=args.epochs,
          imgsz=args.imgsz, batch=args.batch, fp16=args.fp16)
