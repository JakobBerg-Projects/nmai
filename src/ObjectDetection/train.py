"""
Unified training entry point.

Usage:
    # Strategy 1 — Multiclass end-to-end (nc=356):
    python train.py --strategy multiclass --model yolov8x.pt --epochs 100

    # Strategy 2 — Detection-only (nc=1):
    python train.py --strategy detector --model yolov8x.pt --epochs 100

    # Strategy 3 — Two-stage (detector + classifier):
    python train.py --strategy two_stage --model yolov8x.pt --epochs 100 --cls-epochs 30

    # Just the classifier (detector must be trained first):
    python train.py --strategy classifier --cls-epochs 30
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",
                        choices=["multiclass", "detector", "classifier", "two_stage"],
                        default="detector",
                        help="Which strategy to train")
    parser.add_argument("--model", default="yolov8x.pt", help="YOLO model variant")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cls-epochs", type=int, default=30, dest="cls_epochs")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    args = parser.parse_args()

    if args.strategy in ("multiclass", "detector"):
        from training.train_detector import train
        mode = "multi" if args.strategy == "multiclass" else "single"
        train(mode=mode, model_name=args.model, epochs=args.epochs, imgsz=args.imgsz)

    elif args.strategy == "classifier":
        from training.extract_crops import extract_annotation_crops, copy_reference_images
        from training.train_classifier import train
        print("=== Extracting crops ===")
        extract_annotation_crops()
        copy_reference_images()
        print("\n=== Training classifier ===")
        train(epochs=args.cls_epochs, batch_size=args.batch_size, lr=1e-3)

    elif args.strategy == "two_stage":
        from training.train_detector import train as train_det
        from training.extract_crops import extract_annotation_crops, copy_reference_images
        from training.train_classifier import train as train_cls

        print("=== Phase 1: Training single-class detector ===")
        train_det(mode="single", model_name=args.model,
                  epochs=args.epochs, imgsz=args.imgsz)

        print("\n=== Phase 2: Extracting crops ===")
        extract_annotation_crops()
        copy_reference_images()

        print("\n=== Phase 3: Training classifier ===")
        train_cls(epochs=args.cls_epochs, batch_size=args.batch_size, lr=1e-3)


if __name__ == "__main__":
    main()
