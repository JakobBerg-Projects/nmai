"""
Submission entry point for the NM i AI Object Detection task.

The sandbox executes:
    python run.py --input /data/images --output /output/predictions.json

Auto-detects which strategy to use based on available weight files:
    weights/detector.pt + weights/classifier.pt  → two_stage
    weights/detector.pt                          → detection_only
    weights/best.pt                              → yolo_multiclass
    best.pt (legacy)                             → yolo_multiclass
"""
import argparse
import json
from pathlib import Path

import torch

from strategies import auto_detect

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with shelf images")
    parser.add_argument("--output", required=True, help="Path to write predictions.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect strategy, falling back to legacy best.pt location
    try:
        strategy = auto_detect(WEIGHTS_DIR)
    except FileNotFoundError:
        if (SCRIPT_DIR / "best.pt").exists():
            # Legacy: best.pt in root — symlink into weights/
            WEIGHTS_DIR.mkdir(exist_ok=True)
            dst = WEIGHTS_DIR / "best.pt"
            if not dst.exists():
                import shutil
                shutil.copy2(SCRIPT_DIR / "best.pt", dst)
            strategy = auto_detect(WEIGHTS_DIR)
        else:
            raise

    print(f"Strategy: {strategy.name}")
    strategy.load()

    predictions = []
    input_dir = Path(args.input)
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        preds = strategy.predict(str(img_path), device=device)

        for p in preds:
            p["image_id"] = image_id
            predictions.append(p)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
