"""
Submission entry point for the NM i AI Object Detection task.

The sandbox executes:
    python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

from strategies import auto_detect

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    strategy = auto_detect(WEIGHTS_DIR)
    print(f"Strategy: {strategy.name}")
    strategy.load()

    predictions = []
    input_dir = Path(args.input)
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        preds = strategy.predict(str(img_path), device="cuda")

        for p in preds:
            p["image_id"] = image_id
            predictions.append(p)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
