"""Strategy 2: Detection-only with nc=1 (class-agnostic).

All predictions get category_id=0. Maximises the detection component
which is 70 % of the competition score.
"""
import torch
from pathlib import Path
from ultralytics import YOLO
from strategies.base import Strategy


class DetectionOnly(Strategy):
    name = "detection_only"

    def __init__(self, weights_dir: Path):
        super().__init__(weights_dir)
        self.model = None

    def load(self):
        self.model = YOLO(str(self.weights_dir / "detector.onnx"))

    def predict(self, img_path: str, device: str = "cpu") -> list[dict]:
        with torch.no_grad():
            results = self.model(img_path, device=device, verbose=False)

        preds = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                preds.append({
                    "category_id": 0,
                    "bbox": [
                        round(x1, 1), round(y1, 1),
                        round(x2 - x1, 1), round(y2 - y1, 1),
                    ],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })
        return preds
