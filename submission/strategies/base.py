from abc import ABC, abstractmethod
from pathlib import Path


class Strategy(ABC):
    """Base class for all detection/classification strategies."""

    name: str = "base"

    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir

    @abstractmethod
    def load(self):
        """Load model weights into memory."""

    @abstractmethod
    def predict(self, img_path: str, device: str = "cpu") -> list[dict]:
        """Run inference on a single image.

        Returns a list of COCO-format dicts:
            {"image_id": int, "category_id": int,
             "bbox": [x, y, w, h], "score": float}
        (image_id is filled in by the caller)
        """
