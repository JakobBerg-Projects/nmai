from strategies.base import Strategy
from strategies.yolo_multiclass import YoloMulticlass
from strategies.detection_only import DetectionOnly
from strategies.two_stage import TwoStage

STRATEGIES = {
    "yolo_multiclass": YoloMulticlass,
    "detection_only": DetectionOnly,
    "two_stage": TwoStage,
}


def auto_detect(weights_dir):
    """Pick the best available strategy based on which weight files exist."""
    if (weights_dir / "detector.pt").exists() and (weights_dir / "classifier.pt").exists():
        return TwoStage(weights_dir)
    if (weights_dir / "detector.pt").exists():
        return DetectionOnly(weights_dir)
    if (weights_dir / "best.pt").exists():
        return YoloMulticlass(weights_dir)
    raise FileNotFoundError(f"No weight files found in {weights_dir}")
