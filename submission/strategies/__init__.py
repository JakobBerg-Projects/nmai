from strategies.base import Strategy
from strategies.two_stage import TwoStage


def auto_detect(weights_dir):
    """Pick the best available strategy based on which weight files exist."""
    if (weights_dir / "detector.onnx").exists() and (weights_dir / "classifier.pt").exists():
        return TwoStage(weights_dir)
    raise FileNotFoundError(f"No weight files found in {weights_dir}")
