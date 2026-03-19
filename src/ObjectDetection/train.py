import os
from pathlib import Path

from ultralytics import YOLO


def train(data: str, epochs: int = 50, imgsz: int = 640, model: str = "yolov8n.pt"):
    model = YOLO(model)
    results = model.train(data=data, epochs=epochs, imgsz=imgsz)
    return results


if __name__ == "__main__":
    # Prepare dataset (COCO -> YOLO) if not already done
    train_imgs = Path("data/images/train")
    if not train_imgs.exists() or not any(train_imgs.iterdir()):
        print("Preparing dataset...")
        from ObjectDetection.prepare_dataset import prepare, write_yaml
        categories, cat_ids = prepare()
        write_yaml(categories, cat_ids)

    train(data="data/dataset.yaml")
