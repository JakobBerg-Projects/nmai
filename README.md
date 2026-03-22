# nmai

Baseline object detection using YOLOv8.

## Setup

```bash
pip install -r requirements.txt
```

## Training

Edit `data/dataset.yaml` to point to your dataset, then:

```bash
python src/train.py
```

## Inference

```bash
python src/detect.py
```

## Dataset format

Images go in `data/images/train/` and `data/images/val/`.
Labels go in `data/labels/train/` and `data/labels/val/` (YOLO `.txt` format).

Each label file contains one row per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1].
