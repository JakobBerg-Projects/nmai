"""End-to-end validation for the two-stage pipeline.

Caches detector results and crop tensors once, then re-classifies each epoch.
    Score = 0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5
"""
import contextlib
import io
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent

def get_crop_transform(crop_size=224):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_data():
    """Load annotations and return val image IDs matching prepare_dataset split."""
    ann_path = ROOT / "data" / "train" / "annotations.json"
    with open(ann_path) as f:
        coco = json.load(f)

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    annotated_ids = list(anns_by_image.keys())
    random.seed(42)
    random.shuffle(annotated_ids)
    split_idx = int(len(annotated_ids) * 0.8)
    val_ids = set(annotated_ids[split_idx:])

    return val_ids, coco


def cache_detections_and_crops(detector_path, val_ids, coco, imgsz=1280, conf=0.25,
                               crop_size=224):
    """Run detector on val images ONCE and pre-crop all detections.

    Returns (crops_list, val_gt_coco_dict, det_mAP50).
    det_mAP50 is read from the marker file written by detector training.
    """
    from ultralytics import YOLO

    # --- Read det_mAP50 from detector training's marker file ---
    weights_dir = Path(detector_path).parent
    marker_path = weights_dir / "detector_updated"
    if marker_path.exists():
        det_mAP50 = float(marker_path.read_text().strip())
        print(f"  Detector mAP@0.5 (from training): {det_mAP50:.4f}")
    else:
        print("  WARNING: detector_updated marker not found, using det_mAP50=0.0")
        det_mAP50 = 0.0

    model = YOLO(str(detector_path))

    # --- Cache crops at conf=0.25 for classification ---
    crop_transform = get_crop_transform(crop_size)
    img_info = {img["id"]: img for img in coco["images"]}
    images_dir = ROOT / "data" / "train" / "images"

    crops = []
    for img_id in sorted(val_ids):
        info = img_info[img_id]
        img_path = images_dir / info["file_name"]
        if not img_path.exists():
            continue

        results = model(str(img_path), imgsz=imgsz, conf=conf, verbose=False)
        img_pil = Image.open(img_path).convert("RGB")

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            crop = img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.width < 4 or crop.height < 4:
                continue

            crops.append({
                "image_id": img_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "tensor": crop_transform(crop),
                "det_conf": float(box.conf[0].cpu()),
            })

    # Free YOLO model from GPU before classifier loads
    del model
    torch.cuda.empty_cache()

    # Build val ground truth in COCO format
    val_gt = {
        "images": [img for img in coco["images"] if img["id"] in val_ids],
        "annotations": [],
        "categories": coco["categories"],
    }
    ann_id = 1
    for ann in coco["annotations"]:
        if ann["image_id"] in val_ids:
            val_gt["annotations"].append({**ann, "id": ann_id})
            ann_id += 1

    print(f"  Cached {len(crops)} detection crops from {len(val_ids)} val images")
    return crops, val_gt, det_mAP50


def evaluate(crops, val_gt, classifier_model, cat_ids, device, det_mAP50,
             batch_size=256):
    """Classify cached crops and compute combined score.

    Returns (combined_score, det_mAP50, cls_mAP50).
    """
    if not crops:
        return 0.0, det_mAP50, 0.0

    classifier_model.eval()
    predictions = []

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i + batch_size]
        tensors = torch.stack([c["tensor"] for c in batch]).to(device)

        with torch.no_grad():
            logits = classifier_model(tensors)
        probs = torch.softmax(logits, dim=1)
        cls_idxs = probs.argmax(dim=1)
        cls_confs = probs.gather(1, cls_idxs.unsqueeze(1)).squeeze(1)

        for j, crop_info in enumerate(batch):
            predictions.append({
                "image_id": crop_info["image_id"],
                "bbox": crop_info["bbox"],
                "category_id": int(cat_ids[int(cls_idxs[j])]),
                "score": float(crop_info["det_conf"] * float(cls_confs[j])),
            })

    cls_map = _map50(val_gt, predictions, category_agnostic=False)
    combined = 0.7 * det_mAP50 + 0.3 * cls_map
    return combined, det_mAP50, cls_map


def _map50(val_gt, predictions, category_agnostic=False):
    """Compute mAP@0.5 using pycocotools."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if category_agnostic:
        gt = json.loads(json.dumps(val_gt))
        gt["categories"] = [{"id": 0, "name": "product", "supercategory": "product"}]
        for ann in gt["annotations"]:
            ann["category_id"] = 0
        preds = [{**p, "category_id": 0} for p in predictions]
    else:
        gt = val_gt
        preds = predictions

    if not preds:
        return 0.0

    gt_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(gt, f)
            gt_path = f.name

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(gt_path)
            coco_dt = coco_gt.loadRes(preds)
            ev = COCOeval(coco_gt, coco_dt, "bbox")
            ev.params.iouThrs = np.array([0.5])
            ev.evaluate()
            ev.accumulate()
            ev.summarize()

        return float(ev.stats[0])
    finally:
        if gt_path:
            os.unlink(gt_path)
