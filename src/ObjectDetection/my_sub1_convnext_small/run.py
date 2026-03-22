"""
Submission entry point for the NM i AI Object Detection task.

Two-stage pipeline:
  Stage 1: ONNX detector (nc=1) finds all product bounding boxes.
  Stage 2: ONNX classifier embeds each crop, classifies by cosine
           similarity to pre-computed reference embeddings.

The sandbox executes:
    python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DETECT_IMGSZ = 1280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ── Config ────────────────────────────────────────────────────────────

def load_config():
    """Load model config, with fallback defaults for backward compatibility."""
    config_path = SCRIPT_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {"arch": "convnext_small", "crop_size": 224, "embed_dim": 768}


def preprocess_crop(img, crop_size):
    """Resize, normalize, return (1, 3, H, W) float32 array."""
    img = img.resize((crop_size, crop_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr[np.newaxis]  # (1, 3, H, W)


# ── ONNX detector helpers ───────────────────────────────────────────

def letterbox(img, size):
    """Resize with padding to square, return array and scale info."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis], scale, pad_x, pad_y


def nms(boxes, scores, iou_thresh=0.5):
    """Simple greedy NMS."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * \
                 (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def detect_onnx(session, img, conf_thresh=0.25):
    """Run ONNX detector and return list of (x1, y1, x2, y2, conf)."""
    input_tensor, scale, pad_x, pad_y = letterbox(img, DETECT_IMGSZ)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})[0]

    pred = output[0]  # (5, N) for nc=1: [cx, cy, w, h, conf]
    if pred.shape[0] == 5:
        cx, cy, bw, bh, conf = pred
    else:
        cx, cy, bw, bh, conf = pred.T[:5] if pred.shape[1] >= 5 else pred[:, :5].T

    mask = conf > conf_thresh
    cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]

    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, conf)
    return [(x1[i], y1[i], x2[i], y2[i], conf[i]) for i in keep]


def l2_normalize(x):
    """L2-normalize rows of a 2-D array."""
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


# ── Model loading ────────────────────────────────────────────────────

def load_models():
    """Load detector and classifier ONNX models + reference embeddings."""
    config = load_config()
    crop_size = config['crop_size']

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Stage 1 — ONNX detector
    detector = ort.InferenceSession(
        str(SCRIPT_DIR / "detector.onnx"), providers=providers
    )

    # Stage 2 — ONNX classifier (backbone exported to ONNX)
    classifier = ort.InferenceSession(
        str(SCRIPT_DIR / "classifier.onnx"), providers=providers
    )

    # Reference embeddings (numpy .npz)
    data = np.load(str(SCRIPT_DIR / "ref_embeddings.npz"))
    ref_cat_ids = data["cat_ids"].tolist()
    ref_embeds = l2_normalize(data["embeddings"].astype(np.float32))

    print(f"Config: arch={config.get('arch')}, crop_size={crop_size}")
    return detector, classifier, ref_embeds, ref_cat_ids, crop_size


# ── Inference ────────────────────────────────────────────────────────

def predict_image(img_path, detector, classifier, ref_embeds, ref_cat_ids,
                  crop_size):
    """Run two-stage inference on a single image."""
    img = Image.open(img_path).convert("RGB")
    detections = detect_onnx(detector, img)

    cls_input_name = classifier.get_inputs()[0].name

    preds = []
    for x1, y1, x2, y2, det_score in detections:
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        if crop.width < 4 or crop.height < 4:
            continue

        # Classify crop via embedding similarity
        tensor = preprocess_crop(crop, crop_size)
        emb = classifier.run(None, {cls_input_name: tensor})[0]  # (1, D)
        emb = l2_normalize(emb)
        sims = (emb @ ref_embeds.T).squeeze(0)
        best_idx = int(sims.argmax())

        preds.append({
            "category_id": ref_cat_ids[best_idx],
            "bbox": [
                round(float(x1), 1), round(float(y1), 1),
                round(float(x2 - x1), 1), round(float(y2 - y1), 1),
            ],
            "score": round(float(det_score) * float(sims[best_idx]), 3),
        })
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    detector, classifier, ref_embeds, ref_cat_ids, crop_size = load_models()
    print("Models loaded")

    predictions = []
    input_dir = Path(args.input)
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        preds = predict_image(
            str(img_path), detector, classifier, ref_embeds, ref_cat_ids,
            crop_size
        )
        for p in preds:
            p["image_id"] = image_id
            predictions.append(p)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
