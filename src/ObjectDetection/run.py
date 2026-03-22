"""
Submission entry point for the NM i AI Object Detection task.

Two-stage pipeline:
  Stage 1: ONNX detector (nc=1) finds all product bounding boxes.
  Stage 2: Embedding model classifies each crop by cosine similarity
           to pre-computed reference embeddings.

The sandbox executes:
    python run.py --input /data/images --output /output/predictions.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
DETECT_IMGSZ = 1280


# ── Config ────────────────────────────────────────────────────────────

def load_config():
    """Load model config, with fallback defaults for backward compatibility."""
    config_path = SCRIPT_DIR / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {"arch": "resnet50", "crop_size": 224, "embed_dim": 2048}


def build_backbone(arch):
    """Build backbone model for inference (no pretrained weights)."""
    if arch == 'resnet50':
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        model.fc = nn.Identity()
        return model
    elif arch == 'efficientnet_v2_s':
        from torchvision.models import efficientnet_v2_s
        model = efficientnet_v2_s(weights=None)
        model.classifier = nn.Identity()
        return model
    elif arch == 'convnext_tiny':
        from torchvision.models import convnext_tiny
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Identity()
        return model
    elif arch == 'convnext_small':
        from torchvision.models import convnext_small
        model = convnext_small(weights=None)
        model.classifier[2] = nn.Identity()
        return model
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def get_crop_transform(crop_size):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


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


# ── Model loading ────────────────────────────────────────────────────

def load_models():
    """Load detector and classifier models."""
    config = load_config()
    arch = config['arch']
    crop_size = config['crop_size']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage 1 — ONNX detector (no pickle, no ultralytics dependency)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    detector = ort.InferenceSession(
        str(SCRIPT_DIR / "detector.onnx"), providers=providers
    )

    # Stage 2 — embedder + reference gallery
    embedder = build_backbone(arch)
    state = torch.load(
        str(SCRIPT_DIR / "classifier.pt"), map_location=device, weights_only=True
    )
    embedder.load_state_dict(state)
    embedder.to(device).eval()

    refs = torch.load(
        str(SCRIPT_DIR / "ref_embeddings.pt"), map_location=device, weights_only=True
    )
    ref_cat_ids = list(refs.keys())
    ref_embeds = F.normalize(torch.stack([refs[c] for c in ref_cat_ids]), dim=1)

    crop_transform = get_crop_transform(crop_size)

    print(f"Config: arch={arch}, crop_size={crop_size}")
    return detector, embedder, ref_embeds, ref_cat_ids, device, crop_transform


# ── Inference ────────────────────────────────────────────────────────

def predict_image(img_path, detector, embedder, ref_embeds, ref_cat_ids,
                  device, crop_transform):
    """Run two-stage inference on a single image."""
    img = Image.open(img_path).convert("RGB")
    detections = detect_onnx(detector, img)

    preds = []
    for x1, y1, x2, y2, det_score in detections:
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        if crop.width < 4 or crop.height < 4:
            continue

        # Classify crop via embedding similarity
        tensor = crop_transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = F.normalize(embedder(tensor), dim=1)
        sims = (emb @ ref_embeds.T).squeeze(0)
        best_idx = sims.argmax().item()

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

    detector, embedder, ref_embeds, ref_cat_ids, device, crop_transform = load_models()
    print(f"Models loaded on {device}")

    predictions = []
    input_dir = Path(args.input)
    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        image_id = int(img_path.stem.split("_")[-1])
        preds = predict_image(
            str(img_path), detector, embedder, ref_embeds, ref_cat_ids,
            device, crop_transform
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
