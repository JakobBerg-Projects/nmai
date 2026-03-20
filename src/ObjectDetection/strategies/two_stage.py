"""Strategy 3: Two-stage — detect (nc=1) then classify via embedding matching.

Stage 1: ONNX detector (nc=1) finds all product bounding boxes.
Stage 2: A MobileNetV3 backbone embeds each crop and classifies it by
         cosine-similarity to pre-computed reference embeddings.

Weight files expected in weights_dir:
    detector.onnx      — YOLOv8 detector exported to ONNX (nc=1)
    classifier.pt      — MobileNetV3 feature extractor (fine-tuned)
    ref_embeddings.pt  — {category_id: embedding} dict for all 356 classes
"""
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import onnxruntime as ort
from strategies.base import Strategy


CROP_SIZE = 224
EMBED_DIM = 576  # MobileNetV3-Small final feature dim

DETECT_IMGSZ = 640  # must match the ONNX export size

CROP_TRANSFORM = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _letterbox(img: Image.Image, size: int):
    """Resize with padding to square, return tensor and scale info."""
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return arr, scale, pad_x, pad_y


def _postprocess_onnx(output, scale, pad_x, pad_y, conf_thresh=0.25):
    """Parse raw ONNX output (1, 5, N) → list of [x1, y1, x2, y2, conf]."""
    # output shape: (1, 5, num_anchors) for nc=1 → [x, y, w, h, conf]
    pred = output[0]  # (5, N)
    if pred.shape[0] == 5:
        cx, cy, bw, bh, conf = pred
    else:
        pred = pred.T  # (N, 5)
        cx, cy, bw, bh, conf = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]

    mask = conf > conf_thresh
    cx, cy, bw, bh, conf = cx[mask], cy[mask], bw[mask], bh[mask], conf[mask]

    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale

    # NMS
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = _nms(boxes, conf, iou_thresh=0.5)

    return [(x1[i], y1[i], x2[i], y2[i], conf[i]) for i in keep]


def _nms(boxes, scores, iou_thresh=0.5):
    """Simple NMS."""
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
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def _build_embedder(weights_path: Path, device: str):
    """Load MobileNetV3-Small as a feature extractor."""
    model = mobilenet_v3_small(weights=None)
    # Replace classifier head with identity to get embeddings
    model.classifier = torch.nn.Identity()
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


class TwoStage(Strategy):
    name = "two_stage"

    def __init__(self, weights_dir: Path):
        super().__init__(weights_dir)
        self.detector = None
        self.embedder = None
        self.ref_embeds = None   # tensor (num_classes, embed_dim)
        self.ref_cat_ids = None  # list of category_ids aligned with ref_embeds
        self.device = "cpu"

    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stage 1 — ONNX detector
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.detector = ort.InferenceSession(
            str(self.weights_dir / "detector.onnx"), providers=providers
        )

        # Stage 2 — embedder + reference gallery
        self.embedder = _build_embedder(
            self.weights_dir / "classifier.pt", self.device
        )
        refs = torch.load(
            str(self.weights_dir / "ref_embeddings.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.ref_cat_ids = list(refs.keys())
        self.ref_embeds = torch.stack([refs[cid] for cid in self.ref_cat_ids])
        self.ref_embeds = F.normalize(self.ref_embeds, dim=1)

    def _classify_crop(self, crop: Image.Image) -> tuple[int, float]:
        """Embed a crop and return (category_id, confidence)."""
        tensor = CROP_TRANSFORM(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedder(tensor)
        emb = F.normalize(emb, dim=1)
        sims = (emb @ self.ref_embeds.T).squeeze(0)
        best_idx = sims.argmax().item()
        return self.ref_cat_ids[best_idx], sims[best_idx].item()

    def predict(self, img_path: str, device: str = "cpu") -> list[dict]:
        img = Image.open(img_path).convert("RGB")

        # Stage 1 — detect via ONNX
        input_tensor, scale, pad_x, pad_y = _letterbox(img, DETECT_IMGSZ)
        input_name = self.detector.get_inputs()[0].name
        output = self.detector.run(None, {input_name: input_tensor})[0]
        detections = _postprocess_onnx(output, scale, pad_x, pad_y)

        preds = []
        for x1, y1, x2, y2, det_score in detections:
            # Stage 2 — classify crop
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.width < 4 or crop.height < 4:
                continue
            cat_id, cls_score = self._classify_crop(crop)

            preds.append({
                "category_id": cat_id,
                "bbox": [
                    round(float(x1), 1), round(float(y1), 1),
                    round(float(x2 - x1), 1), round(float(y2 - y1), 1),
                ],
                "score": round(float(det_score) * cls_score, 3),
            })
        return preds
