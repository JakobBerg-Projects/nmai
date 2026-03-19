"""Strategy 3: Two-stage — detect (nc=1) then classify via embedding matching.

Stage 1: YOLOv8 detector (nc=1) finds all product bounding boxes.
Stage 2: A MobileNetV3 backbone embeds each crop and classifies it by
         cosine-similarity to pre-computed reference embeddings.

Weight files expected in weights_dir:
    detector.pt        — YOLOv8 detector (nc=1)
    classifier.pt      — MobileNetV3 feature extractor (fine-tuned)
    ref_embeddings.pt  — {category_id: embedding} dict for all 356 classes
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from ultralytics import YOLO
from strategies.base import Strategy


CROP_SIZE = 224
EMBED_DIM = 576  # MobileNetV3-Small final feature dim

CROP_TRANSFORM = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


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

        # Stage 1 — detector
        self.detector = YOLO(str(self.weights_dir / "detector.pt"))

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

        # Stage 1 — detect
        with torch.no_grad():
            results = self.detector(img_path, device=self.device, verbose=False)

        preds = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                det_score = float(r.boxes.conf[i].item())

                # Stage 2 — classify crop
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                if crop.width < 4 or crop.height < 4:
                    continue
                cat_id, cls_score = self._classify_crop(crop)

                preds.append({
                    "category_id": cat_id,
                    "bbox": [
                        round(x1, 1), round(y1, 1),
                        round(x2 - x1, 1), round(y2 - y1, 1),
                    ],
                    "score": round(det_score * cls_score, 3),
                })
        return preds
