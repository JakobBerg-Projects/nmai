"""Train a MobileNetV3 embedding model for crop → product classification.

Uses crops extracted by extract_crops.py (annotation crops + reference images).
Fine-tunes MobileNetV3-Small with classification loss, then saves the backbone
as a feature extractor and pre-computes reference embeddings.

Usage:
    python -m training.train_classifier --epochs 30 --batch-size 64
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT / "data" / "crops"
WEIGHTS_DIR = ROOT / "weights"

CROP_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class CropDataset(Dataset):
    def __init__(self, crops_dir: Path, transform=None):
        self.transform = transform
        self.samples = []  # (path, class_idx)
        self.cat_ids = []  # ordered list of category_ids

        # Each subdirectory is a category_id
        cat_dirs = sorted(
            [d for d in crops_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name),
        )
        self.cat_ids = [int(d.name) for d in cat_dirs]
        self.cat_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

        for d in cat_dirs:
            idx = self.cat_to_idx[int(d.name)]
            for img_path in d.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class EmbeddingClassifier(nn.Module):
    """MobileNetV3-Small backbone + classification head."""
    def __init__(self, num_classes: int, embed_dim: int = 576):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the original classifier
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        return self.head(emb)

    def embed(self, x):
        return self.backbone(x)


def train(epochs: int, batch_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load dataset
    full_dataset = CropDataset(CROPS_DIR, transform=TRAIN_TRANSFORM)
    num_classes = len(full_dataset.cat_ids)
    print(f"Dataset: {len(full_dataset)} images, {num_classes} classes")

    # Train/val split
    val_size = max(1, int(0.15 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Override transform for validation
    val_ds.dataset = CropDataset(CROPS_DIR, transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = EmbeddingClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    WEIGHTS_DIR.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"loss={total_loss/total:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"val_acc={val_acc:.3f}")

        if val_acc >= best_acc:
            best_acc = val_acc
            # Save backbone only (for embedding at inference)
            torch.save(model.backbone.state_dict(), WEIGHTS_DIR / "classifier.pt")

    print(f"\nBest val accuracy: {best_acc:.3f}")
    print(f"Saved backbone to {WEIGHTS_DIR / 'classifier.pt'}")

    # Pre-compute reference embeddings
    print("\nComputing reference embeddings...")
    _compute_reference_embeddings(model, full_dataset.cat_ids, device)


def _compute_reference_embeddings(model, cat_ids, device):
    """Compute average embedding per category from reference images."""
    model.eval()

    transform = VAL_TRANSFORM
    ref_embeds = {}

    for cat_id in cat_ids:
        cat_dir = CROPS_DIR / str(cat_id)
        ref_imgs = [p for p in cat_dir.iterdir()
                    if p.name.startswith("ref_") and p.suffix.lower() in (".jpg", ".jpeg", ".png")]

        if not ref_imgs:
            # Fall back to annotation crops if no reference images
            ref_imgs = list(cat_dir.iterdir())[:10]

        embeddings = []
        for img_path in ref_imgs:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.embed(tensor)
            embeddings.append(emb.squeeze(0))

        if embeddings:
            avg_emb = torch.stack(embeddings).mean(dim=0)
            ref_embeds[cat_id] = avg_emb.cpu()

    torch.save(ref_embeds, WEIGHTS_DIR / "ref_embeddings.pt")
    print(f"Saved reference embeddings for {len(ref_embeds)} categories "
          f"to {WEIGHTS_DIR / 'ref_embeddings.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
