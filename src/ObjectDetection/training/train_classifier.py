"""Train an embedding model for crop → product classification.

Supports multiple backbone architectures via --arch flag.
Fine-tunes with classification loss, then saves the backbone
as a feature extractor and pre-computes reference embeddings.

Usage:
    python -m training.train_classifier --epochs 100 --batch-size 64
    python -m training.train_classifier --arch efficientnet_v2_s --crop-size 384 --epochs 100
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT / "data" / "crops"
WEIGHTS_DIR = ROOT / "weights"

SUPPORTED_ARCHS = {
    'resnet50': {'embed_dim': 2048, 'default_crop': 224},
    'efficientnet_v2_s': {'embed_dim': 1280, 'default_crop': 384},
    'convnext_tiny': {'embed_dim': 768, 'default_crop': 224},
    'convnext_small': {'embed_dim': 768, 'default_crop': 224},
}


def build_backbone(arch, pretrained=True):
    """Build backbone model and return (model, embed_dim)."""
    if arch == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
        return model, 2048
    elif arch == 'efficientnet_v2_s':
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        model.classifier = nn.Identity()
        return model, 1280
    elif arch == 'convnext_tiny':
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier[2] = nn.Identity()
        return model, 768
    elif arch == 'convnext_small':
        from torchvision.models import convnext_small, ConvNeXt_Small_Weights
        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        model = convnext_small(weights=weights)
        model.classifier[2] = nn.Identity()
        return model, 768
    else:
        raise ValueError(f"Unknown arch: {arch}. Supported: {list(SUPPORTED_ARCHS.keys())}")


def get_train_transform(crop_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(crop_size):
    return transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
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
    """Backbone + classification head."""
    def __init__(self, num_classes: int, arch: str = 'resnet50',
                 pretrained: bool = True):
        super().__init__()
        self.backbone, self.embed_dim = build_backbone(arch, pretrained)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        emb = self.backbone(x)
        return self.head(emb)

    def embed(self, x):
        return self.backbone(x)


def train(epochs: int, batch_size: int, lr: float, patience: int = 10,
          arch: str = 'resnet50', crop_size: int | None = None,
          detector_path: str | None = None, det_imgsz: int = 1280):
    if crop_size is None:
        crop_size = SUPPORTED_ARCHS[arch]['default_crop']

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Architecture: {arch} (embed_dim={SUPPORTED_ARCHS[arch]['embed_dim']}, "
          f"crop_size={crop_size})")

    # Build transforms with the configured crop size
    train_transform = get_train_transform(crop_size)
    val_transform = get_val_transform(crop_size)

    # Load dataset
    full_dataset = CropDataset(CROPS_DIR, transform=train_transform)
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
    val_ds.dataset = CropDataset(CROPS_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Combined validation setup — run detector on val images ONCE
    use_combined = False
    val_crops, val_gt, det_mAP50 = None, None, 0.0
    if detector_path and Path(detector_path).exists():
        try:
            from training.validate_pipeline import (
                get_val_data, cache_detections_and_crops,
                evaluate as eval_combined,
            )
            print("\n=== Caching detector results on val set ===")
            val_ids, coco_data = get_val_data()
            val_crops, val_gt, det_mAP50 = cache_detections_and_crops(
                detector_path, val_ids, coco_data, imgsz=det_imgsz,
                crop_size=crop_size,
            )
            use_combined = True
            print("  Combined validation enabled (competition metric)\n")
        except Exception as e:
            print(f"  Warning: combined validation unavailable ({e}), using macro_acc\n")
    else:
        print("No detector path provided — using macro_acc for model selection\n")

    # Model
    model = EmbeddingClassifier(num_classes=num_classes, arch=arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_score = 0.0
    weights_updated = False
    epochs_without_improvement = 0
    WEIGHTS_DIR.mkdir(exist_ok=True)
    marker_path = WEIGHTS_DIR / "classifier_updated"
    marker_path.unlink(missing_ok=True)

    # Evaluate existing weights (only if same architecture)
    existing_weights = WEIGHTS_DIR / "classifier.pt"
    existing_config = WEIGHTS_DIR / "config.json"
    existing_arch_matches = False
    if existing_config.exists():
        with open(existing_config) as f:
            prev_config = json.load(f)
        existing_arch_matches = prev_config.get('arch') == arch

    if existing_weights.exists() and use_combined and existing_arch_matches:
        print("=== Evaluating existing classifier weights ===")
        prev_model = EmbeddingClassifier(
            num_classes=num_classes, arch=arch, pretrained=False,
        ).to(device)
        prev_model.backbone.load_state_dict(
            torch.load(existing_weights, map_location=device, weights_only=True)
        )
        prev_combined, prev_det, prev_cls = eval_combined(
            val_crops, val_gt, prev_model, full_dataset.cat_ids, device,
            det_mAP50=det_mAP50,
        )
        best_score = prev_combined
        del prev_model
        print(f"  Existing model score: combined={prev_combined:.4f}  "
              f"det_mAP50={prev_det:.4f}  cls_mAP50={prev_cls:.4f}")
        print(f"  New model must beat {best_score:.4f} to save weights\n")

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

        # Validate — crop accuracy
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        val_acc = (all_preds == all_labels).mean()

        per_class_acc = []
        for cls in range(num_classes):
            mask = all_labels == cls
            if mask.sum() > 0:
                per_class_acc.append((all_preds[mask] == cls).mean())
        macro_acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0

        # Combined score (competition metric) or fall back to macro_acc
        if use_combined:
            combined, det_map, cls_map = eval_combined(
                val_crops, val_gt, model, full_dataset.cat_ids, device,
                det_mAP50=det_mAP50,
            )
            score = combined
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"loss={total_loss/total:.4f}  "
                  f"train_acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"macro_acc={macro_acc:.3f}  "
                  f"|| combined={combined:.4f}  "
                  f"det_mAP50={det_map:.4f}  "
                  f"cls_mAP50={cls_map:.4f}"
                  f"{'  *' if score > best_score else ''}")
        else:
            score = macro_acc
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"loss={total_loss/total:.4f}  "
                  f"train_acc={train_acc:.3f}  "
                  f"val_acc={val_acc:.3f}  "
                  f"macro_acc={macro_acc:.3f}"
                  f"{'  *' if score > best_score else ''}")

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            weights_updated = True
            torch.save(model.backbone.state_dict(), WEIGHTS_DIR / "classifier.pt")
            marker_path.write_text(f"{best_score:.6f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs")
                break

    metric_name = "combined score" if use_combined else "macro accuracy"
    print(f"\nBest val {metric_name}: {best_score:.4f}")

    if weights_updated:
        print(f"Saved backbone to {WEIGHTS_DIR / 'classifier.pt'}")
        # Re-load best weights before computing reference embeddings
        model.backbone.load_state_dict(
            torch.load(WEIGHTS_DIR / "classifier.pt", map_location=device, weights_only=True)
        )
        print("\nComputing reference embeddings...")
        _compute_reference_embeddings(model, full_dataset.cat_ids, device, crop_size)

        # Save config alongside weights
        config = {
            'arch': arch,
            'crop_size': crop_size,
            'embed_dim': SUPPORTED_ARCHS[arch]['embed_dim'],
        }
        config_path = WEIGHTS_DIR / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")
    else:
        print("No improvement over existing weights — kept previous classifier.pt")


def _compute_reference_embeddings(model, cat_ids, device, crop_size=224):
    """Compute average embedding per category from reference images."""
    model.eval()

    transform = get_val_transform(crop_size)
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--arch", choices=list(SUPPORTED_ARCHS.keys()),
                        default="resnet50",
                        help="Classifier backbone architecture")
    parser.add_argument("--crop-size", type=int, default=None,
                        help="Crop size (default: architecture-specific)")
    parser.add_argument("--detector-path", default=None,
                        help="Path to detector.pt for combined validation")
    parser.add_argument("--det-imgsz", type=int, default=1280)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          patience=args.patience, arch=args.arch, crop_size=args.crop_size,
          detector_path=args.detector_path, det_imgsz=args.det_imgsz)
