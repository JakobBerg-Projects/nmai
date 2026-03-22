# Object Detection Strategies — NM i AI (NorgesGruppen)

## Task Summary

- **Goal:** Detect and classify grocery products on store shelf images
- **Data:** 254 shelf images, ~22,300 annotations, 357 product categories (4 sections: Egg, Frokost, Knekkebrod, Varmedrikker)
- **Scoring:** `0.7 × detection_mAP@0.5 + 0.3 × classification_mAP@0.5`
- **Sandbox:** NVIDIA L4 (24 GB VRAM), 8 GB RAM, 300s timeout, no network, max 420 MB weights
- **Submissions:** 3/day, max 2 in-flight

## Key Observations

| Aspect | Detail | Implication |
|---|---|---|
| Detection = 70% of score | Category ignored for detection component | A detection-only baseline can score up to 0.70 |
| 357 classes, 254 images | ~62 annotations per image, but very few per category | Classification is the hard part — many classes are rare |
| Product reference images | 327 products with multi-angle photos (main, front, back, etc.) | Enables visual matching / few-shot classification |
| L4 GPU, 24 GB VRAM | Large models are feasible within timeout | Use YOLOv8l/x or even RT-DETR, not just nano/small |
| `ensemble-boxes` pre-installed | WBF ensembling available at inference time | Can ensemble multiple models without extra packages |
| 3 submissions/day | Very limited iteration | Test locally with train/val split before submitting |

---

## Strategy 1: Fine-Tuned YOLOv8 (Baseline)

**Approach:** Train YOLOv8 end-to-end with `nc=357` on the competition COCO data.

**Pros:**
- Simplest pipeline — single model does both detection and classification
- `ultralytics==8.1.0` is pre-installed, so `.pt` weights work directly
- Well-documented training workflow

**Cons:**
- 357 classes with imbalanced data makes classification hard
- Small dataset (254 images) risks overfitting

**Implementation:**
1. Convert COCO annotations to YOLO format (already done in `prepare_dataset.py`)
2. Train: `yolo detect train data=data/dataset.yaml model=yolov8x.pt epochs=100 imgsz=640`
3. Use largest model that fits in 420 MB (YOLOv8l or YOLOv8x)
4. Augmentation: mosaic, mixup, multi-scale, copy-paste

**Expected score:** 0.40–0.60 depending on model size and training

---

## Strategy 2: Detection-Only First (Quick Win)

**Approach:** Maximise the detection component (70%) by training with a single "product" class.

**Pros:**
- Much easier learning task — binary "product or not"
- All 22,300 annotations contribute to one class
- Can score up to 0.70

**Cons:**
- Leaves 30% on the table (classification)
- Good as a stepping stone, not a final solution

**Implementation:**
1. Remap all `category_id` to `0` in labels
2. Train YOLOv8x with `nc=1`
3. Submit with `category_id: 0` for all predictions

**Expected score:** 0.50–0.70

---

## Strategy 3: Two-Stage Pipeline (Detection + Classification)

**Approach:** Separate the problem into two stages — detect first, then classify each crop.

**Stage 1 — Detection:**
- Train YOLOv8x with `nc=1` (single class) for maximum detection recall
- Or train with `nc=357` and keep bounding boxes but re-classify

**Stage 2 — Classification:**
- Crop each detected box from the shelf image
- Classify using a separate model (e.g., ResNet/EfficientNet/ViT from `timm`)
- Train classifier on cropped annotations from the training data
- Optionally use product reference images to augment training

**Pros:**
- Detection model sees all data as one class — more robust
- Classifier can be trained with product reference images as extra data
- Can iterate on each stage independently

**Cons:**
- More complex pipeline, two models to fit in 420 MB
- Adds latency (must fit in 300s total)
- Crop quality depends on detection accuracy

**Expected score:** 0.55–0.80

---

## Strategy 4: Two-Stage with Visual Matching (Product Images)

**Approach:** Use the 327 product reference images for classification via embedding similarity.

**Stage 1 — Detection:** Same as Strategy 3.

**Stage 2 — Visual Matching:**
- Use a pre-trained feature extractor (e.g., `timm` EfficientNet or ViT) to embed both:
  - Cropped detections from the shelf
  - Product reference images (multi-angle)
- Classify by nearest-neighbour in embedding space
- Fine-tune the backbone on crop–reference pairs for better embeddings

**Pros:**
- Handles rare categories well (each product has reference images even if few annotations)
- No need for per-class training data balance
- Product reference images are clean, isolated product shots — great for matching

**Cons:**
- Shelf crops contain occlusion, partial views, and shelf context
- Need to handle `unknown_product` (category 356) somehow
- Embedding similarity can be noisy

**Implementation:**
1. Pre-compute embeddings for all reference images (average multi-angle embeddings per product)
2. At inference: detect → crop → embed → nearest-neighbour match
3. Use cosine similarity with a confidence threshold
4. Fine-tune with contrastive/triplet loss for better crop-to-reference matching

**Expected score:** 0.60–0.85

---

## Strategy 5: Ensemble Multiple Models

**Approach:** Train multiple detectors and combine predictions using Weighted Boxes Fusion (WBF).

**Models to ensemble:**
- YOLOv8x (best YOLO variant)
- YOLOv8l at different `imgsz` (640, 1280)
- RT-DETR-l (transformer-based, available in ultralytics)

**Implementation:**
1. Train 2–3 models with different architectures or hyperparameters
2. At inference, run all models on each image
3. Use `ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion()` to merge predictions
4. Fit all model weights within 420 MB total (use FP16)

**Pros:**
- Ensembles consistently improve mAP by 2–5%
- `ensemble-boxes` is pre-installed — no extra dependencies
- Different models catch different products

**Cons:**
- Must fit multiple models in 420 MB and run within 300s
- More complex training and `run.py`

**Expected score:** +2–5% over single-model approaches

---

## Strategy 6: RT-DETR (Transformer Detector)

**Approach:** Use RT-DETR instead of YOLO. RT-DETR is a real-time transformer-based detector available in `ultralytics==8.1.0`.

**Pros:**
- Transformer attention may better handle dense, overlapping products
- No NMS needed (set-based prediction)
- Available in ultralytics — same training workflow as YOLO

**Cons:**
- Larger model, more VRAM
- Less community knowledge for this specific task

**Expected score:** Comparable or slightly better than YOLOv8x for dense scenes

---

## Training Tips

### Data Augmentation
- **Mosaic** — combines 4 images; great for dense scenes (on by default in YOLO)
- **Copy-paste** — paste product crops onto different backgrounds
- **Multi-scale** — train at multiple resolutions (640, 960, 1280)
- **Albumentations** — available in sandbox for custom augmentation pipelines

### Handling Class Imbalance
- Use focal loss (default in YOLO)
- Oversample rare categories during training
- Consider class-agnostic detection + separate classifier (Strategy 3/4)

### Resolution
- Higher `imgsz` (1280) captures small products better but uses more VRAM and is slower
- Test trade-off between 640 and 1280 — shelf images are typically 2000×1500

### Avoiding Overfitting (254 images)
- Strong augmentation (mosaic, mixup, HSV shifts)
- Early stopping on validation mAP
- Freeze backbone for first N epochs, then unfreeze
- Use pre-trained COCO weights (default in YOLO)
- K-fold cross-validation if time allows

### FP16
- Export with `half=True` for half the weight size and faster inference on L4
- `model.export(format="onnx", half=True)` or `model.export(format="engine", half=True)`

---

## Recommended Roadmap

| Phase | Strategy | Target Score |
|---|---|---|
| 1. Baseline | YOLOv8n with nc=357, default settings | ~0.35 |
| 2. Quick win | YOLOv8x detection-only (nc=1) | ~0.55–0.70 |
| 3. Classification | Add two-stage classifier with product images | ~0.65–0.80 |
| 4. Polish | Higher resolution, augmentation tuning, ensemble | ~0.75–0.85+ |

---

## Submission Checklist

- [ ] `run.py` at zip root (not in a subfolder)
- [ ] Accepts `--input` and `--output` arguments
- [ ] Outputs COCO-format JSON: `image_id`, `category_id` (0–356), `bbox` [x,y,w,h], `score`
- [ ] `image_id` parsed from filename (`img_00042.jpg` → `42`)
- [ ] Uses `pathlib` not `os` (security restriction)
- [ ] Auto-detects GPU with `torch.cuda.is_available()`
- [ ] Total weights < 420 MB
- [ ] No disallowed imports (`os`, `subprocess`, `socket`, etc.)
- [ ] Tested locally before submitting
- [ ] Pin `ultralytics==8.1.0` if using YOLO `.pt` weights

./gcp_train.sh --strategy two_stage --model yolov8x.pt --epochs 300 --imgsz 1280 --batch 4 --cls-epochs 100 --gpu t4 --zone europe-west4-a
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/300      14.6G      1.024      0.464      1.232        253       1280: 100%|██████████| 50/50 [01:05<00:00,  1.31s/it]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:05<00:00,  1.40it/s]
                   all         50       4807      0.923      0.919      0.951      0.648

Run around 90 epochs for detection.



./gcp_train.sh --strategy two_stage --model yolov8x.pt --epochs 90 --imgsz 1280 --batch 6 --cls-epochs 100 --patience 15 --gpu l4 --zone europe-west1-b

./gcp_train.sh --strategy two_stage --model yolov8x.pt --epochs 300 --imgsz 1280 --batch 16 --cls-epochs 100 --cls-batch 512 --patience 15 --vm-name instance-20260320-232652 --zone us-central1-f

./gcp_train.sh --strategy two_stage --model yolov8x.pt --epochs 100 --imgsz 1280 --batch 4 \
  --cls-epochs 100 --cls-batch 64 --patience 15 \
  --cls-arch efficientnet_v2_s \
  --vm-name nmai-a100 --zone us-central1-f