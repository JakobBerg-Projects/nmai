"""Extract product crops from training images using COCO annotations.

Creates a directory of cropped product images organised by category_id,
ready for classifier training. Also copies product reference images
into the same structure.

Output:
    data/crops/{category_id}/crop_{ann_id}.jpg      — from shelf annotations
    data/crops/{category_id}/ref_{product_code}_{view}.jpg  — from reference images
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT / "data" / "crops"


def extract_annotation_crops():
    """Crop every annotated bounding box from training images."""
    ann_path = ROOT / "data" / "train" / "annotations.json"
    img_dir = ROOT / "data" / "train" / "images"

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    img_cache = {}

    count = 0
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        out_dir = CROPS_DIR / str(cat_id)
        out_path = out_dir / f"crop_{ann['id']}.jpg"
        if out_path.exists():
            continue

        info = img_info[ann["image_id"]]
        src = img_dir / info["file_name"]
        if not src.exists():
            continue

        # Cache opened images to avoid re-reading
        if ann["image_id"] not in img_cache:
            img_cache[ann["image_id"]] = Image.open(src).convert("RGB")
        img = img_cache[ann["image_id"]]

        x, y, w, h = ann["bbox"]
        crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
        if crop.width < 4 or crop.height < 4:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        crop.save(out_path, quality=95)
        count += 1

    print(f"Extracted {count} annotation crops to {CROPS_DIR}")


def copy_reference_images():
    """Copy product reference images into the crops directory structure."""
    ref_dir = ROOT / "data" / "NM_NGD_product_images"
    meta_path = ref_dir / "metadata.json"
    ann_path = ROOT / "data" / "train" / "annotations.json"

    if not meta_path.exists():
        print("No product reference images found, skipping.")
        return

    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Build product_name → category_id mapping
    name_to_cat = {c["name"]: c["id"] for c in coco["categories"]}

    count = 0
    for product in meta["products"]:
        if not product["has_images"]:
            continue
        cat_id = name_to_cat.get(product["product_name"])
        if cat_id is None:
            continue

        out_dir = CROPS_DIR / str(cat_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        product_dir = ref_dir / product["product_code"]
        for view in product["image_types"]:
            src = product_dir / f"{view}.jpg"
            if not src.exists():
                continue
            dst = out_dir / f"ref_{product['product_code']}_{view}.jpg"
            if dst.exists():
                continue
            # Copy with PIL to ensure consistent format
            img = Image.open(src).convert("RGB")
            img.save(dst, quality=95)
            count += 1

    print(f"Copied {count} reference images to {CROPS_DIR}")


if __name__ == "__main__":
    print("Extracting annotation crops...")
    extract_annotation_crops()
    print("\nCopying reference images...")
    copy_reference_images()
    print("\nDone!")
