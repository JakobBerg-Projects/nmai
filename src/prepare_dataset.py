"""
Convert COCO-format annotations (train/annotations.json) to YOLO format
and split into train/val sets under data/images/ and data/labels/.
"""
import json
import os
import random
import shutil
from pathlib import Path


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] bbox to YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def prepare(
    annotations_path: str = "data/raw/annotations.json",
    images_dir: str = "data/raw/images",
    output_dir: str = "data",
    val_split: float = 0.2,
    seed: int = 42,
):
    random.seed(seed)

    with open(annotations_path) as f:
        coco = json.load(f)

    # Build lookup tables
    img_info = {img["id"]: img for img in coco["images"]}
    # category_id -> contiguous index (0-based)
    cat_ids = sorted({ann["category_id"] for ann in coco["annotations"]})
    cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Only keep images that have annotations
    annotated_ids = list(anns_by_image.keys())
    random.shuffle(annotated_ids)
    split = int(len(annotated_ids) * (1 - val_split))
    train_ids = set(annotated_ids[:split])
    val_ids = set(annotated_ids[split:])

    splits = {"train": train_ids, "val": val_ids}

    for split_name, id_set in splits.items():
        img_out = Path(output_dir) / "images" / split_name
        lbl_out = Path(output_dir) / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in id_set:
            info = img_info[img_id]
            src = Path(images_dir) / info["file_name"]
            if not src.exists():
                print(f"  [warn] image not found: {src}")
                continue

            # Copy image
            shutil.copy2(src, img_out / info["file_name"])

            # Write label file
            stem = Path(info["file_name"]).stem
            label_path = lbl_out / f"{stem}.txt"
            lines = []
            for ann in anns_by_image[img_id]:
                cx, cy, nw, nh = coco_to_yolo(
                    ann["bbox"], info["width"], info["height"]
                )
                # Clamp to [0, 1] to handle any annotation edge cases
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                idx = cat_id_to_idx[ann["category_id"]]
                lines.append(f"{idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_path.write_text("\n".join(lines))

    print(
        f"Dataset prepared: {len(train_ids)} train / {len(val_ids)} val images"
    )
    print(f"Classes: {len(cat_id_to_idx)}")

    # Return metadata for dataset.yaml generation
    return coco["categories"], cat_ids


def write_yaml(
    categories,
    cat_ids,
    output_dir: str = "data",
    yaml_path: str = "data/dataset.yaml",
):
    cat_map = {c["id"]: c["name"] for c in categories}
    names = [cat_map[cid] for cid in cat_ids]
    nc = len(names)

    lines = [
        "# Dataset configuration (YOLO format)",
        f"path: {output_dir}",
        "train: images/train",
        "val: images/val",
        "",
        f"# {nc} product classes",
        f"nc: {nc}",
        "names:",
    ]
    for name in names:
        # Escape special YAML characters
        safe = name.replace("'", "\\'")
        lines.append(f"  - '{safe}'")

    Path(yaml_path).write_text("\n".join(lines) + "\n")
    print(f"Wrote {yaml_path} with {nc} classes")


if __name__ == "__main__":
    categories, cat_ids = prepare()
    write_yaml(categories, cat_ids)
