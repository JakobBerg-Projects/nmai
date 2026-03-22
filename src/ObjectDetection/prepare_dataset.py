"""
Convert COCO-format annotations to YOLO format and split into train/val.

Expected raw data layout (relative to this file):
    data/train/annotations.json
    data/train/images/*.jpg

Output layout:
    data/yolo/images/train/    data/yolo/images/val/
    data/yolo/labels/train/    data/yolo/labels/val/
    data/yolo/dataset.yaml
"""
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x, y, w, h] bbox to YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def prepare(
    val_split: float = 0.2,
    seed: int = 42,
    output_dir: Path | None = None,
    single_class: bool = False,
):
    annotations_path = ROOT / "data" / "train" / "annotations.json"
    images_dir = ROOT / "data" / "train" / "images"
    if output_dir is None:
        output_dir = ROOT / "data" / "yolo"

    random.seed(seed)

    with open(annotations_path, encoding="utf-8") as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Split annotated images into train/val
    annotated_ids = list(anns_by_image.keys())
    random.shuffle(annotated_ids)
    split = int(len(annotated_ids) * (1 - val_split))
    splits = {
        "train": set(annotated_ids[:split]),
        "val": set(annotated_ids[split:]),
    }

    for split_name, id_set in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in id_set:
            info = img_info[img_id]
            src = images_dir / info["file_name"]
            if not src.exists():
                print(f"  [warn] image not found: {src}")
                continue

            # Copy image
            dst = img_out / info["file_name"]
            if not dst.exists():
                shutil.copy2(src, dst)

            # Write YOLO label file
            # category_id is used directly (0-355) — no remapping
            stem = Path(info["file_name"]).stem
            label_path = lbl_out / f"{stem}.txt"
            lines = []
            for ann in anns_by_image[img_id]:
                cx, cy, nw, nh = coco_to_yolo(
                    ann["bbox"], info["width"], info["height"]
                )
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))
                cat = 0 if single_class else ann["category_id"]
                lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_path.write_text("\n".join(lines))

    print(f"Dataset prepared: {len(splits['train'])} train / {len(splits['val'])} val images")
    print(f"Output: {output_dir}")
    return coco["categories"]


def write_yaml(categories):
    output_dir = ROOT / "data" / "yolo"
    yaml_path = output_dir / "dataset.yaml"

    cat_map = {c["id"]: c["name"] for c in categories}
    nc = len(categories)
    # Build names list in ID order (0, 1, 2, ..., 355)
    names = [cat_map[i] for i in range(nc)]

    lines = [
        "# Dataset configuration (YOLO format)",
        f"path: {output_dir}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {nc}",
        "names:",
    ]
    for name in names:
        # YAML single-quoted strings escape ' by doubling it
        safe = name.replace("'", "''")
        lines.append(f"  - '{safe}'")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {yaml_path} with {nc} classes")
    return yaml_path


if __name__ == "__main__":
    categories = prepare()
    write_yaml(categories)
