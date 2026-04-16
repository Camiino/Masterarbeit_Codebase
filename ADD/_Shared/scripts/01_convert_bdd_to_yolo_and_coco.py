#!/usr/bin/env python3
"""
Convert BDD100K detection labels to YOLO format and mapped COCO JSON.

Usage example:
    python project/scripts/01_convert_bdd_to_yolo_and_coco.py \
        --images-root project/data/ad/bdd100k_raw/bdd100k \
        --ann-root project/data/ad/bdd100k_raw/bdd100k \
        --out-yolo-root project/data/ad/bdd_yolo \
        --out-coco-root project/data/ad/bdd_coco \
        --report-raw-categories
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional progress bar; falls back to a passthrough iterator if tqdm is missing.
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


# =========================
# Mapping definitions (UPDATED)
# =========================

# Normalize raw category strings to improve robustness (e.g., "Traffic Light" vs "traffic light").
def _norm_cat(s: str) -> str:
    return " ".join(s.strip().lower().split())


TARGET_CATEGORY_MAP = {
    # vehicles -> car
    _norm_cat("car"): "car",
    _norm_cat("truck"): "car",
    _norm_cat("bus"): "car",

    # pedestrians -> pedestrian
    _norm_cat("person"): "pedestrian",

    # cyclists -> cyclist
    _norm_cat("bike"): "cyclist",
    _norm_cat("rider"): "cyclist",
}

YOLO_CLASS_IDS = {"car": 0, "pedestrian": 1, "cyclist": 2}

COCO_CATEGORIES = [
    {"id": 1, "name": "car"},
    {"id": 2, "name": "pedestrian"},
    {"id": 3, "name": "cyclist"},
]
COCO_NAME_TO_ID = {c["name"]: c["id"] for c in COCO_CATEGORIES}


@dataclass
class NormalizedAnnotation:
    category: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BDD-style labels to YOLO and COCO formats.")
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root directory containing BDD images (train/val/test) or a directory to recurse for images.",
    )
    parser.add_argument("--det-train-json", type=Path, help="Path to det_train.json (optional if using --ann-root).")
    parser.add_argument("--det-val-json", type=Path, help="Path to det_val.json (optional if using --ann-root).")
    parser.add_argument(
        "--ann-root",
        type=Path,
        help="Root containing train/ann and val/ann per-image JSON annotations (DatasetNinja style). Used when det JSONs are absent.",
    )
    parser.add_argument("--out-yolo-root", type=Path, required=True, help="Output root for YOLO dataset (images/all, labels/all).")
    parser.add_argument("--out-coco-root", type=Path, required=True, help="Output root for mapped COCO dataset.")
    parser.add_argument("--default-width", type=int, default=1280, help="Fallback width if labels omit image dimensions.")
    parser.add_argument("--default-height", type=int, default=720, help="Fallback height if labels omit image dimensions.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--report-raw-categories", action="store_true", help="Log unique raw categories encountered per split.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def map_category(raw_category: Optional[str]) -> Optional[str]:
    if raw_category is None:
        return None
    return TARGET_CATEGORY_MAP.get(_norm_cat(raw_category))


def load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ann_directory(ann_dir: Path) -> List[Dict]:
    """
    Load per-image annotations from a directory of JSON files (DatasetNinja style).
    Converts to list-of-items format expected by normalize_annotations.

    Expected schema per file (typical):
      {
        "size": {"width": ..., "height": ...},
        "objects": [
          {"classTitle": "...", "geometryType": "rectangle",
           "points": {"exterior": [[x1,y1],[x2,y2]]}}
        ]
      }
    """
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")

    items: List[Dict] = []
    for json_file in sorted(ann_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        size = data.get("size", {})
        width = int(size.get("width", 0) or 0)
        height = int(size.get("height", 0) or 0)

        labels = []
        for obj in data.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                continue
            points = obj.get("points", {}).get("exterior", [])
            if not points or len(points) < 2:
                continue
            (x1, y1), (x2, y2) = points[0], points[1]
            box = {
                "x1": float(min(x1, x2)),
                "y1": float(min(y1, y2)),
                "x2": float(max(x1, x2)),
                "y2": float(max(y1, y2)),
            }
            labels.append({"category": obj.get("classTitle"), "box2d": box})

        base_name = json_file.name.replace(".json", "")
        normalized_name = Path(base_name).stem + ".jpg"
        items.append({"name": normalized_name, "width": width, "height": height, "labels": labels})

    if not items:
        raise FileNotFoundError(f"No annotation JSON files found in {ann_dir}")
    return items


def normalize_annotations(raw_data: object, default_width: int, default_height: int) -> Dict[str, Dict]:
    """
    Normalize annotations to a per-image structure.
    Supports:
      - COCO-like dict: {"images":[...], "annotations":[...], "categories":[...]}
      - BDD list style: [{"name":..., "labels":[{"category":..., "box2d":...}]}]
    """
    normalized: Dict[str, Dict] = {}

    # COCO-like format
    if isinstance(raw_data, dict) and {"images", "annotations"}.issubset(raw_data.keys()):
        categories = {c.get("id"): c.get("name") for c in raw_data.get("categories", [])}
        images = {img.get("id"): img for img in raw_data.get("images", [])}

        for ann in raw_data.get("annotations", []):
            img_info = images.get(ann.get("image_id"))
            if not img_info:
                continue
            name = Path(img_info.get("file_name") or img_info.get("name") or "").name
            width = int(img_info.get("width", default_width) or default_width)
            height = int(img_info.get("height", default_height) or default_height)

            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
            elif "box2d" in ann:
                box = ann["box2d"]
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            else:
                continue

            category = categories.get(ann.get("category_id")) or ann.get("category")
            entry = normalized.setdefault(name, {"width": width, "height": height, "labels": []})
            entry["labels"].append({"category": category, "bbox": (float(x1), float(y1), float(x2), float(y2))})

        # Ensure images with no annotations still appear
        for img in raw_data.get("images", []):
            name = Path(img.get("file_name") or img.get("name") or "").name
            if not name:
                continue
            normalized.setdefault(
                name,
                {"width": int(img.get("width", default_width) or default_width),
                 "height": int(img.get("height", default_height) or default_height),
                 "labels": []},
            )
        return normalized

    # BDD list format
    if isinstance(raw_data, list):
        for item in raw_data:
            name = Path(item.get("name", "")).name
            if not name:
                continue
            width = int(item.get("width", default_width) or default_width)
            height = int(item.get("height", default_height) or default_height)
            labels = []
            for lbl in item.get("labels", []):
                box = lbl.get("box2d")
                if not box:
                    continue
                labels.append(
                    {"category": lbl.get("category"),
                     "bbox": (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))}
                )
            normalized[name] = {"width": width, "height": height, "labels": labels}
        return normalized

    raise ValueError("Unsupported label JSON structure; expected list or COCO-like dict.")


def collect_images(images_root: Path) -> Dict[str, Path]:
    """
    Find images on disk and map basename -> path.
    Prefers BDD structure train/img etc., otherwise recurses from images_root.
    """
    allowed_exts = {".jpg", ".jpeg", ".png"}
    mapping: Dict[str, Path] = {}
    duplicates: List[str] = []

    search_roots: List[Path] = []
    for split in ["train", "val", "test"]:
        img_dir = images_root / split / "img"
        if img_dir.is_dir():
            search_roots.append(img_dir)

    if not search_roots:
        search_roots = [images_root]

    for root in search_roots:
        for img_path in root.rglob("*"):
            if img_path.suffix.lower() not in allowed_exts or not img_path.is_file():
                continue
            if img_path.name in mapping and mapping[img_path.name] != img_path:
                duplicates.append(img_path.name)
            mapping.setdefault(img_path.name, img_path)

    if duplicates:
        logging.warning("Duplicate image names detected (using first occurrence): %s", sorted(set(duplicates))[:50])
    if not mapping:
        raise FileNotFoundError(f"No images found under {images_root}")
    return mapping


def find_ann_root(images_root: Path, override: Optional[Path]) -> Optional[Path]:
    """
    Find a base directory that contains train/ann and val/ann.
    """
    candidates: List[Path] = []
    bases = [b for b in {override, images_root, images_root.parent, images_root.parent.parent} if b]
    for base in bases:
        if not base.exists():
            continue
        if (base / "train" / "ann").is_dir() and (base / "val" / "ann").is_dir():
            candidates.append(base)
    return candidates[0] if candidates else None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    ensure_parent(dst)
    if copy:
        if dst.is_symlink() or dst.exists():
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
        shutil.copy2(src, dst)
        return
    # Prefer absolute symlink to avoid broken relative links.
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def yolo_format_line(bbox: Tuple[float, float, float, float], width: int, height: int, cls_id: int) -> str:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def convert_split(
    split_name: str,
    labels_path: Path,
    available_images: Dict[str, Path],
    yolo_labels_dir: Path,
    yolo_images_dir: Path,
    coco_images_dir: Path,
    default_width: int,
    default_height: int,
    copy_images: bool,
    report_raw_categories: bool,
) -> Dict:
    if labels_path.is_dir():
        raw = load_ann_directory(labels_path)
    else:
        raw = load_json(labels_path)

    normalized = normalize_annotations(raw, default_width, default_height)

    coco_images: List[Dict] = []
    coco_annotations: List[Dict] = []
    annotation_id = 1

    images_with_mapped_objects = 0
    dropped_boxes = 0
    invalid_boxes = 0
    missing_images = 0

    raw_categories_seen: set[str] = set()

    for img_name in tqdm(sorted(normalized.keys()), desc=f"{split_name} images"):
        info = normalized[img_name]
        width = int(info.get("width", default_width) or default_width)
        height = int(info.get("height", default_height) or default_height)

        if width <= 0 or height <= 0:
            width, height = default_width, default_height

        img_path = available_images.get(img_name)
        if not img_path:
            missing_images += 1
            continue

        mapped_labels: List[NormalizedAnnotation] = []
        for lbl in info.get("labels", []):
            raw_cat = lbl.get("category")
            if raw_cat:
                raw_categories_seen.add(_norm_cat(str(raw_cat)))

            mapped_category = map_category(raw_cat)
            if not mapped_category:
                dropped_boxes += 1
                continue

            x1, y1, x2, y2 = lbl["bbox"]

            # Basic validity checks
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                invalid_boxes += 1
                continue

            mapped_labels.append(NormalizedAnnotation(mapped_category, x1, y1, x2, y2, width, height))

        # YOLO label file (empty file is acceptable).
        yolo_lines = [
            yolo_format_line((ann.x1, ann.y1, ann.x2, ann.y2), ann.width, ann.height, YOLO_CLASS_IDS[ann.category])
            for ann in mapped_labels
        ]
        label_out = yolo_labels_dir / f"{Path(img_name).stem}.txt"
        ensure_parent(label_out)
        label_out.write_text("\n".join(yolo_lines), encoding="utf-8")

        # COCO image entry.
        img_id = len(coco_images) + 1
        coco_images.append(
            {
                "id": img_id,
                "file_name": f"{split_name}/{img_name}",
                "width": width,
                "height": height,
            }
        )

        for ann in mapped_labels:
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": COCO_NAME_TO_ID[ann.category],
                    "bbox": [ann.x1, ann.y1, ann.x2 - ann.x1, ann.y2 - ann.y1],
                    "area": (ann.x2 - ann.x1) * (ann.y2 - ann.y1),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        if mapped_labels:
            images_with_mapped_objects += 1

        # Materialize images (YOLO all + COCO split).
        safe_link_or_copy(img_path, yolo_images_dir / img_name, copy_images)
        safe_link_or_copy(img_path, coco_images_dir / split_name / img_name, copy_images)

    if report_raw_categories:
        logging.info("Split %s: unique raw categories seen (%d): %s",
                     split_name, len(raw_categories_seen), sorted(raw_categories_seen))

    coco_payload = {"images": coco_images, "annotations": coco_annotations, "categories": COCO_CATEGORIES}

    stats = {
        "images_total": len(coco_images),
        "images_with_mapped_objects": images_with_mapped_objects,
        "annotations": len(coco_annotations),
        "dropped_boxes": dropped_boxes,
        "invalid_boxes": invalid_boxes,
        "missing_images": missing_images,
    }
    return {"coco": coco_payload, "stats": stats}


def _basic_self_check() -> None:
    # updated checks for your DatasetNinja labels
    assert map_category("car") == "car"
    assert map_category("truck") == "car"
    assert map_category("bus") == "car"
    assert map_category("person") == "pedestrian"
    assert map_category("rider") == "cyclist"
    assert map_category("bike") == "cyclist"
    assert map_category("traffic light") is None
    assert YOLO_CLASS_IDS["car"] == 0 and COCO_NAME_TO_ID["car"] == 1


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    _basic_self_check()

    if not args.images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {args.images_root}")

    yolo_labels_dir = args.out_yolo_root / "labels" / "all"
    yolo_images_dir = args.out_yolo_root / "images" / "all"
    coco_ann_dir = args.out_coco_root / "annotations"
    coco_images_dir = args.out_coco_root / "images"

    yolo_labels_dir.mkdir(parents=True, exist_ok=True)
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    coco_ann_dir.mkdir(parents=True, exist_ok=True)
    coco_images_dir.mkdir(parents=True, exist_ok=True)

    available_images = collect_images(args.images_root)
    logging.info("Found %d unique images on disk.", len(available_images))

    label_sources: List[Tuple[str, Path]] = []
    if args.det_train_json and args.det_train_json.is_file():
        label_sources.append(("train", args.det_train_json))
    if args.det_val_json and args.det_val_json.is_file():
        label_sources.append(("val", args.det_val_json))

    if not label_sources:
        ann_base = find_ann_root(args.images_root, args.ann_root)
        if not ann_base:
            raise FileNotFoundError(
                "No label sources found. Provide --det-train-json/--det-val-json OR --ann-root containing train/ann and val/ann."
            )
        logging.info("Using per-image annotations under %s (train/ann and val/ann).", ann_base)
        label_sources = [
            ("train", ann_base / "train" / "ann"),
            ("val", ann_base / "val" / "ann"),
        ]

    for split_name, label_path in label_sources:
        result = convert_split(
            split_name=split_name,
            labels_path=label_path,
            available_images=available_images,
            yolo_labels_dir=yolo_labels_dir,
            yolo_images_dir=yolo_images_dir,
            coco_images_dir=coco_images_dir,
            default_width=args.default_width,
            default_height=args.default_height,
            copy_images=args.copy_images,
            report_raw_categories=args.report_raw_categories,
        )

        coco_out = coco_ann_dir / f"bdd_{split_name}_mapped.json"
        ensure_parent(coco_out)
        coco_out.write_text(json.dumps(result["coco"], indent=2), encoding="utf-8")

        logging.info(
            "Split %s: images=%d, annotations=%d, mapped_images=%d, dropped=%d, invalid=%d, missing_images=%d",
            split_name,
            result["stats"]["images_total"],
            result["stats"]["annotations"],
            result["stats"]["images_with_mapped_objects"],
            result["stats"]["dropped_boxes"],
            result["stats"]["invalid_boxes"],
            result["stats"]["missing_images"],
        )

    logging.info("Conversion complete. YOLO at %s, COCO at %s", args.out_yolo_root, args.out_coco_root)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Conversion failed: %s", exc)
        sys.exit(1)
