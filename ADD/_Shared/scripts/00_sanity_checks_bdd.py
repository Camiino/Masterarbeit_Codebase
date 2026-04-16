#!/usr/bin/env python3
"""
Sanity checks for BDD100K detection labels before conversion.

Usage examples:
    python project/scripts/00_sanity_checks_bdd.py \
    --images-root project/data/ad/bdd100k_raw/bdd100k \
    --ann-root project/data/ad/bdd100k_raw/bdd100k \
    --output-report project/data/ad/bdd100k_raw/bdd_sanity_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Mapping from raw BDD categories to target categories.
TARGET_CATEGORY_MAP = {
    "car": "car",
    "truck": "car",
    "bus": "car",
    "train": "car",
    "pedestrian": "pedestrian",
    "rider": "pedestrian",
    "bicycle": "cyclist",
    "motorcycle": "cyclist",
}

TARGET_ORDER = ["car", "pedestrian", "cyclist"]


@dataclass
class NormalizedBBox:
    category: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity checks on BDD100K labels.")
    parser.add_argument("--images-root", type=Path, required=True, help="Root of BDD images (contains train/val/test folders).")
    parser.add_argument(
        "--labels-json",
        type=Path,
        nargs="+",
        default=[],
        help="Optional path(s) to det_train.json and det_val.json.",
    )
    parser.add_argument("--output-report", type=Path, required=True, help="Where to write the JSON report.")
    parser.add_argument("--default-width", type=int, default=1280, help="Fallback width if labels lack dimension metadata.")
    parser.add_argument("--default-height", type=int, default=720, help="Fallback height if labels lack dimension metadata.")
    parser.add_argument(
        "--ann-root",
        type=Path,
        help="Optional root containing train/ann and val/ann directories with per-image JSONs. Used as fallback if labels-json are missing.",
    )
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
    return TARGET_CATEGORY_MAP.get(raw_category)


def load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ann_directory(ann_dir: Path) -> List[Dict]:
    """
    Load per-image annotations from a directory of JSON files that follow the datasetninja structure:
    {
        "size": {"width": ..., "height": ...},
        "objects": [
            {"geometryType": "rectangle", "classTitle": "...", "points": {"exterior": [[x1,y1],[x2,y2]]}}
        ]
    }
    Returns a list of dicts shaped like the native BDD list format for reuse in normalize_annotations.
    """
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {ann_dir}")
    items: List[Dict] = []
    for json_file in sorted(ann_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        size = data.get("size", {})
        width = int(size.get("width", 0))
        height = int(size.get("height", 0))
        labels = []
        for obj in data.get("objects", []):
            if obj.get("geometryType") != "rectangle":
                continue
            points = obj.get("points", {}).get("exterior", [])
            if not points or len(points) < 2:
                continue
            (x1, y1), (x2, y2) = points[0], points[1]
            box = {"x1": float(min(x1, x2)), "y1": float(min(y1, y2)), "x2": float(max(x1, x2)), "y2": float(max(y1, y2))}
            labels.append({"category": obj.get("classTitle"), "box2d": box})
        # Some filenames already include .jpg in the stem (e.g., foo.jpg.json); normalize to a single .jpg.
        base_name = json_file.name.replace(".json", "")
        normalized_name = Path(base_name).stem + ".jpg"
        items.append({"name": normalized_name, "width": width, "height": height, "labels": labels})
    if not items:
        raise FileNotFoundError(f"No annotation JSON files found in {ann_dir}")
    return items


def normalize_annotations(
    raw_data: object,
    default_width: int,
    default_height: int,
) -> Dict[str, Dict]:
    """
    Normalize annotations to a per-image structure:
    {
        image_name: {
            "width": ...,
            "height": ...,
            "labels": [{"category": str, "bbox": (x1, y1, x2, y2)}]
        }
    }
    Supports the native BDD list format and COCO-like dict format.
    """
    normalized: Dict[str, Dict] = {}

    if isinstance(raw_data, dict) and {"images", "annotations"}.issubset(raw_data.keys()):
        categories = {c["id"]: c["name"] for c in raw_data.get("categories", [])}
        images = {img["id"]: img for img in raw_data.get("images", [])}
        for ann in raw_data.get("annotations", []):
            img_info = images.get(ann.get("image_id"))
            if not img_info:
                continue
            name = Path(img_info.get("file_name") or img_info.get("name")).name
            width = int(img_info.get("width", default_width))
            height = int(img_info.get("height", default_height))

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

        # Ensure images without annotations are still tracked.
        for img in raw_data.get("images", []):
            name = Path(img.get("file_name") or img.get("name")).name
            normalized.setdefault(
                name,
                {
                    "width": int(img.get("width", default_width)),
                    "height": int(img.get("height", default_height)),
                    "labels": [],
                },
            )
        return normalized

    if isinstance(raw_data, list):
        for item in raw_data:
            name = Path(item["name"]).name
            width = int(item.get("width", default_width))
            height = int(item.get("height", default_height))
            labels = []
            for lbl in item.get("labels", []):
                box = lbl.get("box2d")
                if not box:
                    continue
                labels.append(
                    {"category": lbl.get("category"), "bbox": (float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]))}
                )
            normalized[name] = {"width": width, "height": height, "labels": labels}
        return normalized

    raise ValueError("Unsupported label JSON structure; expected list or COCO-like dict.")


def count_images_on_disk(images_root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split in ["train", "val", "test"]:
        split_dir = images_root / split
        if (split_dir / "img").is_dir():
            split_dir = split_dir / "img"
        if split_dir.is_dir():
            counts[split] = sum(1 for _ in split_dir.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
    counts["total"] = sum(counts.values())
    return counts


def infer_split_name(path: Path) -> str:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "validation" in name:
        return "val"
    if "test" in name:
        return "test"
    return "unknown"


def ensure_images_exist(images_root: Path) -> Path:
    candidates = [
        images_root,
        images_root.parent,
    ]
    for cand in candidates:
        if cand and cand.is_dir():
            has_images = any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in cand.rglob("*") if p.is_file())
            if has_images:
                if cand != images_root:
                    logging.warning("Using images found at %s instead of provided %s", cand, images_root)
                return cand

    tar_hint = find_tarball()
    hint_text = f"Found tarball at {tar_hint}. Extract with: python project/scripts/00_extract_bdd_tar.py --tar-path {tar_hint} --out-dir {images_root}" if tar_hint else "No tarball detected automatically."
    raise FileNotFoundError(
        f"Images root not found or empty near: {images_root}. "
        "Prepare BDD100K images first (use 00_extract_bdd_tar.py or 00_fetch_bdd_images.py) and ensure train/val/test[/img] subfolders exist. "
        f"{hint_text}"
    )


def find_tarball() -> Optional[Path]:
    search_roots = [
        Path.cwd(),
        Path.cwd() / "project",
        Path.cwd() / "project" / "bdd100k_download",
        Path.cwd() / "project" / "data" / "ad" / "bdd100k_raw" / "bdd100k",
    ]
    candidates: List[Path] = []
    patterns = ["bdd100k*.tar", "bdd100k*.tar.gz", "*.tar", "*.tar.gz"]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
    if not candidates:
        return None
    # Deduplicate and pick the first.
    unique: List[Path] = []
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique.append(cand)
    return unique[0]


def find_ann_root(images_root: Path, override: Optional[Path]) -> Optional[Path]:
    candidates = []
    bases = [b for b in {override, images_root, images_root.parent, images_root.parent.parent} if b]
    for base in bases:
        if not base.exists():
            continue
        has_train = (base / "train" / "ann").is_dir()
        has_val = (base / "val" / "ann").is_dir()
        if has_train and has_val:
            candidates.append(base)
    if not candidates:
        return None
    return candidates[0]


def validate_and_map_bboxes(
    normalized: Dict[str, Dict],
) -> Tuple[List[NormalizedBBox], Counter, int, int]:
    """
    Returns:
        - list of normalized mapped boxes
        - per-class counter
        - dropped boxes count
        - invalid boxes count
    """
    per_class = Counter()
    dropped = 0
    invalid = 0
    mapped_boxes: List[NormalizedBBox] = []

    for name, info in normalized.items():
        width = info.get("width")
        height = info.get("height")
        for lbl in info.get("labels", []):
            mapped_category = map_category(lbl.get("category"))
            if not mapped_category:
                dropped += 1
                continue
            x1, y1, x2, y2 = lbl["bbox"]
            if x2 <= x1 or y2 <= y1:
                invalid += 1
                continue
            if width is not None and height is not None:
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    invalid += 1
                    continue
            mapped_boxes.append(NormalizedBBox(mapped_category, x1, y1, x2, y2, width or 0, height or 0))
            per_class[mapped_category] += 1

    return mapped_boxes, per_class, dropped, invalid


def count_images_with_objects(normalized: Dict[str, Dict]) -> int:
    count = 0
    for info in normalized.values():
        mapped_objects = [lbl for lbl in info.get("labels", []) if map_category(lbl.get("category"))]
        if mapped_objects:
            count += 1
    return count


def write_report(report_path: Path, payload: Dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _basic_self_check() -> None:
    assert map_category("truck") == "car"
    assert map_category("pedestrian") == "pedestrian"
    assert map_category("bicycle") == "cyclist"
    assert map_category("unknown") is None


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    _basic_self_check()

    images_root = ensure_images_exist(args.images_root)
    label_sources: List[Dict] = []
    missing_paths: List[Path] = []
    for lbl_path in args.labels_json:
        if lbl_path.is_file():
            label_sources.append({"source": lbl_path, "loader": load_json, "split": infer_split_name(lbl_path)})
        else:
            missing_paths.append(lbl_path)

    # If no JSONs were provided, auto-fallback to ann-root.
    if not label_sources and not args.labels_json:
        ann_base = find_ann_root(images_root, args.ann_root)
        if ann_base:
            logging.info("Using per-image annotations under %s (train/ann and val/ann).", ann_base)
            for split in ["train", "val"]:
                ann_dir = ann_base / split / "ann"
                if not ann_dir.is_dir():
                    raise FileNotFoundError(f"Annotation directory missing: {ann_dir}")
                label_sources.append({"source": ann_dir, "loader": load_ann_directory, "split": split})

    if missing_paths:
        ann_base = find_ann_root(images_root, args.ann_root)
        if not ann_base:
            missing_str = ", ".join(str(p) for p in missing_paths)
            raise FileNotFoundError(
                f"Label JSON missing: {missing_str}. "
                "Provide det_train/det_val JSONs or supply per-image annotations via --ann-root (train/ann, val/ann)."
            )
        logging.info("Falling back to per-image annotations under %s", ann_base)
        desired_splits = {infer_split_name(p) for p in missing_paths}
        if "unknown" in desired_splits or not desired_splits:
            desired_splits = {"train", "val"}
        for split in ["train", "val", "test"]:
            if split not in desired_splits:
                continue
            ann_dir = ann_base / split / "ann"
            if not ann_dir.is_dir():
                raise FileNotFoundError(f"Annotation directory missing: {ann_dir}")
            label_sources.append({"source": ann_dir, "loader": load_ann_directory, "split": split})

    if not label_sources:
        raise FileNotFoundError("No label sources resolved. Provide --labels-json or --ann-root with train/ann and val/ann.")

    image_counts = count_images_on_disk(images_root)
    logging.info("Image counts on disk: %s", image_counts)

    combined_counts = Counter()
    total_dropped = 0
    total_invalid = 0
    per_label_file_summary = []
    images_with_objects_total = 0
    processed_images = set()

    for src in label_sources:
        raw = src["loader"](src["source"])
        normalized = normalize_annotations(raw, args.default_width, args.default_height)
        mapped_boxes, per_class, dropped, invalid = validate_and_map_bboxes(normalized)
        combined_counts.update(per_class)
        total_dropped += dropped
        total_invalid += invalid
        images_with_objects = count_images_with_objects(normalized)
        images_with_objects_total += images_with_objects
        processed_images.update(normalized.keys())
        per_label_file_summary.append(
            {
                "label_file": str(src["source"]),
                "split": src.get("split", "unknown"),
                "images_in_labels": len(normalized),
                "images_with_mapped_objects": images_with_objects,
                "mapped_box_count": sum(per_class.values()),
                "dropped_boxes": dropped,
                "invalid_boxes": invalid,
            }
        )
        logging.info(
            "Processed %s: %d images, %d mapped boxes, %d dropped, %d invalid",
            src["source"],
            len(normalized),
            sum(per_class.values()),
            dropped,
            invalid,
        )

    report = {
        "image_counts_on_disk": image_counts,
        "label_files": per_label_file_summary,
        "total_images_in_labels": len(processed_images),
        "images_with_mapped_objects": images_with_objects_total,
        "object_counts": {cls: combined_counts.get(cls, 0) for cls in TARGET_ORDER},
        "dropped_boxes": total_dropped,
        "invalid_boxes": total_invalid,
    }

    logging.info("Writing report to %s", args.output_report)
    write_report(args.output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Sanity checks failed: %s", exc)
        sys.exit(1)
