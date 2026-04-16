#!/usr/bin/env python3
"""Prepare KITTI object detection labels as an external ADD YOLO/COCO test split."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

SCENARIO_ROOT = Path(__file__).resolve().parents[2]
AD_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "ad"
CLASS_NAMES = ["car", "pedestrian", "cyclist"]
KITTI_TO_YOLO = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Tram": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}
ALLOWED_EXTS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert KITTI object labels to ADD YOLO external test split.")
    p.add_argument(
        "--kitti-root",
        type=Path,
        default=AD_DATA_ROOT / "kitti_raw",
        help="Root containing KITTI object training/image_2 and training/label_2.",
    )
    p.add_argument("--images-dir", type=Path, help="Override KITTI image_2 directory.")
    p.add_argument("--labels-dir", type=Path, help="Override KITTI label_2 directory.")
    p.add_argument(
        "--out-root",
        type=Path,
        default=AD_DATA_ROOT / "kitti_yolo_splits",
        help="Output materialized YOLO split root.",
    )
    p.add_argument("--split-name", default="test_external", help="Output split folder name.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-images", type=int, help="Optional deterministic subset size.")
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking/hardlinking.")
    p.add_argument(
        "--strict-kitti-classes",
        action="store_true",
        help="Only map KITTI Car/Pedestrian/Cyclist; ignore Van/Truck/Tram/Person_sitting.",
    )
    p.add_argument("--min-box-height", type=float, default=0.0, help="Drop boxes shorter than this many pixels.")
    p.add_argument("--min-box-width", type=float, default=0.0, help="Drop boxes narrower than this many pixels.")
    return p.parse_args()


def first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def resolve_kitti_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    root = args.kitti_root.resolve()
    images_dir = args.images_dir.resolve() if args.images_dir else first_existing(
        [
            root / "training" / "image_2",
            root / "data_object_image_2" / "training" / "image_2",
            root / "image_2" / "training" / "image_2",
        ]
    )
    labels_dir = args.labels_dir.resolve() if args.labels_dir else first_existing(
        [
            root / "training" / "label_2",
            root / "data_object_label_2" / "training" / "label_2",
            root / "label_2" / "training" / "label_2",
        ]
    )
    if images_dir is None:
        raise FileNotFoundError(
            f"Could not find KITTI images under {root}. Expected training/image_2, "
            "or pass --images-dir explicitly."
        )
    if labels_dir is None:
        raise FileNotFoundError(
            f"Could not find KITTI labels under {root}. Expected training/label_2, "
            "or pass --labels-dir explicitly."
        )
    return images_dir, labels_dir


def list_images(images_dir: Path) -> List[Path]:
    images = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not images:
        raise FileNotFoundError(f"No KITTI images found in {images_dir}")
    return images


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def convert_box_to_yolo(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float] | None:
    left, top, right, bottom = box
    left = max(0.0, min(float(width), left))
    right = max(0.0, min(float(width), right))
    top = max(0.0, min(float(height), top))
    bottom = max(0.0, min(float(height), bottom))
    bw = right - left
    bh = bottom - top
    if bw <= 0 or bh <= 0:
        return None
    x_center = (left + right) / 2.0 / width
    y_center = (top + bottom) / 2.0 / height
    return x_center, y_center, bw / width, bh / height


def parse_kitti_label(
    label_path: Path,
    image_size: Tuple[int, int],
    mapping: Dict[str, int],
    min_box_width: float,
    min_box_height: float,
) -> Tuple[List[str], Counter, Counter]:
    width, height = image_size
    out_lines: List[str] = []
    kept = Counter()
    ignored = Counter()
    if not label_path.is_file():
        return out_lines, kept, ignored

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 8:
            continue
        kitti_class = parts[0]
        if kitti_class == "DontCare":
            ignored[kitti_class] += 1
            continue
        if kitti_class not in mapping:
            ignored[kitti_class] += 1
            continue
        left, top, right, bottom = map(float, parts[4:8])
        if (right - left) < min_box_width or (bottom - top) < min_box_height:
            ignored[f"{kitti_class}:small"] += 1
            continue
        yolo_box = convert_box_to_yolo((left, top, right, bottom), width, height)
        if yolo_box is None:
            ignored[f"{kitti_class}:invalid"] += 1
            continue
        cls_id = mapping[kitti_class]
        kept[CLASS_NAMES[cls_id]] += 1
        out_lines.append(f"{cls_id} " + " ".join(f"{v:.8f}" for v in yolo_box))
    return out_lines, kept, ignored


def write_yaml(out_root: Path, split_name: str) -> None:
    yaml_text = (
        f"path: {out_root.resolve()}\n"
        f"train: images/{split_name}\n"
        f"val: images/{split_name}\n"
        f"test: images/{split_name}\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES!r}\n"
    )
    (out_root / "kitti_external.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    images_dir, labels_dir = resolve_kitti_dirs(args)
    images = list_images(images_dir)
    if args.max_images:
        rng = random.Random(args.seed)
        images = sorted(rng.sample(images, min(args.max_images, len(images))))

    out_root = args.out_root.resolve()
    split_name = args.split_name
    images_out = out_root / "images" / split_name
    labels_out = out_root / "labels" / split_name
    splits_out = out_root / "splits"
    if out_root.exists():
        shutil.rmtree(out_root)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    splits_out.mkdir(parents=True, exist_ok=True)

    mapping = dict(KITTI_TO_YOLO)
    if args.strict_kitti_classes:
        mapping = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    kept_total = Counter()
    ignored_total = Counter()
    empty_labels = 0
    stems: List[str] = []

    for img_path in images:
        with Image.open(img_path) as im:
            width, height = im.size
        out_lines, kept, ignored = parse_kitti_label(
            labels_dir / f"{img_path.stem}.txt",
            (width, height),
            mapping,
            args.min_box_width,
            args.min_box_height,
        )
        link_or_copy(img_path, images_out / img_path.name, args.copy)
        (labels_out / f"{img_path.stem}.txt").write_text("\n".join(out_lines), encoding="utf-8")
        if not out_lines:
            empty_labels += 1
        kept_total.update(kept)
        ignored_total.update(ignored)
        stems.append(img_path.stem)

    (splits_out / f"kitti_{split_name}.txt").write_text("\n".join(stems), encoding="utf-8")
    (splits_out / "bdd_test_internal.txt").write_text("\n".join(stems), encoding="utf-8")
    write_yaml(out_root, split_name)

    manifest = {
        "dataset": "kitti_external",
        "source_images": str(images_dir),
        "source_labels": str(labels_dir),
        "out_root": str(out_root),
        "split_name": split_name,
        "image_count": len(images),
        "empty_labels": empty_labels,
        "class_counts": {name: int(kept_total[name]) for name in CLASS_NAMES},
        "ignored_counts": dict(sorted((k, int(v)) for k, v in ignored_total.items())),
        "class_mapping": {k: CLASS_NAMES[v] for k, v in mapping.items()},
    }
    (out_root / "kitti_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"KITTI preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
