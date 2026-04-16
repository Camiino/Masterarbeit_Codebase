#!/usr/bin/env python3
"""
Prepare a YOLO-style detection subset from SYNTHIA-RAND-CITYSCAPES.

This script converts SYNTHIA semantic+instance labels into YOLO txt labels for the
three-class detection setup already used elsewhere in this repo:

- car        <- SYNTHIA {car, truck, bus}
- pedestrian <- SYNTHIA {pedestrian}
- cyclist    <- SYNTHIA {rider, bicycle, motorcycle}

The default mapping is intentionally conservative so the synthetic labels stay as close
as possible to the BDD classes already used by the one-stage and two-stage pipelines.

The script writes:
- images/{train,val,test}
- labels/{train,val,test}
- splits/synthia_{train,val,test}.txt
- splits/bdd_{train,val,test_internal}.txt
- images/test_internal and labels/test_internal symlinks to test
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

YOLO_CLASSES = ["car", "pedestrian", "cyclist"]
SCENARIO_ROOT = Path(__file__).resolve().parents[2]
AD_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "ad"

# SYNTHIA class IDs from the official README in RAND_CITYSCAPES/README.txt
DEFAULT_CLASS_GROUPS = {
    "car": {8, 18, 19},   # car, truck, bus
    "pedestrian": {10},   # pedestrian
    "cyclist": {11, 12, 17},  # bicycle, motorcycle, rider
}


@dataclass(frozen=True)
class Record:
    stem: str
    image_path: Path
    label_lines: List[str]


def parse_id_set(text: str) -> set[int]:
    ids = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        ids.add(int(item))
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert SYNTHIA-RAND-CITYSCAPES into YOLO detection splits.")
    p.add_argument(
        "--synthia-root",
        type=Path,
        default=AD_DATA_ROOT / "synthia_rand_cityscapes" / "RAND_CITYSCAPES",
        help="Extracted SYNTHIA RAND_CITYSCAPES root containing RGB/ and GT/LABELS/.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=AD_DATA_ROOT / "synthia_yolo_splits",
        help="Output root for YOLO-style split folders.",
    )
    p.add_argument("--subset-size", type=int, default=9000, help="Total number of images to sample.")
    p.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    p.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    p.add_argument(
        "--min-instance-pixels",
        type=int,
        default=20,
        help="Discard tiny instances smaller than this many pixels.",
    )
    p.add_argument("--car-ids", type=str, default="8,18,19", help="Comma-separated SYNTHIA IDs grouped as YOLO car.")
    p.add_argument("--ped-ids", type=str, default="10", help="Comma-separated SYNTHIA IDs grouped as YOLO pedestrian.")
    p.add_argument("--cyclist-ids", type=str, default="11,12,17", help="Comma-separated SYNTHIA IDs grouped as YOLO cyclist.")
    return p.parse_args()


def safe_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def find_rgb_images(rgb_dir: Path) -> List[Path]:
    images = sorted(p for p in rgb_dir.glob("*.png") if p.is_file())
    if not images:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")
    return images


def load_label_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read label image: {path}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unexpected label image shape for {path}: {img.shape}")
    return img.astype(np.int32)


def make_instance_map(label_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # OpenCV returns channel order BGR. The README states:
    # - first channel: class id
    # - second channel: instance id
    # In the stored 16-bit PNGs this lands as:
    # - channel 2 (R): class id
    # - channels 0/1 (B/G): instance id split into high/low bytes
    class_ids = label_img[:, :, 2]
    instance_ids = label_img[:, :, 0] * 256 + label_img[:, :, 1]
    return class_ids, instance_ids


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h
    xc = (x1 + x2 + 1) / 2.0 / w
    yc = (y1 + y2 + 1) / 2.0 / h
    return f"{xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def extract_boxes_for_group(
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    source_ids: Iterable[int],
    target_cls: int,
    image_w: int,
    image_h: int,
    min_instance_pixels: int,
) -> List[str]:
    lines: List[str] = []
    for source_id in source_ids:
        class_mask = class_ids == source_id
        if not np.any(class_mask):
            continue
        obj_ids = np.unique(instance_ids[class_mask])
        obj_ids = obj_ids[obj_ids > 0]
        for obj_id in obj_ids:
            ys, xs = np.where(class_mask & (instance_ids == int(obj_id)))
            if ys.size < min_instance_pixels:
                continue
            x1 = int(xs.min())
            x2 = int(xs.max())
            y1 = int(ys.min())
            y2 = int(ys.max())
            lines.append(f"{target_cls} {xyxy_to_yolo(x1, y1, x2, y2, image_w, image_h)}")
    return lines


def build_records(
    synthia_root: Path,
    groups: Dict[str, set[int]],
    min_instance_pixels: int,
) -> List[Record]:
    rgb_dir = synthia_root / "RGB"
    labels_dir = synthia_root / "GT" / "LABELS"
    images = find_rgb_images(rgb_dir)
    records: List[Record] = []

    total = len(images)
    for idx, image_path in enumerate(images, start=1):
        label_path = labels_dir / image_path.name
        if not label_path.is_file():
            continue
        label_img = load_label_image(label_path)
        image_h, image_w = label_img.shape[:2]
        class_ids, instance_ids = make_instance_map(label_img)

        lines: List[str] = []
        lines.extend(
            extract_boxes_for_group(
                class_ids,
                instance_ids,
                groups["car"],
                0,
                image_w,
                image_h,
                min_instance_pixels,
            )
        )
        lines.extend(
            extract_boxes_for_group(
                class_ids,
                instance_ids,
                groups["pedestrian"],
                1,
                image_w,
                image_h,
                min_instance_pixels,
            )
        )
        lines.extend(
            extract_boxes_for_group(
                class_ids,
                instance_ids,
                groups["cyclist"],
                2,
                image_w,
                image_h,
                min_instance_pixels,
            )
        )

        if lines:
            records.append(Record(stem=image_path.stem, image_path=image_path, label_lines=lines))

        if idx == 1 or idx % 250 == 0 or idx == total:
            print(f"[synthia-prepare] scanned {idx}/{total} images; kept {len(records)} with mapped objects", flush=True)

    if not records:
        raise RuntimeError("No labeled SYNTHIA images were converted into detection boxes.")
    return records


def split_records(records: Sequence[Record], subset_size: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Record]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and sum to less than 1.")
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    pool = pool[: min(subset_size, len(pool))]

    n_total = len(pool)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": pool[:n_train],
        "val": pool[n_train : n_train + n_val],
        "test": pool[n_train + n_val :],
    }


def write_split(split_name: str, items: Sequence[Record], out_root: Path, copy: bool) -> None:
    img_dir = out_root / "images" / split_name
    lbl_dir = out_root / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for rec in items:
        dst_img = img_dir / rec.image_path.name
        dst_lbl = lbl_dir / f"{rec.stem}.txt"
        safe_link_or_copy(rec.image_path, dst_img, copy)
        dst_lbl.write_text("\n".join(rec.label_lines), encoding="utf-8")


def write_split_lists(splits: Dict[str, List[Record]], out_root: Path) -> None:
    split_dir = out_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        stems = "\n".join(rec.stem for rec in items)
        (split_dir / f"synthia_{split_name}.txt").write_text(stems, encoding="utf-8")

    (split_dir / "bdd_train.txt").write_text("\n".join(rec.stem for rec in splits["train"]), encoding="utf-8")
    (split_dir / "bdd_val.txt").write_text("\n".join(rec.stem for rec in splits["val"]), encoding="utf-8")
    (split_dir / "bdd_test_internal.txt").write_text("\n".join(rec.stem for rec in splits["test"]), encoding="utf-8")


def ensure_test_internal_alias(out_root: Path) -> None:
    for parent in [out_root / "images", out_root / "labels"]:
        test_dir = parent / "test"
        alias = parent / "test_internal"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(test_dir, target_is_directory=True)


def main() -> None:
    args = parse_args()
    groups = {
        "car": parse_id_set(args.car_ids),
        "pedestrian": parse_id_set(args.ped_ids),
        "cyclist": parse_id_set(args.cyclist_ids),
    }

    records = build_records(args.synthia_root, groups, args.min_instance_pixels)
    splits = split_records(records, args.subset_size, args.train_ratio, args.val_ratio, args.seed)

    for split_name, items in splits.items():
        write_split(split_name, items, args.out_root, args.copy)
        print(f"{split_name}: {len(items)} images", flush=True)

    write_split_lists(splits, args.out_root)
    ensure_test_internal_alias(args.out_root)

    print("Class grouping:", flush=True)
    print(f"  car        <- {sorted(groups['car'])}", flush=True)
    print(f"  pedestrian <- {sorted(groups['pedestrian'])}", flush=True)
    print(f"  cyclist    <- {sorted(groups['cyclist'])}", flush=True)
    print(f"Output root: {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
