#!/usr/bin/env python3
"""
Materialize YOLO split folders and dataset YAML for experiment E1.

Usage example:
    python project/scripts/03_materialize_yolo_split_folders.py \
        --yolo-root project/data/ad/bdd_yolo \
        --split-dir project/data/ad/bdd_yolo/splits \
        --out-yolo-split-root project/data/ad/bdd_yolo_splits \
        --seed 0
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TARGET_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize YOLO split folders for BDD experiment.")
    parser.add_argument("--yolo-root", type=Path, required=True, help="Root where images/all and labels/all reside.")
    parser.add_argument("--split-dir", type=Path, required=True, help="Directory containing bdd_train.txt, bdd_val.txt, bdd_test_internal.txt.")
    parser.add_argument("--out-yolo-split-root", type=Path, required=True, help="Output root for split images/labels folders and YAML.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for naming the dataset YAML.")
    parser.add_argument("--copy-images", action="store_true", help="Copy instead of symlink (useful on platforms without symlink support).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_split_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Split file is empty: {path}")
    return lines


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def find_image_by_stem(stem: str, images_dir: Path) -> Path:
    for ext in ALLOWED_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image for stem '{stem}' not found in {images_dir}")


def materialize_split(
    split_name: str,
    stems: List[str],
    images_all_dir: Path,
    labels_all_dir: Path,
    out_root: Path,
    copy_images: bool,
) -> Dict[str, int]:
    out_images = out_root / "images" / split_name
    out_labels = out_root / "labels" / split_name
    ensure_dir(out_images)
    ensure_dir(out_labels)

    created_empty_labels = 0

    for stem in stems:
        src_img = find_image_by_stem(stem, images_all_dir)
        src_label = labels_all_dir / f"{stem}.txt"
        dst_img = out_images / src_img.name
        dst_label = out_labels / f"{stem}.txt"

        safe_link_or_copy(src_img, dst_img, copy_images)

        if src_label.is_file():
            safe_link_or_copy(src_label, dst_label, copy_images)
        else:
            # Create empty label file for unlabeled images.
            dst_label.write_text("", encoding="utf-8")
            created_empty_labels += 1

    return {
        "images": len(stems),
        "empty_labels_created": created_empty_labels,
    }


def write_dataset_yaml(out_root: Path, seed: int) -> Path:
    yaml_path = out_root / f"ad_bdd_E1_seed{seed}.yaml"
    content = "\n".join(
        [
            f"path: {out_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test_internal",
            "names:",
            "  0: car",
            "  1: pedestrian",
            "  2: cyclist",
        ]
    )
    yaml_path.write_text(content + "\n", encoding="utf-8")
    return yaml_path


def _basic_self_check() -> None:
    assert TARGET_NAMES[0] == "car"
    assert TARGET_NAMES[2] == "cyclist"


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    _basic_self_check()

    images_all_dir = args.yolo_root / "images" / "all"
    labels_all_dir = args.yolo_root / "labels" / "all"
    for required_dir in [images_all_dir, labels_all_dir]:
        if not required_dir.is_dir():
            raise FileNotFoundError(f"Required directory missing: {required_dir}")

    train_list = load_split_list(args.split_dir / "bdd_train.txt")
    val_list = load_split_list(args.split_dir / "bdd_val.txt")
    test_list = load_split_list(args.split_dir / "bdd_test_internal.txt")

    if args.out_yolo_split_root.exists():
        shutil.rmtree(args.out_yolo_split_root)
    args.out_yolo_split_root.mkdir(parents=True, exist_ok=True)

    stats_train = materialize_split("train", train_list, images_all_dir, labels_all_dir, args.out_yolo_split_root, args.copy_images)
    stats_val = materialize_split("val", val_list, images_all_dir, labels_all_dir, args.out_yolo_split_root, args.copy_images)
    stats_test = materialize_split("test_internal", test_list, images_all_dir, labels_all_dir, args.out_yolo_split_root, args.copy_images)

    yaml_path = write_dataset_yaml(args.out_yolo_split_root, args.seed)
    logging.info("Dataset YAML written to %s", yaml_path)
    logging.info("Split stats: train=%s, val=%s, test_internal=%s", stats_train, stats_val, stats_test)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Materialization failed: %s", exc)
        sys.exit(1)
