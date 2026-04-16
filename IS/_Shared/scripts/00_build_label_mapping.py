#!/usr/bin/env python3
"""Build and validate the IS label mapping for a YOLO dataset root."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_NAMES = ["hex", "hex_socket", "phillips", "pozidriv", "slotted", "torx"]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SCENARIO_ROOT = Path(__file__).resolve().parents[2]
IS_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "is"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a label mapping and label summary for IS YOLO data.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=IS_DATA_ROOT / "raw" / "real_yolov9_all_images",
        help="YOLO dataset root containing train/images and train/labels.",
    )
    p.add_argument("--split", default="train", help="Split to inspect. Default: train.")
    p.add_argument(
        "--out",
        type=Path,
        default=IS_DATA_ROOT / "label_mapping_real.json",
        help="Output JSON mapping/summary path.",
    )
    return p.parse_args()


def load_names(dataset_root: Path) -> List[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        return DEFAULT_NAMES

    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            value = line.split(":", 1)[1].strip()
            try:
                names = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                break
            if isinstance(names, list) and all(isinstance(x, str) for x in names):
                return names
    return DEFAULT_NAMES


def image_stems(images_dir: Path) -> set[str]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return {p.stem for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS}


def label_paths(labels_dir: Path) -> List[Path]:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    return sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images_dir = dataset_root / args.split / "images"
    labels_dir = dataset_root / args.split / "labels"

    names = load_names(dataset_root)
    mapping = {str(idx): name for idx, name in enumerate(names)}
    instances_by_class: Dict[str, int] = {str(idx): 0 for idx in range(len(names))}
    images_by_class: Dict[str, int] = {str(idx): 0 for idx in range(len(names))}
    unknown_class_ids: Dict[str, int] = {}
    empty_labels = 0

    stems_with_images = image_stems(images_dir)
    labels = label_paths(labels_dir)
    stems_with_labels = {p.stem for p in labels}

    for label_path in labels:
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels += 1
            continue

        classes_in_image: set[str] = set()
        for line_no, line in enumerate(content.splitlines(), start=1):
            parts = line.split()
            if not parts:
                continue
            class_id = parts[0]
            if class_id not in mapping:
                unknown_class_ids[class_id] = unknown_class_ids.get(class_id, 0) + 1
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO row at {label_path}:{line_no}: expected 5 columns, got {len(parts)}")
            instances_by_class[class_id] += 1
            classes_in_image.add(class_id)

        for class_id in classes_in_image:
            images_by_class[class_id] += 1

    missing_labels = sorted(stems_with_images - stems_with_labels)
    labels_without_images = sorted(stems_with_labels - stems_with_images)
    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "mapping": mapping,
        "num_images": len(stems_with_images),
        "num_labels": len(labels),
        "empty_labels": empty_labels,
        "instances_by_class": {mapping[k]: instances_by_class[k] for k in mapping},
        "images_by_class": {mapping[k]: images_by_class[k] for k in mapping},
        "unknown_class_ids": unknown_class_ids,
        "missing_labels": missing_labels,
        "labels_without_images": labels_without_images,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"dataset_root: {dataset_root}")
    print(f"images: {summary['num_images']}")
    print(f"labels: {summary['num_labels']}")
    print(f"empty_labels: {empty_labels}")
    print("labels:")
    for class_id, name in mapping.items():
        print(f"  {class_id}: {name}")
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"label mapping failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
