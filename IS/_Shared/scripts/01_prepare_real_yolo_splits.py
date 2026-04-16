#!/usr/bin/env python3
"""Prepare deterministic IS YOLO splits for real or synthetic raw roots."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["hex", "hex_socket", "phillips", "pozidriv", "slotted", "torx"]
SCENARIO_ROOT = Path(__file__).resolve().parents[2]
IS_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "is"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize IS YOLO train/val/test_internal splits.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=IS_DATA_ROOT / "raw" / "real_yolov9_all_images",
        help="Raw YOLO root containing train/images and train/labels.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=IS_DATA_ROOT / "real_yolo_splits",
        help="Output YOLO split root.",
    )
    p.add_argument("--name", default="is_real", help="Dataset name prefix for YAML and split files.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-images", type=int, default=488, help="Dataset size to sample. Default: 488.")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking.")
    return p.parse_args()


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def split_pool(images: List[Path], n_images: int, train_ratio: float, val_ratio: float, rng: random.Random) -> Dict[str, List[Path]]:
    if n_images <= 0:
        raise ValueError("n_images must be positive.")
    if n_images > len(images):
        raise ValueError(f"Requested {n_images} images but only {len(images)} are available.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and sum to less than 1.")

    pool = images[:]
    rng.shuffle(pool)
    pool = pool[:n_images]

    train_count = int(n_images * train_ratio)
    val_count = int(n_images * val_ratio)
    test_count = n_images - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("Computed split counts must all be positive.")

    return {
        "train": pool[:train_count],
        "val": pool[train_count : train_count + val_count],
        "test_internal": pool[train_count + val_count :],
    }


def materialize(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def write_yaml(out_root: Path, name: str) -> None:
    yaml_text = (
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test_internal\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES!r}\n"
    )
    (out_root / f"{name}.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_images = dataset_root / "train" / "images"
    source_labels = dataset_root / "train" / "labels"
    out_root = args.out_root.resolve()
    split_dir = out_root / "splits"

    images = list_images(source_images)
    rng = random.Random(args.seed)
    splits = split_pool(images, args.n_images, args.train_ratio, args.val_ratio, rng)

    if out_root.exists():
        shutil.rmtree(out_root)

    manifest = {
        "dataset": args.name,
        "source_root": str(dataset_root),
        "seed": args.seed,
        "n_images": args.n_images,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "counts": {},
        "empty_labels": {},
    }

    for split_name, split_images in splits.items():
        stems = []
        empty_labels = 0
        for image_path in split_images:
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path.name}: {label_path}")
            materialize(image_path, out_root / "images" / split_name / image_path.name, args.copy)
            materialize(label_path, out_root / "labels" / split_name / label_path.name, args.copy)
            stems.append(image_path.stem)
            if label_path.stat().st_size == 0:
                empty_labels += 1

        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"{args.name}_{split_name}.txt").write_text("\n".join(stems), encoding="utf-8")
        # Compatibility with ADD two-stage scripts that expect bdd_* names.
        compat_name = "test_internal" if split_name == "test_internal" else split_name
        (split_dir / f"bdd_{compat_name}.txt").write_text("\n".join(stems), encoding="utf-8")
        manifest["counts"][split_name] = len(stems)
        manifest["empty_labels"][split_name] = empty_labels

    write_yaml(out_root, args.name)
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote: {out_root}")
    print(f"dataset size: {args.n_images}")
    print(f"train: {manifest['counts']['train']}")
    print(f"val: {manifest['counts']['val']}")
    print(f"test_internal: {manifest['counts']['test_internal']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"split preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
