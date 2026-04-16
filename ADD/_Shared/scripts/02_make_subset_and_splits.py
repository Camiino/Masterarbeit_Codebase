#!/usr/bin/env python3
"""
Create deterministic BDD100K subsets and splits for YOLO training.

Usage example:
    python project/scripts/02_make_subset_and_splits.py \
        --yolo-images-dir project/data/ad/bdd_yolo/images/all \
        --seed 0 \
        --n-images 9000 \
        --out-split-dir project/data/ad/bdd_yolo/splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic subset and splits for BDD YOLO data.")
    parser.add_argument("--yolo-images-dir", type=Path, required=True, help="Path to YOLO images/all directory.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Optional labels directory. Defaults to sibling labels/all under YOLO root.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic sampling.")
    parser.add_argument("--n-images", type=int, default=9000, help="Total images to select for the pool.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio (default 0.7).")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (default 0.2).")
    parser.add_argument("--allow-empty-images", action="store_true", help="Allow empty-label images if needed to reach N.")
    parser.add_argument("--out-split-dir", type=Path, required=True, help="Output directory for split text files and manifest.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def locate_labels_dir(images_dir: Path, override: Path | None) -> Path:
    if override:
        return override
    return images_dir.parent.parent / "labels" / "all"


def list_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images: List[Path] = []
    for p in images_dir.iterdir():
        if p.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        # Accept real files OR symlinks that resolve to real files
        try:
            if p.is_file() or (p.is_symlink() and p.resolve().is_file()):
                images.append(p)
        except FileNotFoundError:
            # Broken symlink
            continue

    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return sorted(images)



def image_has_labels(image_path: Path, labels_dir: Path) -> bool:
    label_path = labels_dir / f"{image_path.stem}.txt"
    if not label_path.is_file():
        return False
    content = label_path.read_text(encoding="utf-8").strip()
    return bool(content)


def choose_pool(
    images: List[Path],
    labels_dir: Path,
    n_images: int,
    allow_empty: bool,
    rng: random.Random,
) -> Tuple[List[Path], List[Path]]:
    with_labels = []
    without_labels = []
    for img in images:
        if image_has_labels(img, labels_dir):
            with_labels.append(img)
        else:
            without_labels.append(img)

    rng.shuffle(with_labels)
    rng.shuffle(without_labels)

    pool = list(with_labels[:n_images])
    if len(pool) < n_images:
        needed = n_images - len(pool)
        if not allow_empty and needed > 0:
            raise RuntimeError(f"Only {len(with_labels)} labeled images available, but N={n_images} requested and --allow-empty-images is False.")
        pool.extend(without_labels[:needed])

    if len(pool) < n_images:
        raise RuntimeError(f"Unable to build pool of size {n_images}. Found {len(pool)} images total.")

    return pool, with_labels


def split_pool(pool: List[Path], n_images: int, train_ratio: float, val_ratio: float, rng: random.Random) -> Dict[str, List[Path]]:
    if n_images <= 0:
        raise ValueError("n_images must be positive.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and sum to less than 1.")

    train_count = int(n_images * train_ratio)
    val_count = int(n_images * val_ratio)
    test_count = n_images - train_count - val_count

    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("Computed split counts must all be positive.")

    rng.shuffle(pool)
    train_split = pool[:train_count]
    val_split = pool[train_count : train_count + val_count]
    test_split = pool[train_count + val_count : n_images]

    assert len(train_split) == train_count
    assert len(val_split) == val_count
    assert len(test_split) == test_count

    return {"train": train_split, "val": val_split, "test_internal": test_split}


def write_list(path: Path, items: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join([p.stem for p in items])
    path.write_text(content, encoding="utf-8")


def checksum_from_names(names: List[str]) -> str:
    md5 = hashlib.md5()
    for name in names:
        md5.update(name.encode("utf-8"))
    return md5.hexdigest()


def _basic_self_check() -> None:
    assert checksum_from_names(["a", "b"]) == checksum_from_names(["a", "b"])
    assert "jpg" not in ALLOWED_EXTENSIONS  # simple guard that extensions are lower-case dotted.


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    _basic_self_check()

    labels_dir = locate_labels_dir(args.yolo_images_dir, args.labels_dir)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    rng = random.Random(args.seed)
    images = list_images(args.yolo_images_dir)
    pool, labeled_images = choose_pool(images, labels_dir, args.n_images, args.allow_empty_images, rng)
    splits = split_pool(pool.copy(), args.n_images, args.train_ratio, args.val_ratio, rng)

    args.out_split_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.out_split_dir / f"bdd_pool_{args.n_images}.txt"
    write_list(pool_path, pool)
    write_list(args.out_split_dir / "bdd_train.txt", splits["train"])
    write_list(args.out_split_dir / "bdd_val.txt", splits["val"])
    write_list(args.out_split_dir / "bdd_test_internal.txt", splits["test_internal"])

    manifest = {
        "seed": args.seed,
        "n_images": args.n_images,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "allow_empty_images": args.allow_empty_images,
        "pool_count": len(pool),
        "labeled_available": len(labeled_images),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_internal_count": len(splits["test_internal"]),
        "checksum": checksum_from_names([p.stem for p in pool]),
    }

    manifest_path = args.out_split_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Pool and splits created at %s", args.out_split_dir)
    logging.info("Counts: train=%d val=%d test_internal=%d", len(splits["train"]), len(splits["val"]), len(splits["test_internal"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Split creation failed: %s", exc)
        sys.exit(1)
