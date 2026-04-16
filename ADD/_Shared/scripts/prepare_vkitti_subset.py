#!/usr/bin/env python3
"""
Prepare a YOLO-style detection subset from the Virtual KITTI 1.3.1 dataset.

Steps performed:
1) Scan extracted VKITTI RGB frames (Camera_0) and MOT ground-truth files.
2) Convert KITTI-format boxes to YOLO txt labels for classes {car, pedestrian, cyclist}.
   - Maps Car/Van/Truck -> car, Pedestrian/Person_sitting -> pedestrian, Cyclist -> cyclist.
   - Ignores other classes (Tram, Misc, DontCare, etc.).
3) Sample a subset of frames (default 9,000) and split into train/val/test (80/10/10).
4) Write images/{train,val,test} and labels/{train,val,test}; uses symlinks by default to avoid copying.

Expected input layout (after extracting vkitti_1.3.1_rgb.tar and vkitti_1.3.1_motgt.tar.gz):
<vkitti_root>/vkitti_1.3.1_rgb/Scene01/clone/frames/rgb/Camera_0/00000.png
<vkitti_root>/vkitti_1.3.1_motgt/Scene01/clone.txt

Usage example:
python _Shared/scripts/prepare_vkitti_subset.py \
  --vkitti-root /data/vkitti \
  --out-root _Shared/data/ad/vkitti_yolo_splits \
  --subset-size 9000 \
  --seed 1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

# YOLO class order expected elsewhere in this project
YOLO_CLASSES = ["car", "pedestrian", "cyclist"]
LABEL_MAP = {
    "Car": "car",
    "Van": "car",
    "Truck": "car",
    "Pedestrian": "pedestrian",
    "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a 20k-frame YOLO subset from Virtual KITTI 1.3.1.")
    p.add_argument("--vkitti-root", type=Path, required=True, help="Root containing vkitti_1.3.1_rgb and vkitti_1.3.1_motgt folders.")
    p.add_argument("--out-root", type=Path, required=True, help="Output root for YOLO-style splits.")
    p.add_argument("--subset-size", type=int, default=9000, help="Total frames to sample (default 9k).")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8).")
    p.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (default 0.1). Test gets the remainder.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking (default symlink).")
    return p.parse_args()


def load_mot_annotations(mot_file: Path) -> Dict[int, List[Tuple[str, float, float, float, float]]]:
    """Read KITTI-format MOT GT file and return per-frame bboxes."""
    annos: Dict[int, List[Tuple[str, float, float, float, float]]] = {}
    if not mot_file.is_file():
        return annos
    for line in mot_file.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        try:
            frame = int(parts[0])
        except ValueError:
            continue  # skip header or malformed lines
        label = parts[2]
        if label not in LABEL_MAP:
            continue
        l, t, r, b = map(float, parts[6:10])
        annos.setdefault(frame, []).append((LABEL_MAP[label], l, t, r, b))
    return annos


def convert_to_yolo(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    l, t, r, b = box
    x_c = (l + r) / 2.0 / w
    y_c = (t + b) / 2.0 / h
    bw = (r - l) / w
    bh = (b - t) / h
    return x_c, y_c, bw, bh


def collect_frames(rgb_root: Path, mot_root: Path) -> List[Dict]:
    """
    VKITTI 1.3.1 layout (as extracted from the official tarballs):
      vkitti_1.3.1_rgb/<scene>/<variant>/<frame>.png
      vkitti_1.3.1_motgt/<scene>_<variant>.txt
    where scene is e.g. 0001, variant is e.g. clone, morning, rain, 15-deg-right, etc.
    """
    frames: List[Dict] = []
    rgb_base = rgb_root / "vkitti_1.3.1_rgb"
    mot_base = mot_root / "vkitti_1.3.1_motgt"

    for scene_dir in sorted(p for p in rgb_base.iterdir() if p.is_dir()):
        scene = scene_dir.name
        for variant_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            variant = variant_dir.name
            mot_file = mot_base / f"{scene}_{variant}.txt"
            mot = load_mot_annotations(mot_file)
            for img_path in sorted(variant_dir.glob("*.png")):
                try:
                    frame_idx = int(img_path.stem)
                except ValueError:
                    continue
                frames.append(
                    {
                        "scene": scene,
                        "variant": variant,
                        "frame": frame_idx,
                        "img": img_path,
                        "boxes": mot.get(frame_idx, []),
                    }
                )
    return frames


def write_label(label_path: Path, boxes: List[Tuple[str, float, float, float, float]], w: int, h: int) -> None:
    lines = []
    for cls, l, t, r, b in boxes:
        cls_id = YOLO_CLASSES.index(cls)
        x_c, y_c, bw, bh = convert_to_yolo((l, t, r, b), w, h)
        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rgb_root = args.vkitti_root
    mot_root = args.vkitti_root

    frames = collect_frames(rgb_root, mot_root)
    if not frames:
        raise RuntimeError("No VKITTI frames found. Check --vkitti-root and extraction.")

    random.shuffle(frames)
    subset_size = min(args.subset_size, len(frames))
    frames = frames[:subset_size]

    n_train = int(subset_size * args.train_ratio)
    n_val = int(subset_size * args.val_ratio)
    splits = {
        "train": frames[:n_train],
        "val": frames[n_train : n_train + n_val],
        "test": frames[n_train + n_val :],
    }

    out_img = args.out_root / "images"
    out_lbl = args.out_root / "labels"
    split_dir = args.out_root / "splits"
    for split in splits:
        (out_img / split).mkdir(parents=True, exist_ok=True)
        (out_lbl / split).mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    for split, items in splits.items():
        split_list: List[str] = []
        for rec in items:
            img_path: Path = rec["img"]
            with Image.open(img_path) as im:
                w, h = im.size
            # unique stem to avoid collisions across scenes/variants
            stem = f"{rec['scene']}_{rec['variant']}_{rec['frame']:05d}"
            # write label
            label_path = out_lbl / split / f"{stem}.txt"
            write_label(label_path, rec["boxes"], w, h)
            # link or copy image
            dest_img = out_img / split / f"{stem}{img_path.suffix}"
            if dest_img.exists():
                dest_img.unlink()
            if args.copy:
                dest_img.write_bytes(img_path.read_bytes())
            else:
                dest_img.symlink_to(img_path.resolve())
            split_list.append(stem)
        # write split file listing stems (one per line)
        (split_dir / f"vkitti_{split}.txt").write_text("\n".join(split_list), encoding="utf-8")

    # Provide YOLOv8-friendly alias (test_internal) expected by some scripts
    for alias_parent in [out_img, out_lbl]:
        test_dir = alias_parent / "test"
        alias = alias_parent / "test_internal"
        if alias.exists():
            continue
        if not test_dir.exists():
            continue
        alias.symlink_to(test_dir, target_is_directory=True)

    print(f"Done. Total frames: {subset_size} -> train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
    print(f"Images root: {out_img}")
    print(f"Labels root: {out_lbl}")


if __name__ == "__main__":
    main()
