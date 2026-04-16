#!/usr/bin/env python3
"""
Prepare a YOLO-style subset from Virtual KITTI 2 with cars/pedestrians/cyclists.

Assumptions about VKITTI2 layout (after extracting the official tarballs):
  vkitti_2.0.3_rgb/SceneXX/clone/frames/rgb/Camera_0/00000.png   (RGB)
  vkitti_2.0.3_instanceSegmentation/SceneXX/clone/frames/instanceSegmentation/Camera_0/00000.png (instance masks)

Instance mask encoding (from VKITTI2 docs):
  value = class_id * 1000 + instance_id
Class IDs (VKITTI2):
  0 background, 1 car, 2 truck, 3 van, 4 misc, 5 person, 6 sitting person, 7 cyclist, 8 tram

We map to YOLO classes:
  car: {1,2,3,4,8}
  pedestrian: {5,6}
  cyclist: {7}

Outputs:
  images/{train,val,test}  (symlinks by default)
  labels/{train,val,test}  (YOLO txt)
  splits/vkitti2_{train,val,test}.txt (stems)
  test_internal symlinks to test for YOLOv8 compatibility
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image

YOLO_CLASSES = ["car", "pedestrian", "cyclist"]

CAR_IDS = {1, 2, 3, 4, 8}
PED_IDS = {5, 6}
CYCLIST_IDS = {7}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a YOLO subset from VKITTI2.")
    p.add_argument("--vkitti2-root", type=Path, required=True, help="Root containing vkitti_2.0.3_rgb and instanceSegmentation.")
    p.add_argument("--out-root", type=Path, required=True, help="Output root for YOLO splits.")
    p.add_argument("--subset-size", type=int, default=9000, help="Total frames to sample (default 9k).")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlink.")
    return p.parse_args()


def find_rgb_frames(rgb_root: Path) -> List[Path]:
    patterns = [
        "Scene*/**/frames/rgb/Camera_0/*.jpg",
        "vkitti_2.0.3_rgb/Scene*/**/frames/rgb/Camera_0/*.jpg",
        "vkitti_2.0.3_rgb/Scene*/**/rgb/Camera_0/*.png",
        "Scene*/**/rgb/Camera_0/*.png",
    ]
    frames: List[Path] = []
    for pat in patterns:
        frames.extend(rgb_root.glob(pat))
    return sorted(set(frames))


def mask_path_for_rgb(rgb_path: Path, inst_root: Path) -> Path:
    parts = list(rgb_path.parts)
    if "vkitti_2.0.3_rgb" in parts:
        parts[parts.index("vkitti_2.0.3_rgb")] = "vkitti_2.0.3_instanceSegmentation"
    # replace frames/rgb -> frames/instanceSegmentation and filename prefix
    for i, p in enumerate(parts):
        if p == "rgb":
            parts[i] = "instanceSegmentation"
    stem = Path(parts[-1]).stem
    parts[-1] = f"instancegt_{stem.split('_')[-1]}.png"
    return Path(*parts)


def mask_to_boxes(mask: np.ndarray) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    VKITTI2 instanceSegmentation PNGs are palette images with values equal to class IDs (no instance ID).
    We extract connected components per class to get approximate instance boxes.
    """
    boxes: Dict[int, List[Tuple[float, float, float, float]]] = {0: [], 1: [], 2: []}
    h, w = mask.shape
    def add_boxes_for_ids(id_set, cls_id):
        cls_mask = np.isin(mask, list(id_set)).astype(np.uint8)
        if cls_mask.max() == 0:
            return
        num, labels = cv2.connectedComponents(cls_mask, connectivity=8)
        for lab in range(1, num):
            ys, xs = np.where(labels == lab)
            if ys.size == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            boxes[cls_id].append((x1, y1, x2, y2))

    add_boxes_for_ids(CAR_IDS, 0)
    add_boxes_for_ids(PED_IDS, 1)
    add_boxes_for_ids(CYCLIST_IDS, 2)
    return boxes


def xyxy_to_yolo(box, w, h):
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    rgb_root = args.vkitti2_root
    inst_root = args.vkitti2_root

    rgb_frames = find_rgb_frames(rgb_root)
    if not rgb_frames:
        raise SystemExit("No VKITTI2 RGB frames found. Check --vkitti2-root and extraction.")

    random.shuffle(rgb_frames)
    rgb_frames = rgb_frames[: args.subset_size]

    n_train = int(len(rgb_frames) * args.train_ratio)
    n_val = int(len(rgb_frames) * args.val_ratio)
    splits = {
        "train": rgb_frames[:n_train],
        "val": rgb_frames[n_train : n_train + n_val],
        "test": rgb_frames[n_train + n_val :],
    }

    out_img = args.out_root / "images"
    out_lbl = args.out_root / "labels"
    out_split = args.out_root / "splits"
    for split in splits:
        (out_img / split).mkdir(parents=True, exist_ok=True)
        (out_lbl / split).mkdir(parents=True, exist_ok=True)
    out_split.mkdir(parents=True, exist_ok=True)

    for split, frames in splits.items():
        stems = []
        for rgb_path in frames:
            stem = "_".join(rgb_path.with_suffix("").parts[-4:])  # scene_variant_cam_frame
            stems.append(stem)
            dest_img = out_img / split / f"{stem}.png"
            if dest_img.exists():
                dest_img.unlink()
            if args.copy:
                dest_img.write_bytes(rgb_path.read_bytes())
            else:
                dest_img.symlink_to(rgb_path.resolve())

            # load mask -> boxes
            mask_path = mask_path_for_rgb(rgb_path, inst_root)
            if not mask_path.is_file():
                continue
            mask = np.array(Image.open(mask_path).convert("I"))
            boxes = mask_to_boxes(mask)
            lines = []
            h, w = mask.shape
            for cls_id, blist in boxes.items():
                for b in blist:
                    xc, yc, bw, bh = xyxy_to_yolo(b, w, h)
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (out_lbl / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        (out_split / f"vkitti2_{split}.txt").write_text("\n".join(stems), encoding="utf-8")

    # YOLOv8 alias
    for parent in (out_img, out_lbl):
        test_dir = parent / "test"
        alias = parent / "test_internal"
        if alias.exists():
            continue
        if test_dir.exists():
            alias.symlink_to(test_dir, target_is_directory=True)

    print("Done VKITTI2 subset.")
    print(f"Images root: {out_img}")
    print(f"Labels root: {out_lbl}")


if __name__ == "__main__":
    main()
