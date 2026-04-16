#!/usr/bin/env python3
"""
Create a YOLO dataset YAML for the prepared Virtual KITTI subset.

Defaults target the 20k split we built at:
  _Shared/data/ad/vkitti_yolo_splits/{images,labels}/{train,val,test}

Usage (defaults):
  python _Shared/scripts/08_make_vkitti_yaml.py

Custom paths:
  python _Shared/scripts/08_make_vkitti_yaml.py \
    --split-root /data/vkitti_yolo_splits \
    --out-yaml /data/vkitti_yolo_splits/vkitti_det.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SPLIT_ROOT = Path(__file__).resolve().parents[2] / "_Shared" / "data" / "ad" / "vkitti_yolo_splits"
DEFAULT_OUT = DEFAULT_SPLIT_ROOT / "vkitti_det.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write YOLO dataset YAML for VKITTI subset.")
    p.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT, help="Root containing images/ and labels/ splits.")
    p.add_argument("--out-yaml", type=Path, default=DEFAULT_OUT, help="Path to write YAML file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    split_root = args.split_root
    out_yaml = args.out_yaml

    # Basic sanity check
    required_dirs = [
        split_root / "images" / "train",
        split_root / "images" / "val",
        split_root / "images" / "test",
        split_root / "labels" / "train",
        split_root / "labels" / "val",
        split_root / "labels" / "test",
    ]
    missing = [d for d in required_dirs if not d.is_dir()]
    if missing:
        missing_str = "\n".join(str(m) for m in missing)
        raise SystemExit(f"Missing expected split folders:\n{missing_str}")

    yaml_text = "\n".join(
        [
            f"path: {split_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: car",
            "  1: pedestrian",
            "  2: cyclist",
            "",
        ]
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {out_yaml}")


if __name__ == "__main__":
    main()
