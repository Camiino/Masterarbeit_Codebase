#!/usr/bin/env python3
"""
Train YOLOv8m for Autonomous Driving Experiment E1 (real-only) on the
materialized YOLO split dataset created by your 03_materialize script.

This script:
- Verifies dataset structure (images/labels for train/val/test_internal)
- Verifies the dataset YAML exists
- Trains YOLOv8m (pretrained) with a specified seed
- Optionally runs evaluation on the internal test split after training

Example:
    python project/scripts/04_train_yolov8m_e1.py \
    --data-yaml project/data/ad/bdd_yolo_splits/ad_bdd_E1_seed0.yaml \
    --run-root project/runs_ad \
    --name E1_yolov8m_real_seed0 \
    --seed 0 \
    --imgsz 640 \
    --epochs 100 \
    --batch 32 \
    --device 0 \
    --workers 12 \
    --do-test


Notes on "correct backbone":
- yolov8m.pt selects the YOLOv8m architecture variant (fixed backbone/neck/head).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8m for experiment E1 (Autonomous Driving, real-only).")
    p.add_argument("--data-yaml", type=Path, required=True, help="Path to dataset YAML produced by materialize step.")
    p.add_argument("--model", type=str, default="yolov8m.pt", help="Ultralytics model/weights (default: yolov8m.pt).")
    p.add_argument("--run-root", type=Path, default=Path("project/runs_ad"), help="Where to store runs (project=...).")
    p.add_argument("--name", type=str, default="E1_yolov8m_real", help="Run name (name=...).")
    p.add_argument("--seed", type=int, default=0, help="Training seed.")
    p.add_argument("--imgsz", type=int, default=640, help="Image size (default 640).")
    p.add_argument("--epochs", type=int, default=100, help="Epochs (default 100).")
    p.add_argument("--batch", type=int, default=16, help="Batch size (default 16). Reduce if OOM.")
    p.add_argument("--device", type=str, default="0", help="Device, e.g. '0' or 'cpu' (default 0).")
    p.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    p.add_argument("--cache", action="store_true", help="Enable caching (default off).")
    p.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights (default True).")
    p.add_argument("--deterministic", action="store_true", default=True, help="Enable deterministic behavior where possible.")
    p.add_argument("--do-test", action="store_true", help="After training, run 'val' on split=test.")
    p.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _require_dir(p: Path, what: str) -> None:
    if not p.is_dir():
        raise FileNotFoundError(f"Missing {what} directory: {p}")


def _require_file(p: Path, what: str) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing {what} file: {p}")


def verify_materialized_dataset(data_yaml: Path) -> Path:
    """
    The YAML produced by your materialize script contains:
      path: <ABSOLUTE OUT_ROOT>
      train: images/train
      val: images/val
      test: images/test_internal

    We read the first line to extract the absolute root.
    This avoids having to re-parse YAML with dependencies.
    """
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    path_line = next((ln for ln in lines if ln.strip().startswith("path:")), None)
    if not path_line:
        raise ValueError(f"YAML {data_yaml} does not contain a 'path:' entry.")
    root_str = path_line.split(":", 1)[1].strip()
    root = Path(root_str)

    _require_dir(root, "dataset root")
    _require_dir(root / "images" / "train", "images/train")
    _require_dir(root / "images" / "val", "images/val")
    _require_dir(root / "images" / "test_internal", "images/test_internal")
    _require_dir(root / "labels" / "train", "labels/train")
    _require_dir(root / "labels" / "val", "labels/val")
    _require_dir(root / "labels" / "test_internal", "labels/test_internal")

    # Quick sanity counts (not strict)
    n_train = len(list((root / "images" / "train").glob("*.jpg")))
    n_val = len(list((root / "images" / "val").glob("*.jpg")))
    n_test = len(list((root / "images" / "test_internal").glob("*.jpg")))
    logging.info("Dataset root: %s", root)
    logging.info("Image counts (jpg only): train=%d val=%d test_internal=%d", n_train, n_val, n_test)
    if n_train == 0 or n_val == 0:
        logging.warning("Train/val appear empty (jpg count=0). If images are .png/.jpeg, this is ok.")

    return root


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    _require_file(args.data_yaml, "data yaml")
    dataset_root = verify_materialized_dataset(args.data_yaml)

    # Import ultralytics lazily so the script can fail early on path issues.
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required. Install with:\n"
            "  pip install ultralytics\n"
            "and ensure 'yolo' CLI works."
        ) from exc

    # Log model info to confirm YOLOv8m variant
    logging.info("Loading model: %s", args.model)
    model = YOLO(args.model)
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        logging.info("Model loaded. Approx parameter count: %d", n_params)
    except Exception:
        logging.info("Model loaded.")

    # Train
    logging.info("Starting training: %s", args.name)
    results = model.train(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.run_root),
        name=args.name,
        pretrained=bool(args.pretrained),
        deterministic=bool(args.deterministic),
        cache=bool(args.cache),
    )

    # Find best.pt (Ultralytics writes to project/name/weights/best.pt)
    best_pt = args.run_root / args.name / "weights" / "best.pt"
    if best_pt.is_file():
        logging.info("Best weights: %s", best_pt)
    else:
        logging.warning("Could not find best.pt at %s (check Ultralytics output).", best_pt)

    # Optional test evaluation on internal test split
    if args.do_test:
        if not best_pt.is_file():
            logging.warning("Skipping test eval because best.pt was not found.")
        else:
            logging.info("Running evaluation on split=test (test_internal)...")
            test_model = YOLO(str(best_pt))
            test_model.val(
                data=str(args.data_yaml),
                imgsz=args.imgsz,
                split="test",
                device=args.device,
            )

    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Training script failed: %s", exc)
        sys.exit(1)
