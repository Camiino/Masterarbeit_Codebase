#!/usr/bin/env python3
"""
COCO-style evaluation for one or two real-world test sets (e.g., internal split
and external KITTI) with a unified protocol:

- Overall COCO metrics: AP@[0.50:0.95], AP50 (plus AR breakdown)
- Per-class AP: car, pedestrian, cyclist
- No bbox-area thresholds or custom scale filters; scale is analyzed at the
  scenario level outside this script.

Assumptions:
- Materialized YOLO dataset root has images/labels for the requested split
  (defaults: images/test_internal, labels/test_internal).
- YOLO label ids: 0 car, 1 pedestrian, 2 cyclist.

Example:
  python project/scripts/05_eval_internal_test_coco_custom_scale.py \\
    --model project/runs_ad/E1_yolov8m_real_seed0/weights/best.pt \\
    --dataset-root project/data/ad/bdd_yolo_splits \\
    --out-dir project/metrics/E1_seed0_yolov8m \\
    --extra-dataset-root project/data/ad/kitti_yolo_splits \\
    --extra-name kitti

Outputs (per dataset, under out_dir/<name>/):
  - gt_<split>.json
  - preds_<split>.json
  - metrics.json (all, per_class)
  - console summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore

# Ultralytics
from ultralytics import YOLO  # type: ignore


CATEGORY_ID_TO_NAME = {1: "car", 2: "pedestrian", 3: "cyclist"}  # COCO category ids we will use
YOLO_ID_TO_COCO_ID = {0: 1, 1: 2, 2: 3}  # yolo label ids -> coco category ids
ALLOWED_EXTS = (".jpg", ".jpeg", ".png")


@dataclass
class ImgInfo:
    id: int
    file_name: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True, help="Path to trained weights (.pt).")
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Primary dataset root (default internal test)."
    )
    p.add_argument(
        "--extra-dataset-root",
        type=Path,
        help="Optional second dataset root (e.g., KITTI) evaluated with the same split logic.",
    )
    p.add_argument("--extra-name", type=str, default="kitti", help="Name for the extra dataset block (default: kitti).")
    p.add_argument("--split-name", type=str, default="test_internal", help="Split folder name under images/ and labels/ (default: test_internal).")
    p.add_argument(
        "--extra-split-name",
        type=str,
        default="test",
        help="Split folder name for the extra dataset (default: test).",
    )
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default 640).")
    p.add_argument("--device", type=str, default="0", help="Device, e.g. '0' or 'cpu'.")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for predictions (keep low for COCO eval).")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold for predictions.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for GT/preds/metrics.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def list_test_images(images_dir: Path) -> List[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in ALLOWED_EXTS and p.is_file()]
    if not imgs:
        raise FileNotFoundError(f"No test images found in: {images_dir}")
    return imgs


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return int(w), int(h)


def yolo_label_to_xywh_abs(line: str, w: int, h: int) -> Tuple[int, float, float, float, float]:
    """
    YOLO label format: cls x_center y_center width height (all normalized)
    Returns: (cls_id, x, y, bw, bh) in absolute pixels, where (x,y) is top-left.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Bad YOLO label line: {line}")
    cls = int(float(parts[0]))
    xc = float(parts[1]) * w
    yc = float(parts[2]) * h
    bw = float(parts[3]) * w
    bh = float(parts[4]) * h
    x = xc - bw / 2.0
    y = yc - bh / 2.0
    return cls, x, y, bw, bh


def build_coco_gt_from_yolo(dataset_root: Path, split_name: str, out_gt: Path) -> Dict:
    """
    Build COCO GT JSON from YOLO labels in labels/test_internal corresponding to images/test_internal.
    """
    images_dir = dataset_root / "images" / split_name
    labels_dir = dataset_root / "labels" / split_name
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    test_images = list_test_images(images_dir)

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories = [{"id": cid, "name": name} for cid, name in CATEGORY_ID_TO_NAME.items()]
    ann_id = 1

    img_infos: Dict[str, ImgInfo] = {}
    for i, img_path in enumerate(test_images, start=1):
        w, h = read_image_size(img_path)
        file_name = img_path.name  # relative within split folder for our own tracking
        info = ImgInfo(id=i, file_name=file_name, width=w, height=h)
        img_infos[file_name] = info
        images.append({"id": i, "file_name": file_name, "width": w, "height": h})

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            # treat as empty labels
            continue
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            cls, x, y, bw, bh = yolo_label_to_xywh_abs(ln, w, h)
            if cls not in YOLO_ID_TO_COCO_ID:
                continue
            if bw <= 0 or bh <= 0:
                continue
            coco_cat = YOLO_ID_TO_COCO_ID[cls]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i,
                    "category_id": coco_cat,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    out_gt.parent.mkdir(parents=True, exist_ok=True)
    out_gt.write_text(json.dumps(coco_gt, indent=2), encoding="utf-8")
    logging.info("Wrote GT COCO to %s (images=%d, anns=%d)", out_gt, len(images), len(annotations))
    return coco_gt


def run_inference_to_coco_preds(
    model_path: Path,
    dataset_root: Path,
    split_name: str,
    out_preds: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
) -> List[Dict]:
    """
    Run Ultralytics inference on images/test_internal and export COCO detections JSON.
    COCO detections format: [{image_id, category_id, bbox[x,y,w,h], score}, ...]
    """
    images_dir = dataset_root / "images" / split_name
    test_images = list_test_images(images_dir)

    model = YOLO(str(model_path))

    # Build a stable mapping: filename -> image_id (we will use 1..N in sorted order)
    filename_to_id: Dict[str, int] = {p.name: idx for idx, p in enumerate(test_images, start=1)}

    preds: List[Dict] = []
    logging.info("Running inference on %d test images...", len(test_images))

    # Batch inference by passing the directory; results returned in filename order
    results = model.predict(
        source=str(images_dir),
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
        save=False,
    )

    for r in results:
        # Ultralytics returns a Results object per image
        img_path = Path(r.path)
        img_name = img_path.name
        image_id = filename_to_id.get(img_name)
        if image_id is None:
            continue

        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), score, cls in zip(boxes_xyxy, scores, cls_ids):
            if cls not in YOLO_ID_TO_COCO_ID:
                continue
            coco_cat = YOLO_ID_TO_COCO_ID[int(cls)]
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(coco_cat),
                    "bbox": [x, y, w, h],
                    "score": float(score),
                }
            )

    out_preds.parent.mkdir(parents=True, exist_ok=True)
    out_preds.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    logging.info("Wrote preds COCO detections to %s (dets=%d)", out_preds, len(preds))
    return preds


def coco_eval(
    coco_gt_path: Path,
    coco_dt_path: Path,
    cat_ids: List[int] | None = None,
) -> Dict[str, float]:
    """
    Run COCOeval for bbox, optionally restricted to specific categories.
    Returns COCOeval stats in a dict.
    """
    coco_gt = COCO(str(coco_gt_path))
    coco_dt = coco_gt.loadRes(str(coco_dt_path))

    coco_eval_obj = COCOeval(coco_gt, coco_dt, iouType="bbox")

    coco_eval_obj.params.imgIds = list(coco_gt.imgs.keys())
    coco_eval_obj.params.catIds = cat_ids or list(coco_gt.cats.keys())

    coco_eval_obj.evaluate()
    coco_eval_obj.accumulate()
    coco_eval_obj.summarize()

    # stats indices: 0=AP@[.50:.95], 1=AP@.50, 2=AP@.75, 3=AP_small, 4=AP_medium, 5=AP_large, ...
    # We report overall AP and AP50/AP75 plus ARs for completeness.
    return {
        "AP_50_95": float(coco_eval_obj.stats[0]),
        "AP_50": float(coco_eval_obj.stats[1]),
        "AP_75": float(coco_eval_obj.stats[2]),
        "AR_1": float(coco_eval_obj.stats[6]),
        "AR_10": float(coco_eval_obj.stats[7]),
        "AR_100": float(coco_eval_obj.stats[8]),
    }


def evaluate_dataset(
    name: str,
    dataset_root: Path,
    split_name: str,
    model_path: Path,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
    out_dir: Path,
) -> Dict:
    """Run GT build, inference, and metric breakdown for one dataset."""
    images_dir = dataset_root / "images" / split_name
    labels_dir = dataset_root / "labels" / split_name
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing: {labels_dir}")

    dataset_out = out_dir / name
    dataset_out.mkdir(parents=True, exist_ok=True)
    gt_path = dataset_out / f"gt_{split_name}.json"
    pred_path = dataset_out / f"preds_{split_name}.json"
    metrics_path = dataset_out / "metrics.json"

    build_coco_gt_from_yolo(dataset_root, split_name, gt_path)
    run_inference_to_coco_preds(
        model_path=model_path,
        dataset_root=dataset_root,
        split_name=split_name,
        out_preds=pred_path,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
    )

    logging.info("=== COCO evaluation: %s (ALL classes) ===", name)
    metrics_all = coco_eval(gt_path, pred_path)

    # Per-class AP
    per_class = {}
    for cid, cname in CATEGORY_ID_TO_NAME.items():
        logging.info("=== COCO evaluation: %s (class=%s) ===", name, cname)
        per_class[cname] = coco_eval(gt_path, pred_path, cat_ids=[cid])

    out = {
        "dataset": str(dataset_root.resolve()),
        "split": split_name,
        "all": metrics_all,
        "per_class": per_class,
    }

    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logging.info(
        "[%s] SUMMARY: mAP=%.4f | AP50=%.4f | AP_car=%.4f | AP_ped=%.4f | AP_cyc=%.4f",
        name,
        out["all"]["AP_50_95"],
        out["all"]["AP_50"],
        out["per_class"]["car"]["AP_50_95"],
        out["per_class"]["pedestrian"]["AP_50_95"],
        out["per_class"]["cyclist"]["AP_50_95"],
    )

    return {
        "name": name,
        "paths": {
            "gt": str(gt_path),
            "preds": str(pred_path),
            "metrics": str(metrics_path),
        },
        "results": out,
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    summaries.append(
        evaluate_dataset(
            name="internal",
            dataset_root=args.dataset_root,
            split_name=args.split_name,
            model_path=args.model,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            out_dir=args.out_dir,
        )
    )

    if args.extra_dataset_root:
        summaries.append(
            evaluate_dataset(
                name=args.extra_name,
                dataset_root=args.extra_dataset_root,
                split_name=args.extra_split_name,
                model_path=args.model,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                out_dir=args.out_dir,
            )
        )

    # Master summary file
    master = {
        "model": str(args.model),
        "imgsz": args.imgsz,
        "device": args.device,
        "datasets": summaries,
    }
    (args.out_dir / "metrics_all.json").write_text(json.dumps(master, indent=2), encoding="utf-8")
    logging.info("Wrote consolidated metrics to %s", args.out_dir / "metrics_all.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logging.error("Eval failed: %s", exc)
        sys.exit(1)
