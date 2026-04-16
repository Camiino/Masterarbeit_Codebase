#!/usr/bin/env python3
"""
Evaluate a trained Faster R-CNN checkpoint on a COCO-formatted dataset (internal test split).

Inputs
- --weights: path to .pt state_dict saved from 06_train_frcnn_e6.py
- --data-root: COCO root with images/<split>/ and annotations/bdd_<split>_mapped.json
- --split: which split name to use (default: val)
- --out: output metrics JSON path

Outputs
- metrics JSON with overall AP/AR and per-class AP via pycocotools COCOeval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from pycocotools.coco import COCO  # type: ignore
from pycocotools.cocoeval import COCOeval  # type: ignore
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

SCENARIO_ROOT = Path(__file__).resolve().parents[2]
IS_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "is"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval Faster R-CNN on COCO split")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split", type=str, default="val", help="COCO split name (val) or 'test_internal_yolo'")
    p.add_argument(
        "--yolo-split-root",
        type=Path,
        default=IS_DATA_ROOT / "real_yolo_splits",
        help="Root of materialized YOLO splits (used when split=test_internal_yolo)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.5)
    return p.parse_args()


def build_dataset(root: Path, split: str) -> CocoDetection:
    ann = root / "annotations" / f"bdd_{split}_mapped.json"
    # file_name fields include split prefixes; point to images root.
    imgs = root / "images"
    to_tensor = transforms.ToTensor()

    def coco_transform(img, target):
        return to_tensor(img), target

    return CocoDetection(imgs, ann, transforms=coco_transform)


# --- YOLO split helpers for test_internal ---

ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
YOLO_ID_TO_COCO_ID = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
CATEGORY_ID_TO_NAME = {
    1: "hex",
    2: "hex_socket",
    3: "phillips",
    4: "pozidriv",
    5: "slotted",
    6: "torx",
}


def list_images(images_dir: Path) -> List[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in ALLOWED_EXTS and p.is_file()]


def read_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def yolo_label_to_xywh_abs(line: str, w: int, h: int) -> Tuple[int, float, float, float, float]:
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


def build_coco_gt_from_yolo_split(split_root: Path) -> Path:
    images_dir = split_root / "images" / "test_internal"
    labels_dir = split_root / "labels" / "test_internal"
    out_gt = split_root / "annotations_test_internal_gt.json"

    images = []
    annotations = []
    ann_id = 1
    categories = [{"id": cid, "name": name} for cid, name in CATEGORY_ID_TO_NAME.items()]

    imgs = list_images(images_dir)
    for idx, img_path in enumerate(imgs, start=1):
        w, h = read_image_size(img_path)
        images.append({"id": idx, "file_name": img_path.name, "width": w, "height": h})
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            continue
        lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            cls, x, y, bw, bh = yolo_label_to_xywh_abs(ln, w, h)
            if cls not in YOLO_ID_TO_COCO_ID:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": YOLO_ID_TO_COCO_ID[cls],
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco_gt = {"images": images, "annotations": annotations, "categories": categories}
    out_gt.write_text(json.dumps(coco_gt, indent=2), encoding="utf-8")
    return out_gt


def coco_eval_on_split(model, dataset: CocoDetection, device: torch.device, conf: float) -> Dict:
    model.eval()
    coco_gt = dataset.coco
    results: List[Dict] = []
    with torch.no_grad():
        for img, target in dataset:
            img = img.to(device)
            outputs = model([img])[0]
            boxes = outputs["boxes"].cpu()
            scores = outputs["scores"].cpu()
            labels = outputs["labels"].cpu()

            if isinstance(target, dict):
                image_id = target["image_id"]
            elif target:
                image_id = target[0]["image_id"]
            else:
                raise ValueError("Dataset returned an empty target without image_id metadata.")
            image_id = int(image_id)
            for box, score, label in zip(boxes, scores, labels):
                if score.item() < conf:
                    continue
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    }
                )

    if not results:
        raise RuntimeError("No detections produced; try lowering --conf")

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "AP_50_95": float(coco_eval.stats[0]),
        "AP_50": float(coco_eval.stats[1]),
        "AP_75": float(coco_eval.stats[2]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
    }

    per_class = {}
    for cid, cat in coco_gt.cats.items():
        coco_eval_cat = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval_cat.params.catIds = [cid]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        coco_eval_cat.summarize()
        per_class[cat["name"]] = {
            "AP_50_95": float(coco_eval_cat.stats[0]),
            "AP_50": float(coco_eval_cat.stats[1]),
            "AP_75": float(coco_eval_cat.stats[2]),
        }

    return {"all": stats, "per_class": per_class}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Build dataset / GT depending on split choice
    if args.split == "test_internal_yolo":
        gt_path = build_coco_gt_from_yolo_split(args.yolo_split_root)
        # build a dummy CocoDetection to reuse coco_eval
        coco_gt = COCO(str(gt_path))
        img_root = args.yolo_split_root / "images" / "test_internal"
        to_tensor = transforms.ToTensor()

        class YoloSplitDataset(CocoDetection):
            def __init__(self, coco_gt, root, tfm):
                self.coco = coco_gt
                self.ids = list(sorted(self.coco.imgs.keys()))
                self.root = root
                self.tfm = tfm

            def __getitem__(self, index):
                img_id = self.ids[index]
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                path = Path(self.coco.loadImgs(img_id)[0]["file_name"])
                img = Image.open(self.root / path).convert("RGB")
                img, _ = self.tfm(img, anns)
                target = {"image_id": img_id, "annotations": anns}
                return img, target

            def __len__(self):
                return len(self.ids)

        dataset = YoloSplitDataset(coco_gt, img_root, lambda img, tgt: (to_tensor(img), tgt))
    else:
        if args.data_root is None:
            raise ValueError("--data-root is required unless --split test_internal_yolo is used.")
        dataset = build_dataset(args.data_root, args.split)

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=7)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    metrics = coco_eval_on_split(model, dataset, device, args.conf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
