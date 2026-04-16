#!/usr/bin/env python3
"""
Train Faster R-CNN (ResNet-50 FPN, ImageNet-pretrained) for experiment E6 (two-stage, real-only).

This version trains on the same YOLO materialized splits used by the one-stage experiments:
- images and labels from bdd_yolo_splits (train/val)
- split lists from bdd_yolo/splits (bdd_train.txt, bdd_val.txt)

Classes: car (id=1), pedestrian (id=2), cyclist (id=3)
Images are resized to 640x640 to mirror the YOLO input size.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
YOLO_ID_TO_COCO_ID = {0: 1, 1: 2, 2: 3}
SCENARIO_ROOT = Path(__file__).resolve().parents[2]
AD_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "ad"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Faster R-CNN (E6 real-only) on YOLO splits")
    p.add_argument(
        "--yolo-split-root",
        type=Path,
        default=AD_DATA_ROOT / "bdd_yolo_splits",
        help="Materialized YOLO splits root (images/labels train/val/test_internal)",
    )
    p.add_argument(
        "--split-dir",
        type=Path,
        default=AD_DATA_ROOT / "bdd_yolo" / "splits",
        help="Directory containing bdd_train.txt and bdd_val.txt",
    )
    p.add_argument("--run-root", type=Path, required=True, help="Output root for runs")
    p.add_argument("--run-name", type=str, default="E6_frcnn_real_seed0", help="Run name")
    # COCO-style 2× schedule: 24 epochs (default)
    p.add_argument("--epochs", type=int, default=24, help="Number of epochs (default: 24 = 2× schedule)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 avoids multiprocessing hangs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu (default: cuda)")
    p.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Optional wall-clock budget (hours). Training stops after this time (default: unlimited).",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split_stems(split_dir: Path, name: str) -> List[str]:
    path = split_dir / f"bdd_{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    stems = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return stems


class YoloSplitDataset(Dataset):
    def __init__(self, split_root: Path, split_name: str, stems: List[str], resize_to: int = 640) -> None:
        self.images_dir = split_root / "images" / split_name
        self.labels_dir = split_root / "labels" / split_name
        self.stems = stems
        self.resize = transforms.Resize((resize_to, resize_to))
        self.to_tensor = transforms.ToTensor()
        self.resize_to = resize_to

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]
        img_path = None
        for ext in ALLOWED_EXTS:
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.is_file():
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for stem {stem} in {self.images_dir}")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        img = self.resize(img)
        img_tensor = self.to_tensor(img)
        sx = self.resize_to / w
        sy = self.resize_to / h

        label_path = self.labels_dir / f"{stem}.txt"
        boxes = []
        labels = []
        if label_path.is_file():
            for ln in label_path.read_text().splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if cls not in YOLO_ID_TO_COCO_ID:
                    continue
                xc, yc, bw, bh = map(float, parts[1:])
                # convert from normalized xywh to absolute xyxy using original size
                x = (xc * w) - (bw * w) / 2.0
                y = (yc * h) - (bh * h) / 2.0
                bw_abs = bw * w
                bh_abs = bh * h
                # scale to resized image coordinates
                boxes.append([x * sx, y * sy, (x + bw_abs) * sx, (y + bh_abs) * sy])
                labels.append(YOLO_ID_TO_COCO_ID[cls])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
        }
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Please enable GPU or set --device cpu.")
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    train_stems = load_split_stems(args.split_dir, "train")
    val_stems = load_split_stems(args.split_dir, "val")

    train_ds = YoloSplitDataset(args.yolo_split_root, "train", train_stems)
    val_ds = YoloSplitDataset(args.yolo_split_root, "val", val_stems)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Train batches per epoch: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

    # Load COCO-pretrained backbone/head, then replace the predictor for 3 classes (+background)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=4)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # StepLR in the original recipe: drop LR every 5 epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    run_dir = args.run_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_trace: List[Dict] = []

    max_seconds = args.max_hours * 3600 if args.max_hours else None
    wall_start = time.time()

    for epoch in range(args.epochs):
        if max_seconds is not None and (time.time() - wall_start) >= max_seconds:
            print(
                f"Reached max_hours={args.max_hours:.2f} (elapsed {(time.time()-wall_start)/3600:.2f}h); stopping.",
                flush=True,
            )
            break
        epoch_start = time.time()
        model.train()
        loss_sum = 0.0
        for it, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device) for img in images]
            t_targets = []
            for tgt in targets:
                boxes = tgt["boxes"].to(device)
                labels = tgt["labels"].to(device)
                t_targets.append({"boxes": boxes, "labels": labels})

            loss_dict = model(images, t_targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            loss_sum += losses.item()
            if it == 1 or it % 50 == 0 or it == len(train_loader):
                print(f"Epoch {epoch+1} iter {it}/{len(train_loader)} loss {losses.item():.4f}", flush=True)

        lr_scheduler.step()

        # quick val loss (use train-mode forward to get loss dict, but keep grads off)
        was_training = model.training
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                t_targets = []
                for tgt in targets:
                    boxes = tgt["boxes"].to(device)
                    labels = tgt["labels"].to(device)
                    t_targets.append({"boxes": boxes, "labels": labels})
                loss_dict = model(images, t_targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        if not was_training:
            model.eval()

        epoch_rec = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, len(val_loader)),
            "lr": lr_scheduler.get_last_lr()[0],
            "time_sec": time.time() - epoch_start,
        }
        metrics_trace.append(epoch_rec)
        print(
            f"Epoch {epoch_rec['epoch']:03d}/{args.epochs} | "
            f"train_loss={epoch_rec['train_loss']:.4f} | "
            f"val_loss={epoch_rec['val_loss']:.4f} | "
            f"lr={epoch_rec['lr']:.6f} | "
            f"time={epoch_rec['time_sec']:.1f}s",
            flush=True,
        )

        # save last weights every epoch
        torch.save(model.state_dict(), run_dir / "last.pt")

    torch.save(model.state_dict(), run_dir / "final.pt")
    (run_dir / "last_metrics.json").write_text(str(metrics_trace), encoding="utf-8")


if __name__ == "__main__":
    main()
