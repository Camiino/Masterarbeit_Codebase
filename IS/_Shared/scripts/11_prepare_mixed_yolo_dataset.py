#!/usr/bin/env python3
"""Create a mixed real/synthetic IS YOLO dataset from prepared split roots."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

ALLOWED_EXTS = (".jpg", ".jpeg", ".png")
CLASS_NAMES = ["hex", "hex_socket", "phillips", "pozidriv", "slotted", "torx"]
SCENARIO_ROOT = Path(__file__).resolve().parents[2]
IS_DATA_ROOT = SCENARIO_ROOT / "_Shared" / "data" / "is"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a mixed real/synthetic IS YOLO dataset.")
    p.add_argument(
        "--real-root",
        type=Path,
        default=IS_DATA_ROOT / "real_yolo_splits",
        help="Prepared real IS YOLO split root.",
    )
    p.add_argument(
        "--synth-root",
        type=Path,
        default=IS_DATA_ROOT / "synthetic_yolo_splits",
        help="Prepared synthetic IS YOLO split root.",
    )
    p.add_argument("--real-pct", type=int, required=True, help="Percentage of real data in train/val.")
    p.add_argument("--synth-pct", type=int, required=True, help="Percentage of synthetic data in train/val.")
    p.add_argument("--out-root", type=Path, help="Optional output root.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    p.add_argument("--train-total", type=int, help="Override total train count.")
    p.add_argument("--val-total", type=int, help="Override total val count.")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    return p.parse_args()


def load_stems(split_root: Path, split_name: str) -> List[str]:
    image_dir = split_root / "images" / split_name
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {image_dir}")
    stems = [p.stem for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
    if not stems:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return stems


def find_image(split_root: Path, split_name: str, stem: str) -> Path:
    for ext in ALLOWED_EXTS:
        candidate = split_root / "images" / split_name / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for stem {stem} in {split_root}/images/{split_name}")


def find_label(split_root: Path, split_name: str, stem: str) -> Path:
    label = split_root / "labels" / split_name / f"{stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Label not found for stem {stem} in {split_root}/labels/{split_name}")
    return label


def safe_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def compute_counts(total: int, real_pct: int, synth_pct: int) -> Tuple[int, int]:
    denom = real_pct + synth_pct
    if denom <= 0:
        raise ValueError("real_pct + synth_pct must be positive.")
    real_count = round(total * real_pct / denom)
    return real_count, total - real_count


def sample_split(
    real_stems: List[str],
    synth_stems: List[str],
    total: int,
    real_pct: int,
    synth_pct: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    real_count, synth_count = compute_counts(total, real_pct, synth_pct)
    if real_count > len(real_stems):
        raise ValueError(f"Requested {real_count} real samples but only {len(real_stems)} available.")
    if synth_count > len(synth_stems):
        raise ValueError(f"Requested {synth_count} synthetic samples but only {len(synth_stems)} available.")

    mixed = [("real", stem) for stem in rng.sample(real_stems, real_count)]
    mixed.extend(("synth", stem) for stem in rng.sample(synth_stems, synth_count))
    rng.shuffle(mixed)
    return mixed


def write_yaml(out_root: Path, slug: str) -> Path:
    yaml_path = out_root / f"{slug}.yaml"
    yaml_text = (
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test_internal\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES!r}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    if args.real_pct < 0 or args.synth_pct < 0:
        raise ValueError("Percentages must be non-negative.")

    slug = f"mixed_real{args.real_pct}_synth{args.synth_pct}_seed{args.seed}"
    out_root = (args.out_root or (IS_DATA_ROOT / slug)).resolve()
    rng = random.Random(args.seed)

    real_train = load_stems(args.real_root, "train")
    real_val = load_stems(args.real_root, "val")
    synth_train = load_stems(args.synth_root, "train")
    synth_val = load_stems(args.synth_root, "val")

    train_total = args.train_total or min(len(real_train), len(synth_train))
    val_total = args.val_total or min(len(real_val), len(synth_val))

    train_mix = sample_split(real_train, synth_train, train_total, args.real_pct, args.synth_pct, rng)
    val_mix = sample_split(real_val, synth_val, val_total, args.real_pct, args.synth_pct, rng)

    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name in ["train", "val", "test_internal"]:
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(parents=True, exist_ok=True)

    def materialize(split_name: str, items: List[Tuple[str, str]]) -> List[str]:
        out_stems: List[str] = []
        for source_name, stem in items:
            source_root = args.real_root if source_name == "real" else args.synth_root
            img = find_image(source_root, split_name, stem)
            lbl = find_label(source_root, split_name, stem)
            out_stem = f"{source_name}__{stem}"
            safe_link_or_copy(img, out_root / "images" / split_name / f"{out_stem}{img.suffix}", args.copy)
            safe_link_or_copy(lbl, out_root / "labels" / split_name / f"{out_stem}.txt", args.copy)
            out_stems.append(out_stem)
        return out_stems

    train_out = materialize("train", train_mix)
    val_out = materialize("val", val_mix)
    (out_root / "splits" / "bdd_train.txt").write_text("\n".join(train_out), encoding="utf-8")
    (out_root / "splits" / "bdd_val.txt").write_text("\n".join(val_out), encoding="utf-8")
    (out_root / "splits" / "bdd_test_internal.txt").write_text("", encoding="utf-8")

    yaml_path = write_yaml(out_root, slug)
    manifest = {
        "seed": args.seed,
        "real_pct": args.real_pct,
        "synth_pct": args.synth_pct,
        "train_total": train_total,
        "val_total": val_total,
        "train_real": sum(1 for source, _ in train_mix if source == "real"),
        "train_synth": sum(1 for source, _ in train_mix if source == "synth"),
        "val_real": sum(1 for source, _ in val_mix if source == "real"),
        "val_synth": sum(1 for source, _ in val_mix if source == "synth"),
        "yaml": str(yaml_path),
    }
    (out_root / "mix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
