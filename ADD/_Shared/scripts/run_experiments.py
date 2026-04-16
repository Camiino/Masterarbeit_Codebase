#!/usr/bin/env python3
"""
Unified entry point (CLI UI) to run the full pipeline for one-stage (YOLO) and two-stage
(Faster R-CNN) experiments. Actions cover data prep, split creation, training, and eval.

Supported actions
- extract-bdd       : extract downloaded BDD100K tarball
- sanity-bdd        : run label sanity checks
- convert-bdd       : convert BDD to YOLO + mapped COCO
- make-splits       : build pooled subset + train/val/test_internal lists
- materialize       : materialize YOLO split folders + dataset YAML
- yolo-train        : train YOLOv8m real-only (E1) for a seed
- yolo-eval         : eval YOLO model on internal test split for a seed
- frcnn-train       : train Faster R-CNN real-only (E6) for a seed
- frcnn-eval        : eval Faster R-CNN on COCO val for a seed

Examples
  python run_experiments.py --action make-splits --seed 0 --n-images 9000
  python run_experiments.py --action materialize --seed 0
  python run_experiments.py --action yolo-train --seed 1
  python run_experiments.py --action yolo-eval  --seed 1
  python run_experiments.py --action frcnn-train --seed 1
  python run_experiments.py --action frcnn-eval  --seed 1
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scenario root, e.g. ADD/

# Shared roots
SHARED_SCRIPTS = ROOT / "_Shared" / "scripts"
SHARED_DATA = ROOT / "_Shared" / "data"
DOWNLOADS = ROOT / "_Shared" / "downloads"

# Common data locations (shared)
BDD_RAW = SHARED_DATA / "ad" / "bdd100k_raw" / "bdd100k"
BDD_YOLO_ROOT = SHARED_DATA / "ad" / "bdd_yolo"
BDD_COCO_ROOT = SHARED_DATA / "ad" / "bdd_coco"
BDD_YOLO_SPLITS_ROOT = SHARED_DATA / "ad" / "bdd_yolo_splits"
BDD_YAML_DEFAULT = BDD_YOLO_SPLITS_ROOT / "ad_bdd_E1_seed0.yaml"

# Virtual KITTI paths (synthetic)
VKITTI_RGB_TAR = DOWNLOADS / "vkitti_1.3.1_rgb.tar"
VKITTI_MOT_TAR = DOWNLOADS / "vkitti_1.3.1_motgt.tar.gz"
VKITTI_ROOT = SHARED_DATA / "ad" / "vkitti"
VKITTI_YOLO_SPLITS_ROOT = SHARED_DATA / "ad" / "vkitti_yolo_splits"
VKITTI_YAML = VKITTI_YOLO_SPLITS_ROOT / "vkitti_det.yaml"

# Virtual KITTI 2 (synthetic with pedestrians/cyclists)
VKITTI2_ROOT = SHARED_DATA / "ad" / "vkitti2"
VKITTI2_YOLO_SPLITS_ROOT = SHARED_DATA / "ad" / "vkitti2_yolo_splits"
VKITTI2_YAML = VKITTI2_YOLO_SPLITS_ROOT / "vkitti2_det.yaml"
VKITTI2_RGB_TAR = DOWNLOADS / "vkitti_2.0.3_rgb.tar"
VKITTI2_INST_TAR = DOWNLOADS / "vkitti_2.0.3_instanceSegmentation.tar"
VKITTI2_CLASS_TAR = DOWNLOADS / "vkitti_2.0.3_classSegmentation.tar"
VKITTI2_TEXT_TAR = DOWNLOADS / "vkitti_2.0.3_textgt.tar.gz"

# SYNTHIA RAND CITYSCAPES (synthetic)
SYNTHIA_RAND_CITYSCAPES_ROOT = SHARED_DATA / "ad" / "synthia_rand_cityscapes" / "RAND_CITYSCAPES"
SYNTHIA_YOLO_SPLITS_ROOT = SHARED_DATA / "ad" / "synthia_yolo_splits"
SYNTHIA_YAML = SYNTHIA_YOLO_SPLITS_ROOT / "synthia_det.yaml"

# One-stage project (keeps its own runs/metrics)
YOLO_PROJECT = ROOT / "One-Stage" / "Real" / "project"
YOLO_RUN_ROOT = YOLO_PROJECT / "runs_ad"
YOLO_METRICS_CROSSDOMAIN = YOLO_PROJECT / "metrics_synthia_transfer"
YOLO_VKITTI_RUN_ROOT = YOLO_PROJECT / "runs_vkitti"
# Synthetic YOLO project (new)
YOLO_VKITTI_PROJECT = ROOT / "One-Stage" / "Synthetic" / "project"
YOLO_VKITTI_RUN_ROOT = YOLO_VKITTI_PROJECT / "runs_vkitti"
YOLO_VKITTI_METRICS = YOLO_VKITTI_PROJECT / "metrics"
YOLO_VKITTI2_RUN_ROOT = YOLO_VKITTI_PROJECT / "runs_vkitti2"
YOLO_VKITTI2_METRICS = YOLO_VKITTI_PROJECT / "metrics_vkitti2"

# Two-stage project (keeps its own runs/metrics)
FRCNN_PROJECT = ROOT / "Two-Stage" / "Real" / "project"
FRCNN_RUN_ROOT = FRCNN_PROJECT / "runs_ad_frcnn"
FRCNN_METRICS_CROSSDOMAIN = FRCNN_PROJECT / "metrics_synthia_transfer"
FRCNN_VKITTI_PROJECT = ROOT / "Two-Stage" / "Synthetic" / "project"
FRCNN_VKITTI_RUN_ROOT = FRCNN_VKITTI_PROJECT / "runs_vkitti_frcnn"
YOLO_SYNTHIA_PROJECT = ROOT / "One-Stage" / "Synthetic" / "project"
YOLO_SYNTHIA_RUN_ROOT = YOLO_SYNTHIA_PROJECT / "runs_synthia"
YOLO_SYNTHIA_METRICS = YOLO_SYNTHIA_PROJECT / "metrics_synthia"
YOLO_SYNTHIA_METRICS_INDOMAIN = YOLO_SYNTHIA_PROJECT / "metrics_synthia_indomain"
FRCNN_SYNTHIA_PROJECT = ROOT / "Two-Stage" / "Synthetic" / "project"
FRCNN_SYNTHIA_RUN_ROOT = FRCNN_SYNTHIA_PROJECT / "runs_synthia_frcnn"


def run_cmd(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_vkitti(dataset_root: Path | None = None) -> int:
    """
    Extract Virtual KITTI RGB and MOT ground-truth tarballs from _Shared/downloads into dataset_root.
    """
    import tarfile

    dataset_root = dataset_root or VKITTI_ROOT
    dataset_root.mkdir(parents=True, exist_ok=True)

    def _extract(tar_path: Path):
        if not tar_path.is_file():
            print(f"Missing archive: {tar_path}")
            return 1
        print(f"Extracting {tar_path} -> {dataset_root}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=dataset_root)
        return 0

    rc_rgb = _extract(VKITTI_RGB_TAR)
    rc_mot = _extract(VKITTI_MOT_TAR)
    return max(rc_rgb, rc_mot)


def prepare_vkitti_subset(dataset_root: Path | None, subset_size: int, seed: int) -> int:
    """
    Build YOLO-style 20k subset and splits from extracted VKITTI.
    """
    dataset_root = dataset_root or VKITTI_ROOT
    out_root = VKITTI_YOLO_SPLITS_ROOT
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "prepare_vkitti_subset.py"),
        "--vkitti-root",
        str(dataset_root),
        "--out-root",
        str(out_root),
        "--subset-size",
        str(subset_size),
        "--seed",
        str(seed),
    ]
    return run_cmd(cmd)


def extract_vkitti2(dataset_root: Path | None = None) -> int:
    """
    Extract VKITTI2 RGB + instance/class/text tarballs.
    """
    import tarfile

    dataset_root = dataset_root or VKITTI2_ROOT
    dataset_root.mkdir(parents=True, exist_ok=True)

    def _extract(tar_path: Path):
        if not tar_path.is_file():
            print(f"Missing archive: {tar_path}")
            return 1
        print(f"Extracting {tar_path} -> {dataset_root}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(path=dataset_root)
        return 0

    rc = 0
    for t in [VKITTI2_RGB_TAR, VKITTI2_INST_TAR, VKITTI2_CLASS_TAR, VKITTI2_TEXT_TAR]:
        rc = max(rc, _extract(t))
    return rc


def prepare_vkitti2_subset(dataset_root: Path | None, subset_size: int, seed: int) -> int:
    dataset_root = dataset_root or VKITTI2_ROOT
    out_root = VKITTI2_YOLO_SPLITS_ROOT
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "09_prepare_vkitti2_subset.py"),
        "--vkitti2-root",
        str(dataset_root),
        "--out-root",
        str(out_root),
        "--subset-size",
        str(subset_size),
        "--seed",
        str(seed),
    ]
    return run_cmd(cmd)


def prepare_synthia_subset(dataset_root: Path | None, subset_size: int, seed: int) -> int:
    dataset_root = dataset_root or SYNTHIA_RAND_CITYSCAPES_ROOT
    out_root = SYNTHIA_YOLO_SPLITS_ROOT
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "10_prepare_synthia_rand_cityscapes.py"),
        "--synthia-root",
        str(dataset_root),
        "--out-root",
        str(out_root),
        "--subset-size",
        str(subset_size),
        "--seed",
        str(seed),
    ]
    return run_cmd(cmd)


def make_vkitti2_yaml(dataset_root: Path | None = None, out_yaml: Path | None = None) -> int:
    split_root = dataset_root or VKITTI2_YOLO_SPLITS_ROOT
    out = out_yaml or (split_root / "vkitti2_det.yaml")
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "08_make_vkitti_yaml.py"),  # reuse writer
        "--split-root",
        str(split_root),
        "--out-yaml",
        str(out),
    ]
    return run_cmd(cmd)


def make_synthia_yaml(dataset_root: Path | None = None, out_yaml: Path | None = None) -> int:
    split_root = dataset_root or SYNTHIA_YOLO_SPLITS_ROOT
    out = out_yaml or (split_root / "synthia_det.yaml")
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "08_make_vkitti_yaml.py"),
        "--split-root",
        str(split_root),
        "--out-yaml",
        str(out),
    ]
    return run_cmd(cmd)


def make_vkitti_yaml(dataset_root: Path | None = None, out_yaml: Path | None = None) -> int:
    """
    Write a YOLO dataset YAML for the prepared VKITTI subset.
    """
    split_root = dataset_root or VKITTI_YOLO_SPLITS_ROOT
    out = out_yaml or (split_root / "vkitti_det.yaml")
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "08_make_vkitti_yaml.py"),
        "--split-root",
        str(split_root),
        "--out-yaml",
        str(out),
    ]
    return run_cmd(cmd)


def ensure_vkitti_bdd_style_splits() -> Path:
    """
    06_train_frcnn_e6.py expects split files named bdd_train/val/test.
    Mirror VKITTI splits into that naming under vkitti_yolo_splits/splits_bddstyle.
    """
    src = VKITTI_YOLO_SPLITS_ROOT / "splits"
    dst = VKITTI_YOLO_SPLITS_ROOT / "splits_bddstyle"
    dst.mkdir(parents=True, exist_ok=True)
    mapping = {
        "bdd_train.txt": "vkitti_train.txt",
        "bdd_val.txt": "vkitti_val.txt",
        "bdd_test_internal.txt": "vkitti_test.txt",
    }
    for out_name, src_name in mapping.items():
        src_file = src / src_name
        if not src_file.is_file():
            raise FileNotFoundError(f"Missing split file: {src_file}")
        dst_file = dst / out_name
        dst_file.write_text(src_file.read_text(), encoding="utf-8")
    return dst


def extract_bdd() -> int:
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "00_extract_bdd_tar.py"),
        "--out-dir",
        str(BDD_RAW),
    ]
    return run_cmd(cmd)


def sanity_bdd() -> int:
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "00_sanity_checks_bdd.py"),
        "--images-root",
        str(BDD_RAW),
        "--ann-root",
        str(BDD_RAW),
        "--output-report",
        str(BDD_RAW / "bdd_sanity_report.json"),
    ]
    return run_cmd(cmd)


def convert_bdd() -> int:
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "01_convert_bdd_to_yolo_and_coco.py"),
        "--images-root",
        str(BDD_RAW),
        "--ann-root",
        str(BDD_RAW),
        "--out-yolo-root",
        str(BDD_YOLO_ROOT),
        "--out-coco-root",
        str(BDD_COCO_ROOT),
    ]
    return run_cmd(cmd)


def make_splits(seed: int, n_images: int) -> int:
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "02_make_subset_and_splits.py"),
        "--yolo-images-dir",
        str(BDD_YOLO_ROOT / "images" / "all"),
        "--seed",
        str(seed),
        "--n-images",
        str(n_images),
        "--out-split-dir",
        str(BDD_YOLO_ROOT / "splits"),
    ]
    return run_cmd(cmd)


def materialize(seed: int) -> int:
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "03_materialize_yolo_split_folders.py"),
        "--yolo-root",
        str(BDD_YOLO_ROOT),
        "--split-dir",
        str(BDD_YOLO_ROOT / "splits"),
        "--out-yolo-split-root",
        str(BDD_YOLO_SPLITS_ROOT),
        "--seed",
        str(seed),
    ]
    return run_cmd(cmd)


def yolo_train(seed: int, device: str) -> int:
    name = f"E1_yolov8m_real_seed{seed}"
    # Use the fixed split YAML (seed0) to keep data constant across seeds.
    data_yaml = BDD_YAML_DEFAULT
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "04_train_yolov8m_e1.py"),
        "--data-yaml",
        str(data_yaml),
        "--run-root",
        str(YOLO_RUN_ROOT),
        "--name",
        name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_eval(seed: int, device: str) -> int:
    name = f"E1_yolov8m_real_seed{seed}"
    weights = YOLO_RUN_ROOT / name / "weights" / "best.pt"
    out_dir = YOLO_PROJECT / "metrics" / f"E1_seed{seed}_yolov8m_eval"
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "05_eval_internal_test_coco_custom_scale.py"),
        "--model",
        str(weights),
        "--dataset-root",
        str(BDD_YOLO_SPLITS_ROOT),
        "--split-name",
        "test_internal",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_eval_synthia(seed: int, device: str) -> int:
    name = f"E1_yolov8m_real_seed{seed}"
    weights = YOLO_RUN_ROOT / name / "weights" / "best.pt"
    out_dir = YOLO_METRICS_CROSSDOMAIN / f"E1_seed{seed}_yolov8m_on_synthia_eval"
    ensure_dir(out_dir)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "05_eval_internal_test_coco_custom_scale.py"),
        "--model",
        str(weights),
        "--dataset-root",
        str(SYNTHIA_YOLO_SPLITS_ROOT),
        "--split-name",
        "test",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_vkitti_train(seed: int, device: str) -> int:
    name = f"VKITTI_yolov8m_seed{seed}"
    data_yaml = VKITTI_YAML
    ensure_dir(YOLO_VKITTI_RUN_ROOT)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "04_train_yolov8m_e1.py"),
        "--data-yaml",
        str(data_yaml),
        "--run-root",
        str(YOLO_VKITTI_RUN_ROOT),
        "--name",
        name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_vkitti_eval(seed: int, device: str) -> int:
    name = f"VKITTI_yolov8m_seed{seed}"
    weights = YOLO_VKITTI_RUN_ROOT / name / "weights" / "best.pt"
    out_dir = YOLO_VKITTI_METRICS / f"VKITTI_seed{seed}_yolov8m_eval"
    ensure_dir(out_dir)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "05_eval_internal_test_coco_custom_scale.py"),
        "--model",
        str(weights),
        "--dataset-root",
        str(VKITTI_YOLO_SPLITS_ROOT),
        "--split-name",
        "test",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_vkitti2_train(seed: int, device: str) -> int:
    name = f"VKITTI2_yolov8m_seed{seed}"
    data_yaml = VKITTI2_YAML
    ensure_dir(YOLO_VKITTI2_RUN_ROOT)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "04_train_yolov8m_e1.py"),
        "--data-yaml",
        str(data_yaml),
        "--run-root",
        str(YOLO_VKITTI2_RUN_ROOT),
        "--name",
        name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_vkitti2_eval(seed: int, device: str) -> int:
    name = f"VKITTI2_yolov8m_seed{seed}"
    weights = YOLO_VKITTI2_RUN_ROOT / name / "weights" / "best.pt"
    out_dir = YOLO_VKITTI2_METRICS / f"VKITTI2_seed{seed}_yolov8m_eval"
    ensure_dir(out_dir)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "05_eval_internal_test_coco_custom_scale.py"),
        "--model",
        str(weights),
        "--dataset-root",
        str(VKITTI2_YOLO_SPLITS_ROOT),
        "--split-name",
        "test",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def frcnn_train(seed: int, device: str, max_hours: float | None = None) -> int:
    run_name = f"E6_frcnn_real_seed{seed}"
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "06_train_frcnn_e6.py"),
        "--yolo-split-root",
        str(BDD_YOLO_SPLITS_ROOT),
        "--split-dir",
        str(BDD_YOLO_ROOT / "splits"),
        "--run-root",
        str(FRCNN_RUN_ROOT),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if max_hours is not None:
        cmd += ["--max-hours", str(max_hours)]
    return run_cmd(cmd)


def frcnn_vkitti_train(seed: int, device: str) -> int:
    run_name = f"VKITTI_frcnn_seed{seed}"
    split_dir = ensure_vkitti_bdd_style_splits()
    ensure_dir(FRCNN_VKITTI_RUN_ROOT)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "06_train_frcnn_e6.py"),
        "--yolo-split-root",
        str(VKITTI_YOLO_SPLITS_ROOT),
        "--split-dir",
        str(split_dir),
        "--run-root",
        str(FRCNN_VKITTI_RUN_ROOT),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_synthia_train(seed: int, device: str) -> int:
    name = f"SYNTHIA_yolov8m_seed{seed}"
    ensure_dir(YOLO_SYNTHIA_RUN_ROOT)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "04_train_yolov8m_e1.py"),
        "--data-yaml",
        str(SYNTHIA_YAML),
        "--run-root",
        str(YOLO_SYNTHIA_RUN_ROOT),
        "--name",
        name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_synthia_eval(seed: int, device: str) -> int:
    name = f"SYNTHIA_yolov8m_seed{seed}"
    weights = YOLO_SYNTHIA_RUN_ROOT / name / "weights" / "best.pt"
    out_dir = YOLO_SYNTHIA_METRICS / f"SYNTHIA_seed{seed}_yolov8m_on_bdd_eval"
    ensure_dir(out_dir)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "05_eval_internal_test_coco_custom_scale.py"),
        "--model",
        str(weights),
        "--dataset-root",
        str(BDD_YOLO_SPLITS_ROOT),
        "--split-name",
        "test_internal",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def yolo_synthia_eval_indomain(seed: int, device: str) -> int:
    name = f"SYNTHIA_yolov8m_seed{seed}"
    weights = YOLO_SYNTHIA_RUN_ROOT / name / "weights" / "best.pt"
    out_dir = YOLO_SYNTHIA_METRICS_INDOMAIN / f"SYNTHIA_seed{seed}_yolov8m_eval"
    ensure_dir(out_dir)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "05_eval_internal_test_coco_custom_scale.py"),
        "--model",
        str(weights),
        "--dataset-root",
        str(SYNTHIA_YOLO_SPLITS_ROOT),
        "--split-name",
        "test",
        "--out-dir",
        str(out_dir),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def frcnn_synthia_train(seed: int, device: str, max_hours: float | None = None) -> int:
    run_name = f"SYNTHIA_frcnn_seed{seed}"
    ensure_dir(FRCNN_SYNTHIA_RUN_ROOT)
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "06_train_frcnn_e6.py"),
        "--yolo-split-root",
        str(SYNTHIA_YOLO_SPLITS_ROOT),
        "--split-dir",
        str(SYNTHIA_YOLO_SPLITS_ROOT / "splits"),
        "--run-root",
        str(FRCNN_SYNTHIA_RUN_ROOT),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if max_hours is not None:
        cmd += ["--max-hours", str(max_hours)]
    return run_cmd(cmd)


def frcnn_synthia_eval(seed: int, device: str) -> int:
    run_name = f"SYNTHIA_frcnn_seed{seed}"
    weights = FRCNN_SYNTHIA_RUN_ROOT / run_name / "final.pt"
    out = FRCNN_SYNTHIA_PROJECT / "metrics" / f"SYNTHIA_seed{seed}_frcnn_on_bdd_eval.json"
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "07_eval_frcnn_coco.py"),
        "--weights",
        str(weights),
        "--data-root",
        str(BDD_COCO_ROOT),
        "--split",
        "test_internal_yolo",
        "--yolo-split-root",
        str(BDD_YOLO_SPLITS_ROOT),
        "--out",
        str(out),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def frcnn_synthia_eval_indomain(seed: int, device: str) -> int:
    run_name = f"SYNTHIA_frcnn_seed{seed}"
    weights = FRCNN_SYNTHIA_RUN_ROOT / run_name / "final.pt"
    out = FRCNN_SYNTHIA_PROJECT / "metrics_indomain" / f"SYNTHIA_seed{seed}_frcnn_eval.json"
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "07_eval_frcnn_coco.py"),
        "--weights",
        str(weights),
        "--data-root",
        str(BDD_COCO_ROOT),
        "--split",
        "test_internal_yolo",
        "--yolo-split-root",
        str(SYNTHIA_YOLO_SPLITS_ROOT),
        "--out",
        str(out),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def frcnn_eval(seed: int, device: str) -> int:
    run_name = f"E6_frcnn_real_seed{seed}"
    weights = FRCNN_RUN_ROOT / run_name / "final.pt"
    out = FRCNN_PROJECT / "metrics" / f"E6_seed{seed}_frcnn_eval.json"
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "07_eval_frcnn_coco.py"),
        "--weights",
        str(weights),
        "--data-root",
        str(BDD_COCO_ROOT),
        "--split",
        "test_internal_yolo",
        "--yolo-split-root",
        str(BDD_YOLO_SPLITS_ROOT),
        "--out",
        str(out),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def frcnn_eval_synthia(seed: int, device: str) -> int:
    run_name = f"E6_frcnn_real_seed{seed}"
    weights = FRCNN_RUN_ROOT / run_name / "final.pt"
    out = FRCNN_METRICS_CROSSDOMAIN / f"E6_seed{seed}_frcnn_on_synthia_eval.json"
    cmd = [
        "python",
        str(SHARED_SCRIPTS / "07_eval_frcnn_coco.py"),
        "--weights",
        str(weights),
        "--data-root",
        str(BDD_COCO_ROOT),
        "--split",
        "test_internal_yolo",
        "--yolo-split-root",
        str(SYNTHIA_YOLO_SPLITS_ROOT),
        "--out",
        str(out),
        "--device",
        device,
    ]
    return run_cmd(cmd)


def main() -> None:
    actions = {
        "vkitti-extract": lambda seed, dataset_root: extract_vkitti(dataset_root),
        "vkitti-prepare": lambda seed, dataset_root, subset_size: prepare_vkitti_subset(dataset_root, subset_size, seed),
        "vkitti-yaml": lambda seed, dataset_root, out_yaml: make_vkitti_yaml(dataset_root, out_yaml),
        "vkitti-yolo-train": yolo_vkitti_train,
        "vkitti-yolo-eval": yolo_vkitti_eval,
        "vkitti-frcnn-train": frcnn_vkitti_train,
        "vkitti2-extract": lambda seed, dataset_root: extract_vkitti2(dataset_root),
        "vkitti2-prepare": lambda seed, dataset_root, subset_size: prepare_vkitti2_subset(dataset_root, subset_size, seed),
        "vkitti2-yaml": lambda seed, dataset_root, out_yaml: make_vkitti2_yaml(dataset_root, out_yaml),
        "vkitti2-yolo-train": yolo_vkitti2_train,
        "vkitti2-yolo-eval": yolo_vkitti2_eval,
        "synthia-prepare": lambda seed, dataset_root, subset_size: prepare_synthia_subset(dataset_root, subset_size, seed),
        "synthia-yaml": lambda seed, dataset_root, out_yaml: make_synthia_yaml(dataset_root, out_yaml),
        "synthia-yolo-train": yolo_synthia_train,
        "synthia-yolo-eval": yolo_synthia_eval,
        "synthia-yolo-eval-indomain": yolo_synthia_eval_indomain,
        "synthia-frcnn-train": frcnn_synthia_train,
        "synthia-frcnn-eval": frcnn_synthia_eval,
        "synthia-frcnn-eval-indomain": frcnn_synthia_eval_indomain,
        "extract-bdd": lambda seed: extract_bdd(),
        "sanity-bdd": lambda seed: sanity_bdd(),
        "convert-bdd": lambda seed: convert_bdd(),
        "make-splits": lambda seed, n_images: make_splits(seed, n_images),
        "materialize": lambda seed: materialize(seed),
        "yolo-train": yolo_train,
        "yolo-eval": yolo_eval,
        "yolo-eval-synthia": yolo_eval_synthia,
        "frcnn-train": frcnn_train,
        "frcnn-eval": frcnn_eval,
        "frcnn-eval-synthia": frcnn_eval_synthia,
    }

    ap = argparse.ArgumentParser(description="Unified experiment runner")
    ap.add_argument("--action", choices=actions.keys(), required=True)
    ap.add_argument("--seed", type=int, default=0, help="Random seed (used by make-splits/materialize/train/eval)")
    ap.add_argument("--n-images", type=int, default=9000, help="Pool size for make-splits")
    ap.add_argument("--subset-size", type=int, default=9000, help="Subset size for synthetic prepares")
    ap.add_argument("--dataset-root", type=Path, help="Optional dataset root override (VKITTI actions)")
    ap.add_argument("--out-yaml", type=Path, help="Optional YAML output override (vkitti-yaml)")
    ap.add_argument("--device", type=str, default="cuda", help="Device for train/eval (cuda or cpu)")
    ap.add_argument("--max-hours", type=float, default=None, help="Optional wall-clock budget for frcnn-train only.")
    args = ap.parse_args()

    if args.action == "make-splits":
        rc = actions[args.action](args.seed, args.n_images)
    elif args.action == "vkitti-prepare":
        rc = actions[args.action](args.seed, args.dataset_root, args.subset_size)
    elif args.action == "vkitti-extract":
        rc = actions[args.action](args.seed, args.dataset_root)
    elif args.action == "vkitti2-prepare":
        rc = actions[args.action](args.seed, args.dataset_root, args.subset_size)
    elif args.action == "vkitti2-extract":
        rc = actions[args.action](args.seed, args.dataset_root)
    elif args.action == "vkitti-yaml":
        rc = actions[args.action](args.seed, args.dataset_root, args.out_yaml)
    elif args.action == "vkitti2-yaml":
        rc = actions[args.action](args.seed, args.dataset_root, args.out_yaml)
    elif args.action == "synthia-prepare":
        rc = actions[args.action](args.seed, args.dataset_root, args.subset_size)
    elif args.action == "synthia-yaml":
        rc = actions[args.action](args.seed, args.dataset_root, args.out_yaml)
    elif args.action in {
        "yolo-train",
        "yolo-eval",
        "yolo-eval-synthia",
        "frcnn-train",
        "frcnn-eval",
        "frcnn-eval-synthia",
        "vkitti-yolo-train",
        "vkitti-yolo-eval",
        "vkitti-frcnn-train",
        "synthia-yolo-train",
        "synthia-yolo-eval",
        "synthia-yolo-eval-indomain",
        "synthia-frcnn-train",
        "synthia-frcnn-eval",
        "synthia-frcnn-eval-indomain",
    }:
        if args.action in {"frcnn-train", "synthia-frcnn-train"}:
            rc = actions[args.action](args.seed, args.device, args.max_hours)
        else:
            rc = actions[args.action](args.seed, args.device)
    elif args.action in {"vkitti2-yolo-train", "vkitti2-yolo-eval"}:
        rc = actions[args.action](args.seed, args.device)
    else:
        rc = actions[args.action](args.seed)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
