#!/usr/bin/env python3
"""Shared labels and formatting helpers for thesis analysis outputs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
QUAL_DIR = OUT_DIR / "qualitative"

REGIME_ORDER = ["real_only", "synthetic_only", "hybrid_70_30", "hybrid_50_50", "hybrid_30_70"]
REGIME_LABELS = {
    "real_only": "Real-only",
    "synthetic_only": "Synthetic-only",
    "hybrid_70_30": "Hybrid 70/30",
    "hybrid_50_50": "Hybrid 50/50",
    "hybrid_30_70": "Hybrid 30/70",
}
ARCH_ORDER = ["YOLOv8m", "FasterRCNN"]
ARCH_LABELS = {
    "YOLOv8m": "YOLOv8m",
    "FasterRCNN": "Faster R-CNN",
}
SCENARIO_LABELS = {
    "ADD": "ADD",
    "IS": "IS",
}
DATASET_LABELS = {
    "BDD": "ADD Real Test Set",
    "KITTI": "KITTI",
    "SYNTHIA": "SYNTHIA",
    "IS_real": "IS Real Test Set",
    "IS_synth": "IS Synthetic Test Set",
}
METRIC_ORDER = ["AP_50_95", "AP_50", "AP_75", "AR_1", "AR_10", "AR_100"]
METRIC_LABELS = {
    "AP_50_95": r"AP$_{50:95}$",
    "AP_50": r"AP$_{50}$",
    "AP_75": r"AP$_{75}$",
    "AR_1": r"AR$_{1}$",
    "AR_10": r"AR$_{10}$",
    "AR_100": r"AR$_{100}$",
}


def scenario_title(scenario: str, dataset: str, external: bool = False) -> str:
    if external:
        return "External Evaluation on KITTI"
    if scenario == "ADD":
        return "Internal Evaluation on the ADD Real Test Set"
    return "Internal Evaluation on the IS Real Test Set"


def pretty_arch(architecture: str) -> str:
    return ARCH_LABELS.get(architecture, architecture)


def pretty_regime(regime: str) -> str:
    return REGIME_LABELS.get(regime, regime)


def pretty_metric(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)
