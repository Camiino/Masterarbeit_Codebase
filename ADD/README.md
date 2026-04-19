# ADD Scenario: Autonomous Driving Detection

ADD evaluates synthetic-to-real transfer for autonomous-driving object detection.
The real target domain is BDD100K-derived data; the synthetic domain is SYNTHIA.
The external benchmark is KITTI.

## Label Space

All ADD experiments use three detector classes:

```text
0 car
1 pedestrian
2 cyclist
```

Mappings are intentionally grouped to keep BDD, SYNTHIA, and KITTI comparable:

| Source | Mapped to `car` | Mapped to `pedestrian` | Mapped to `cyclist` | Ignored |
| --- | --- | --- | --- | --- |
| BDD | `car`, `truck`, `bus` | `person` | `bike`, `rider` | all other labels |
| SYNTHIA | `car`, `truck`, `bus` | `pedestrian` | `bicycle`, `motorcycle`, `rider` | all other labels |
| KITTI | `Car`, `Van`, `Truck`, `Tram` | `Pedestrian`, `Person_sitting` | `Cyclist` | `Misc`, `DontCare` |

## Data Roots

```text
_Shared/data/ad/bdd_yolo_splits/          # prepared real BDD split
_Shared/data/ad/synthia_yolo_splits/      # prepared synthetic SYNTHIA split
_Shared/data/ad/mixed_bdd*_synth*_seed0/  # prepared mixed datasets
_Shared/data/ad/kitti_yolo_splits/        # prepared KITTI external split
_Shared/data/ad/kitti_raw/                # extracted KITTI source data
_Shared/downloads/KITTI/                  # KITTI zip archives
```

Current KITTI external split:

```text
images: 7481
car objects: 33261
pedestrian objects: 4709
cyclist objects: 1627
ignored: Misc, DontCare
```

## Experiment Outputs

```text
One-Stage/Real/project/                   # YOLO real-only runs and metrics
One-Stage/Synthetic/project/              # YOLO SYNTHIA-only runs and metrics
One-Stage/Mixed/project/                  # YOLO mixed runs and metrics
One-Stage/External/project/metrics_kitti/ # YOLO KITTI metrics

Two-Stage/Real/project/                   # Faster R-CNN real-only runs and metrics
Two-Stage/Synthetic/project/              # Faster R-CNN SYNTHIA-only runs and metrics
Two-Stage/Mixed/project/                  # Faster R-CNN mixed runs and metrics
Two-Stage/External/project/metrics_kitti/ # Faster R-CNN KITTI metrics

External/project/kitti_summary.csv
External/project/kitti_summary.md
External/project/best_seed_checkpoints.json
```

The repository-level thesis analysis pipeline aggregates ADD outputs into:

```text
../analysis/outputs/figures/add_internal_real_regime_comparison.{png,pdf}
../analysis/outputs/figures/add_kitti_external_comparison.{png,pdf}
../analysis/outputs/figures/class_heatmap_add_real_internal.{png,pdf}
../analysis/outputs/figures/scale_delta_comparison.{png,pdf}
../analysis/outputs/figures/synthetic_to_real_gap.{png,pdf}
../analysis/outputs/figures/architecture_gap_summary.{png,pdf}
../analysis/outputs/figures/seed_stability_internal_ap50_95.{png,pdf}

../analysis/outputs/tables/table_add_internal.tex
../analysis/outputs/tables/table_add_kitti.tex
../analysis/outputs/tables/table_class_ap_add.tex
../analysis/outputs/tables/table_effect_deltas.tex
../analysis/outputs/tables/table_scale_delta_comparison.tex
../analysis/outputs/tables/table_seed_stability_internal.tex
```

Supporting markdown summaries are generated at:

```text
../analysis/outputs/RESULTS_INVENTORY.md
../analysis/outputs/THESIS_RESULTS_SUMMARY.md
../analysis/outputs/qualitative/QUALITATIVE_STATUS.md
```

## Main Commands

Run from `ADD/`.

Real-only BDD:

```bash
_Shared/scripts/run_real_all.sh cuda
```

Synthetic-only SYNTHIA:

```bash
_Shared/scripts/run_synthetic_all.sh cuda
```

Mixed real/synthetic:

```bash
_Shared/scripts/run_mixed_all.sh cuda 70 30 0 4
_Shared/scripts/run_mixed_all.sh cuda 50 50 0 4
_Shared/scripts/run_mixed_all.sh cuda 30 70 0 4
```

KITTI external evaluation using the best seed per setup:

```bash
_Shared/scripts/run_kitti_external_best_seed_eval.sh cuda
```

The legacy script names remain available for reproducibility:

```text
run_budgeted.sh           -> real-only
run_budgeted_synthia.sh   -> synthetic-only
run_budgeted_mixed.sh     -> mixed
run_external_kitti_eval.sh
run_kitti_external_all_overnight.sh
```

## Methodological Notes

- KITTI external evaluation intentionally uses the labeled KITTI training set, because official KITTI test labels are withheld.
- External evaluation is run only for selected best-seed checkpoints, chosen by BDD real-target `AP_50_95` with `AP_50` as the tie-breaker.
- The KITTI summary table is produced by `_Shared/scripts/13_summarize_kitti_external.py`.
- The current KITTI trend is scientifically plausible: synthetic-only transfers poorly; mixed training can improve or preserve external generalization, especially for Faster R-CNN.
- Thesis-ready qualitative overlay examples are not currently committed, because the repository excludes the underlying images and the ignored `gt_*.json` / `preds_*.json` evaluation dumps required to reconstruct them deterministically.
