# Synthetic-to-Real Object Detection Experiments

This workspace contains two scenario-level studies of synthetic data for object detection.
The repository is organized by scenario first, then by detector family and training data
recipe. This keeps the autonomous-driving and industrial-screw experiments comparable
without mixing their data, class mappings, or evaluation assumptions.

## Scenarios

| Scenario | Folder | Real data | Synthetic data | Classes |
| --- | --- | --- | --- | --- |
| Autonomous Driving Detection | `ADD/` | BDD100K-derived detection subset | SYNTHIA RAND CITYSCAPES | `car`, `pedestrian`, `cyclist` |
| Industrial Screws | `IS/` | Roboflow screw-head images | Roboflow synthetic screw-head images | `hex`, `hex_socket`, `phillips`, `pozidriv`, `slotted`, `torx` |

## High-Level Layout

```text
analysis/
  collect_metrics.py       # normalize ADD/IS metric JSONs into one long CSV
  aggregate_results.py     # aggregate internal metrics, KITTI metrics, deltas, gaps
  summarize_effects.py     # descriptive effect summaries for thesis research questions
  plot_results.py          # thesis-ready PDF/PNG figures
  export_latex_tables.py   # LaTeX tables for thesis inclusion
  run_analysis.sh          # one-command analysis pipeline
  outputs/                 # generated CSVs, figures, and LaTeX tables

<SCENARIO>/
  One-Stage/
    Real/project/         # YOLO trained on real data
    Synthetic/project/    # YOLO trained on synthetic data
    Mixed/project/        # YOLO trained on real/synthetic mixtures
    External/project/     # external evaluation outputs where applicable
  Two-Stage/
    Real/project/         # Faster R-CNN trained on real data
    Synthetic/project/    # Faster R-CNN trained on synthetic data
    Mixed/project/        # Faster R-CNN trained on real/synthetic mixtures
    External/project/     # external evaluation outputs where applicable
  External/project/       # scenario-level external summaries
  _Shared/
    data/                 # raw and prepared datasets
    downloads/            # source archives
    scripts/              # data preparation, training, and evaluation scripts
    tools/                # local tooling and packages
```

The result directories preserve their historical names so completed runs remain traceable.
New user-facing shell entry points use clearer names such as `run_real_all.sh`,
`run_synthetic_all.sh`, and `run_mixed_all.sh`.

## Experimental Design

Each scenario evaluates the same conceptual training recipes:

- **Real only:** train on the prepared real split and evaluate on the real test split.
- **Synthetic only:** train on the prepared synthetic split and evaluate both in-domain and on the real test split.
- **Mixed:** train on real/synthetic train and validation mixtures. Current mixtures are `70/30`, `50/50`, and `30/70`, where the first number is the real-data percentage.
- **External:** for ADD, selected best-seed models are additionally evaluated on KITTI.

Each main training recipe was run for seeds `1`, `2`, and `3`. External KITTI evaluation
uses the best seed per model/setup selected from the existing BDD real-target metrics,
to avoid unnecessary repeated external evaluation.

## Metrics

All reported detector metrics use COCO-style bounding-box evaluation:

- `AP_50_95`
- `AP_50`
- `AP_75`
- per-class AP

The external KITTI split uses KITTI training images because public KITTI test labels are
not available. KITTI classes are mapped into the ADD label space consistently with the
BDD and SYNTHIA mappings:

```text
Car, Van, Truck, Tram -> car
Pedestrian, Person_sitting -> pedestrian
Cyclist -> cyclist
Misc, DontCare -> ignored
```

## Data Availability and Local Placement

The datasets and downloaded archives are not tracked in git. They are intentionally
excluded through `.gitignore` because the raw archives, extracted image folders,
prepared YOLO/COCO splits, and model checkpoints are too large for normal repository
history and may be governed by separate dataset licenses.
Large derived evaluation dumps such as `gt_*.json` and `preds_*.json` are also
ignored; the compact metric JSON summaries and `analysis/outputs/*.csv` files are
kept for reproducible result aggregation.

Ignored local data locations:

```text
ADD/_Shared/downloads/
ADD/_Shared/data/
IS/_Shared/downloads/
IS/_Shared/data/
ADD/_Shared/models/
ADD/**/runs_*/
IS/**/runs_*/
```

The code and generated metric summaries remain in the repository, but a fresh clone
requires users to obtain the datasets separately and place/extract them into the
expected local folders before rerunning preparation or training scripts.

### Autonomous Driving Data

**BDD100K real data**

- Source: BDD100K official download portal: https://bdd-data.berkeley.edu/
- Required for this project: 100K images and detection labels, converted to the
  three-class ADD label space.
- Local expected root after extraction/conversion:

```text
ADD/_Shared/data/ad/bdd100k_raw/bdd100k/
ADD/_Shared/data/ad/bdd_yolo/
ADD/_Shared/data/ad/bdd_coco/
ADD/_Shared/data/ad/bdd_yolo_splits/
```

The implemented experiments use a deterministic 9,000-image BDD subset with a
70/20/10 train/validation/internal-test split.

**SYNTHIA-RAND-CITYSCAPES synthetic data**

- Source: SYNTHIA dataset project / RAND-CITYSCAPES release. The commonly used
  archive is `SYNTHIA_RAND_CITYSCAPES.rar` from the SYNTHIA dataset site:
  http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar
- Required for this project: RGB images and `GT/LABELS` instance/semantic labels.
- Local expected root:

```text
ADD/_Shared/data/ad/synthia_rand_cityscapes/RAND_CITYSCAPES/
ADD/_Shared/data/ad/synthia_yolo_splits/
```

The implemented experiments use a deterministic 9,000-image SYNTHIA subset converted
to the ADD classes `car`, `pedestrian`, and `cyclist`.

**KITTI external real benchmark**

- Source: KITTI object detection benchmark:
  https://www.cvlibs.net/datasets/kitti/eval_object.php
- Required archives:

```text
data_object_image_2.zip
data_object_label_2.zip
```

- Place archives under:

```text
ADD/_Shared/downloads/KITTI/
```

The external evaluation script extracts these archives into:

```text
ADD/_Shared/data/ad/kitti_raw/
ADD/_Shared/data/ad/kitti_yolo_splits/
```

KITTI uses the labeled training split as the external test set because public KITTI
test labels are not distributed.

### Industrial Screw Data

The industrial screw experiments use the six-class real and synthetic screw-head data
exports described in the thesis methodology. These data are not included in the
repository.

Expected local raw roots:

```text
IS/_Shared/data/is/raw/real_yolov9_all_images/
IS/_Shared/data/is/raw/synthetic_yolov9_all_images/
```

Prepared split roots generated by the project scripts:

```text
IS/_Shared/data/is/real_yolo_splits/
IS/_Shared/data/is/synthetic_yolo_splits/
IS/_Shared/data/is/mixed_real*_synth*_seed0/
```

If using the original exported archive, place it under:

```text
IS/_Shared/downloads/
```

Then run the preparation scripts from `IS/`:

```bash
_Shared/scripts/run_real_only_prepare.sh
_Shared/scripts/run_synthetic_prepare.sh
```

## Results Analysis Pipeline

The `analysis/` folder contains a reproducible script-based pipeline for thesis result
aggregation, descriptive effect summaries, plots, and LaTeX tables. It is designed to
answer the thesis research questions directly:

- **RQ1:** regime comparison on real test data (`real_only`, `synthetic_only`, hybrid ratios).
- **RQ2:** one-stage vs two-stage architecture gaps.
- **RQ3:** scale comparison between ADD (`large`) and IS (`small`).
- **RQ4:** hybrid-regime deltas relative to `real_only`.

Run from the repository root:

```bash
analysis/run_analysis.sh
```

The pipeline runs these steps in order:

```bash
python analysis/collect_metrics.py
python analysis/aggregate_results.py
python analysis/summarize_effects.py
python analysis/plot_results.py
python analysis/export_latex_tables.py
```

Generated CSV outputs:

```text
analysis/outputs/results_long.csv
analysis/outputs/internal_summary.csv
analysis/outputs/class_summary.csv
analysis/outputs/kitti_summary.csv
analysis/outputs/delta_vs_real.csv
analysis/outputs/delta_vs_synth.csv
analysis/outputs/architecture_gap.csv
analysis/outputs/effect_summary.csv
analysis/outputs/consistency_summary.csv
analysis/outputs/rankings.csv
```

Generated figures are written as both PDF and PNG under:

```text
analysis/outputs/figures/
```

The main figures include ADD and IS internal real-test regime comparisons, ADD KITTI
external comparison, deltas versus real-only training, ADD-vs-IS scale comparison,
class-AP heatmaps, and synthetic-to-real gap plots.

Generated LaTeX tables are written under:

```text
analysis/outputs/tables/
```

Current exported tables:

```text
table_add_internal.tex
table_add_kitti.tex
table_is_internal.tex
table_class_ap_add.tex
table_class_ap_is.tex
table_effect_deltas.tex
```

KITTI remains a separate external generalization evaluation. KITTI selected-seed
metadata is retained in the CSV files for reproducibility, but the thesis-facing KITTI
figure and LaTeX table do not display the seed.

## Main Commands

Run commands from inside the relevant scenario root.

ADD:

```bash
cd ADD
_Shared/scripts/run_real_all.sh cuda
_Shared/scripts/run_synthetic_all.sh cuda
_Shared/scripts/run_mixed_all.sh cuda 70 30 0 4
_Shared/scripts/run_mixed_all.sh cuda 50 50 0 4
_Shared/scripts/run_mixed_all.sh cuda 30 70 0 4
_Shared/scripts/run_kitti_external_best_seed_eval.sh cuda
```

IS:

```bash
cd IS
_Shared/scripts/run_real_all.sh cuda 4
_Shared/scripts/run_synthetic_all.sh cuda 4
_Shared/scripts/run_mixed_all.sh cuda 70 30 0 4
_Shared/scripts/run_mixed_all.sh cuda 50 50 0 4
_Shared/scripts/run_mixed_all.sh cuda 30 70 0 4
```

## Scientific Caveats

- ADD has a meaningful external benchmark through KITTI. IS does not currently have an external screw benchmark in this workspace.
- IS real labels are highly imbalanced in the available data: most real annotations are `phillips`, with very few or no examples for several other screw classes. IS conclusions should therefore be framed as results for the available real IS distribution, not as final evidence for all screw-head types.
- Synthetic-only performance should be interpreted as a domain-gap diagnostic. High synthetic in-domain AP does not imply strong real-domain transfer.
- Mixed-data comparisons should be read together with the mixture manifests under `_Shared/data/.../mixed_*_seed0/mix_manifest.json`, because dataset budgets differ by scenario.
