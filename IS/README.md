# IS Scenario: Industrial Screw Detection

IS evaluates synthetic-to-real transfer for small industrial screw-head detection.
The scenario mirrors the ADD experiment families, but it uses a six-class screw
label space and separate scripts/data roots.

## Label Space

All IS experiments use:

```text
0 hex
1 hex_socket
2 phillips
3 pozidriv
4 slotted
5 torx
```

The raw real and synthetic YOLO exports use the same class order.

## Data Roots

```text
_Shared/data/is/raw/real_yolov9_all_images/       # raw real YOLO export
_Shared/data/is/raw/synthetic_yolov9_all_images/  # raw synthetic YOLO export
_Shared/data/is/real_yolo_splits/                 # prepared real split
_Shared/data/is/synthetic_yolo_splits/            # prepared synthetic split
_Shared/data/is/mixed_real*_synth*_seed0/         # prepared mixed datasets
```

Prepared dataset sizes:

| Dataset | Train | Val | Test | Notes |
| --- | ---: | ---: | ---: | --- |
| Real | 341 | 97 | 50 | 488 images total |
| Synthetic | 341 | 97 | 50 | 488-image controlled subset from 1728 available images |

Mixed datasets use the real split size as the common budget:

| Mixture | Train real | Train synthetic | Val real | Val synthetic |
| --- | ---: | ---: | ---: | ---: |
| 70/30 | 239 | 102 | 68 | 29 |
| 50/50 | 170 | 171 | 48 | 49 |
| 30/70 | 102 | 239 | 29 | 68 |

## Important Data Caveat

The available real IS labels are highly imbalanced:

```text
hex=1
hex_socket=1
phillips=2120
pozidriv=0
slotted=30
torx=0
```

The real test split is therefore mostly a `phillips` evaluation. IS conclusions
should be stated as conclusions for the available real data distribution, not as a
complete assessment of all six screw classes.

## Experiment Outputs

```text
One-Stage/Real/project/       # YOLO real-only runs and metrics
One-Stage/Synthetic/project/  # YOLO synthetic-only runs and metrics
One-Stage/Mixed/project/      # YOLO mixed runs and metrics

Two-Stage/Real/project/       # Faster R-CNN real-only runs and metrics
Two-Stage/Synthetic/project/  # Faster R-CNN synthetic-only runs and metrics
Two-Stage/Mixed/project/      # Faster R-CNN mixed runs and metrics
```

## Main Commands

Run from `IS/`.

Prepare real and synthetic splits:

```bash
_Shared/scripts/run_real_only_prepare.sh
_Shared/scripts/run_synthetic_prepare.sh
```

Real-only:

```bash
_Shared/scripts/run_real_all.sh cuda 4
```

Synthetic-only:

```bash
_Shared/scripts/run_synthetic_all.sh cuda 4
```

Mixed:

```bash
_Shared/scripts/run_mixed_all.sh cuda 70 30 0 4
_Shared/scripts/run_mixed_all.sh cuda 50 50 0 4
_Shared/scripts/run_mixed_all.sh cuda 30 70 0 4
```

The older names remain available:

```text
run_budgeted_real.sh       -> real-only
run_budgeted_synthetic.sh  -> synthetic-only
run_budgeted_mixed.sh      -> mixed
```

## Methodological Notes

- YOLO and Faster R-CNN are evaluated with the same COCO-style metric protocol used in ADD.
- Synthetic-only models are evaluated both in-domain and on the real test split.
- Mixed models are evaluated on both real and synthetic test splits.
- The current IS pattern is scientifically plausible for a strong synthetic-to-real gap: synthetic-only performs well on synthetic data but transfers poorly to the real screw images; naive mixing does not outperform real-only on the current real test distribution.
