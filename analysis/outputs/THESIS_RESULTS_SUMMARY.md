# Thesis Results Summary

## Best Internal Real-Test Regime per Scenario and Architecture
- ADD / YOLOv8m: Real-only (0.390 AP_{50:95})
- ADD / Faster R-CNN: Hybrid 70/30 (0.219 AP_{50:95})
- IS / YOLOv8m: Real-only (0.064 AP_{50:95})
- IS / Faster R-CNN: Real-only (0.044 AP_{50:95})

## Hybrid vs Real-only
- ADD / Faster R-CNN / Hybrid 70/30 improved by 0.002 AP_{50:95}.

## One-stage vs Two-stage
- YOLOv8m outperforms Faster R-CNN on internal real-test AP_{50:95} in both scenarios and all reported regimes.

## ADD vs IS
- The ADD scenario reaches substantially higher real-domain AP than the IS scenario.
- The IS scenario shows a stronger synthetic-to-real collapse despite high synthetic in-domain AP.

## Qualitative Examples
- Qualitative overlay export status is documented in `qualitative/QUALITATIVE_STATUS.md`.
- No thesis-ready overlay examples could be reconstructed from the committed repository artifacts alone.

## Caveats
- IS real labels are strongly imbalanced and dominated by the phillips class.
- Undefined or absent class metrics in IS are treated as missing during aggregation.
