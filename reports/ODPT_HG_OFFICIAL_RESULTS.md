# SSPC-Net ODPT-HG full-run results (adapted implementation)

Note: results were produced by the dataset-adapted SSPC-Net implementation (modified for the custom ODPT-HG dataset), not the paper authors' official code.

Status: **verified**. Metrics were independently recomputed from the saved point-weighted confusion matrices.

Protocol: canonical Area 1 10%/20% scene splits, seed 1, 400 epochs, no validation during training, then one final Area 3 evaluation using 10 deterministic samples.

Area 3: 3 scenes and 2,757,311 original points.

| Method | Budget | OA | mAcc | mIoU |
|---|---:|---:|---:|---:|
| SSPC-Net | 10% | 86.80% | 72.81% | 62.47% |
| SemiGMMPoint | 10% | 76.84% | 62.42% | 45.58% |
| SSPC-Net | 20% | 86.05% | 71.43% | 60.71% |
| SemiGMMPoint | 20% | 75.69% | 67.73% | 46.44% |

## SSPC-Net 10% per-class metrics

| Class | Support | Accuracy | IoU |
|---|---:|---:|---:|
| pipeline | 475,322 | 81.55% | 67.30% |
| steel_frame | 271,222 | 44.41% | 37.08% |
| elbow_pipe | 251,727 | 90.94% | 77.20% |
| valve_guardrail | 96,732 | 46.29% | 31.71% |
| gate_valve | 106,067 | 75.28% | 69.74% |
| Christmas_tree_body | 1,556,241 | 98.42% | 91.81% |

Difference versus SemiGMMPoint (SSPC-Net minus baseline): OA +9.96 pp, mAcc +10.39 pp, mIoU +16.89 pp.

Label extension evidence: 117 checks, 0 non-empty checks, 0 generated extension points.

## SSPC-Net 20% per-class metrics

| Class | Support | Accuracy | IoU |
|---|---:|---:|---:|
| pipeline | 475,322 | 72.92% | 63.25% |
| steel_frame | 271,222 | 51.68% | 40.19% |
| elbow_pipe | 251,727 | 95.61% | 76.52% |
| valve_guardrail | 96,732 | 34.56% | 21.22% |
| gate_valve | 106,067 | 75.40% | 70.81% |
| Christmas_tree_body | 1,556,241 | 98.43% | 92.25% |

Difference versus SemiGMMPoint (SSPC-Net minus baseline): OA +10.36 pp, mAcc +3.70 pp, mIoU +14.27 pp.

Label extension evidence: 108 checks, 0 non-empty checks, 0 generated extension points.

## Validation evidence

For each budget, the collector verified epoch 399 completion, finite final loss, the final checkpoint, command-line protocol flags, Area 3 appearing only after the final-evaluation marker, all three prediction vectors and their manifest-derived superpoint counts, confusion shape/count/support, agreement with `scores_val.txt`, and label-extension counts parsed from the training log.
