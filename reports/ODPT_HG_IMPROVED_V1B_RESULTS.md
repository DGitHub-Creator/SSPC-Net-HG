# SSPC-Net-Improved-v1b ODPT-HG full-run results (adapted implementation)

Note: results were produced by the dataset-adapted SSPC-Net implementation (modified for the custom ODPT-HG dataset), not the paper authors' official code.

Status: **verified**. Metrics were independently recomputed from the saved point-weighted confusion matrices.

Protocol: canonical Area 1 10%/20% scene splits, seed 1, 400 epochs, no validation during training, then one final Area 3 evaluation using 10 deterministic samples.

Area 3: 3 scenes and 2,757,311 original points.

| Method | Budget | OA | mAcc | mIoU |
|---|---:|---:|---:|---:|
| SSPC-Net-Improved-v1b | 10% | 86.11% | 69.58% | 59.51% |
| SemiGMMPoint | 10% | 76.84% | 62.42% | 45.58% |
| SSPC-Net original | 10% | 86.80% | 72.81% | 62.47% |
| SSPC-Net-Improved-v1b | 20% | 87.10% | 71.64% | 61.72% |
| SemiGMMPoint | 20% | 75.69% | 67.73% | 46.44% |
| SSPC-Net original | 20% | 86.05% | 71.43% | 60.71% |

## SSPC-Net-Improved-v1b 10% per-class metrics

| Class | Support | Accuracy | IoU |
|---|---:|---:|---:|
| pipeline | 475,322 | 78.60% | 67.81% |
| steel_frame | 271,222 | 43.89% | 37.58% |
| elbow_pipe | 251,727 | 96.37% | 78.42% |
| valve_guardrail | 96,732 | 35.35% | 21.36% |
| gate_valve | 106,067 | 64.52% | 61.04% |
| Christmas_tree_body | 1,556,241 | 98.72% | 90.85% |

Difference versus SemiGMMPoint (SSPC-Net-Improved-v1b minus baseline): OA +9.27 pp, mAcc +7.16 pp, mIoU +13.93 pp.

Difference versus SSPC-Net original (SSPC-Net-Improved-v1b minus reference): OA -0.69 pp, mAcc -3.24 pp, mIoU -2.96 pp.

Label extension evidence: 162 checks, 162 non-empty checks, 8,106 generated extension points.
Prototype seed evidence: 162 checks, 162 non-empty checks, 2,778 generated seeds.

## SSPC-Net-Improved-v1b 20% per-class metrics

| Class | Support | Accuracy | IoU |
|---|---:|---:|---:|
| pipeline | 475,322 | 83.00% | 67.78% |
| steel_frame | 271,222 | 52.38% | 42.74% |
| elbow_pipe | 251,727 | 91.17% | 74.05% |
| valve_guardrail | 96,732 | 31.73% | 25.15% |
| gate_valve | 106,067 | 73.45% | 68.21% |
| Christmas_tree_body | 1,556,241 | 98.11% | 92.40% |

Difference versus SemiGMMPoint (SSPC-Net-Improved-v1b minus baseline): OA +11.41 pp, mAcc +3.91 pp, mIoU +15.28 pp.

Difference versus SSPC-Net original (SSPC-Net-Improved-v1b minus reference): OA +1.05 pp, mAcc +0.21 pp, mIoU +1.01 pp.

Label extension evidence: 162 checks, 162 non-empty checks, 8,114 generated extension points.
Prototype seed evidence: 162 checks, 162 non-empty checks, 2,745 generated seeds.

## Validation evidence

For each budget, the collector verified epoch 399 completion, finite final loss, the final checkpoint, command-line protocol flags, Area 3 appearing only after the final-evaluation marker, all three prediction vectors and their manifest-derived superpoint counts, confusion shape/count/support, agreement with `scores_val.txt`, and label-extension counts parsed from the training log.
