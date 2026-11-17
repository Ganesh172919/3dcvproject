# TinyDepth and visible-surface evaluation

TinyDepth is a 24,369-parameter network trained from scratch on the project's procedural RGB and depth pairs. A full-resolution convolution produces 16 detail channels. A stride-two branch with a dilated convolution adds scene context. The network upsamples that branch, concatenates it with the detail features, and uses a small head and softplus to predict positive axial depth.

## Training and camera geometry

The renderer scales each mesh to unit maximum extent and stores visible-surface depth in 16-bit PNGs at 1,000 counts per normalized unit. Eight views of each mesh share one split. The reference run used 48 training views across six meshes, eight validation views of a capsule, and eight test views of a cylinder.

Let $r_i = \log \hat z_i - \log z_i$ for valid foreground pixels. The loss combines $\sqrt{\overline{r^2} - 0.85\bar{r}^2}$ with the mean absolute differences of neighboring residuals at three scales. The validation MAE selects the checkpoint. The exporter compares its FP32 predictions with ONNX Runtime at multiple input shapes.

For the pixel center $(u,v)$ and known focal length $f$, the evaluator computes $X = (u-c_x)Z/f$ and $Y = (c_y-v)Z/f$. The same foreground pixel sample becomes a predicted, reference, and constant-depth cloud. The constant uses the training-set median of foreground medians. The known silhouette and camera make this a controlled visible-surface test.

## Recorded CPU results

| Model | Depth MAE ↓ | Unsquared Chamfer ↓ | F-score at 0.01 ↑ |
| --- | ---: | ---: | ---: |
| Constant depth | 0.1591 | 0.1358 | 0.0211 |
| SILog only | 0.1267 | 0.0989 | 0.0424 |
| SILog + gradients | 0.1239 | 0.0980 | 0.0456 |

Scores average the eight views of a single held-out mesh. Chamfer averages unsquared nearest-neighbor distances in both directions. F-score is the harmonic mean of the fractions falling within 0.01 normalized units in each direction. The training set includes a wider cylinder, so this experiment measures transfer to another instance of a familiar shape family.

The recorded summaries, training histories, and per-view CSVs are in `results/course17/`. See the [experiment guide](../handbook/reproducibility.md) to plan a repeat run. The mesh benchmark uses a different protocol and independently normalized full meshes.
