# 3DFUN AI — Training, Fine-Tuning & Model Adaptation Suite

This directory contains the self-contained training and domain adaptation tools for **3DFUN AI**, engineered for the 3D Computer Vision curriculum at the Indian Institute of Information Technology (IIIT), Dharwad.

---

## 1. Module Overview

```
training/
├── configs/
│   └── train_config.yaml            # Master configuration schema
├── dataset_gen/
│   └── synthetic_renderer.py        # Multi-view RGB, 16-bit depth & normal generator
├── fine_tune_depth/
│   ├── train_depth.py               # PyTorch training script with SILog & gradient loss
│   └── export_onnx.py               # INT8 ONNX graph exporter with dynamic batching
├── calibration/
│   └── optimize_parameters.py       # Optuna / Bayesian hyperparameter tuner
└── README.md                        # This documentation
```

---

## 2. Step 1: Synthetic Dataset Generation

To train or fine-tune monocular depth estimators without expensive 3D scanning equipment, `synthetic_renderer.py` renders multi-view synthetic datasets directly from closed 3D OBJ or GLB meshes:

```bash
python training/dataset_gen/synthetic_renderer.py \
  --input_dir benchmarks/data/synthetic/gt \
  --output_dir training/data/synthetic_renders \
  --num_views 16 \
  --resolution 512
```

### Outputs Generated:
- `rgb/*.png`: Shaded RGB images with directional lighting.
- `depth/*.png`: 16-bit unsigned integer depth maps ($1\text{ unit} = 1\text{ mm}$).
- `normals/*.png`: Unit surface normals mapped into RGB color space.
- `mask/*.png`: Binary silhouette foreground segmentations.

---

## 3. Step 2: Depth Fine-Tuning with SILog Loss

Fine-tuning uses the Scale-Invariant Logarithmic Loss ($\mathcal{L}_{\text{SILog}}$) combined with multi-scale gradient matching:

$$\mathcal{L}_{\text{SILog}} = \sqrt{\frac{1}{N}\sum_i d_i^2 - \frac{\lambda}{N^2}\left(\sum_i d_i\right)^2}, \quad d_i = \ln(y_i + \epsilon) - \ln(y_i^* + \epsilon)$$

$$\mathcal{L}_{\text{grad}} = \frac{1}{N}\sum_i \left(|\nabla_x d_i| + |\nabla_y d_i|\right)$$

Run training with:
```bash
python training/fine_tune_depth/train_depth.py \
  --data_dir training/data/synthetic_renders \
  --output_dir training/checkpoints \
  --epochs 20 \
  --lr 2e-5
```

---

## 4. Step 3: Export & INT8 Quantization

Convert fine-tuned weights into an optimized ONNX graph with dynamic batch axes (`batch_size`, `height`, `width`) and apply post-training INT8 quantization:

```bash
python training/fine_tune_depth/export_onnx.py \
  --source training/checkpoints/best_model.pth \
  --output engine/models/depth_anything_v2_vits_finetuned.onnx \
  --quantize
```

---

## 5. Step 4: Hyperparameter Calibration via Optuna

To maximize geometric reconstruction fidelity for specific object categories, tune the volumetric inflation parameters ($t_{\max}, f, g, \gamma, \sigma_{\text{sdf}}$) using Optuna:

```bash
python training/calibration/optimize_parameters.py \
  --trials 50 \
  --output training/checkpoints/optimal_params.json
```

The optimizer searches the continuous manifold parameter space to minimize Chamfer Distance against analytic ground truth meshes.

---

## Team & Attribution
- **J Ganesh (23BDS024)** — Dept. of Data Science and Artificial Intelligence
- **M Jagadeshwar (23BDS033)** — Dept. of Data Science and Artificial Intelligence
- **B Varshith (23BDS011)** — Dept. of Data Science and Artificial Intelligence
- **Pranav Kashyap (23BCS098)** — Dept. of Computer Science and Engineering
- **C Sai Aravind (23BDS075)** — Dept. of Data Science and Artificial Intelligence
- **Course Supervisor:** **Dr. Shirshendu Layek**, Assistant Professor, Dept. of Data Science & AI (DSAI), IIIT Dharwad
