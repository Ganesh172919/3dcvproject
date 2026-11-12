# How-To Guide: Fine-Tuning Depth Estimators on Domain-Specific Data

This guide explains how to fine-tune monocular depth models (such as Depth Anything V2 or MiDaS) on custom 3D domains (e.g. footwear, industrial components, or dental casts) and export quantized INT8 ONNX graphs for production inference in 3DFUN AI.

---

## 1. Preparing the Training Dataset

Monocular depth models require paired RGB images and metric depth maps. You can generate synthetic training pairs using `training/dataset_gen/synthetic_renderer.py` from 3D CAD/OBJ meshes:

```bash
python training/dataset_gen/synthetic_renderer.py \
  --input_dir data/cad_models \
  --output_dir data/rendered_depth \
  --num_views 24 \
  --resolution 512
```

This generates:
- `rgb/*.png`: Multi-view rendered images under random directional lighting.
- `depth/*.png`: 16-bit metric depth maps where pixel intensity encodes Euclidean camera distance.
- `mask/*.png`: Binary silhouette foreground segmentations.

---

## 2. Configuring the Fine-Tuning Pipeline

Edit `training/configs/train_config.yaml` to specify hyperparameters and loss weights:

```yaml
training:
  batch_size: 16
  learning_rate: 1e-5
  weight_decay: 0.01
  epochs: 25
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"

losses:
  silog_weight: 1.0         # Scale-Invariant Logarithmic Loss
  grad_weight: 0.5          # Multi-Scale Gradient Loss
  smoothness_weight: 0.1    # Edge-Aware Smoothness Loss

model:
  backbone: "vits"          # Small ViT encoder
  input_resolution: 518
```

---

## 3. Running the Training Script

Launch the fine-tuning script:

```bash
python training/fine_tune_depth/train_depth.py \
  --config training/configs/train_config.yaml \
  --data_dir data/rendered_depth \
  --checkpoint_dir checkpoints/depth_v2_finetuned
```

### Loss Formulations Used:
- **Scale-Invariant Logarithmic Loss ($\mathcal{L}_{\text{SILog}}$):**
  $$\mathcal{L}_{\text{SILog}} = \frac{1}{N}\sum_{i} d_i^2 - \frac{\lambda}{N^2}\left(\sum_i d_i\right)^2, \quad d_i = \ln(y_i) - \ln(y_i^*)$$
  This invariant loss decouples absolute scene scale from relative depth gradients, preventing global offset collapse.
- **Multi-Scale Gradient Matching ($\mathcal{L}_{\text{grad}}$):** Penalizes surface normal discrepancies and preserves sharp occluding boundaries.

---

## 4. Exporting to Quantized INT8 ONNX

Once fine-tuning completes, export the PyTorch model checkpoint to an ONNX graph and apply dynamic range INT8 quantization:

```bash
python training/fine_tune_depth/export_onnx.py \
  --checkpoint checkpoints/depth_v2_finetuned/best_model.pth \
  --output engine/models/depth_anything_v2_vits.onnx \
  --quantize int8 \
  --calib_dir data/rendered_depth/rgb
```

The script verifies:
1. Valid ONNX graph topology via `onnx.checker.check_model()`.
2. Numerical tolerance: Mean Absolute Error between FP32 PyTorch and INT8 ONNX must remain $\le 0.008$.
3. File size reduction from $\approx 98$ MB (FP32) to $\approx 27$ MB (INT8).

---

## 5. Integrating with the Reconstruction Pipeline

To deploy your fine-tuned model into 3DFUN AI:
1. Place the generated `.onnx` file in `engine/models/`.
2. Restart the FastAPI server.
3. Validate reconstruction quality using the benchmark harness:
   ```bash
   python benchmarks/run_benchmarks.py --method proposed
   ```

---
*Evidence: Model training code in `training/fine_tune_depth/train_depth.py`; ONNX exporter in `training/fine_tune_depth/export_onnx.py`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
