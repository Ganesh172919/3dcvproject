"""Lab 1: Monocular Depth Estimator Fine-Tuning (TinyDepth).

Course: 3D Computer Vision · Student Educational Laboratory
Theoretical Foundation:
Combines Scale-Invariant Logarithmic (SILog) loss, Multiscale Gradient loss,
and Surface Normal Consistency loss:
    L = alpha * L_SILog + beta * L_grad + gamma * L_normal

Where:
    L_SILog = sqrt( 1/N sum(d_i^2) - lambda/N^2 (sum(d_i))^2 )  with d_i = ln(y_i) - ln(y_i*)
    L_grad  = sum_s 1/N_s sum( |grad_x(d_i^s)| + |grad_y(d_i^s)| )
    L_normal = 1/N sum( 1 - cos(n_pred, n_gt) )
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lab1.tinydepth")


# ---------------------------------------------------------------------------
# Architecture: Convolutional encoder-decoder
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TinyDepth(nn.Module):
    """Educational lightweight monocular depth estimator."""
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, stride=1)      # S
        self.enc2 = ConvBlock(base_channels, base_channels * 2, stride=2) # S/2
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2) # S/4
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2) # S/8

        # Convolutional bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GELU(),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Positive depth in the units used by the 3D render dataset
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels // 2, 1, 3, padding=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        depth = self.head(d1) + 1e-3
        return depth


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
def silog_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    """Scale-Invariant Logarithmic (SILog) loss (Eigen et al., 2014)."""
    d = (torch.log(pred) - torch.log(target.clamp_min(1e-4))) * mask
    n = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    d2 = (d ** 2).sum(dim=(1, 2, 3)) / n
    d_mean_sq = ((d.sum(dim=(1, 2, 3)) / n) ** 2)
    loss = torch.sqrt((d2 - lam * d_mean_sq).clamp_min(1e-7)).mean()
    return loss


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scales: int = 3) -> torch.Tensor:
    """Multiscale gradient difference loss enforcing sharp object silhouettes."""
    d = (torch.log(pred) - torch.log(target.clamp_min(1e-4))) * mask
    loss = torch.tensor(0.0, device=pred.device)
    curr_d, curr_m = d, mask
    for _ in range(scales):
        dx_p = curr_d[:, :, :, 1:] - curr_d[:, :, :, :-1]
        dx_m = curr_m[:, :, :, 1:] * curr_m[:, :, :, :-1]
        dy_p = curr_d[:, :, 1:, :] - curr_d[:, :, :-1, :]
        dy_m = curr_m[:, :, 1:, :] * curr_m[:, :, :-1, :]
        loss += (dx_p.abs() * dx_m).sum() / dx_m.sum().clamp_min(1.0)
        loss += (dy_p.abs() * dy_m).sum() / dy_m.sum().clamp_min(1.0)
        curr_d = F.avg_pool2d(curr_d, 2)
        curr_m = (F.avg_pool2d(curr_m.float(), 2) > 0.5)
    return loss


def normal_consistency_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes surface normal vectors from depth spatial derivatives and penalizes angular error."""
    # Approximate normals via spatial finite differences
    def depth_to_normals(z: torch.Tensor) -> torch.Tensor:
        dz_dx = (z[:, :, :, 2:] - z[:, :, :, :-2]) * 0.5
        dz_dy = (z[:, :, 2:, :] - z[:, :, :-2, :]) * 0.5
        dz_dx = dz_dx[:, :, 1:-1, :]
        dz_dy = dz_dy[:, :, :, 1:-1]
        ones = torch.ones_like(dz_dx)
        n = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
        norm = torch.norm(n, p=2, dim=1, keepdim=True).clamp_min(1e-6)
        return n / norm

    if pred.shape[-1] < 5 or pred.shape[-2] < 5:
        return torch.tensor(0.0, device=pred.device)

    n_pred = depth_to_normals(pred)
    n_gt = depth_to_normals(target)
    m = mask[:, :, 1:-1, 1:-1]
    cos_sim = (n_pred * n_gt).sum(dim=1, keepdim=True)
    loss = ((1.0 - cos_sim) * m).sum() / m.sum().clamp_min(1.0)
    return loss


# ---------------------------------------------------------------------------
# 3D Render Dataset Generator for Student Experiments
# ---------------------------------------------------------------------------
class Render3DStudentDepthDataset(Dataset):
    """Generates 3D render geometric objects with analytical ground-truth depth."""
    def __init__(self, num_samples: int = 40, size: int = 128):
        self.num_samples = num_samples
        self.size = size
        self.data = []
        # Pre-generate 3D render samples (spheres, cubes, cones, cylinders)
        for i in range(num_samples):
            obj_type = i % 4
            rgb, depth, mask = self._generate_sample(obj_type, size)
            self.data.append((rgb, depth, mask))

    def _generate_sample(self, obj_type: int, S: int):
        y, x = np.mgrid[-1:1:complex(0, S), -1:1:complex(0, S)]
        r = np.sqrt(x**2 + y**2)
        mask = np.zeros((S, S), dtype=bool)
        depth = np.ones((S, S), dtype=np.float32) * 3.0  # background depth

        color = np.array([random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)], dtype=np.float32)
        rgb = np.ones((S, S, 3), dtype=np.float32) * 0.1

        if obj_type == 0:  # Sphere
            radius = 0.55
            m = r <= radius
            mask[m] = True
            depth[m] = 2.0 - np.sqrt(np.maximum(0, radius**2 - r[m]**2))
            shading = (1.0 - 0.4 * (depth[m] - 1.45))[:, None]
            rgb[m] = color[None, :] * shading
        elif obj_type == 1:  # Cube
            m = (np.abs(x) <= 0.45) & (np.abs(y) <= 0.45)
            mask[m] = True
            depth[m] = 1.6 + 0.3 * np.abs(x[m])
            rgb[m] = color[None, :] * 0.9
        elif obj_type == 2:  # Cone
            m = (r <= 0.5) & (y <= 0.5)
            mask[m] = True
            depth[m] = 1.5 + 1.2 * r[m]
            shading = (1.0 - 0.5 * r[m])[:, None]
            rgb[m] = color[None, :] * shading
        else:  # Cylinder
            m = (np.abs(x) <= 0.4) & (np.abs(y) <= 0.6)
            mask[m] = True
            depth[m] = 1.7 - np.sqrt(np.maximum(0, 0.4**2 - x[m]**2))
            rgb[m] = color[None, :]

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))
        depth_t = torch.from_numpy(depth[None, ...].astype(np.float32))
        mask_t = torch.from_numpy(mask[None, ...].astype(bool))
        return rgb_t, depth_t, mask_t

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Training & Evaluation Loop
# ---------------------------------------------------------------------------
def evaluate_depth(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mae_list, rmse_list, delta1_list = [], [], []
    with torch.no_grad():
        for rgb, depth, mask in loader:
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
            pred = model(rgb)
            p_val = pred[mask]
            t_val = depth[mask]
            if len(p_val) == 0:
                continue
            mae = (p_val - t_val).abs().mean().item()
            rmse = torch.sqrt(((p_val - t_val) ** 2).mean()).item()
            ratio = torch.max(p_val / t_val.clamp_min(1e-4), t_val / p_val.clamp_min(1e-4))
            delta1 = (ratio < 1.25).float().mean().item()

            mae_list.append(mae)
            rmse_list.append(rmse)
            delta1_list.append(delta1)

    return {
        "mae": float(np.mean(mae_list)) if mae_list else 0.0,
        "rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "delta_1": float(np.mean(delta1_list)) if delta1_list else 0.0,
    }


def train_student_depth(
    epochs: int = 5,
    lr: float = 1e-3,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
    batch_size: int = 4,
    out_dir: str | Path = "out/student_labs/lab1",
    device_name: str = "cpu"
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    log.info("Starting Lab 1: TinyDepth fine-tuning on %s (%d epochs)", device, epochs)
    dataset_train = Render3DStudentDepthDataset(num_samples=32, size=128)
    dataset_val = Render3DStudentDepthDataset(num_samples=8, size=128)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    model = TinyDepth(base_channels=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"epochs": [], "loss": [], "silog": [], "grad": [], "normal": [], "val_metrics": []}

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, ep_silog, ep_grad, ep_norm = [], [], [], []

        for rgb, depth, mask in loader_train:
            rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(rgb)

            l_silog = silog_loss(pred, depth, mask)
            l_grad = gradient_loss(pred, depth, mask)
            l_norm = normal_consistency_loss(pred, depth, mask)
            loss = alpha * l_silog + beta * l_grad + gamma * l_norm

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss.append(loss.item())
            ep_silog.append(l_silog.item())
            ep_grad.append(l_grad.item())
            ep_norm.append(l_norm.item())

        scheduler.step()
        val_m = evaluate_depth(model, loader_val, device)

        avg_loss = float(np.mean(ep_loss))
        avg_silog = float(np.mean(ep_silog))
        avg_grad = float(np.mean(ep_grad))
        avg_norm = float(np.mean(ep_norm))

        history["epochs"].append(ep)
        history["loss"].append(round(avg_loss, 4))
        history["silog"].append(round(avg_silog, 4))
        history["grad"].append(round(avg_grad, 4))
        history["normal"].append(round(avg_norm, 4))
        history["val_metrics"].append(val_m)

        log.info("Epoch %d/%d — Loss: %.4f (SILog: %.4f, Grad: %.4f) — Val MAE: %.4f, δ1: %.3f",
                 ep, epochs, avg_loss, avg_silog, avg_grad, val_m["mae"], val_m["delta_1"])

    # Export checkpoint
    ckpt_path = out_path / "tinydepth_student.pth"
    torch.save({"model_state": model.state_dict(), "history": history}, ckpt_path)

    # Export ONNX
    onnx_path = out_path / "tinydepth_student.onnx"
    dummy_input = torch.randn(1, 3, 128, 128, device=device)
    model.eval()
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["rgb_input"],
            output_names=["depth_output"],
            dynamic_axes={"rgb_input": {0: "batch"}, "depth_output": {0: "batch"}},
            opset_version=14,
            dynamo=False,
        )
        log.info("Exported ONNX checkpoint to %s", onnx_path)
    except Exception as e:
        log.warning("ONNX export skipped: %s", e)

    results = {
        "status": "completed",
        "final_loss": history["loss"][-1] if history["loss"] else 0.0,
        "final_mae": history["val_metrics"][-1]["mae"] if history["val_metrics"] else 0.0,
        "final_delta1": history["val_metrics"][-1]["delta_1"] if history["val_metrics"] else 0.0,
        "history": history,
        "checkpoint": str(ckpt_path),
        "onnx": str(onnx_path) if onnx_path.exists() else None,
    }
    (out_path / "lab1_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 1: Monocular Depth Estimator Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out-dir", type=str, default="out/student_labs/lab1", help="Output directory")
    args = parser.parse_args()
    train_student_depth(epochs=args.epochs, lr=args.lr, out_dir=args.out_dir)
