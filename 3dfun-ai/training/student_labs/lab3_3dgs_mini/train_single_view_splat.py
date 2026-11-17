"""Lab 3: 3D Gaussian Splatting (3DGS) Fundamentals & Optimization.

Course: 3D Computer Vision · Student Educational Laboratory
Theoretical Foundation:
Each 3D Gaussian is parameterized by:
    - Mean position: mu in R^3
    - Covariance matrix: Sigma = R * S * S^T * R^T (parameterized by rotation quaternion q and scale s)
    - Opacity: alpha in [0, 1]
    - Color: RGB (or Spherical Harmonics coefficients)

The 3D Gaussian is projected onto the 2D image plane via Jacobian J:
    Sigma' = J * W * Sigma * W^T * J^T
The 2D projected Gaussian footprint has covariance:
    G(x) = exp( - 1/2 (x - mu')^T Sigma'^-1 (x - mu') )
Blended via alpha-compositing:
    C = sum_i c_i * alpha_i * prod_{j < i} (1 - alpha_j)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lab3.splat")


# ---------------------------------------------------------------------------
# Differentiable Mini 3D Gaussian Splatter
# ---------------------------------------------------------------------------
class MiniGaussianSplatter(nn.Module):
    def __init__(self, num_points: int = 1000):
        super().__init__()
        self.num_points = num_points
        # 3D Means (positions in [-0.8, 0.8]^3)
        self.means = nn.Parameter(torch.randn(num_points, 3) * 0.3)
        # Log-scales (s_x, s_y, s_z)
        self.log_scales = nn.Parameter(torch.ones(num_points, 3) * -3.5)
        # Logit-opacities
        self.logit_opacities = nn.Parameter(torch.ones(num_points, 1) * 0.0)
        # RGB colors (logit or raw in [0, 1])
        self.colors = nn.Parameter(torch.rand(num_points, 3))

    def render(self, img_size: int = 128, focal_length: float = 180.0) -> torch.Tensor:
        """Differentiable 2D projection and accumulation rasterizer."""
        device = self.means.device
        H, W = img_size, img_size

        # Unpack parameters
        mu = self.means  # (N, 3)
        scales = torch.exp(self.log_scales)  # (N, 3)
        opacities = torch.sigmoid(self.logit_opacities)  # (N, 1)
        colors = torch.clamp(self.colors, 0.0, 1.0)  # (N, 3)

        # Perspective projection onto image plane:
        # X' = (X / Z) * fx + W/2, Y' = (Y / Z) * fy + H/2
        z = (mu[:, 2] + 2.5).clamp_min(0.1)
        u = (mu[:, 0] / z) * focal_length + (W / 2.0)
        v = (mu[:, 1] / z) * focal_length + (H / 2.0)

        # 2D projected radius r_2d proportional to scale / z
        r_2d = (scales.mean(dim=-1) / z) * focal_length
        r_2d = torch.clamp(r_2d, 1.5, 25.0)

        # Pixel grid coordinates
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # Sort Gaussians front-to-back by depth z
        depth_order = torch.argsort(z)
        u_sorted = u[depth_order, None, None]           # (N, 1, 1)
        v_sorted = v[depth_order, None, None]           # (N, 1, 1)
        rad_sorted = r_2d[depth_order, None, None]       # (N, 1, 1)
        alpha_sorted = opacities[depth_order, :, None]   # (N, 1, 1)
        col_sorted = colors[depth_order, :, None, None]  # (N, 3, 1, 1)

        x_2d = x_grid[None, :, :]  # (1, H, W)
        y_2d = y_grid[None, :, :]  # (1, H, W)

        dist_sq = ((x_2d - u_sorted) ** 2 + (y_2d - v_sorted) ** 2) / (2.0 * (rad_sorted ** 2) + 1e-4)
        weights = alpha_sorted * torch.exp(-dist_sq)  # (N, H, W)
        weights = torch.clamp(weights, min=0.0, max=0.99)

        # Transmittance along ray: T_i = prod_{j < i} (1 - w_j)
        one_minus = torch.clamp(1.0 - weights, min=1e-4, max=1.0)
        prefix = torch.ones(1, H, W, device=device)
        cum_trans = torch.cumprod(torch.cat([prefix, one_minus[:-1]], dim=0), dim=0)

        # Differentiable alpha-blended rendering
        contrib = (weights * cum_trans)[:, None, :, :] * col_sorted  # (N, 3, H, W)
        canvas = torch.sum(contrib, dim=0)  # (3, H, W)
        return canvas


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_gaussian_splat(
    target_image_path: str | Path | None = None,
    num_points: int = 1200,
    iterations: int = 40,
    lr: float = 0.02,
    out_dir: str | Path = "out/student_labs/lab3",
    device_name: str = "cpu"
) -> dict[str, Any]:
    if num_points < 1 or iterations < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Use positive point and iteration counts and a finite learning rate.")
    img_size = 64
    if target_image_path is not None:
        img_bgr = cv2.imread(str(target_image_path))
        if img_bgr is None:
            raise ValueError(f"Could not read the target image: {target_image_path}")
        resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        # Create a synthetic target (multi-colored concentric geometric icon)
        y, x = np.mgrid[-1:1:complex(0, img_size), -1:1:complex(0, img_size)]
        r = np.sqrt(x**2 + y**2)
        img_rgb = np.zeros((img_size, img_size, 3), dtype=np.float32)
        img_rgb[r < 0.6] = [0.85, 0.35, 0.2]   # coral
        img_rgb[r < 0.35] = [0.2, 0.75, 0.85]  # cyan core
        img_rgb[np.abs(x) < 0.08] = [0.95, 0.85, 0.15] # gold stripe

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    target_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).to(device)

    model = MiniGaussianSplatter(num_points=num_points).to(device)
    optimizer = torch.optim.Adam([
        {"params": [model.means], "lr": lr},
        {"params": [model.log_scales], "lr": lr * 0.5},
        {"params": [model.logit_opacities], "lr": lr},
        {"params": [model.colors], "lr": lr * 1.5},
    ])

    log.info("Starting Lab 3: 3D Gaussian Splatting (%d Gaussians, %d iterations)", num_points, iterations)
    history = {"iterations": [], "loss": [], "psnr": []}

    for it in range(1, iterations + 1):
        optimizer.zero_grad()
        rendered = model.render(img_size=img_size)

        loss = F.l1_loss(rendered, target_t)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            mse = F.mse_loss(rendered, target_t).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-8))

        history["iterations"].append(it)
        history["loss"].append(round(loss.item(), 4))
        history["psnr"].append(round(psnr, 2))

        if it % 10 == 0 or it == iterations:
            log.info("Iteration %d/%d — L1 Loss: %.4f, PSNR: %.2f dB", it, iterations, loss.item(), psnr)

    # Save rendered output
    with torch.no_grad():
        rendered = model.render(img_size=img_size)
        final_loss = F.l1_loss(rendered, target_t).item()
        final_mse = F.mse_loss(rendered, target_t).item()
        final_render = rendered.cpu().numpy().transpose(1, 2, 0)
        final_render_u8 = (np.clip(final_render, 0, 1) * 255).astype(np.uint8)
        if not cv2.imwrite(str(out_path / "splat_render.png"), cv2.cvtColor(final_render_u8, cv2.COLOR_RGB2BGR)):
            raise OSError("Could not save the splat render.")

    # Export Gaussian Point Cloud to PLY
    ply_path = out_path / "gaussian_splats.ply"
    means = model.means.detach().cpu().numpy()
    colors = (np.clip(model.colors.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(means)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header"
    ]
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for i in range(len(means)):
            f.write(f"{means[i,0]:.5f} {means[i,1]:.5f} {means[i,2]:.5f} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    results = {
        "status": "completed",
        "num_gaussians": num_points,
        "final_loss": round(final_loss, 4),
        "final_psnr": round(10 * math.log10(1.0 / max(final_mse, 1e-8)), 2),
        "history": history,
        "render_path": str(out_path / "splat_render.png"),
        "ply_path": str(ply_path),
    }
    (out_path / "lab3_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 3: 3D Gaussian Splatting Fundamentals")
    parser.add_argument("--num-points", type=int, default=800)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--image", type=Path, help="Use a photo in place of the generated target")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--out-dir", type=str, default="out/student_labs/lab3")
    args = parser.parse_args()
    try:
        train_gaussian_splat(target_image_path=args.image, num_points=args.num_points,
                             iterations=args.iterations, lr=args.lr, out_dir=args.out_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
