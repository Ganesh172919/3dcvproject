"""Lab 2: Implicit Neural Representation & Occupancy Network (MLP-SDF).

Course: 3D Computer Vision · Student Educational Laboratory
Theoretical Foundation:
An implicit function represents geometry as the level set of a continuous neural field:
    f_theta(x, y, z) -> o in [0, 1]  (Occupancy)
    or
    Phi_theta(x, y, z) -> s in R     (Signed Distance Field)

Trained with Binary Cross-Entropy Loss:
    L_BCE = - sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
Extracts the zero-isosurface via Marching Cubes over an evaluation grid [-1, 1]^3.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from skimage import measure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("lab2.occupancy")


# ---------------------------------------------------------------------------
# Positional Encoding & Implicit MLP
# ---------------------------------------------------------------------------
class SinusoidalEncoding(nn.Module):
    """Fourier feature positional encoding: gamma(v) = [sin(2^k pi v), cos(2^k pi v)]."""
    def __init__(self, in_features: int = 3, num_frequencies: int = 6):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3) -> (B, 3 * num_frequencies * 2 + 3)
        b, c = x.shape
        x_proj = x.unsqueeze(-1) * self.freqs  # (B, 3, F)
        sin_feat = torch.sin(x_proj).view(b, -1)
        cos_feat = torch.cos(x_proj).view(b, -1)
        return torch.cat([x, sin_feat, cos_feat], dim=-1)


class OccupancyNetwork(nn.Module):
    """Implicit Neural Representation MLP mapping 3D coordinates to occupancy probability."""
    def __init__(self, hidden_dim: int = 128, num_layers: int = 5, num_frequencies: int = 6):
        super().__init__()
        self.encoding = SinusoidalEncoding(in_features=3, num_frequencies=num_frequencies)
        in_dim = 3 + 3 * num_frequencies * 2

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, 3) in [-1, 1]
        feat = self.encoding(coords)
        logits = self.mlp(feat)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Analytical Ground-Truth CAD Primitives for Training
# ---------------------------------------------------------------------------
def sample_ground_truth_primitive(primitive: str, num_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Generates 3D points inside bounding box [-1, 1]^3 with analytical occupancy labels."""
    pts = np.random.uniform(-1.0, 1.0, size=(num_points, 3)).astype(np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    occupancy = np.zeros(num_points, dtype=np.float32)

    if primitive == "sphere":
        # Sphere of radius 0.65
        r = np.sqrt(x**2 + y**2 + z**2)
        occupancy[r <= 0.65] = 1.0
    elif primitive == "cube":
        # Rounded cube
        m = (np.abs(x) <= 0.5) & (np.abs(y) <= 0.5) & (np.abs(z) <= 0.5)
        occupancy[m] = 1.0
    elif primitive == "cylinder":
        # Cylinder aligned with Y axis
        r = np.sqrt(x**2 + z**2)
        m = (r <= 0.5) & (np.abs(y) <= 0.6)
        occupancy[m] = 1.0
    elif primitive == "torus":
        # Torus with major radius R=0.55, minor radius r=0.25
        R, r_torus = 0.55, 0.25
        dist = np.sqrt((np.sqrt(x**2 + z**2) - R)**2 + y**2)
        occupancy[dist <= r_torus] = 1.0
    else:
        # Dual-sphere composite
        r1 = np.sqrt((x - 0.3)**2 + y**2 + z**2)
        r2 = np.sqrt((x + 0.3)**2 + y**2 + z**2)
        occupancy[(r1 <= 0.45) | (r2 <= 0.45)] = 1.0

    return pts, occupancy


# ---------------------------------------------------------------------------
# Marching Cubes Isosurface Extraction
# ---------------------------------------------------------------------------
def extract_implicit_mesh(model: nn.Module, res: int = 48, device: torch.device = torch.device("cpu")):
    model.eval()
    grid_coords = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    pts_eval = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    preds = []
    chunk_size = 32768
    with torch.no_grad():
        for i in range(0, len(pts_eval), chunk_size):
            chunk = torch.from_numpy(pts_eval[i:i + chunk_size]).to(device)
            occ = model(chunk).squeeze(-1).cpu().numpy()
            preds.append(occ)

    vol = np.concatenate(preds).reshape(res, res, res)
    try:
        spacing = 2.0 / (res - 1)
        verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5, spacing=(spacing, spacing, spacing))
        verts -= 1.0  # center from [0, 2] to [-1, 1]
        return verts, faces
    except Exception as e:
        log.warning("Marching cubes failed: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------
def train_occupancy_network(
    primitive: str = "torus",
    epochs: int = 15,
    hidden_dim: int = 128,
    lr: float = 2e-3,
    out_dir: str | Path = "out/student_labs/lab2",
    device_name: str = "cpu",
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")

    log.info("Starting Lab 2: Implicit Neural Representation (%s primitive) on %s", primitive, device)
    model = OccupancyNetwork(hidden_dim=hidden_dim, num_layers=4, num_frequencies=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Pre-generate point dataset
    pts, gt_occ = sample_ground_truth_primitive(primitive, num_points=30000)
    pts_t = torch.from_numpy(pts).to(device)
    gt_occ_t = torch.from_numpy(gt_occ[:, None]).to(device)

    batch_size = 2048
    num_batches = len(pts) // batch_size
    history = {"epochs": [], "loss": [], "accuracy": [], "intersection_over_union": []}

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(pts))
        ep_loss = []
        ep_acc = []

        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            b_pts = pts_t[idx]
            b_gt = gt_occ_t[idx]

            optimizer.zero_grad()
            pred = model(b_pts)
            loss = criterion(pred, b_gt)
            loss.backward()
            optimizer.step()

            ep_loss.append(loss.item())
            acc = ((pred > 0.5) == (b_gt > 0.5)).float().mean().item()
            ep_acc.append(acc)

        mean_loss = float(np.mean(ep_loss))
        mean_acc = float(np.mean(ep_acc))

        # Test IOU on validation points
        with torch.no_grad():
            val_pts, val_gt = sample_ground_truth_primitive(primitive, num_points=5000)
            val_pred = (model(torch.from_numpy(val_pts).to(device)).squeeze(-1).cpu().numpy() > 0.5)
            val_target = (val_gt > 0.5)
            intersection = np.logical_and(val_pred, val_target).sum()
            union = np.logical_or(val_pred, val_target).sum()
            iou = float(intersection / max(union, 1))

        history["epochs"].append(ep)
        history["loss"].append(round(mean_loss, 4))
        history["accuracy"].append(round(mean_acc, 4))
        history["intersection_over_union"].append(round(iou, 4))

        log.info("Epoch %d/%d — Loss: %.4f, Acc: %.2f%%, IoU: %.3f", ep, epochs, mean_loss, mean_acc * 100, iou)

    # Extract Isosurface Mesh
    verts, faces = extract_implicit_mesh(model, res=50, device=device)
    mesh_path = out_path / f"reconstructed_{primitive}.obj"
    if verts is not None and faces is not None:
        with open(mesh_path, "w", encoding="utf-8") as f:
            f.write(f"# Lab 2 Implicit Reconstructed {primitive}\n")
            for v in verts:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        log.info("Extracted 3D isosurface: %d vertices, %d faces -> %s", len(verts), len(faces), mesh_path)

    results = {
        "status": "completed",
        "primitive": primitive,
        "final_loss": history["loss"][-1],
        "final_accuracy": history["accuracy"][-1],
        "final_iou": history["intersection_over_union"][-1],
        "history": history,
        "mesh_path": str(mesh_path) if mesh_path.exists() else None,
        "vertices": len(verts) if verts is not None else 0,
        "faces": len(faces) if faces is not None else 0,
    }
    (out_path / "lab2_results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 2: Implicit Neural Representation & Occupancy Network")
    parser.add_argument("--primitive", type=str, default="torus", choices=["sphere", "cube", "cylinder", "torus"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--out-dir", type=str, default="out/student_labs/lab2")
    args = parser.parse_args()
    train_occupancy_network(primitive=args.primitive, epochs=args.epochs, out_dir=args.out_dir)
