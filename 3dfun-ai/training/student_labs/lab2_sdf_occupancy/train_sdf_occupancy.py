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

import os
import sys

# Re-export from occupancy_network implementation
from ..lab2_occupancy_network.train_occupancy_sdf import (
    SinusoidalEncoding,
    OccupancyNetwork,
    sample_ground_truth_primitive,
    extract_mesh_marching_cubes,
    train_occupancy_network,
)

train_sdf_occupancy = train_occupancy_network

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lab 2: Implicit MLP Occupancy & SDF Training")
    parser.add_argument("--primitive", choices=["torus", "sphere", "cylinder", "cube"], default="torus")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out-dir", type=str, default="./out/lab2_sdf")
    args = parser.parse_args()
    res = train_occupancy_network(primitive=args.primitive, epochs=args.epochs, lr=args.lr, out_dir=args.out_dir)
    print(f"Completed! Mesh saved at: {res['mesh_path']}")
