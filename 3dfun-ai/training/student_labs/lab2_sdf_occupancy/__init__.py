"""Lab 2: Implicit Neural Representation & Occupancy Network (MLP-SDF)."""
from ..lab2_occupancy_network.train_occupancy_sdf import (
    SinusoidalEncoding,
    OccupancyNetwork,
    sample_ground_truth_primitive,
    extract_mesh_marching_cubes,
    train_occupancy_network,
)

train_sdf_occupancy = train_occupancy_network

__all__ = [
    "SinusoidalEncoding",
    "OccupancyNetwork",
    "sample_ground_truth_primitive",
    "extract_mesh_marching_cubes",
    "train_occupancy_network",
    "train_sdf_occupancy",
]
