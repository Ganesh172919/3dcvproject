# ADR 0003: Extract a mesh from a signed distance field

## Decision

Convert the silhouette-constrained occupancy grid to a signed distance field and extract triangles with marching cubes. A mesh gives Studio an explicit surface to inspect and export.

## Consequences

Padding helps keep the zero crossing away from the grid boundary. The mesher measures closure after cleanup, including attempted repair of small holes. Check the resulting topology and scale before using an exported mesh for fabrication. Memory use grows with voxel resolution, so the pipeline estimates the grid size for the requested triangle budget.
