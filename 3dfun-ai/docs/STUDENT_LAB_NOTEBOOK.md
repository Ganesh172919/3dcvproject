# Lab notebook exercises

## Compare the engines in Studio

1. Open the teapot sample and run **Depth Anything V2** in Draft quality.
2. Run the same photo with **MiDaS Small**, keeping the other settings fixed. Both engines feed the same geometry pipeline.
3. Run the photo with **TripoSR INT8** to compare a learned 3D density field with the two depth-based results.
4. Rotate each mesh to the same angle. Record what happens to the handle, spout, and inferred back surface.
5. Open **Dashboard → Experiment notebook**, select the teapot, and export the CSV. Record the provider and settings alongside any observation about runtime.

## Read the course experiment

- [Dataset and protocol](BENCHMARK.md) explains the synthetic and photo splits. Reference geometry is available for the synthetic split.
- [Raw rows](../results/raw/runs.csv) record the per-run measurements behind the [summary table](../results/summary/main_table.csv).
- [Failure rows](../results/summary/failures.csv) identify cases to inspect. A completed job can still produce an open mesh or miss a geometric quality threshold.
- [Ablation results](../results/summary/ablation_table.csv) show how the measured pipeline responded to individual changes.

## Record a new observation

| Photo and quality | Engine and provider | Runtime | Triangles | Watertight? | What changed in the mesh? |
|---|---|---:|---:|---|---|
| | | | | | |
