# Benchmark hardware and recorded timings

The course benchmark ran on the Linux host described in [environment.json](../results/environment.json). The record lists a two-core Intel Xeon at 2.60 GHz, roughly 2 GB of RAM, Python 3.13.16, and CPU inference.

The benchmark contains 25 scenes and three repetitions for each method. These end-to-end means and standard deviations come from [main_table.csv](../results/summary/main_table.csv).

| Method | Runs | Time per run |
|---|---:|---:|
| Course pipeline | 75 | 27.73 ± 5.26 s |
| TSDF fusion | 75 | 1.43 ± 0.24 s |
| Visual hull | 75 | 2.13 ± 0.32 s |

The reported runtime includes each method's configured output steps. See [stage timings](../results/summary/stage_timings.csv) and the [benchmark protocol](BENCHMARK.md) when comparing speed.

For a new hardware comparison, keep the image, quality, preview settings, and export formats fixed. Repeat the run, record the inference provider and machine details, and export the measurements from the Experiment notebook. The reported course benchmark covers the recorded Linux CPU host.
