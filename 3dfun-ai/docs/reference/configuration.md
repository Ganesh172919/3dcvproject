# Reference: Configuration & Parameter Reference

This document provides a comprehensive reference for environment variables, reconstruction pipeline parameters, meshing budgets, and server operational settings.

---

## 1. Environment Variables

Variables can be defined in a `.env` file at the project root or passed via the runtime environment:

| Variable | Type | Default | Description |
|---|---|---|---|
| `APP_ENV` | string | `development` | Environment mode (`development`, `testing`, `production`). |
| `DATABASE_URL` | string | `sqlite:///data/3dfun.db` | SQLAlchemy database connection URI. |
| `SECRET_KEY` | string | *auto-generated* | HMAC secret used to sign JWT tokens. |
| `ONNX_THREADS` | integer | `4` | Number of intra-op threads allocated to ONNX Runtime sessions. |
| `ONNX_INTER_THREADS` | integer | `1` | Number of inter-op threads allocated to ONNX Runtime sessions. |
| `MAX_UPLOAD_MB` | integer | `25` | Maximum permissible file upload size in megabytes. |
| `CACHE_DIR` | path | `engine/models` | Local directory for cached ONNX weight graphs. |
| `DEFAULT_TRIANGLE_BUDGET` | integer | `8000` | Target triangle polygon count when omitted by caller. |

---

## 2. Volumetric Inflation Parameters

The geometric core inflates foreground silhouette pixels into volumetric solids according to:
$$t(p) = t_{\min} + (t_{\max} - t_{\min}) \cdot \left(f + g \cdot d(p)^\gamma\right)$$

| Parameter | Symbol | Default | Acceptable Range | Physical Meaning |
|---|---|---|---|---|
| `thickness_max` | $t_{\max}$ | `0.70` | $[0.20, 1.20]$ | Maximum volumetric depth relative to silhouette width $W$. |
| `thickness_min` | $t_{\min}$ | `0.02` | $[0.005, 0.10]$ | Minimum structural core thickness preventing zero-volume collapse. |
| `constant_core` | $f$ | `0.55` | $[0.30, 0.80]$ | Constant inflation fraction guaranteeing closed rim watertightness. |
| `depth_scale` | $g$ | `0.45` | $[0.20, 0.70]$ | Depth modulation scale carving front-facing surface relief. |
| `depth_gamma` | $\gamma$ | `0.85` | $[0.50, 1.50]$ | Non-linear gamma curve adjusting mid-tone relief slope. |
| `sdf_sigma` | $\sigma_{\text{sdf}}$ | `0.80` | $[0.40, 2.00]$ | Standard deviation (voxels) for 3D Gaussian field smoothing. |

---

## 3. Surface Extraction & Smoothing Settings

| Parameter | Default | Range | Description |
|---|---|---|---|
| `target_triangles` | `8000` | $[1000, 50000]$ | Desired mesh complexity used to calibrate Cartesian grid resolution. |
| `grid_min` | `32` | $[16, 64]$ | Minimum allowed grid dimension along shortest bounding axis. |
| `grid_max` | `224` | $[128, 512]$ | Maximum allowed grid dimension along longest bounding axis. |
| `taubin_iterations` | `12` | $[4, 30]$ | Number of dual-step Taubin non-shrinking smoothing iterations. |
| `taubin_lambda` | `0.50` | $(0.0, 1.0]$ | Positive forward contraction parameter. |
| `taubin_mu` | `-0.53` | $(-\infty, -\lambda)$ | Negative backward expansion parameter ($\mu < -\lambda < 0$). |

---
*Evidence: Configuration schemas in `engine/recon/config.py` and `app/core/config.py`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
