# Lab 6: Physically Based Rendering (PBR) Material Decomposition & Delighting

## Academic Context
- **Course**: 3D Computer Vision (Group 17) · IIIT Dharwad
- **Supervisor**: Dr. Shirshendu Layek
- **Paradigm**: Physically Based Rendering (PBR) Texture Synthesis and Single-View Intrinsic Image Delighting

---

## 1. Pedagogical Objectives
1. Understand why naïve camera back-projection fails in real-world rendering due to lighting-baked artifacts.
2. Formulate intrinsic image decomposition: separating observed image radiance into intrinsic diffuse Albedo and low-frequency shading fields.
3. Compute tangent-space surface normal maps from spatial luminance and depth gradients.
4. Estimate microfacet roughness and metallic values to produce industry-standard ORM packed texture maps.

---

## 2. Mathematical Formulation

### 2.1 Cook-Torrance BRDF Model
$$f_r(\mathbf{x}, \omega_i, \omega_o) = k_d \frac{\mathbf{A}}{\pi} + k_s \frac{D(\mathbf{h}) F(\omega_o, \mathbf{h}) G(\omega_i, \omega_o, \mathbf{h})}{4 (\mathbf{n} \cdot \omega_i) (\mathbf{n} \cdot \omega_o)}$$

### 2.2 Delighting Filter
$$\mathbf{A}(u, v) = \text{clamp}\left(\frac{\mathbf{I}(u, v)}{\max(S(u, v), \epsilon)} \cdot \bar{S}, 0, 1\right)$$

### 2.3 Tangent-Space Normal Generation
$$\mathbf{N}(u, v) = \frac{(-\partial Z/\partial u, -\partial Z/\partial v, 1)^T}{\sqrt{(\partial Z/\partial u)^2 + (\partial Z/\partial v)^2 + 1}}$$

---

## 3. How to Run via CLI

```bash
python training/student_labs/lab6_pbr_delighting/pbr_material_decomposition.py
```
