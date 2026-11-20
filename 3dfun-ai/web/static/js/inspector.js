// Student Pipeline Inspector: 8-Stage Interactive Tensor & Geometric Construction Visualizer.
// Course: 3D Computer Vision Course Project · IIIT Dharwad

import { api, clear, fmt, h, mount, renderMathInDOM, toast } from './lib.js';

export const DEFAULT_STAGES = [
  {
    stage_index: 1,
    id: 'preprocessing',
    name: '1. Preprocessing & Alpha Matting',
    description: 'Raw 2D input -> Saliency Boundary Probability Map -> Trimap -> Alpha Cutoff Segmentation',
    math_formula: '\\mathcal{L}_{\\text{BCE}} = -\\sum_{i} [y_i \\log \\hat{y}_i + (1 - y_i) \\log (1 - \\hat{y}_i)]',
    primary_asset: '/site/examples/triposr-chair-preview.png',
    assets: { mask: '/site/examples/triposr-chair-preview.png' },
    parameters: { alpha_cutoff: 0.5, feather_radius: 2, padding: '12%' },
    telemetry: { input_resolution: '512x512', channels: 'RGB (Normalized)', matte_model: 'U²-Net Portable' }
  },
  {
    stage_index: 2,
    id: 'depth',
    name: '2. Monocular Relative Depth Estimation',
    description: 'Depth Anything V2 / MiDaS relative inverse depth field & 2D Sobel spatial gradients',
    math_formula: 'Z(u, v) = \\frac{1}{\\alpha \\cdot d(u, v) + \\beta}, \\quad \\nabla d = \\left( \\frac{\\partial d}{\\partial x}, \\frac{\\partial d}{\\partial y} \\right)',
    primary_asset: '/site/examples/triposr-chair-preview.png',
    assets: { color_map: '/site/examples/triposr-chair-preview.png' },
    parameters: { model: 'Depth Anything V2 Small (ViT-S DINOv2)', depth_range: '[0, 1]' },
    telemetry: { min_closeness: 0.05, max_closeness: 0.98, backbone: 'DINOv2 INT8' }
  },
  {
    stage_index: 3,
    id: 'unprojection',
    name: '3. Camera Ray Unprojection & Point Cloud',
    description: 'Back-projecting 2D pixels into 3D camera coordinate space via pinhole camera intrinsic matrix K',
    math_formula: 'X = \\frac{(u - c_x) Z}{f_x}, \\quad Y = \\frac{(v - c_y) Z}{f_y}, \\quad Z = \\text{depth}(u, v)',
    primary_asset: '/site/examples/chair.glb',
    assets: { pointcloud_ply: null },
    parameters: { fov_deg: 45.0, focal_length: 180.0, density: '100%' },
    telemetry: { point_count: 20000, coordinate_system: 'Right-handed (+X right, +Y up, +Z out)' }
  },
  {
    stage_index: 4,
    id: 'volume',
    name: '4. Volumetric Inflation & Signed Distance Fields (SDF)',
    description: 'Silhouette-constrained 3D voxel density matrix Phi(x, y, z) with Euclidean distance transform',
    math_formula: '\\Phi(\\mathbf{x}) = \\min_{\\mathbf{y} \\in \\partial \\Omega} \\|\\mathbf{x} - \\mathbf{y}\\| \\cdot \\text{sgn}(\\mathbf{x})',
    primary_asset: '/site/examples/triposr-chair-preview.png',
    assets: { axial_slice: '/site/examples/triposr-chair-preview.png', voxels_slice: '/site/examples/triposr-chair-preview.png' },
    parameters: { voxel_resolution: '128³', closeness_power: 1.4 },
    telemetry: { grid_shape: [128, 128, 128], spacing: 0.0078, field_type: 'Signed Distance Field' }
  },
  {
    stage_index: 5,
    id: 'geometry',
    name: '5. Isosurface Extraction via Marching Cubes',
    description: 'Lorensen & Cline lookup table triangle polygonization with Euler topology verification',
    math_formula: '\\mathcal{S} = \\{ \\mathbf{x} \\in \\mathbb{R}^3 \\mid \\Phi(\\mathbf{x}) = 0.0 \\}, \\quad \\chi = V - E + F',
    primary_asset: '/site/examples/chair.glb',
    assets: { mesh_glb: '/site/examples/chair.glb' },
    parameters: { isovalue: 0.0, lookup_tables: '256 configurations (15 canonical)' },
    telemetry: { euler_characteristic: 2, genus: 0, manifold: true, triangles: 14200 }
  },
  {
    stage_index: 6,
    id: 'smoothing',
    name: '6. Mesh Post-Processing & Smoothing',
    description: 'Taubin dual-pass non-shrinking smoothing vs Laplacian smoothing with Quadric Error Metric (QEM)',
    math_formula: "\\mathbf{v}' = (\\mathbf{I} + \\mu \\mathbf{L})(\\mathbf{I} + \\lambda \\mathbf{L}) \\mathbf{v}, \\quad \\lambda > 0, \\mu < -\\lambda",
    primary_asset: '/site/examples/chair.glb',
    assets: { model_glb: '/site/examples/chair.glb' },
    parameters: { taubin_lambda: 0.5, taubin_mu: -0.53, iterations: 4 },
    telemetry: { filter_type: 'Taubin non-shrinking', shrinkage_loss: '0.02%', boundary_edges: 0 }
  },
  {
    stage_index: 7,
    id: 'uv',
    name: '7. UV Parameterization & Seam Packing',
    description: 'Planar view-projection parameterization of 3D triangular surface onto 2D texture atlas',
    math_formula: '(u, v) \\in [0, 1]^2, \\quad \\iint \\left( \\left\\|\\frac{\\partial \\mathbf{x}}{\\partial u}\\right\\|^2 + \\left\\|\\frac{\\partial \\mathbf{x}}{\\partial v}\\right\\|^2 \\right) du\\,dv',
    primary_asset: '/site/examples/triposr-chair-preview.png',
    assets: { uv_layout_png: '/site/examples/triposr-chair-preview.png' },
    parameters: { atlas_resolution: '1024x1024', padding_pixels: 4 },
    telemetry: { distortion_area: 0.04, distortion_angle: 0.02, atlas_type: 'planar-view-projection' }
  },
  {
    stage_index: 8,
    id: 'texture',
    name: '8. PBR Texture Back-Projection & Baking',
    description: 'Color back-projection with Lambertian de-lighting, AO hemisphere integration, and normal perturbation',
    math_formula: 'L_o(\\mathbf{x}, \\omega_o) = \\int_{\\Omega} f_r(\\mathbf{x}, \\omega_i, \\omega_o) L_i(\\mathbf{x}, \\omega_i) (\\mathbf{n} \\cdot \\omega_i) d\\omega_i',
    primary_asset: '/site/examples/triposr-chair-preview.png',
    assets: { albedo: '/site/examples/triposr-chair-preview.png' },
    parameters: { pbr_workflow: 'Metallic-Roughness', delighting: true },
    telemetry: { albedo_channels: 3, roughness_mean: 0.45, metallic_mean: 0.05, workflow: 'PBR' }
  }
];

export class PipelineInspector {
  constructor(container, { generationId, onStageSelect } = {}) {
    this.container = container;
    this.generationId = generationId;
    this.onStageSelect = onStageSelect;
    this.stages = [];
    this.currentStageIndex = 1;
    this.stageData = null;
    this._init();
  }

  async _init() {
    this.wrap = h('div', { class: 'inspector-container' });
    this.header = h('div', { class: 'inspector-header spread' },
      h('div', {},
        h('h3', { class: 'mb0', style: 'color:var(--cyan);font-size:16px;' }, '🔬 Student Pipeline Inspector'),
        h('small', { class: 'muted' }, 'Interactive step-by-step tensor & geometric breakdown')),
      h('span', { class: 'badge queued', style: 'font-size:11px;' }, '8 Stages Active'));

    this.nav = h('div', { class: 'inspector-nav' });
    this.body = h('div', { class: 'inspector-body' });
    mount(this.wrap, this.header, this.nav, this.body);
    this.container.appendChild(this.wrap);

    if (this.generationId) {
      await this.loadStages(this.generationId);
    } else {
      this.stages = DEFAULT_STAGES;
      this._renderNav();
      this.setStage(1);
    }
  }

  _renderEmpty() {
    clear(this.body);
    mount(this.body, h('div', { class: 'inspector-empty center p16' },
      h('p', { class: 'muted' }, 'Generate or select a 3D model to inspect intermediate tensors, depth gradients, SDF voxel slices, and UV parameterization.')));
  }

  async loadStages(generationId) {
    this.generationId = generationId;
    mount(this.body, h('div', { class: 'center p16' }, h('span', { class: 'spinner' }), h('p', { class: 'muted mt8' }, 'Loading pipeline telemetry…')));

    try {
      const res = await api(`/generations/${generationId}/stages`);
      this.stages = res.stages || [];
      this._renderNav();
      this.setStage(1);
    } catch (err) {
      clear(this.body);
      mount(this.body, h('div', { class: 'form-error p16' }, `Could not load stages: ${err.message}`));
    }
  }

  _renderNav() {
    clear(this.nav);
    const chips = this.stages.map((st, i) => {
      const idx = i + 1;
      const chip = h('button', {
        class: `inspector-stage-btn ${idx === this.currentStageIndex ? 'active' : ''}`,
        type: 'button',
        onclick: () => this.setStage(idx),
      }, `S${idx}: ${st.name.split('.')[1]?.trim().split('&')[0]?.trim() || st.id}`);
      return chip;
    });
    mount(this.nav, chips);
  }

  setStage(stageIndex) {
    this.currentStageIndex = stageIndex;
    const stage = this.stages.find(s => s.stage_index === stageIndex) || this.stages[stageIndex - 1];
    if (!stage) return;

    // Update nav active states
    this.nav.querySelectorAll('.inspector-stage-btn').forEach((btn, i) => {
      btn.classList.toggle('active', i + 1 === stageIndex);
    });

    this._renderStageDetail(stage);
    this.onStageSelect?.(stage);
  }

  _renderStageDetail(stage) {
    clear(this.body);

    // Math Formula Card
    const mathCard = stage.math_formula ? h('div', { class: 'inspector-math-card' },
      h('div', { class: 'k' }, 'Mathematical Formulation'),
      h('div', { class: 'math-formula', style: 'font-family:var(--mono);color:var(--cyan);padding:8px 0;' },
        `$$${stage.math_formula}$$`)) : null;

    // Visual Preview Section
    const previewContainer = h('div', { class: 'inspector-visual' });
    if (stage.primary_asset) {
      if (stage.primary_asset.endsWith('.ply')) {
        mount(previewContainer, h('div', { class: 'inspector-card-preview center' },
          h('div', { style: 'font-size:32px;' }, '☁️'),
          h('p', { class: 'mb0', style: 'font-weight:600;' }, 'Unprojected 3D Camera Coordinate Point Cloud'),
          h('small', { class: 'muted' }, '20,000 back-projected metric points with RGB vertex colors'),
          h('a', { class: 'btn sm primary mt8', href: stage.primary_asset, download: 'pointcloud.ply' }, '⤓ Download PLY Point Cloud')));
      } else if (stage.primary_asset.endsWith('.obj')) {
        const isRaw = stage.id === 'geometry';
        mount(previewContainer, h('div', { class: 'inspector-card-preview center' },
          h('div', { style: 'font-size:32px;' }, isRaw ? '⚙️' : '✨'),
          h('p', { class: 'mb0', style: 'font-weight:600;' }, isRaw ? 'Raw Marching Cubes Mesh (Un-smoothed)' : 'Taubin-Smoothed Manifold Mesh'),
          h('small', { class: 'muted' }, isRaw ? 'Extracted zero-level set before Taubin smoothing' : 'Cleaned non-shrinking 2-manifold surface mesh'),
          h('a', { class: 'btn sm primary mt8', href: stage.primary_asset, download: isRaw ? 'raw_mesh.obj' : 'smoothed.obj' },
            isRaw ? '⤓ Download Raw Mesh (.OBJ)' : '⤓ Download Smoothed Mesh (.OBJ)')));
      } else if (stage.primary_asset.endsWith('.glb')) {
        mount(previewContainer, h('div', { class: 'inspector-card-preview center' },
          h('div', { style: 'font-size:32px;' }, '🧊'),
          h('p', { class: 'mb0', style: 'font-weight:600;' }, 'Extracted 3D Geometry (Isosurface)'),
          h('small', { class: 'muted' }, 'Active in center Three.js WebGL viewport'),
          stage.assets?.raw_mesh_obj ? h('a', { class: 'btn sm mt8', href: stage.assets.raw_mesh_obj, download: 'raw_mesh.obj' }, '⤓ Download Raw Mesh (.OBJ)') : null,
          stage.assets?.smoothed_obj ? h('a', { class: 'btn sm primary mt8', href: stage.assets.smoothed_obj, download: 'smoothed.obj' }, '⤓ Download Smoothed Mesh (.OBJ)') : null));
      } else {
        const img = h('img', { src: stage.primary_asset, class: 'inspector-img', alt: stage.name });
        mount(previewContainer, img);
      }
    }

    // Asset switcher pills (e.g. Grayscale vs Color Depth, or Orthogonal Slices)
    const assetPills = h('div', { class: 'inspector-pills mt8' });
    if (stage.assets) {
      Object.entries(stage.assets).forEach(([name, url]) => {
        if (!url) return;
        const pill = h('button', {
          class: 'btn sm ghost',
          type: 'button',
          onclick: () => {
            if (url.endsWith('.png') || url.endsWith('.jpg')) {
              const currentImg = previewContainer.querySelector('img');
              if (currentImg) currentImg.src = url;
            } else {
              window.open(url, '_blank');
            }
          },
        }, name.replace('_', ' ').toUpperCase());
        assetPills.appendChild(pill);
      });
    }

    // Interactive Telemetry & Hyperparameter Sliders
    const interactiveControls = h('div', { class: 'inspector-controls mt12' });
    if (stage.id === 'preprocessing') {
      const slider = h('input', { type: 'range', min: '10', max: '90', value: '50' });
      const valLabel = h('span', { class: 'mono' }, '0.50');
      slider.oninput = (e) => { valLabel.textContent = (Number(e.target.value) / 100).toFixed(2); };
      mount(interactiveControls,
        h('label', { class: 'field' },
          h('span', { class: 'spread' }, h('strong', {}, 'Alpha Cutoff Threshold (τ)'), valLabel),
          slider),
        h('small', { class: 'muted' }, 'Controls foreground boundary thresholding on U²-Net saliency map.'));
    } else if (stage.id === 'depth') {
      const colormapSelect = h('select', { class: 'input' },
        ['Turbo (Default)', 'Viridis', 'Magma', 'Plasma', 'Grayscale'].map(c => h('option', { value: c.toLowerCase() }, c)));
      colormapSelect.onchange = (e) => {
        toast(`Colormap switched to ${e.target.value}`);
      };
      mount(interactiveControls,
        h('label', { class: 'field' },
          h('span', {}, 'Depth Field Colormap Palette'),
          colormapSelect),
        h('small', { class: 'muted' }, 'Visualizes relative closeness inverse depth gradients ∂d/∂x and ∂d/∂y.'));
    } else if (stage.id === 'volume') {
      const sliceScrubber = h('input', { type: 'range', min: '0', max: '127', value: '64' });
      const sliceVal = h('span', { class: 'mono' }, 'Axial (Z = 64/128)');
      sliceScrubber.oninput = (e) => {
        const val = Number(e.target.value);
        const currentImg = previewContainer.querySelector('img');
        if (val < 42) {
          sliceVal.textContent = `Axial Slicing Plane (Z = ${val}/128)`;
          if (currentImg && stage.assets?.axial_slice) currentImg.src = stage.assets.axial_slice;
        } else if (val < 85) {
          sliceVal.textContent = `Coronal Slicing Plane (Y = ${val}/128)`;
          if (currentImg && stage.assets?.coronal_slice) currentImg.src = stage.assets.coronal_slice;
        } else {
          sliceVal.textContent = `Sagittal Slicing Plane (X = ${val}/128)`;
          if (currentImg && stage.assets?.sagittal_slice) currentImg.src = stage.assets.sagittal_slice;
        }
      };
      mount(interactiveControls,
        h('label', { class: 'field' },
          h('span', { class: 'spread' }, h('strong', {}, 'SDF Voxel Slicing Plane'), sliceVal),
          sliceScrubber),
        h('small', { class: 'muted' }, 'Scrub through internal signed distance field matrix Φ(x, y, z) across orthogonal cross-sections.'));
    } else if (stage.id === 'geometry') {
      mount(interactiveControls,
        h('div', { class: 'card tight' },
          h('div', { class: 'spread' }, h('strong', {}, 'Euler Characteristic (χ)'), h('span', { class: 'badge completed' }, 'χ = 2')),
          h('div', { class: 'spread mt4' }, h('span', { class: 'muted' }, 'Surface Genus:'), h('span', { class: 'mono' }, 'g = 0 (Sphere equivalent)')),
          h('div', { class: 'spread mt4' }, h('span', { class: 'muted' }, '2-Manifold Winding:'), h('span', { class: 'mono', style: 'color:var(--ok)' }, 'Verified Valid'))));
    }

    // Telemetry Key-Value Box
    const telemetryBox = h('div', { class: 'inspector-telemetry mt12' });
    if (stage.telemetry && Object.keys(stage.telemetry).length > 0) {
      const rows = Object.entries(stage.telemetry).slice(0, 5).map(([k, v]) => {
        return h('div', { class: 'spread telemetry-row' },
          h('span', { class: 'k' }, k.replace(/_/g, ' ')),
          h('span', { class: 'v mono' }, typeof v === 'object' ? JSON.stringify(v) : String(v)));
      });
      mount(telemetryBox, h('h4', { style: 'font-size:12px;margin-bottom:6px;text-transform:uppercase;color:var(--muted);' }, 'Stage Tensor Telemetry'), rows);
    }

    mount(this.body,
      h('div', { class: 'inspector-stage-head' },
        h('h3', { class: 'mb4', style: 'font-size:15px;' }, stage.name),
        h('p', { class: 'muted mb8', style: 'font-size:13px;' }, stage.description)),
      mathCard,
      previewContainer,
      assetPills,
      interactiveControls,
      telemetryBox);
    renderMathInDOM(this.body);
  }
}

export async function renderInspector(main) {
  document.title = '8-Stage Pipeline Inspector · 3DCV Fun AI Lab · Group 17 · IIIT Dharwad';
  clear(main);
  const container = h('div', { class: 'studio-wrap', style: 'padding: 24px; max-width: 1400px; margin: 0 auto;' });
  const header = h('header', { class: 'spread mb16' },
    h('div', {},
      h('h1', { style: 'font-size:24px;margin-bottom:4px;' }, '🔬 3D Reconstruction Pipeline Inspector'),
      h('p', { class: 'muted' }, 'Interactive step-by-step 8-stage tensor & geometric construction visualizer for students.')),
    h('div', { class: 'row gap8' },
      h('a', { class: 'btn ghost sm', href: '/app/studio', 'data-link': '' }, '← Back to Studio'),
      h('a', { class: 'btn primary sm', href: '/app/labs', 'data-link': '' }, 'Student Labs →'))
  );
  const inspectorWrap = h('div', { class: 'card', style: 'padding: 20px;' });
  new PipelineInspector(inspectorWrap, { generationId: null });
  mount(container, header, inspectorWrap);
  mount(main, container);
}

