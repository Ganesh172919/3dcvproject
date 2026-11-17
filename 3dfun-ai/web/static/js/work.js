import { h, mount } from './lib.js';

const STAGES = [
  ['1. Foreground Matting & Saliency Boundary', 'Extract high-contrast alpha channel from raw 2D RGB image using U²-Net saliency detection. Automatically centres and normalizes the target object onto a padded square bounding frame.'],
  ['2. Monocular Metric Relative Depth Estimation', 'Infers continuous metric relative depth fields using Depth Anything V2 (ViT-S DINOv2 backbone) or MiDaS v2.1. Extracts spatial Sobel derivatives (∂d/∂x, ∂d/∂y) to compute surface normal orientation fields.'],
  ['3. Bilateral Edge Filtering & Symmetry Priors', 'Applies edge-preserving bilateral filtering to eliminate depth boundary bleeding. For bilateral symmetric objects, computes horizontal reflection depth averaging within the silhouette constraint.'],
  ['4. Camera Ray Backprojection into 3D Coordinates', 'Unprojects 2D pixel coordinates (u, v) into metric 3D camera coordinate space (X, Y, Z) using calibrated pinhole camera intrinsic matrix K: X = (u - c_x)*Z/f_x, Y = (v - c_y)*Z/f_y, Z = depth(u, v).'],
  ['5. Volumetric Inflation & Euclidean Distance Transform (TSDF)', 'Projects silhouette-bounded thickness along camera viewing rays into a 128³ to 256³ voxel density grid. Computes Signed Distance Fields (SDF) Φ(x) = min_{y ∈ ∂Ω} ||x - y|| * sgn(x).'],
  ['6. Marching Cubes Isosurface Extraction', 'Extracts triangular meshes at the zero-crossing isovalue using Lorensen & Cline lookup tables. Validates Euler characteristic (χ = V - E + F = 2) for spherical topological equivalence (genus g = 0).'],
  ['7. Taubin Non-Shrinking Dual-Pass Smoothing', 'Attenuates high-frequency discretization voxel noise using alternating positive (λ) and negative (μ) Laplacian passes satisfying 1/λ + 1/μ < 0, completely avoiding mesh shrinkage.'],
  ['8. UV Parameterization & Multi-Channel PBR Texture Baking', 'Generates 2D planar UV seam atlas and back-projects image color onto the 3D surface. Delights Lambertian shading and bakes Albedo, Tangent-Space Normal Maps, and Roughness channels into a production .GLB container.'],
];

const STARTUP_COMPARISONS = [
  {
    name: '3DFUN AI (Our Platform)',
    architecture: 'Hybrid DINOv2 + TSDF Voxel Grid + TripoSR INT8',
    runtime: 'Local ONNX (CPU / DirectML / CUDA)',
    latency: '12 – 45 s',
    cost: '50 Free Trial Credits + 10 Daily Streak (+0$ local offline)',
    transparency: '100% Transparent: 8-stage tensor inspector, 4 PyTorch student labs, open math',
  },
  {
    name: 'Tripo3D (VAST AI Research)',
    architecture: 'Large Reconstruction Model (LRM) Transformer',
    runtime: 'Cloud GPU Farm (H100/A100)',
    latency: '8 – 15 s',
    cost: 'Daily trial tokens, monthly subscription tier',
    transparency: 'Commercial SaaS API (Closed weights & cloud inference)',
  },
  {
    name: 'Meshy.ai',
    architecture: 'Multi-view Diffusion + Marching Tetrahedra (DMTet)',
    runtime: 'Cloud GPU Clusters',
    latency: '45 – 90 s',
    cost: '200 credits on signup, pay-per-model credit packs',
    transparency: 'Commercial SaaS API (Proprietary cloud pipeline)',
  },
  {
    name: 'Luma AI Genie',
    architecture: 'Generative 3D Diffusion + Gaussian Splatting (3DGS)',
    runtime: 'Cloud Compute Infrastructure',
    latency: '20 – 60 s',
    cost: 'Free tier sandbox with generation limits',
    transparency: 'Commercial SaaS API (Closed NeRF/Splat extraction)',
  },
  {
    name: 'Kaedim 3D',
    architecture: 'Neural Net Generation + Human-in-the-Loop Topology Refinement',
    runtime: 'Cloud Production Pipeline',
    latency: '15 – 30 min',
    cost: 'Enterprise subscription ($150-$1000/mo)',
    transparency: 'Commercial B2B Game Development Asset Service',
  },
  {
    name: 'Spline 3D',
    architecture: 'Parametric WebGL/WebGPU + AI Texture Generation',
    runtime: 'Browser Canvas + Cloud AI backend',
    latency: 'Interactive (<1s / 20s AI)',
    cost: 'Free tier with watermarks, Pro subscription ($24/mo)',
    transparency: 'Commercial Web-Native 3D Design Suite',
  },
  {
    name: 'Rodin (Deemos Tech)',
    architecture: 'Multi-View Diffusion + Micro-Geometry Sculpting',
    runtime: 'Cloud Supercomputing Clusters',
    latency: '60 – 180 s',
    cost: 'Monthly subscription & credit packs',
    transparency: 'Commercial High-Fidelity Character & Asset SaaS',
  },
];

const FIGURES = [
  ['Qualitative Comparison', 'fig_qualitative.png', 'The original images and reconstructions from the course pipeline, TSDF and visual hull.'],
  ['Surface Error Heatmaps', 'fig_error_heatmaps.png', 'Local distance to the reference surfaces for selected synthetic examples, shown in normalized units.'],
  ['Stage Timing Breakdown', 'fig_time_stages.png', 'Elapsed time for each stage in the saved CPU benchmark, including rendering.'],
  ['Chamfer Distance Quantitative Benchmark', 'fig_chamfer_bar.png', 'Nearest-neighbour surface distance on scenes with reference meshes. Smaller values indicate closer surface agreement.'],
];

function createRecommenderWidget() {
  const container = h('div', { class: 'recommender-widget mt20', style: 'background:rgba(99,102,241,0.04);border:1px solid var(--border);border-radius:12px;padding:18px;' });
  const options = [
    {
      id: 'gamedev',
      label: '🎮 Game Dev & Real-Time XR',
      rec: 'Meshy.ai / Kaedim 3D / 3DFUN AI (Decimated)',
      arch: 'Multi-view Diffusion + Quad Remesh / QEM Polycount Decimation',
      budget: 'Low poly (5k-15k faces), PBR Texture Maps (Albedo, Roughness, Normal)',
      detail: 'Game engines (Unity/Unreal) require strict triangle budgets and clean topology loops for character skinning and GPU rasterization.',
    },
    {
      id: 'printing',
      label: '🖨 3D Printing & Fabrication',
      rec: '3DFUN AI (This Platform) / Marching Cubes SDF',
      arch: 'Signed Distance Field (SDF) + Watertight Isosurface Extraction',
      budget: 'High density (20k-50k triangles), 100% Watertight (χ = 2, 0 open boundary loops)',
      detail: 'Slicers (PrusaSlicer, Cura) require strictly closed 2-manifold STL/OBJ geometry without self-intersecting facets or zero-thickness walls.',
    },
    {
      id: 'ecommerce',
      label: '🛍 E-Commerce & AR QuickLook',
      rec: 'Tripo3D / Luma Genie / 3DFUN AI glTF',
      arch: 'Feed-Forward Triplane Transformer / DINOv2 Hybrid with Texture Baking',
      budget: 'glTF 2.0 / USDZ (<10MB file size), PBR Metallic-Roughness workflow',
      detail: 'Mobile web browsers require fast asynchronous downloading, glTF DRACO compression, and iOS QuickLook USDZ compatibility.',
    },
    {
      id: 'cvresearch',
      label: '🎓 Academic Research & Student Labs',
      rec: '3DFUN AI Open Core (100% Recommended)',
      arch: 'Deterministic Hybrid Depth + Voxel Grid + 8-Stage Tensors + 4 PyTorch Labs',
      budget: '100% Free, zero telemetry leakage, inspectable intermediate matrices',
      detail: 'Commercial platforms are closed SaaS black boxes. 3DFUN AI demystifies every tensor transformation from raw image to textured manifold.',
    },
  ];

  let currentId = 'gamedev';
  const detailBox = h('div', { class: 'recommender-detail mt14 p14', style: 'background:var(--panel-2);border-radius:8px;border:1px solid var(--border);' });

  function update() {
    const opt = options.find(o => o.id === currentId);
    detailBox.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
        <span style="font-weight:700;color:var(--cyan);font-size:14px;">Optimal Architecture: ${opt.rec}</span>
        <span class="badge active" style="font-size:11px;">Recommended Workflow</span>
      </div>
      <div style="font-size:13px;line-height:1.6;margin-bottom:6px;"><strong>Core Engine:</strong> ${opt.arch}</div>
      <div style="font-size:13px;line-height:1.6;margin-bottom:6px;"><strong>Asset Budget:</strong> ${opt.budget}</div>
      <p class="muted mb0" style="font-size:12.5px;">${opt.detail}</p>
    `;
  }

  const buttons = h('div', { class: 'row wrap gap8' },
    options.map(opt => {
      const b = h('button', {
        class: `btn sm ${opt.id === currentId ? 'primary' : 'ghost'}`,
        type: 'button',
        onclick: () => {
          currentId = opt.id;
          buttons.querySelectorAll('button').forEach((btn, i) => {
            btn.className = `btn sm ${options[i].id === currentId ? 'primary' : 'ghost'}`;
          });
          update();
        }
      }, opt.label);
      return b;
    }));

  mount(container,
    h('div', { style: 'margin-bottom:12px;' },
      h('h3', { style: 'margin:0 0 4px 0;font-size:15px;color:var(--accent);' }, '⚡ Interactive Startup Architecture & Use-Case Recommender'),
      h('p', { class: 'muted mb0', style: 'font-size:13px;' }, 'Select your deployment target to see optimal representation, hardware budget, and topological requirements.')),
    buttons,
    detailBox);

  update();
  return container;
}

export function renderWork(main) {
  mount(main,
    h('header', { class: 'studio-title' },
      h('div', { class: 'row gap8 mb12', style: 'align-items:center' },
        h('span', { class: 'badge completed' }, 'Peer-Reviewed CV Architecture'),
        h('span', { class: 'badge queued' }, 'Course 17 · IIIT Dharwad'),
        h('span', { class: 'badge active' }, 'Startup Benchmark Standard')),
      h('h1', {}, '3D Reconstruction Architecture & Research Methodology'),
      h('p', { class: 'lead muted', style: 'max-width:80ch;' },
        '3DFUN AI synthesizes single-view RGB photographs into watertight, calibrated 3D meshes. Inspired by leading generative 3D startups (Tripo3D, Meshy, Luma, Kaedim), our platform provides both local execution and complete mathematical transparency across every tensor transformation.'),
      h('div', { class: 'row wrap gap8 mt16' },
        h('a', { class: 'btn primary', href: '/app/studio?sample=chair&engine=triposr', 'data-link': '' }, 'Launch Creative Studio'),
        h('a', { class: 'btn', href: '/app/showcase', 'data-link': '', style: 'color:var(--cyan);font-weight:600;' }, '✦ Showcase & Hub'),
        h('a', { class: 'btn', href: '/app/inspector', 'data-link': '' }, '🔬 8-Stage Pipeline Inspector'),
        h('a', { class: 'btn', href: '/app/labs', 'data-link': '' }, '🎓 Student Curriculum Labs'),
        h('a', { class: 'btn', href: '/docs/STARTUPS_COMPARATIVE_SURVEY.md', target: '_blank' }, '🏢 2026 Startups Survey'),
        h('a', { class: 'btn', href: '/docs/MATHEMATICAL_FOUNDATIONS_AND_PROOFS.md', target: '_blank' }, '📐 Math Proofs'),
        h('a', { class: 'btn', href: '/docs/STARTUP_LANDSCAPE_AND_BUSINESS_MODELS.md', target: '_blank' }, '💰 Unit Economics'),
        h('a', { class: 'btn', href: '/paper/main.pdf', target: '_blank' }, '📄 Download IEEE Paper (.pdf)'),
        h('a', { class: 'btn', href: '/report.pdf', target: '_blank' }, '📊 Download Project Report (.pdf)'))),

    // Real-World Startup Taxonomy & Comparison
    h('section', { class: 'card mt24' },
      h('h2', {}, 'Commercial 3D Startup Landscape & Architecture Taxonomy'),
      h('p', { class: 'muted' }, 'Generative 3D startups operate across three primary paradigms: Feed-Forward Transformers (Tripo3D), Multi-view Diffusion (Meshy), and Radiance Field Splats (Luma). 3DFUN AI uniquely combines local hybrid inference with educational transparency.'),
      h('div', { class: 'table-scroll mt16' },
        h('table', { class: 'tbl' },
          h('thead', {},
            h('tr', {},
              h('th', {}, 'Platform'),
              h('th', {}, 'Core Architecture'),
              h('th', {}, 'Runtime Environment'),
              h('th', {}, 'Draft Latency'),
              h('th', {}, 'Access / Pricing'),
              h('th', {}, 'Educational Openness'))),
          h('tbody', {},
            STARTUP_COMPARISONS.map(s => h('tr', {},
              h('td', {}, h('strong', {}, s.name)),
              h('td', {}, s.architecture),
              h('td', {}, s.runtime),
              h('td', {}, h('span', { class: 'mono' }, s.latency)),
              h('td', {}, s.cost),
              h('td', {}, s.transparency)))))),
      createRecommenderWidget()),

    // 8-Stage Breakdown
    h('h2', { class: 'mt32' }, 'Eight-Stage Image-to-3D Pipeline'),
    h('p', { class: 'muted' }, 'Every transformation in our reconstruction pipeline is deterministic, mathematically bounded, and inspectable in real time.'),
    h('ol', { class: 'col gap12 mt16', style: 'padding-left:0;list-style:none;' },
      STAGES.map(([title, description]) => h('li', { class: 'card tight' },
        h('h3', { style: 'color:var(--accent);font-size:15px;' }, title),
        h('p', { class: 'mb0', style: 'font-size:13.5px;line-height:1.6;' }, description)))),

    // Mathematical Foundations
    h('section', { class: 'card mt24' },
      h('h2', {}, 'Mathematical Formulations & Differential Geometry'),
      h('div', { class: 'grid c2 gap16 mt16' },
        h('div', { class: 'card tight', style: 'background:#07090e;' },
          h('h4', { style: 'color:var(--cyan);' }, 'Scale-Invariant Logarithmic (SILog) Loss'),
          h('pre', { class: 'mono', style: 'color:#a5b4fc;font-size:12px;padding:8px;' }, 'L_SILog = sqrt( (1/N) Σ d_i² - (λ/N²) (Σ d_i)² )'),
          h('p', { class: 'muted mb0', style: 'font-size:12px;' }, 'Penalizes relative geometric errors independent of global scale, preserving fine object silhouettes.')),
        h('div', { class: 'card tight', style: 'background:#07090e;' },
          h('h4', { style: 'color:var(--cyan);' }, 'Camera Ray Unprojection Matrix K'),
          h('pre', { class: 'mono', style: 'color:#a5b4fc;font-size:12px;padding:8px;' }, 'X = (u - c_x)*Z/f_x,  Y = (v - c_y)*Z/f_y,  Z = depth(u, v)'),
          h('p', { class: 'muted mb0', style: 'font-size:12px;' }, 'Back-projects 2D pixels into 3D metric camera space using pinhole camera intrinsic parameters.')),
        h('div', { class: 'card tight', style: 'background:#07090e;' },
          h('h4', { style: 'color:var(--cyan);' }, 'Continuous Signed Distance Field (SDF)'),
          h('pre', { class: 'mono', style: 'color:#a5b4fc;font-size:12px;padding:8px;' }, 'Φ(x) = min_{y ∈ ∂Ω} ||x - y|| * sgn(x)'),
          h('p', { class: 'muted mb0', style: 'font-size:12px;' }, 'Euclidean distance transform defining the 3D surface as the zero-level isosurface crossing.')),
        h('div', { class: 'card tight', style: 'background:#07090e;' },
          h('h4', { style: 'color:var(--cyan);' }, 'Taubin Dual-Pass Non-Shrinking Smoothing'),
          h('pre', { class: 'mono', style: 'color:#a5b4fc;font-size:12px;padding:8px;' }, "x' = x + λ L(x),  x'' = x' + μ L(x'),  (1/λ + 1/μ < 0)"),
          h('p', { class: 'muted mb0', style: 'font-size:12px;' }, 'Attenuates high-frequency voxel noise without contracting the overall mesh volume.')),
        h('div', { class: 'card tight', style: 'background:#07090e;' },
          h('h4', { style: 'color:var(--cyan);' }, 'Euler-Poincaré Characteristic (χ = V - E + F)'),
          h('pre', { class: 'mono', style: 'color:#a5b4fc;font-size:12px;padding:8px;' }, 'χ = V - E + F = 2(1 - g) - b'),
          h('p', { class: 'muted mb0', style: 'font-size:12px;' }, 'Topological invariant validating 2-manifold surfaces. For closed sphere-like meshes with genus g = 0 and boundary loops b = 0, χ = 2.')),
        h('div', { class: 'card tight', style: 'background:#07090e;' },
          h('h4', { style: 'color:var(--cyan);' }, 'Quadric Error Metric (QEM) Decimation'),
          h('pre', { class: 'mono', style: 'color:#a5b4fc;font-size:12px;padding:8px;' }, 'Δ(v) = vᵀ Q v = Σ_{p ∈ planes(v)} (pᵀ v)²'),
          h('p', { class: 'muted mb0', style: 'font-size:12px;' }, 'Calculates the sum of squared distances to incident planes, enabling extreme polygon reduction while preserving boundary creases.')))),

    // Benchmark Evidence
    h('h2', { class: 'mt32' }, 'Empirical Benchmark Evidence & Error Analysis'),
    h('p', { class: 'muted' }, 'CPU and DirectML benchmarks against ground-truth CAD reference surfaces show superior Chamfer distance compared to classical TSDF fusion and visual hull baselines.'),
    h('div', { class: 'grid c2 gap16 mt16' },
      FIGURES.map(([title, file, description]) => {
        const url = '/figures/' + file;
        return h('article', { class: 'card col gap8' },
          h('h3', { class: 'mb0', style: 'font-size:15px;' }, title),
          h('p', { class: 'muted mb0', style: 'font-size:12.5px;' }, description),
          h('img', { src: url, alt: title, loading: 'lazy', style: 'width:100%;height:auto;border-radius:8px;border:1px solid var(--border);' }),
          h('a', { class: 'btn sm ghost mt4', href: url, target: '_blank' }, 'View High-Resolution Figure'));
      })),

    h('div', { class: 'row wrap gap12 mt24' },
      h('a', { class: 'btn primary', href: '/app/benchmark', 'data-link': '' }, 'View Quantitative Benchmark Tables'),
      h('a', { class: 'btn', href: '/app/inspector', 'data-link': '' }, 'Inspect Intermediate Stage Tensors')));
}

