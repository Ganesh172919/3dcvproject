// 3DCV Fun AI Lab · Interactive Model & Website Architecture Mindmaps
// Group 17 · 3D Computer Vision Course Project · IIIT Dharwad
// Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
// Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
//          Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

import { h, mount, renderMathInDOM } from './lib.js';

export function renderMindmapsPage(main) {
  document.title = 'Model Architecture & Website Mindmaps · 3DCV Fun AI Lab · Group 17 · IIIT Dharwad';

  const container = h('div', { class: 'page-container', style: 'max-width: 1300px; margin: 0 auto; padding: 24px;' });

  // 1. Header Banner
  const header = h('header', { class: 'mb24' },
    h('div', { class: 'spread wrap gap12' },
      h('div', {},
        h('span', { class: 'badge', style: 'background: rgba(99, 102, 241, 0.1); color: var(--accent); border: 1px solid rgba(99, 102, 241, 0.2); margin-bottom: 6px; display: inline-block;' }, 'Interactive System Mindmaps & Tensor Topologies'),
        h('h1', { style: 'font-size: 26px; margin: 4px 0 8px 0;' }, '🧠 Model Architecture & Website Stack Mindmaps'),
        h('p', { class: 'muted mb0', style: 'font-size: 14px;' },
          'Visual interactive decomposition of the complete 3DCV Fun AI system: Web Application Stack, 3D Neural Pipeline, 8 Reconstruction Engines, and PyTorch Student Training Labs.')),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn sm', href: '/app/studio?sample=teapot&engine=local&quality=draft', 'data-link': '' }, '🎨 Open 3D Studio'),
        h('a', { class: 'btn sm', href: '/app/architecture', 'data-link': '' }, 'System Architecture →'),
        h('a', { class: 'btn sm', href: '/app/optimization', 'data-link': '' }, 'Performance Optimization →'),
        h('a', { class: 'btn sm', href: '/app/deployment', 'data-link': '' }, 'Model Deployment →'))));

  // 2. High-Level Summary Stat Cards
  const statGrid = h('div', { class: 'grid-4 gap16 mb24' },
    [
      ['8 Engines', 'Integrated 3D Reconstruction', 'Depth Anything, TripoSR, SIREN, 3DGS', '#6366f1'],
      ['8 Stages', 'Geometric Tensor Pipeline', 'Alpha Matting to PBR Texture Baking', '#06b6d4'],
      ['6 Labs', 'PyTorch Student Curriculum', 'SILog Depth, Occupancy, Splats, PBR', '#10b981'],
      ['< 1 ms', 'Zero-Copy Streaming Latency', 'Direct In-Memory RAM to WebGL Pipe', '#f59e0b']
    ].map(([val, title, desc, color]) =>
      h('div', { class: 'card tight', style: `background: #ffffff; border-top: 3px solid ${color};` },
        h('div', { style: `font-size: 24px; font-weight: 700; color: ${color}; line-height: 1.2;` }, val),
        h('div', { style: 'font-size: 13px; font-weight: 600; margin: 4px 0 2px 0;' }, title),
        h('div', { class: 'muted', style: 'font-size: 11px;' }, desc)))
  );

  // 3. Interactive Mindmap Canvas (Multi-Branch Visual Taxonomy)
  let activeBranch = 'all';
  const branchCardHost = h('div', { class: 'mt16' });

  const BRANCHES = {
    webstack: {
      title: 'Branch A: Client Web Application Stack (Pure White SPA & Three.js)',
      color: '#6366f1',
      items: [
        { name: 'Pure White Theme (#ffffff)', detail: 'High-contrast typography, accessible color palette, and zero dark-mode artifacts.' },
        { name: 'Three.js WebGL2 CAD Viewport', detail: 'Real-time rendering canvas at 60 FPS with OrbitControls and camera bookmarks.' },
        { name: '8 Diagnostic Shading Modes', detail: 'PBR Texturized, Wireframe-on-Shaded, White Clay, Red Wax, Chrome, Normals, Depth, Albedo.' },
        { name: 'Zero-Copy ArrayBuffer Receiver', detail: 'Direct binary stream consumption via Starlette chunked streaming without disk I/O.' },
        { name: '60 FPS Frame-Time Budget', detail: '16.6 ms budget allocation: 1.8ms JS physics, 1.2ms draw calls, 4.5ms vertex, 6.8ms fragment, 1.5ms post-fx.' }
      ]
    },
    backend: {
      title: 'Branch B: FastAPI ASGI Gateway & Security Layer',
      color: '#0284c7',
      items: [
        { name: 'FastAPI ASGI Framework', detail: 'Non-blocking asynchronous event loop handling uploads, streaming, and SSE updates.' },
        { name: 'Argon2id & JWT Authentication', detail: 'RFC 7519 tokens, zxcvbn password meter, and HttpOnly SameSite=Lax secure cookies.' },
        { name: 'Two-Phase Credit Ledger', detail: '50-credit free trial wallet with atomic reservation (balance -= 10) and settlement on success.' },
        { name: 'Failure Rollback Protection', detail: '100% automated refund (balance += 10, reserved -= 10) on reconstruction errors.' },
        { name: 'SQLite WAL Concurrency', detail: 'Write-Ahead Logging (PRAGMA journal_mode=WAL;) for concurrent non-blocking reads during writes.' }
      ]
    },
    pipeline: {
      title: 'Branch C: 8-Stage 3D Neural Reconstruction Pipeline',
      color: '#059669',
      items: [
        { name: 'Stage 1: U²-Net Alpha Matting', detail: 'Saliency boundary probability estimation mapping RGB to alpha cutoff in [0, 1].' },
        { name: 'Stage 2: Monocular Depth Prior', detail: 'Depth Anything V2 INT8 ONNX predicting metric inverse depth Z(u, v) and surface normal fields.' },
        { name: 'Stage 3: Metric Ray Unprojection', detail: 'Pinhole camera intrinsic matrix K unprojecting pixels into normalized 3D coordinates [-0.5, 0.5]³.' },
        { name: 'Stage 4: Silhouette Volumetric Inflation', detail: '128³ voxel Euclidean distance transform bounded by visual hull silhouette rays.' },
        { name: 'Stage 5: Marching Cubes Isosurface', detail: 'Watertight boundary extraction satisfying Euler invariant chi = V - E + F = 2 (genus g = 0).' },
        { name: 'Stage 6: Taubin Volume-Preserving Smoothing', detail: 'Dual-pass Laplacian filtering (lambda=0.5, mu=-0.53) removing stairsteps without shrinkage.' },
        { name: 'Stage 7: Conformal UV Parameterization', detail: 'Minimizing Dirichlet conformal energy E_D(phi) to eliminate texture stretching.' },
        { name: 'Stage 8: PBR Material Atlas Baking', detail: '1024² texture baking: Base Color, Normal Map, and packed ORM (Radiance Equation).' }
      ]
    },
    engines: {
      title: 'Branch D: 8 Integrated 3D Reconstruction Engines Matrix',
      color: '#d97706',
      items: [
        { name: '1. Depth Anything V2 (Local Default)', detail: 'INT8 quantized foundation depth estimator with metric back-projection (CPU/DirectML).' },
        { name: '2. TripoSR LRM (Learned Triplane)', detail: 'Fast feed-forward Large Reconstruction Model predicting continuous triplane density.' },
        { name: '3. MiDaS Small (Relative Depth)', detail: 'Multi-scale relative depth transformer with affine metric calibration.' },
        { name: '4. TinyDepth CNN (Student Lab 1)', detail: 'Lightweight student CNN trained with Scale-Invariant Logarithmic (SILog) loss.' },
        { name: '5. SIREN Implicit SDF (Student Lab 2)', detail: 'Periodic sinusoidal representation sin(omega_0 Wx + b) for continuous solids.' },
        { name: '6. 3D Gaussian Splatting (Student Lab 3)', detail: 'Anisotropic 3D ellipsoids (mu, Sigma, alpha, c) rasterized via tile sorting.' },
        { name: '7. Hybrid Triplane LRM (Student Lab 5)', detail: 'Three orthogonal planes (T_xy, T_xz, T_yz) with a coordinate query MLP.' },
        { name: '8. InstantMesh DiT (Student Lab 6)', detail: 'Multi-view diffusion with Deep Marching Tetrahedra (DMTet) 2-manifold extraction.' }
      ]
    },
    labs: {
      title: 'Branch E: 6 PyTorch Student Training Labs Curriculum',
      color: '#8b5cf6',
      items: [
        { name: 'Lab 1: Depth Fine-Tuning', detail: 'Scale-Invariant Logarithmic (SILog) loss + multiscale gradient supervision on TinyDepth.' },
        { name: 'Lab 2: Occupancy & SDF Networks', detail: 'Implicit coordinate MLP fitting continuous Signed Distance Fields with Eikonal loss.' },
        { name: 'Lab 3: 3D Gaussian Splatting Mini', detail: 'Single-view 3DGS optimization with spherical harmonics and tile-based rendering.' },
        { name: 'Lab 4: 3D Metric Evaluator Benchmark', detail: 'Chamfer Distance, Earth Mover\'s Distance (EMD), and F-Score at tau = 0.01 / 0.05.' },
        { name: 'Lab 5: Triplane LRM Generator', detail: 'Orthogonal feature plane generation from single-image tokens with cross-attention.' },
        { name: 'Lab 6: PBR De-Lighting & Atlas Baker', detail: 'Inverse rendering splitting photograph into Albedo, Normal, and Roughness/Metallic.' }
      ]
    }
  };

  function renderBranchCards() {
    branchCardHost.innerHTML = '';
    const branchesToRender = activeBranch === 'all'
      ? Object.entries(BRANCHES)
      : Object.entries(BRANCHES).filter(([k]) => k === activeBranch);

    const grid = h('div', { class: 'grid-2 gap16' });
    branchesToRender.forEach(([key, branch]) => {
      const card = h('div', { class: 'card p16 rounded', style: `background: #ffffff; border-top: 4px solid ${branch.color}; box-shadow: 0 2px 8px rgba(0,0,0,0.04);` },
        h('h3', { style: `color: ${branch.color}; margin: 0 0 12px 0; font-size: 15px;` }, branch.title),
        h('ul', { style: 'padding-left: 20px; font-size: 13px; line-height: 1.7; color: var(--text-2); margin: 0;' },
          branch.items.map(item =>
            h('li', { style: 'margin-bottom: 6px;' },
              h('strong', { style: 'color: var(--text);' }, `${item.name}: `),
              item.detail))));
      grid.appendChild(card);
    });
    branchCardHost.appendChild(grid);
  }

  // Branch Selector Tabs
  const branchTabs = h('div', { class: 'row wrap gap8 mb16' },
    [
      ['all', '🌟 View All Branches'],
      ['webstack', '🌐 Web Stack'],
      ['backend', '⚡ FastAPI & Ledger'],
      ['pipeline', '🔬 8-Stage 3D Pipeline'],
      ['engines', '🚀 8 Integrated Engines'],
      ['labs', '📚 6 Student Labs']
    ].map(([key, label]) => {
      const btn = h('button', { class: 'btn sm', type: 'button' }, label);
      btn.addEventListener('click', () => {
        branchTabs.querySelectorAll('button').forEach(b => b.classList.remove('primary'));
        btn.classList.add('primary');
        activeBranch = key;
        renderBranchCards();
      });
      if (key === activeBranch) btn.classList.add('primary');
      return btn;
    })
  );

  const mindmapSection = h('section', { class: 'card mb24', style: 'background: #ffffff;' },
    h('div', { class: 'spread' },
      h('h3', { style: 'color: var(--accent); margin: 0;' }, 'Visual Component Hierarchy & Mindmap Explorer'),
      h('span', { class: 'runtime-chip' }, '5 Interactive Architectural Pillars')),
    h('p', { class: 'muted mt8', style: 'font-size: 13px;' },
      'Select any architectural branch to inspect its design patterns, invariants, and implementation mechanics:'),
    branchTabs,
    branchCardHost
  );
  renderBranchCards();

  // 4. End-to-End Tensor Transformation Pipeline Flowchart
  const tensorSection = h('section', { class: 'card mb24', style: 'background: #ffffff;' },
    h('h3', { style: 'color: #0284c7; margin-top: 0;' }, '🔬 End-to-End Tensor Dimension Transformation Pipeline'),
    h('p', { class: 'muted', style: 'font-size: 13px;' },
      'The mathematical trajectory of data representations as a 2D photograph is converted into an interactive 3D WebGL mesh:'),
    h('div', { class: 'table-scroll mt12' },
      h('table', { class: 'tbl' },
        h('thead', {},
          h('tr', {},
            h('th', {}, 'Pipeline Step'),
            h('th', {}, 'Input Tensor / Representation'),
            h('th', {}, 'Operation / Kernel'),
            h('th', {}, 'Output Tensor / Representation'),
            h('th', {}, 'Mathematical Invariant'))),
        h('tbody', {},
          [
            ['1. Alpha Matting', 'RGB Photo [3, H, W]', 'U²-Net Saliency CNN', 'Alpha Matte [1, 512, 512]', 'L_BCE = -sum(M log alpha + (1-M) log(1-alpha))'],
            ['2. Depth Estimation', 'Foreground [3, 518, 518]', 'Depth Anything V2 INT8', 'Inverse Depth d in [0, 1]', 'Z(u,v) = 1 / (alpha * d + beta)'],
            ['3. Ray Unprojection', 'Depth Map Z [H, W]', 'Pinhole K^-1 Back-projection', 'Metric Point Cloud [N, 3]', 'X = (u - cx)*Z/fx, Y = (v - cy)*Z/fy'],
            ['4. Volumetric Inflation', 'Point Cloud [N, 3] + Mask', 'Euclidean Distance Transform', 'Voxel SDF Phi in R^(128^3)', 'Phi(x) < 0 <=> x in Volume'],
            ['5. Isosurface Extraction', 'Voxel SDF Phi in R^(128^3)', 'Marching Cubes 256 Lookup', 'Triangle Mesh (V, F)', 'Euler Invariant: chi = V - E + F = 2 (genus g = 0)'],
            ['6. Taubin Smoothing', 'Raw Triangles (V, F)', 'Dual-Pass Laplacian Filter', 'Smoothed Mesh (V, F)', '(I + mu L)(I + lambda L) preserves volume'],
            ['7. UV Parameterization', 'Manifold Surface M', 'Dirichlet Energy Minimizer', 'Planar UV Atlas [N, 2]', 'min E_D(phi) = 1/2 int ||grad phi||^2 dA'],
            ['8. PBR Texture Baking', 'UV Atlas + Original Photo', 'Camera Ray Back-Projection', 'PBR Atlas [1024, 1024, 4]', 'Radiance Equation L_o(x, omega_o)']
          ].map(([step, inp, op, out, inv]) =>
            h('tr', {},
              h('td', { style: 'font-weight: 600;' }, step),
              h('td', { class: 'mono', style: 'color: #0284c7; font-size: 11px;' }, inp),
              h('td', {}, op),
              h('td', { class: 'mono', style: 'color: #059669; font-size: 11px;' }, out),
              h('td', { style: 'font-size: 11px; color: var(--text-2); font-style: italic;' }, inv))))))
  );

  // 5. Two-Phase Credit Ledger State Machine Flow
  const ledgerSection = h('section', { class: 'card mb24', style: 'background: #ffffff;' },
    h('h3', { style: 'color: #059669; margin-top: 0;' }, '💳 Two-Phase Credit Ledger & Atomic State Machine'),
    h('p', { class: 'muted', style: 'font-size: 13px;' },
      'How the student credit wallet prevents double-charging and ensures 100% automated refund guarantees:'),
    h('div', { class: 'grid-3 gap12 mt12' },
      h('div', { class: 'p12 rounded', style: 'background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #0284c7;' },
        h('strong', { style: 'color: #0284c7; font-size: 13px; display: block; margin-bottom: 4px;' }, '1. Atomic Reservation Phase'),
        h('p', { style: 'font-size: 12px; color: var(--text-2); margin: 0; line-height: 1.6;' },
          'User clicks "Generate Model". FastAPI locks wallet row with SELECT ... FOR UPDATE. If balance >= 10, balance -= 10 and reserved += 10. Status set to QUEUED.')),
      h('div', { class: 'p12 rounded', style: 'background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #059669;' },
        h('strong', { style: 'color: #059669; font-size: 13px; display: block; margin-bottom: 4px;' }, '2. Success Settlement Phase'),
        h('p', { style: 'font-size: 12px; color: var(--text-2); margin: 0; line-height: 1.6;' },
          'Reconstruction pipeline completes watertight mesh extraction. Worker commits settled ledger transaction: reserved -= 10, lifetime_spent += 10. Status set to COMPLETED.')),
      h('div', { class: 'p12 rounded', style: 'background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #ef4444;' },
        h('strong', { style: 'color: #b91c1c; font-size: 13px; display: block; margin-bottom: 4px;' }, '3. Failure Rollback Phase'),
        h('p', { style: 'font-size: 12px; color: var(--text-2); margin: 0; line-height: 1.6;' },
          'On any unhandled exception or timeout (>300s), reaper task releases lock: balance += 10, reserved -= 10. Student is refunded 100% with error reason recorded.')))
  );

  // 6. Navigation Actions
  const footerNav = h('div', { class: 'card mt16', style: 'background: #f8fafc;' },
    h('div', { class: 'spread wrap gap12' },
      h('div', {},
        h('strong', {}, '3DCV Fun AI Lab · Course Project Documentation & Verification'),
        h('p', { class: 'muted mb0', style: 'font-size: 12px;' }, 'Group 17 · IIIT Dharwad · Department of Data Science and Artificial Intelligence')),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary sm', href: '/app/studio?sample=teapot&engine=local&quality=draft', 'data-link': '' }, '🎨 Launch 3D Studio'),
        h('a', { class: 'btn sm', href: '/app/labs', 'data-link': '' }, '📚 Student Labs'),
        h('a', { class: 'btn sm', href: '/app/inspector', 'data-link': '' }, '🔬 Pipeline Inspector'),
        h('a', { class: 'btn sm', href: '#', onclick: (e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); } }, '↑ Top'))));

  mount(container, header, statGrid, mindmapSection, tensorSection, ledgerSection, footerNav);
  renderMathInDOM(container);
  mount(main, container);
}
