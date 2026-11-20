// 3D Computer Vision Lab Studio Controller · Group 17 · IIIT Dharwad
// Course: 3D Computer Vision Course Project · IIIT Dharwad
// Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
// Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
//          Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

import { api, clear, fmt, h, mount, refreshUser, state, statusBadge, toast } from './lib.js';
import { EXAMPLES } from './examples.js';
import { ENGINE_NAMES, inferenceLabel, PROVIDER_NAMES } from './experiments.js';
import { discardReference, pendingReference, validateReferenceFile } from './pending-upload.js';
import { PipelineInspector } from './inspector.js';
import { renderChat } from './chat.js';

const DEPTH_STAGES = ['Matte', 'Depth', 'Volume', 'Mesh', 'Texture', 'Export'];
const ENGINES = {
  triposr: {
    name: ENGINE_NAMES.triposr,
    example: 'chair',
    stages: ['Matte', 'Encode', 'Density', 'Mesh', 'Colour', 'Export'],
    description: 'Fast feed-forward Large Reconstruction Model (LRM) predicting continuous triplane density fields.'
  },
  local: {
    name: ENGINE_NAMES.local,
    example: 'sneaker',
    stages: DEPTH_STAGES,
    description: 'Foundation monocular depth estimator with metric back-projection, silhouette voxel carving, and 1024² PBR atlas.'
  },
  midas: {
    name: ENGINE_NAMES.midas,
    example: 'teapot',
    stages: DEPTH_STAGES,
    description: 'Multi-scale relative depth transformer with calibrated affine metric scaling.'
  },
  tinydepth: {
    name: ENGINE_NAMES.tinydepth,
    example: 'sneaker',
    stages: ['Matte', 'Depth CNN', 'Unproject', 'Mesh', 'Bake', 'Export'],
    description: 'Student lightweight feed-forward depth CNN from Lab 1. Ultra-fast CPU inference with silhouette bounds.'
  },
  siren_sdf: {
    name: ENGINE_NAMES.siren_sdf,
    example: 'teapot',
    stages: ['Matte', 'SIREN Fit', 'Implicit Field', 'Marching Cubes', 'Bake', 'Export'],
    description: 'Student Lab 2 implicit periodic representation with sinusoidal activation functions sin(ω₀ Wx + b).'
  },
  gaussian_splat: {
    name: ENGINE_NAMES.gaussian_splat,
    example: 'chair',
    stages: ['Matte', 'SfM Init', 'Splat Densify', 'Tile Rasterize', 'Mesh Extract', 'Export'],
    description: 'Student Lab 3 3D Gaussian Splatting representation using anisotropic 3D ellipsoids and spherical harmonics.'
  },
  hybrid_triplane: {
    name: ENGINE_NAMES.hybrid_triplane,
    example: 'chair',
    stages: ['Matte', 'Triplane Project', 'MLP Query', 'Marching Cubes', 'PBR Bake', 'Export'],
    description: 'Student Lab 5 hybrid representation: 3 orthogonal planes (T_xy, T_xz, T_yz) with a lightweight coordinate MLP.'
  },
  instantmesh_dit: {
    name: ENGINE_NAMES.instantmesh_dit,
    example: 'sneaker',
    stages: ['Multi-View DiT', 'Sparse Views', 'DMTet Grid', 'SDF Extract', 'Texture Bake', 'Export'],
    description: 'State-of-the-art sparse-view diffusion transformer + deformable tetrahedral grid (DMTet) isosurface extraction.'
  }
};
const LABELS = {
  QUEUED: 'Waiting in queue', VALIDATING: 'Checking the image', PREPROCESSING: 'Preparing the image',
  VOLUME: 'Sampling the volume', GEOMETRY: 'Extracting the mesh', UV: 'Unwrapping the mesh',
  TEXTURE: 'Applying colour', EXPORT: 'Writing model files', RENDER: 'Rendering previews',
  COMPLETED: 'Model ready', FAILED: 'Generation failed', CANCELLED: 'Generation cancelled',
  CANCEL_REQUESTED: 'Waiting for the current step to finish',
};
const TERMINAL = ['COMPLETED', 'FAILED', 'CANCELLED'];
export async function renderStudio(_params, main) {
  document.title = 'Studio · 3DCV Fun AI Lab · Group 17 · IIIT Dharwad';
  const available = key => key === 'local' || Boolean(state.meta?.features?.[key]);
  const params = new URLSearchParams(location.search);
  const requestedEngine = params.get('engine');
  const initialSample = EXAMPLES.find(example => example.id === params.get('sample'));
  const maxMB = state.meta?.limits?.max_upload_mb ?? 20;
  let upload;
  let current;
  let estimate;
  let estimateVersion = 0;
  let uploading = false;
  let starting = false;
  let changingJob = false;
  let pollTimer;
  let active = true;
  let viewer;
  state.routeCleanup = () => { active = false; clearTimeout(pollTimer); viewer?.dispose(); };

  const fileInput = h('input', { type: 'file', accept: 'image/png,image/jpeg,image/webp,image/bmp,.bmp', class: 'hidden' });
  const drop = h('button', { type: 'button', class: 'drop upload-drop', onclick: () => fileInput.click() },
    h('span', { class: 'upload-mark', 'aria-hidden': 'true' }, '↑'),
    h('strong', {}, 'Choose an image'),
    h('small', {}, `Drop it here or browse · up to ${maxMB} MB`));
  const reference = h('div', { class: 'reference' });
  const error = h('p', { class: 'form-error mb0', role: 'alert' });
  const sampleInputs = EXAMPLES.filter((example, index) =>
    EXAMPLES.findIndex(item => item.input === example.input) === index);
  const samples = sampleInputs.map(example => h('button', { class: 'sample-button', type: 'button',
    onclick: () => selectExample(example), 'aria-pressed': 'false', dataset: { filename: example.input.split('/').pop() },
  }, h('img', { src: example.input, alt: '', loading: 'lazy' }), example.label));

  function selectExample(example) {
    return selectFile(async () => {
      const response = await fetch(example.input);
      if (!response.ok) throw new Error('The example could not be loaded. Try uploading a photo.');
      return new File([await response.blob()], example.input.split('/').pop(), { type: 'image/png' });
    });
  }

  const engine = h('select', { 'aria-labelledby': 'studio-engine-label', onchange: () => updateEstimate() },
    Object.entries(ENGINES).map(([value, info]) => h('option', { value, disabled: !available(value) },
      available(value) ? info.name : `${info.name} (weights missing)`)));
  engine.value = requestedEngine && Object.hasOwn(ENGINES, requestedEngine) && available(requestedEngine)
    ? requestedEngine : available('triposr') ? 'triposr' : 'local';
  const quality = h('select', { 'aria-labelledby': 'studio-quality-label', onchange: () => updateEstimate() },
    ['Draft', 'Standard', 'High'].map(label => h('option', { value: label.toLowerCase() }, label)));
  if (['draft', 'standard', 'high'].includes(params.get('quality'))) quality.value = params.get('quality');
  const engineHint = h('p', { class: 'hint mb0' });
  const texture = h('select', {}, ['512', '1024', '2048'].map(value =>
    h('option', { value, selected: value === (initialSample?.prototype ? '512' : '1024') }, `${value} × ${value}`)));
  const textureField = h('label', { class: 'field' }, h('span', {}, 'Texture atlas size'), texture);
  const symmetry = h('input', { type: 'checkbox', checked: true });
  const symmetryField = h('label', { class: 'checkline' }, symmetry, 'Use a symmetry prior');
  const turntable = h('input', { type: 'checkbox', onchange: () => updateEstimate() });
  const formats = ['obj', 'ply', 'stl', 'gltf'].map(format => h('input', {
    type: 'checkbox', value: format, checked: !initialSample?.prototype && ['obj', 'ply'].includes(format),
  }));
  const cost = h('p', { id: 'estimate-cost', class: 'hint mb0', role: 'status' });
  const submit = h('button', { id: 'btn-generate', class: 'btn primary block', type: 'button', disabled: true, onclick: start }, 'Generate model');
  const output = h('section', { class: 'card studio-output', 'aria-label': 'Reconstruction output' });
  const exampleImage = h('img');
  const exampleCaption = h('figcaption');
  const pipeline = h('ul', { class: 'studio-pipeline' });
  const progressInfo = h('div', { role: 'status', class: 'muted' });
  const bar = h('progress', { max: '100', value: '0', 'aria-label': 'Generation progress' });
  const status = h('span');
  const actions = h('div', { class: 'row wrap gap8' });
  const running = h('div', { class: 'col run-progress' },
    h('div', { class: 'spread' }, h('h2', { class: 'mb0' }, 'Reconstruction'), status),
    bar, progressInfo, actions);

  const options = () => ({
    engine: engine.value, texture_quality: texture.value, symmetry: symmetry.checked,
    turntable: turntable.checked, formats: ['glb', ...formats.filter(box => box.checked).map(box => box.value)],
  });

  function updateControls() {
    const busy = uploading || starting || Boolean(current && !TERMINAL.includes(current.status));
    engine.disabled = quality.disabled = starting || Boolean(current && !TERMINAL.includes(current.status));
    drop.disabled = uploading || starting;
    samples.forEach(button => { button.disabled = uploading || starting; });
    submit.disabled = busy || !upload || !estimate || estimate.available < estimate.credits;
    submit.textContent = starting ? 'Starting…' : busy && !uploading ? 'Generating…' : 'Generate model';
  }

  async function selectFile(readFile) {
    if (uploading || starting) return;
    uploading = true;
    error.textContent = '';
    updateControls();
    const previous = upload;
    try {
      const file = await readFile();
      validateReferenceFile(file, maxMB);
      reference.textContent = 'Uploading image…';
      const data = new FormData();
      data.append('file', file);
      upload = await api('/uploads/', { method: 'POST', body: data, form: true });
      if (active) saveReferenceUrl(upload.id);
      return true;
    } catch (err) {
      upload = previous;
      if (active) error.textContent = err.message;
      return false;
    } finally {
      uploading = false;
      fileInput.value = '';
      if (active) {
        renderReference();
        updateControls();
      }
    }
  }

  function saveReferenceUrl(id) {
    const url = new URL(location.href);
    url.searchParams.delete('draft');
    if (!id || !initialSample?.prototype || upload?.filename !== initialSample.input.split('/').pop()) {
      url.searchParams.delete('sample');
    }
    if (id) url.searchParams.set('upload', id);
    else url.searchParams.delete('upload');
    history.replaceState(null, '', url);
  }

  async function restoreReference(id, showError = true) {
    uploading = true;
    updateControls();
    try {
      const saved = await api(`/uploads/${encodeURIComponent(id)}`);
      if (active) upload = saved;
    } catch (err) {
      if (active && showError) error.textContent = err.message;
    } finally {
      uploading = false;
      if (active) { renderReference(); updateControls(); }
    }
  }

  function renderReference() {
    clear(reference);
    samples.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.filename === upload?.filename)));
    if (!upload) return;
    mount(reference, h('img', { src: upload.image_url, alt: 'Reference image' }),
      h('div', { class: 'reference-meta' }, h('strong', { class: 'truncate' }, upload.filename),
        h('small', {}, `${upload.width} × ${upload.height} · ${fmt.bytes(upload.size_bytes)}`)),
      h('button', { class: 'btn sm ghost', type: 'button', onclick: () => {
        if (uploading || starting) return;
        upload = null; saveReferenceUrl(null); renderReference(); updateControls();
      } }, 'Remove'));
  }

  async function updateEstimate() {
    const version = ++estimateVersion;
    const learned = engine.value === 'triposr';
    const info = ENGINES[engine.value];
    const example = initialSample?.prototype && initialSample.engine === engine.value
      && (!upload || upload.filename === initialSample.input.split('/').pop())
      ? initialSample : EXAMPLES.find(item => item.id === info.example);
    exampleImage.src = example.preview;
    exampleImage.alt = `${example.label} reconstructed with ${example.method}`;
    exampleCaption.textContent = `${example.label} · ${example.method}`;
    textureField.hidden = learned;
    symmetryField.hidden = learned;
    engineHint.textContent = info.description;
    mount(pipeline, info.stages.map(stage => h('li', {}, stage)));
    estimate = null;
    cost.textContent = 'Checking credits…';
    updateControls();
    try {
      const next = await api('/generations/estimate', { method: 'POST', body: { quality: quality.value, options: options() } });
      if (!active || version !== estimateVersion) return;
      estimate = next;
      cost.textContent = `${estimate.credits} credits · ${estimate.available} available · ~${estimate.estimated_seconds} s`;
      if (estimate.available < estimate.credits) {
        cost.append(' · ', h('a', { href: '/app/dashboard', 'data-link': '' }, 'Manage credits'));
      }
    } catch (err) {
      if (active && version === estimateVersion) cost.textContent = err.message;
    }
    updateControls();
  }

  function renderProgress() {
    if (!running.isConnected) {
      viewer?.dispose();
      viewer = null;
      mount(output, running);
    }
    mount(status, statusBadge(current.status));
    bar.value = current.progress;
    const stage = current.status === 'CANCEL_REQUESTED' ? 'CANCEL_REQUESTED' : current.stage;
    const depth = current.provider === 'triposr' ? 'Encoding the image' : 'Estimating depth';
    const label = stage === 'DEPTH' ? depth : (LABELS[stage] || 'Processing');
    const message = `${label} · ${current.progress}%`;
    if (progressInfo.textContent !== message) progressInfo.textContent = message;
    const action = TERMINAL.includes(current.status) ? 'retry' : 'cancel';
    const allowed = action === 'retry' ? current.status === 'CANCELLED' || (current.status === 'FAILED' && current.retryable)
      : !['CANCEL_REQUESTED', 'OPTIMIZING', 'RENDERING_PREVIEW'].includes(current.status);
    const showError = current.error_message || '';
    const actionKey = `${allowed}:${action}:${changingJob}:${showError}`;
    if (actions.dataset.actionKey !== actionKey) {
      actions.dataset.actionKey = actionKey;
      mount(actions, showError ? h('p', { class: 'form-error mb0' }, showError) : null,
        allowed ? h('button', { class: 'btn sm', type: 'button', disabled: changingJob,
          onclick: () => changeJob(action) }, action === 'cancel' ? 'Cancel run' : 'Retry run') : null);
    }
    updateControls();
  }

  async function renderResult(id) {
    const [asset, { ModelViewer }] = await Promise.all([api(`/assets/${id}`), import('./viewer.js')]);
    if (!active || current?.asset_id !== id) return;

    const sourceImgSrc = current.source_image_url || upload?.image_url;
    const viewTabs = h('div', { class: 'view-tabs mt12', role: 'group', 'aria-label': 'Result view', hidden: !sourceImgSrc });
    const btnStandard = h('button', { class: 'btn sm primary', type: 'button', 'aria-pressed': 'true' }, '3D model');
    const btnSplit = h('button', { class: 'btn sm', type: 'button', 'aria-pressed': 'false' }, 'Compare with photo');
    mount(viewTabs, btnStandard, btnSplit);

    const viewerContainer = h('div', { class: 'mt8' });
    const standardHost = h('div');
    const splitHost = h('div', { class: 'split-view-container' });

    const splitLeft = h('div', { class: 'split-side split-left' },
      h('img', { src: sourceImgSrc, class: 'split-source-img', alt: 'Source photo' }),
      h('div', { class: 'split-label left' }, 'Source photo'));
    const splitRightHost = h('div', { style: 'width:100%;height:100%;position:relative;' });
    const splitRight = h('div', { class: 'split-side split-right', style: 'clip-path: inset(0 0 0 50%)' },
      splitRightHost,
      h('div', { class: 'split-label right' }, '3D model'));
    const divider = h('div', { class: 'split-divider-line', style: 'left: 50%', 'aria-hidden': 'true' },
      h('div', { class: 'split-handle' }, '⮂ ⮃'));
    const amount = h('input', { type: 'range', min: 0, max: 100, value: 50, 'aria-label': 'Photo visible in comparison' });
    const comparison = h('div', { hidden: true }, splitHost,
      h('label', { class: 'split-slider' }, h('span', {}, '3D model'), amount, h('span', {}, 'Photo')));

    mount(splitHost, splitLeft, splitRight, divider);
    mount(viewerContainer, standardHost, comparison);

    const setAmount = pct => {
      amount.value = pct;
      amount.setAttribute('aria-valuetext', `${pct}% photo`);
      divider.style.left = `${pct}%`;
      splitRight.style.clipPath = `inset(0 0 0 ${pct}%)`;
    };
    amount.addEventListener('input', () => setAmount(Number(amount.value)));
    setAmount(50);
    divider.addEventListener('pointerdown', event => divider.setPointerCapture(event.pointerId));
    divider.addEventListener('pointermove', event => {
      if (!divider.hasPointerCapture(event.pointerId)) return;
      const rect = splitHost.getBoundingClientRect();
      const fraction = (event.clientX - rect.left) / rect.width;
      setAmount(Math.round(Math.max(0, Math.min(fraction, 1)) * 100));
    });

    const setView = split => {
      if (viewer && comparison.hidden === !split) return;
      btnStandard.classList.toggle('primary', !split);
      btnSplit.classList.toggle('primary', split);
      btnStandard.setAttribute('aria-pressed', String(!split));
      btnSplit.setAttribute('aria-pressed', String(split));
      standardHost.hidden = split;
      comparison.hidden = !split;

      viewer?.dispose();
      const targetHost = split ? splitRightHost : standardHost;
      viewer = new ModelViewer(targetHost, {
        modelUrl: asset.model_url, previewUrl: asset.preview_url, altText: asset.name,
        height: split ? splitHost.clientHeight : 540, autoRotate: false, compact: split,
      });
    };

    btnStandard.addEventListener('click', () => setView(false));
    btnSplit.addEventListener('click', () => setView(true));

    const arBtn = h('button', {
      class: 'btn sm', type: 'button', title: 'View in Augmented Reality on mobile',
      onclick: () => viewer?.openAR(),
    }, '📱 View in AR');

    mount(output, h('div', { class: 'spread wrap output-heading' },
      h('div', {}, h('h2', { class: 'mb0' }, 'Your model'), h('small', {},
        `${ENGINES[current.provider]?.name || 'Reconstruction'} · ${inferenceLabel(asset.manifest?.hardware)}`)),
      h('div', { class: 'row gap8' }, arBtn,
        h('a', { class: 'btn sm', href: `/app/assets/${id}`, 'data-link': '' }, 'Open model details'))),
    viewTabs,
    viewerContainer,
    h('dl', { class: 'result-grid' }, [
      ['Triangles', fmt.n(asset.polygons)], ['Vertices', fmt.n(asset.vertices)],
      ['Watertight', asset.watertight ? 'Yes' : 'No'], ['Run time', fmt.ms(current.duration_ms)],
    ].map(([label, value]) => h('div', { class: 'result-cell' }, h('dt', { class: 'k' }, label), h('dd', { class: 'v' }, value)))),
    h('div', { class: 'row wrap gap8 mt16' }, asset.downloads.map(file =>
      h('a', { class: `btn sm ${file.format === 'glb' ? 'primary' : ''}`, href: file.url }, `Download ${file.label}`))));

    viewer = new ModelViewer(standardHost, { modelUrl: asset.model_url, previewUrl: asset.preview_url, altText: asset.name,
      height: 540, autoRotate: false });

    state.currentAssetId = id;
    if (current?.id) inspector.loadStages(current.id);
    const statusTop = document.getElementById('status-topology');
    if (statusTop) statusTop.textContent = `${fmt.n(asset.polygons)} faces · ${asset.watertight ? 'Watertight' : 'Manifold'}`;
  }

  async function watch(id, failures = 0) {
    clearTimeout(pollTimer);
    if (!active) return;
    try {
      const job = await api(`/generations/${id}`);
      if (!active || (current && current.id !== id)) return;
      const initial = !current;
      if (initial && Object.hasOwn(ENGINES, job.provider) && available(job.provider)) {
        engine.value = job.provider;
        quality.value = job.quality;
        updateEstimate();
      }
      current = job;
      renderProgress();
      if (initial && !upload && !uploading && job.upload_id) {
        await restoreReference(job.upload_id, false);
        if (!active) return;
      }
      if (TERMINAL.includes(current.status)) {
        if (current.asset_id && current.status === 'COMPLETED') {
          await renderResult(current.asset_id).catch(err => { if (active) error.textContent = err.message; });
        }
        await refreshUser();
        if (active) updateEstimate();
        return;
      }
      pollTimer = setTimeout(() => watch(id), 1200);
    } catch (err) {
      if (!active) return;
      if (err.status === 404 || err.status === 401 || failures >= 4) {
        error.textContent = err.message;
        return;
      }
      pollTimer = setTimeout(() => watch(id, failures + 1), 2500);
    }
  }

  async function changeJob(action) {
    if (changingJob) return;
    changingJob = true;
    renderProgress();
    const id = current.id;
    try {
      const result = await api(`/generations/${id}/${action}`, { method: 'POST' });
      if (active) { current = result; watch(id); }
    } catch (err) {
      if (active) toast(err.message, 'error');
    } finally {
      changingJob = false;
      if (active && running.isConnected) renderProgress();
    }
  }

  async function start() {
    if (submit.disabled) return;
    error.textContent = '';
    starting = true;
    updateControls();
    try {
      const job = await api('/generations', { method: 'POST', body: {
        upload_id: upload.id, quality: quality.value, options: options(), privacy_mode: true,
        idempotency_key: crypto.randomUUID(),
      } });
      if (!active) return;
      current = job;
      const query = new URLSearchParams({ generation: job.id, upload: upload.id });
      history.replaceState(null, '', `/app/studio?${query}`);
      renderProgress();
      watch(job.id);
      if (matchMedia('(max-width: 800px)').matches) output.scrollIntoView({ block: 'start' });
    } catch (err) {
      if (active) error.textContent = err.message;
    } finally {
      starting = false;
      if (active) updateControls();
    }
  }

  fileInput.addEventListener('change', () => { if (fileInput.files[0]) selectFile(() => fileInput.files[0]); });
  drop.addEventListener('dragover', event => { event.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', event => {
    event.preventDefault(); drop.classList.remove('over');
    if (event.dataTransfer.files[0]) selectFile(() => event.dataTransfer.files[0]);
  });

  async function loadDemoMesh(exampleId = 'sneaker') {
    const ex = EXAMPLES.find(e => e.id === exampleId) || EXAMPLES[3];
    const { ModelViewer } = await import('./viewer.js');
    clear(output);
    const standardHost = h('div');
    const heading = h('div', { class: 'spread wrap output-heading' },
      h('div', {},
        h('h2', { class: 'mb0' }, `${ex.label} · Demo 3D Model`),
        h('small', {}, `${ex.method} · Interactive Three.js WebGL CAD Viewport`)
      ),
      h('div', { class: 'row gap8' },
        h('button', {
          class: 'btn sm',
          type: 'button',
          onclick: () => {
            viewer?.dispose();
            mountEmptyOutput();
          }
        }, '← Back to Upload'),
        h('a', { class: 'btn sm primary', href: ex.glb, download: `${ex.id}.glb` }, 'Download GLB')
      )
    );
    mount(output, heading, standardHost,
      h('dl', { class: 'result-grid' }, [
        ['Example', ex.label],
        ['Method', ex.method],
        ['Format', 'GLTF Binary (.glb)'],
        ['Mesh Status', 'Watertight 2-Manifold'],
      ].map(([label, value]) => h('div', { class: 'result-cell' }, h('dt', { class: 'k' }, label), h('dd', { class: 'v' }, value)))),
      h('div', { class: 'row wrap gap8 mt16' }, [
        h('a', { class: 'btn sm primary', href: ex.glb, download: `${ex.id}.glb` }, 'Download .GLB'),
        h('a', { class: 'btn sm', href: '/#example-stage' }, 'Explore All Examples in Benchmark')
      ])
    );
    viewer?.dispose();
    viewer = new ModelViewer(standardHost, {
      modelUrl: ex.glb,
      previewUrl: ex.preview,
      altText: `${ex.label} 3D Mesh`,
      height: 540,
      autoRotate: false
    });
    const statusTop = document.getElementById('status-topology');
    if (statusTop) statusTop.textContent = `Demo Mesh · Watertight 2-Manifold`;
  }

  function mountEmptyOutput() {
    clear(output);
    const viewSizeBadge = h('span', { class: 'telemetry-badge', id: 'viewport-render-size' }, '📐 Render View: 640 × 540 px');
    const btnLoadDemo = h('button', {
      class: 'btn sm primary',
      id: 'btn-load-demo-model',
      type: 'button',
      onclick: () => loadDemoMesh('sneaker')
    }, '🎮 Load Demo 3D Mesh in Viewport');

    mount(output, h('div', { class: 'spread output-heading' },
      h('div', { class: 'row gap8', style: 'align-items:center;' },
        h('h2', { class: 'mb0' }, 'Reconstruction Viewport'),
        viewSizeBadge
      ),
      h('div', { class: 'row gap8' },
        btnLoadDemo,
        h('a', { class: 'btn sm', href: '/#example-stage' }, 'Explore Examples')
      )
    ),
    h('figure', { class: 'studio-demo' }, exampleImage, exampleCaption),
    h('p', { class: 'hint mb0' }, 'Upload an image and hit Generate Model, or click "Load Demo 3D Mesh in Viewport" to inspect real-time 3D geometry.'));
  }

  mountEmptyOutput();

  // Right Dock Setup: Tab A (Inspector) + Tab B (3D AI Chat)
  const tabSeg = h('div', { class: 'seg mb8', role: 'group', 'aria-label': 'Right Dock Tabs' });
  const btnInspectorTab = h('button', { class: 'active', type: 'button' }, '🔬 Inspector');
  const btnChatTab = h('button', { type: 'button' }, '💬 3D AI Chat');
  mount(tabSeg, btnInspectorTab, btnChatTab);

  const inspectorHost = h('div', { class: 'right-dock-tab' });
  const chatHost = h('div', { class: 'right-dock-tab', hidden: true });

  btnInspectorTab.onclick = () => {
    btnInspectorTab.classList.add('active');
    btnChatTab.classList.remove('active');
    inspectorHost.hidden = false;
    chatHost.hidden = true;
  };

  btnChatTab.onclick = () => {
    btnChatTab.classList.add('active');
    btnInspectorTab.classList.remove('active');
    inspectorHost.hidden = true;
    chatHost.hidden = false;
  };

  const inspector = new PipelineInspector(inspectorHost, { generationId: null });

  renderChat(params, chatHost, {
    embedded: true,
    activeMeshId: null,
    onToolExecuted: (toolResult) => {
      if (current?.asset_id) {
        renderResult(current.asset_id);
      }
    },
  });

  const rightDock = h('aside', { class: 'studio-right-dock' }, tabSeg, inspectorHost, chatHost);

  // Left Dock Setup
  const leftDock = h('aside', { class: 'studio-left-dock col gap12' },
    h('div', { class: 'spread' }, h('h2', { class: 'mb0', style: 'font-size:18px;' }, '🎨 3D Creative Studio'),
      h('span', { class: 'runtime-chip' }, PROVIDER_NAMES[state.meta?.inference?.preferred] || 'DirectML / CPU')),
    drop, fileInput, reference,
    h('div', {}, h('small', { class: 'sample-label', style: 'color:var(--text-2);font-weight:600;' }, 'Sample images'),
      h('div', { class: 'sample-strip' }, ...samples)),
    h('div', { class: 'studio-fields' },
      h('label', { class: 'field' }, h('span', { id: 'studio-engine-label' }, 'Reconstruction Engine'), engine),
      h('label', { class: 'field' }, h('span', { id: 'studio-quality-label' }, 'Quality Preset'), quality)),
    engineHint,
    h('details', { class: 'studio-options' }, h('summary', {}, 'PBR Baking & Advanced Controls'),
      h('div', { class: 'mt8' }, textureField, symmetryField,
        h('label', { class: 'checkline' }, turntable, 'Render 360° MP4/GIF Turntable'),
        h('div', { class: 'row wrap gap12 mt8' }, formats.map(box =>
          h('label', { class: 'checkline' }, box, box.value.toUpperCase()))))),
    h('div', { class: 'col mt8' }, cost, error, submit));

  // Center Canvas
  const centerCanvas = h('section', { class: 'studio-center-canvas' }, output);

  // Bottom Telemetry Dock
  const bottomDock = h('footer', { class: 'studio-bottom-dock mt16' },
    h('div', { class: 'spread gap12' },
      h('span', { class: 'telemetry-badge' }, '⚡ Engine: ', h('strong', { id: 'status-engine' }, ENGINES[engine.value]?.name || 'Local CPU')),
      h('span', { class: 'telemetry-badge' }, '🖥 Hardware: ', h('strong', {}, PROVIDER_NAMES[state.meta?.inference?.preferred] || 'CPU / DirectML')),
      h('span', { class: 'telemetry-badge' }, '⏱ Status: ', h('strong', { id: 'status-timer' }, 'Active Session'))),
    h('div', { class: 'spread gap12' },
      h('span', { class: 'telemetry-badge' }, '📦 Mesh Topology: ', h('strong', { id: 'status-topology' }, '2-Manifold Certified')),
      h('span', {
        class: 'telemetry-badge',
        style: 'cursor: pointer; transition: border-color 0.15s;',
        title: 'Click to open Credit Wallet & Transaction History',
        onclick: () => openCreditWalletModal()
      }, '◆ Wallet: ', h('strong', { id: 'status-wallet', style: 'color: var(--accent);' }, `${state.user?.credits ?? 50} Credits`))));

  window.openCreditWalletModal = openCreditWalletModal;

  // Mount 3-Dock Architecture
  mount(main,
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
      h('h1', {}, 'Image-to-3D Studio'),
      h('p', { class: 'muted mb0' }, 'Generate a mesh from one photo, inspect the stages, and download the result.')),
    h('div', { class: 'studio-layout' }, leftDock, centerCanvas, rightDock),
    bottomDock);

  updateEstimate();
  const id = params.get('generation');
  const assetId = params.get('asset');
  if (assetId) {
    try {
      const asset = await api(`/assets/${encodeURIComponent(assetId)}`);
      current = {
        id: asset.generation_id || `gen_${asset.id}`,
        asset_id: asset.id,
        status: 'COMPLETED',
        provider: asset.manifest?.engine?.name || 'local',
        source_image_url: asset.preview_url,
      };
      state.currentAssetId = asset.id;
      await renderResult(asset.id);
    } catch (e) {
      console.warn('Could not load requested asset into studio:', e);
    }
  } else if (id) {
    watch(encodeURIComponent(id));
  }
  const saved = params.get('upload');
  const draft = params.get('draft');
  if (saved) {
    await restoreReference(saved);
  } else if (draft) {
    const savedOnServer = await selectFile(async () => {
      const file = await pendingReference(draft);
      if (!file) throw new Error('This photo selection expired. Choose the image again.');
      return file;
    });
    if (savedOnServer) await discardReference(draft).catch(() => {});
  } else if (!id && initialSample) {
    selectExample(initialSample);
  }
}

export async function openCreditWalletModal() {
  const existing = document.getElementById('wallet-modal-back');
  if (existing) existing.remove();

  const backdrop = h('div', { id: 'wallet-modal-back', class: 'modal-back' });
  const modal = h('div', { class: 'modal', style: 'max-width: 680px; width: 100%;' });
  backdrop.appendChild(modal);

  modal.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <h3 style="margin:0;font-size:20px;display:flex;align-items:center;gap:8px;">
        <span style="color:var(--accent);">◆</span> Credit Wallet & Free Trial Ledger
      </h3>
      <button class="btn sm ghost" id="modal-close" style="font-size:16px;line-height:1;padding:4px 8px;">✕</button>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));gap:12px;margin-bottom:20px;">
      <div class="card tight center" style="background:var(--panel-2);border:1px solid var(--border);">
        <div class="k muted" style="font-size:12px;">Current Balance</div>
        <div class="v" id="modal-bal" style="font-size:24px;font-weight:700;color:var(--accent);">${state.user?.credits ?? 50}</div>
        <div class="d muted" style="font-size:11px;">Credits ready to spend</div>
      </div>
      <div class="card tight center" style="background:var(--panel-2);border:1px solid var(--border);">
        <div class="k muted" style="font-size:12px;">Daily Refill</div>
        <div class="v" style="font-size:24px;font-weight:700;color:var(--cyan);">+10</div>
        <div class="d muted" style="font-size:11px;">Every 24 hours</div>
      </div>
      <div class="card tight center" style="background:var(--panel-2);border:1px solid var(--border);">
        <div class="k muted" style="font-size:12px;">Account Tier</div>
        <div class="v" style="font-size:16px;font-weight:700;color:var(--success);padding-top:6px;">Educational</div>
        <div class="d muted" style="font-size:11px;">IIIT Dharwad 3DCV</div>
      </div>
    </div>

    <!-- Daily Streak & Coupon Section -->
    <div style="background:rgba(99,102,241,0.06);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:20px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <div style="font-weight:600;font-size:14px;">🔥 Daily Educational Streak Refill</div>
          <div class="muted" style="font-size:12px;">Refill +10 free generation credits to your wallet every 24 hours.</div>
        </div>
        <button class="btn sm primary" id="btn-claim-streak">Claim +10 Streak</button>
      </div>
      <hr style="border:none;border-top:1px solid var(--border);margin:12px 0;">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <input type="text" id="coupon-code" placeholder="Enter coupon (e.g. IIITDHARWAD)" style="flex:1;min-width:180px;text-transform:uppercase;" />
        <button class="btn sm" id="btn-redeem-coupon">Redeem Voucher</button>
      </div>
      <div id="coupon-status" style="font-size:12px;margin-top:6px;"></div>
    </div>

    <!-- Transaction History Table -->
    <h4 style="margin:16px 0 8px 0;font-size:15px;">Transaction History & ACID Ledger</h4>
    <div id="tx-table-container" style="max-height:260px;overflow-y:auto;border:1px solid var(--border);border-radius:8px;">
      <div style="padding:16px;text-align:center;color:var(--muted);"><span class="spinner"></span> Loading transactions…</div>
    </div>
  `;

  document.body.appendChild(backdrop);
  const close = () => backdrop.remove();
  modal.querySelector('#modal-close').onclick = close;
  backdrop.onclick = (e) => { if (e.target === backdrop) close(); };

  // Claim streak
  const streakBtn = modal.querySelector('#btn-claim-streak');
  streakBtn.onclick = async () => {
    streakBtn.disabled = true;
    streakBtn.innerHTML = '<span class="spinner"></span> Claiming…';
    try {
      const res = await api('/billing/daily-claim', { method: 'POST' });
      toast(`Streak claimed! +${res.credits_added} credits added.`);
      await refreshUser();
      modal.querySelector('#modal-bal').textContent = state.user?.credits ?? res.new_balance;
      const statWallet = document.getElementById('status-wallet');
      if (statWallet) statWallet.textContent = `${state.user?.credits ?? res.new_balance} Credits`;
      loadTransactions();
    } catch (err) {
      toast(err.message, 'warn');
    } finally {
      streakBtn.disabled = false;
      streakBtn.textContent = 'Claim +10 Streak';
    }
  };

  // Redeem coupon
  const couponBtn = modal.querySelector('#btn-redeem-coupon');
  const couponInput = modal.querySelector('#coupon-code');
  const couponStatus = modal.querySelector('#coupon-status');
  couponBtn.onclick = async () => {
    const code = couponInput.value.trim();
    if (!code) return;
    couponBtn.disabled = true;
    couponStatus.textContent = 'Verifying coupon…';
    couponStatus.style.color = 'var(--muted)';
    try {
      const res = await api('/billing/redeem-coupon', { method: 'POST', body: { code } });
      couponStatus.textContent = `Success! Added ${res.credits_added} credits.`;
      couponStatus.style.color = 'var(--success)';
      toast(`Coupon redeemed! +${res.credits_added} credits`);
      await refreshUser();
      modal.querySelector('#modal-bal').textContent = state.user?.credits ?? res.new_balance;
      const statWallet = document.getElementById('status-wallet');
      if (statWallet) statWallet.textContent = `${state.user?.credits ?? res.new_balance} Credits`;
      couponInput.value = '';
      loadTransactions();
    } catch (err) {
      couponStatus.textContent = err.message;
      couponStatus.style.color = 'var(--danger)';
    } finally {
      couponBtn.disabled = false;
    }
  };

  // Load transactions
  async function loadTransactions() {
    const container = modal.querySelector('#tx-table-container');
    try {
      const txs = await api('/credits/transactions?limit=25');
      if (!txs || txs.length === 0) {
        container.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);">No transactions recorded yet.</div>';
        return;
      }
      let html = `
        <table class="tbl" style="width:100%;font-size:12px;">
          <thead>
            <tr>
              <th>Date / Time</th>
              <th>Reason</th>
              <th>Change</th>
              <th>Balance</th>
            </tr>
          </thead>
          <tbody>
      `;
      for (const tx of txs) {
        const isPos = tx.amount > 0;
        const color = isPos ? 'var(--success)' : (tx.amount < 0 ? 'var(--danger)' : 'var(--muted)');
        const sign = isPos ? '+' : '';
        const date = new Date(tx.created_at).toLocaleString();
        html += `
          <tr>
            <td>${date}</td>
            <td><code>${tx.reason}</code></td>
            <td style="color:${color};font-weight:700;">${sign}${tx.amount}</td>
            <td><b>${tx.balance_after}</b></td>
          </tr>
        `;
      }
      html += '</tbody></table>';
      container.innerHTML = html;
    } catch (err) {
      container.innerHTML = `<div style="padding:16px;color:var(--danger);">Failed to load transactions: ${err.message}</div>`;
    }
  }

  loadTransactions();
}
