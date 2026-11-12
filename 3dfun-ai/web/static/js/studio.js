import { api, clear, fmt, h, mount, refreshUser, state, statusBadge, toast } from './lib.js';
import { EXAMPLES } from './examples.js';
import { ENGINE_NAMES, inferenceLabel, PROVIDER_NAMES } from './experiments.js';
import { discardReference, pendingReference, validateReferenceFile } from './pending-upload.js';

const DEPTH_STAGES = ['Matte', 'Depth', 'Volume', 'Mesh', 'Texture', 'Export'];
const ENGINES = {
  triposr: { name: ENGINE_NAMES.triposr, example: 'chair', stages: ['Matte', 'Encode', 'Density', 'Mesh', 'Colour', 'Export'],
    description: 'Predicts a 3D shape and stores colour on the mesh vertices.' },
  local: { name: ENGINE_NAMES.local, example: 'sneaker', stages: DEPTH_STAGES,
    description: 'The course method uses estimated depth and silhouette to build a mesh with a texture atlas.' },
  midas: { name: ENGINE_NAMES.midas, example: 'teapot', stages: DEPTH_STAGES,
    description: 'Uses MiDaS depth with the same silhouette pipeline and texture atlas.' },
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
  const available = key => key === 'local' || Boolean(state.meta?.features?.[key]);
  const params = new URLSearchParams(location.search);
  const requestedEngine = params.get('engine');
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
  const samples = EXAMPLES.map(example => h('button', { class: 'sample-button', type: 'button',
    onclick: () => selectExample(example), 'aria-pressed': 'false', dataset: { filename: example.input.split('/').pop() },
  }, h('img', { src: example.input, alt: '' }), example.label));

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
    h('option', { value, selected: value === '1024' }, `${value} × ${value}`)));
  const textureField = h('label', { class: 'field' }, h('span', {}, 'Texture atlas size'), texture);
  const symmetry = h('input', { type: 'checkbox', checked: true });
  const symmetryField = h('label', { class: 'checkline' }, symmetry, 'Use a symmetry prior');
  const turntable = h('input', { type: 'checkbox', onchange: () => updateEstimate() });
  const formats = ['obj', 'ply', 'stl', 'gltf'].map(format => h('input', {
    type: 'checkbox', value: format, checked: ['obj', 'ply'].includes(format),
  }));
  const cost = h('p', { class: 'hint mb0', role: 'status' });
  const submit = h('button', { class: 'btn primary block', type: 'button', disabled: true, onclick: start }, 'Generate model');
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
    url.searchParams.delete('sample');
    if (id) url.searchParams.set('upload', id);
    else url.searchParams.delete('upload');
    history.replaceState(null, '', url);
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
    const example = EXAMPLES.find(item => item.id === info.example);
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

    let isSplitView = false;
    const viewTabs = h('div', { class: 'view-tabs mt12' });
    const btnStandard = h('button', { class: 'btn sm primary', type: 'button' }, 'Interactive 3D Viewer');
    const btnSplit = h('button', { class: 'btn sm', type: 'button' }, 'Split-Screen 2D vs 3D');
    mount(viewTabs, btnStandard, btnSplit);

    const viewerContainer = h('div', { class: 'mt8' });
    const standardHost = h('div', {});
    const splitHost = h('div', { class: 'split-view-container hidden' });

    const sourceImgSrc = upload?.image_url || asset.manifest?.source_preview || asset.preview_url;
    const splitLeft = h('div', { class: 'split-side split-left' },
      h('img', { src: sourceImgSrc, class: 'split-source-img', alt: 'Source photo' }),
      h('div', { class: 'split-silhouette-overlay' }),
      h('div', { class: 'split-label left' }, '2D Source & Silhouette Boundary'));
    const splitRightHost = h('div', { style: 'width:100%;height:100%;position:relative;' });
    const splitRight = h('div', { class: 'split-side split-right', style: 'clip-path: inset(0 0 0 50%)' },
      splitRightHost,
      h('div', { class: 'split-label right' }, 'Reconstructed 3D Geometry'));
    const divider = h('div', { class: 'split-divider-line', style: 'left: 50%' },
      h('div', { class: 'split-handle' }, '⮂ ⮃'));

    mount(splitHost, splitLeft, splitRight, divider);
    mount(viewerContainer, standardHost, splitHost);

    let dragging = false;
    const updateDivider = clientX => {
      const rect = splitHost.getBoundingClientRect();
      const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
      const pct = (x / rect.width) * 100;
      divider.style.left = `${pct}%`;
      splitRight.style.clipPath = `inset(0 0 0 ${pct}%)`;
    };

    divider.addEventListener('mousedown', () => { dragging = true; });
    window.addEventListener('mousemove', e => { if (dragging) updateDivider(e.clientX); });
    window.addEventListener('mouseup', () => { dragging = false; });
    divider.addEventListener('touchstart', () => { dragging = true; });
    window.addEventListener('touchmove', e => { if (dragging && e.touches[0]) updateDivider(e.touches[0].clientX); });
    window.addEventListener('touchend', () => { dragging = false; });

    const setView = split => {
      isSplitView = split;
      btnStandard.classList.toggle('primary', !split);
      btnSplit.classList.toggle('primary', split);
      standardHost.classList.toggle('hidden', split);
      splitHost.classList.toggle('hidden', !split);

      viewer?.dispose();
      const targetHost = split ? splitRightHost : standardHost;
      viewer = new ModelViewer(targetHost, {
        modelUrl: asset.model_url, previewUrl: asset.preview_url, altText: asset.name,
        height: split ? 460 : 420, autoRotate: !split,
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
      height: 420, autoRotate: false });
  }

  async function watch(id, failures = 0) {
    clearTimeout(pollTimer);
    if (!active) return;
    try {
      const job = await api(`/generations/${id}`);
      if (!active || (current && current.id !== id)) return;
      if (!current && Object.hasOwn(ENGINES, job.provider) && available(job.provider)) {
        engine.value = job.provider;
        quality.value = job.quality;
        updateEstimate();
      }
      current = job;
      renderProgress();
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

  mount(output, h('div', { class: 'spread output-heading' }, h('h2', { class: 'mb0' }, 'Reconstruction preview'),
    h('a', { class: 'btn sm', href: '/#example-stage' }, 'Interactive examples')),
    h('figure', { class: 'studio-demo' }, exampleImage, exampleCaption),
    h('p', { class: 'hint mb0' }, 'Your model and download links will appear here after generation.'));
  mount(main, h('header', { class: 'studio-title' }, h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
    h('h1', {}, 'Image to 3D Studio'), h('p', { class: 'mb0' }, 'Turn one object photo into a mesh you can rotate and download.')),
  h('div', { class: 'studio' }, h('div', { class: 'col' },
    h('section', { class: 'card col' }, h('h2', { class: 'mb0' }, 'Reference image'), drop, fileInput, reference,
      h('div', {}, h('small', { class: 'sample-label' }, 'Try an example'),
        h('div', { class: 'sample-strip' }, ...samples))),
    h('section', { class: 'card' }, h('div', { class: 'spread wrap' }, h('h2', {}, 'Reconstruction settings'),
      h('span', { class: 'runtime-chip' }, `Preferred · ${PROVIDER_NAMES[state.meta?.inference?.preferred] || 'CPU'}`)),
      h('div', { class: 'studio-fields' }, h('label', { class: 'field' }, h('span', { id: 'studio-engine-label' }, 'Engine'), engine),
        h('label', { class: 'field' }, h('span', { id: 'studio-quality-label' }, 'Quality'), quality)), engineHint,
      h('details', { class: 'studio-options' }, h('summary', {}, 'Export and advanced options'),
        h('div', { class: 'mt8' }, textureField, symmetryField,
          h('label', { class: 'checkline' }, turntable, 'Render a turntable'),
          h('p', { class: 'hint mb0' }, 'GLB is included for the browser viewer.'),
          h('div', { class: 'row wrap gap12 mt8' }, formats.map(box =>
            h('label', { class: 'checkline' }, box, box.value.toUpperCase()))))),
      h('div', { class: 'col mt16' }, cost, error, submit))),
    h('div', { class: 'col studio-right' }, output, h('section', { class: 'card' }, h('h3', {}, 'Pipeline'), pipeline))));

  updateEstimate();
  const id = params.get('generation');
  if (id) watch(encodeURIComponent(id));
  const saved = params.get('upload');
  const draft = params.get('draft');
  if (saved) {
    uploading = true;
    updateControls();
    try { upload = await api(`/uploads/${encodeURIComponent(saved)}`); }
    catch (err) { if (active) error.textContent = err.message; }
    finally { uploading = false; if (active) { renderReference(); updateControls(); } }
  } else if (draft) {
    const savedOnServer = await selectFile(async () => {
      const file = await pendingReference(draft);
      if (!file) throw new Error('This photo selection expired. Choose the image again.');
      return file;
    });
    if (savedOnServer) await discardReference(draft).catch(() => {});
  } else if (!id) {
    const sample = EXAMPLES.find(item => item.id === params.get('sample'));
    if (sample) selectExample(sample);
  }
}
