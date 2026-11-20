import { EXAMPLES, homepageExampleUrl, studioExampleUrl } from './examples.js';
import { $, api, fmt, h, mount } from './lib.js';
import { rememberReference, validateReferenceFile } from './pending-upload.js';

const stage = $('#example-stage');
const demoName = $('#example-name');
const source = $('#example-source');
const runLink = $('#example-run');
const download = $('#example-download');
const status = $('#example-status');
const observation = $('#example-observation');
const photo = $('#example-photo');
const mesh = $('#example-mesh');
const wire = $('#example-wire');
const tabs = [[photo, 'photo'], [mesh, 'pbr'], [wire, 'wire']];
const sampleButtons = [];
const requested = new URLSearchParams(location.search).get('example');
let current = EXAMPLES.find(example => example.id === requested) || EXAMPLES[0];
let viewer;
let mode = 'pbr';
let ready = false;
let renderVersion = 0;

function selectMode(nextMode) {
  mode = nextMode;
  for (const [button, value] of tabs) button.setAttribute('aria-pressed', String(value === mode));
}

async function showExample(example, nextMode) {
  selectMode(nextMode);
  if (example === current && viewer && mode !== 'photo') {
    if (ready) viewer.setMode(mode);
    return;
  }
  const showMesh = mode !== 'photo';
  const version = ++renderVersion;
  current = example;
  ready = false;
  wire.disabled = showMesh;
  viewer?.dispose();
  viewer = null;
  demoName.textContent = `${example.label} · ${example.method}`;
  observation.textContent = example.observation || '';
  source.href = example.source || example.input;
  source.textContent = example.credit || 'Image source';
  runLink.href = studioExampleUrl(example);
  download.href = example.glb;
  download.download = `${example.id}.glb`;
  sampleButtons.forEach(([item, button]) => button.setAttribute('aria-pressed', String(item === example)));

  mount(stage, h('img', {
    src: showMesh ? example.preview : example.input,
    alt: showMesh ? `Rendered mesh of the ${example.label.toLowerCase()}` : `${example.label} reference photo`,
  }));
  status.textContent = showMesh ? 'Loading interactive model…' : 'Original reference image';
  if (!showMesh) return;

  try {
    const { ModelViewer } = await import('./viewer.js');
    if (version !== renderVersion) return;
    mount(stage);
    viewer = new ModelViewer(stage, { modelUrl: example.glb, previewUrl: example.preview,
      altText: `${example.label} reconstructed with ${example.method}`, autoRotate: false,
      compact: false,
      height: stage.clientHeight || 400,
      onReady: () => {
        if (version !== renderVersion) return;
        ready = true;
        wire.disabled = false;
        viewer.setMode(mode);
        status.textContent = 'Drag to rotate · Scroll or pinch to zoom';
      },
      onFallback: () => {
        if (version === renderVersion) status.textContent = 'Rendered preview · GLB available below';
      },
    });
  } catch {
    if (version === renderVersion) {
      mount(stage, h('img', { src: example.preview, alt: `Rendered mesh of the ${example.label.toLowerCase()}` }));
      status.textContent = 'Rendered preview · GLB available below';
    }
  }
}

for (const example of EXAMPLES) {
  const button = h('button', { type: 'button', class: 'sample-button', 'aria-pressed': 'false',
    onclick: () => showExample(example, mode) }, h('img', { src: example.input, alt: '', loading: 'lazy' }), example.label);
  sampleButtons.push([example, button]);
  $('#landing-samples').append(button);
}

const library = $('#prototype-library');
const prototypeEngine = $('#prototype-engine');
const prototypeCards = [];
for (const example of EXAMPLES) {
  const inspect = h('a', { class: 'btn sm primary', href: homepageExampleUrl(example) }, 'Inspect model');
  inspect.addEventListener('click', event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    history.pushState(null, '', inspect.href);
    showExample(example, 'pbr');
    demoName.focus({ preventScroll: true });
    $('#examples').scrollIntoView({ block: 'start' });
  });
  const card = h('article', { class: 'card lab-prototype' },
    h('img', { src: example.preview, alt: `Rendered 3D mesh of the ${example.label.toLowerCase()}`, loading: 'lazy', decoding: 'async' }),
    h('div', { class: 'lab-prototype-body' },
      h('h3', {}, example.label),
      h('p', { class: 'lab-prototype-method' }, example.method),
      h('p', {}, example.observation),
      example.prototype ? h('p', { class: 'lab-prototype-method' },
        `${fmt.n(example.prototype.triangles)} triangles · ${example.prototype.watertight ? 'Watertight' : 'Open mesh'}`) : null,
      h('div', { class: 'row wrap gap8 lab-prototype-actions' }, inspect,
        h('a', { class: 'btn sm', href: studioExampleUrl(example) }, 'Run in Studio'),
        h('a', { class: 'btn sm', href: example.glb, download: `${example.id}.glb`, 'aria-label': `Download ${example.label} GLB` }, 'GLB ↓'))));
  prototypeCards.push([example.engine, card]);
  library.append(card);
}
function filterPrototypes() {
  let count = 0;
  for (const [engine, card] of prototypeCards) {
    card.hidden = prototypeEngine.value !== 'all' && prototypeEngine.value !== engine;
    if (!card.hidden) count += 1;
  }
  $('#prototype-count').textContent = `${count} ${count === 1 ? 'example' : 'examples'}`;
}
prototypeEngine.addEventListener('change', filterPrototypes);
filterPrototypes();
demoName.tabIndex = -1;
window.addEventListener('popstate', () => {
  const id = new URLSearchParams(location.search).get('example');
  const example = EXAMPLES.find(item => item.id === id) || EXAMPLES[0];
  if (current !== example) showExample(example, 'pbr');
});

for (const [button, value] of tabs) button.addEventListener('click', () => showExample(current, value));
showExample(current, mode);

const menu = $('#home-menu');
const nav = $('#home-nav');
function setMenu(open) {
  menu.setAttribute('aria-expanded', String(open));
  nav.classList.toggle('open', open);
}
menu.addEventListener('click', () => setMenu(!nav.classList.contains('open')));
nav.addEventListener('click', event => { if (event.target.closest('a')) setMenu(false); });
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && nav.classList.contains('open')) {
    setMenu(false);
    menu.focus();
  }
});
document.addEventListener('click', event => {
  if (!menu.contains(event.target) && !nav.contains(event.target)) setMenu(false);
});
matchMedia('(max-width: 950px)').addEventListener('change', () => setMenu(false));

const meta = api('/meta').catch(() => null);
meta.then(data => {
  if (data?.limits?.max_upload_mb) $('#home-upload-limit').textContent = `· Up to ${data.limits.max_upload_mb} MB`;
});

async function showBenchmark() {
  const labels = { proposed: 'Course pipeline', tsdf: 'TSDF fusion', visual_hull: 'Visual hull' };
  const data = await api('/results/summary').catch(() => null);
  if (!data?.available || !data.dataset?.scenes) return;
  const scenes = data.dataset.scenes;
  const methods = (data.methods || []).filter(row => labels[row.method] && row.chamfer_mean != null
    && row.chamfer_mean !== '' && Number.isFinite(Number(row.chamfer_mean)));
  if (!methods.length) return;
  const max = Math.max(...methods.map(row => Number(row.chamfer_mean))) || 1;
  const refCount = scenes['3D render'] ?? scenes['rendered'] ?? 13;
  $('#home-benchmark-counts').textContent = `${fmt.n(scenes.total)} images · ${fmt.n(refCount)} with reference meshes`;
  mount($('#home-benchmark-values'), methods.map(row => {
    const value = Number(row.chamfer_mean);
    return h('li', { class: row.method === 'proposed' ? 'lab-benchmark-course' : null },
      h('span', {}, labels[row.method]),
      h('span', { class: 'lab-benchmark-track', 'aria-hidden': 'true' },
        h('span', { style: `width:${value / max * 100}%` })),
      h('strong', {}, value.toFixed(4)));
  }));
  $('#home-benchmark').hidden = false;
  $('#home-benchmark-link').hidden = true;
}
showBenchmark();

const device = $('#training-device');
const preset = $('#training-preset');
const ablation = $('#training-ablation');
const command = $('#training-command');
const copyStatus = $('#training-status');
const trainingHint = $('#training-hint');
const trainingPresets = {
  reference: ['', ''],
  quick: [' --views 3 --size 64 --epochs 2 --output-dir out/tinydepth-quick',
    'Uses the built-in shapes with fewer views and epochs.'],
  custom: [' --mesh-dir meshes --output-dir out/custom-depth',
    'Put at least three distinct OBJ or GLB meshes in meshes/.'],
  cached: [' --data-dir out/training_renders --output-dir out/reused-depth',
    'Use the folder that already contains your RGB, depth, masks and camera records.'],
};
function showTrainingCommand() {
  const [options, hint] = trainingPresets[preset.value];
  command.textContent = `python -m training.pipeline --device ${device.value}`
    + options
    + (ablation.checked ? ' --ablation' : '');
  trainingHint.textContent = hint;
  copyStatus.textContent = '';
}
for (const input of [device, preset, ablation]) input.addEventListener('change', showTrainingCommand);
showTrainingCommand();
$('#training-copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(command.textContent);
    copyStatus.textContent = 'Command copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(command);
    command.focus();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Command selected. Use your keyboard or browser menu to copy it.';
  }
});

const uploadButton = $('#home-upload');
const fileInput = $('#home-file');
const uploadStatus = $('#home-upload-status');
uploadButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => queueReference(fileInput.files[0]));
uploadButton.addEventListener('dragover', event => { event.preventDefault(); uploadButton.classList.add('over'); });
uploadButton.addEventListener('dragleave', () => uploadButton.classList.remove('over'));
uploadButton.addEventListener('drop', event => {
  event.preventDefault();
  uploadButton.classList.remove('over');
  queueReference(event.dataTransfer.files[0]);
});

async function queueReference(file) {
  if (!file || uploadButton.disabled) return;
  uploadButton.disabled = true;
  uploadStatus.textContent = 'Preparing your photo…';
  try {
    const settings = await meta;
    validateReferenceFile(file, settings?.limits?.max_upload_mb ?? 20);
    const draft = await rememberReference(file);
    location.assign(`/app/studio?draft=${draft}`);
  } catch (err) {
    uploadStatus.textContent = err.message || 'Could not prepare this photo. Open the Studio to upload it directly.';
    fileInput.value = '';
    uploadButton.disabled = false;
  }
}
