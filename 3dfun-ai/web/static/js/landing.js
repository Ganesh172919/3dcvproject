import { EXAMPLES, studioExampleUrl } from './examples.js';
import { $, api, h, mount } from './lib.js';
import { rememberReference, validateReferenceFile } from './pending-upload.js';

const stage = $('#example-stage');
const demoName = $('#example-name');
const source = $('#example-source');
const runLink = $('#example-run');
const download = $('#example-download');
const status = $('#example-status');
const photo = $('#example-photo');
const mesh = $('#example-mesh');
const sampleButtons = [];
let current = EXAMPLES[0];
let viewer;
let renderVersion = 0;

async function showExample(example, showMesh) {
  const version = ++renderVersion;
  current = example;
  viewer?.dispose();
  viewer = null;
  demoName.textContent = `${example.label} · ${example.method}`;
  source.href = example.source || example.input;
  runLink.href = studioExampleUrl(example);
  download.href = example.glb;
  download.download = `${example.id}.glb`;
  photo.setAttribute('aria-pressed', String(!showMesh));
  mesh.setAttribute('aria-pressed', String(showMesh));
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
      height: matchMedia('(max-width: 620px)').matches ? 300 : 360 });
    status.textContent = 'Drag to rotate · Scroll or pinch to zoom';
  } catch {
    if (version === renderVersion) status.textContent = 'Rendered preview';
  }
}

for (const example of EXAMPLES.filter(item => item.glb)) {
  const button = h('button', { type: 'button', class: 'sample-button', 'aria-pressed': 'false',
    onclick: () => showExample(example, true) }, h('img', { src: example.input, alt: '' }), example.label);
  sampleButtons.push([example, button]);
  $('#landing-samples').append(button);
}

photo.addEventListener('click', () => showExample(current, false));
mesh.addEventListener('click', () => showExample(current, true));
showExample(current, true);

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
  uploadStatus.textContent = '';
  try {
    const meta = await api('/meta').catch(() => null);
    validateReferenceFile(file, meta?.limits?.max_upload_mb ?? 20);
    const draft = await rememberReference(file);
    location.assign(`/app/studio?draft=${draft}`);
  } catch (err) {
    uploadStatus.textContent = err.message || 'Could not prepare this photo. Open the Studio to upload it directly.';
    fileInput.value = '';
    uploadButton.disabled = false;
  }
}
