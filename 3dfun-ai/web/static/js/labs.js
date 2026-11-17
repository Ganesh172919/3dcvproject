import { api, fmt, h, mount, refreshUser, state, statusBadge, toast } from './lib.js';

const INPUTS = {
  lab1_depth: [
    ['epochs', 'Epochs', 1, 20, 1], ['lr', 'Learning rate', 0.000001, 0.1, 'any'],
    ['alpha', 'SILog weight', 0, 10, 'any'], ['beta', 'Gradient weight', 0, 10, 'any'],
    ['gamma', 'Normal weight', 0, 10, 'any'],
  ],
  lab2_occupancy: [
    ['primitive', 'Shape', ['torus', 'sphere', 'cylinder', 'cube']],
    ['epochs', 'Epochs', 1, 30, 1], ['hidden_dim', 'Hidden width', 16, 256, 1],
    ['lr', 'Learning rate', 0.000001, 0.1, 'any'],
  ],
  lab3_3dgs: [
    ['num_points', 'Gaussians', 100, 3000, 1], ['iterations', 'Steps', 1, 100, 1],
    ['lr', 'Learning rate', 0.000001, 0.1, 'any'],
  ],
  lab4_evaluator: [['num_points', 'Sample points', 16, 100000, 1], ['seed', 'Seed', 0, 2 ** 32 - 1, 1]],
};

const METRICS = {
  mae: 'MAE', delta1: 'δ1', final_loss: 'Final loss', epochs: 'Epochs', accuracy: 'Accuracy', iou: 'IoU',
  vertices: 'Vertices', faces: 'Faces', num_gaussians: 'Gaussians', final_psnr: 'PSNR',
  chamfer_symmetric: 'Chamfer', f_score_tau_01: 'F-score (0.01)', f_score_tau_05: 'F-score (0.05)', emd: 'Approx. EMD',
};

const active = run => run.status === 'queued' || run.status === 'running';
const display = value => typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(4) : (value ?? '—');

export async function renderLabs(main) {
  const history = h('section', { class: 'card mt24', 'aria-labelledby': 'lab-history' },
    h('h2', { id: 'lab-history' }, 'Recent runs'));
  const historyBody = h('div', {});
  history.append(historyBody);
  const catalog = h('div', { class: 'grid c2 gap16' });
  mount(main, h('div', { class: 'course-results' },
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, 'IIIT DHARWAD · GROUP 17'),
      h('h1', {}, 'Student labs'),
      h('p', { class: 'lead' }, 'Try a small depth, occupancy or splatting exercise. Use the metrics lab to compare a clean point cloud with a noisy one.'),
      h('a', { href: '/#training' }, 'Plan a TinyDepth experiment')),
    catalog, history));

  let closed = false;
  let timer;
  let selectedId;
  let runs = [];
  state.routeCleanup = () => {
    closed = true;
    clearTimeout(timer);
  };

  function makeCard(lab) {
    const config = { ...lab.default_config };
    const fields = INPUTS[lab.id].map(([key, label, options, max, step]) => {
      const input = Array.isArray(options)
        ? h('select', {}, options.map(option => h('option', { value: option, selected: option === config[key] }, option)))
        : h('input', { type: 'number', min: options, max, step, value: config[key], required: true });
      input.addEventListener('change', () => { config[key] = input.type === 'number' ? input.valueAsNumber : input.value; });
      return h('label', { class: 'field' }, h('span', {}, label), input);
    });
    const status = h('p', { class: 'form-error', role: 'status' });
    const action = lab.id === 'lab4_evaluator' ? 'Run evaluation' : 'Start training';
    const button = h('button', { class: 'btn primary sm', type: 'submit', disabled: !lab.available },
      `${action} · ${lab.cost_credits} credits`);
    return h('form', {
      class: 'card col gap8 lab-run-form',
      onsubmit: async event => {
        event.preventDefault();
        button.disabled = true;
        status.textContent = '';
        try {
          const result = await api(`/student-labs/${lab.id}/run`, { method: 'POST', body: { config } });
          if (closed) return;
          selectedId = result.run_id;
          await refreshUser();
          await updateRuns();
          toast(result.message);
        } catch (err) {
          const detail = err.detail?.details?.[0];
          status.textContent = detail ? `${err.message} ${detail.field}: ${detail.issue}` : err.message;
        } finally {
          button.disabled = false;
        }
      },
    },
    h('h2', { class: 'mb0' }, lab.title), h('p', { class: 'muted mb8' }, lab.description),
    h('div', { class: 'grid c2 gap8' }, fields),
    lab.available ? null : h('p', { class: 'muted' },
      'Install PyTorch to start this lab. ', h('a', { href: '/project-docs/reproducibility.md' }, 'Experiment guide')),
    button, status);
  }

  async function showDetails(id) {
    const detail = await api(`/student-labs/runs/${id}`);
    if (closed || id !== selectedId) return;
    const panel = history.querySelector('.lab-detail');
    if (!panel) return;
    mount(panel,
      h('h3', {}, detail.name),
      detail.metrics ? h('dl', { class: 'grid c2 gap8' }, Object.entries(detail.metrics).map(([key, value]) =>
        h('div', {}, h('dt', { class: 'muted' }, METRICS[key] || key), h('dd', {}, display(value))))) : null,
      h('div', { class: 'row wrap gap8' }, Object.values(detail.artifacts || {}).map(filename =>
        h('a', { class: 'btn sm', download: filename, href: `/api/v1/student-labs/runs/${id}/artifacts/${encodeURIComponent(filename)}` },
          `Download ${filename}`))),
      h('p', { class: 'muted mt8' }, detail.logs || 'This run is still in progress.'));
  }

  async function updateRuns() {
    clearTimeout(timer);
    try {
      runs = await api('/student-labs/runs');
      if (closed) return;
      mount(historyBody,
        runs.length ? h('div', { class: 'table-scroll' }, h('table', { class: 'tbl' },
          h('thead', {}, h('tr', {}, ['Lab', 'Status', 'Started', ''].map(text => h('th', { scope: 'col' }, text)))),
          h('tbody', {}, runs.map(run => h('tr', {},
            h('td', {}, run.name), h('td', {}, statusBadge(run.status)), h('td', {}, fmt.date(run.created_at)),
            h('td', {}, h('button', { class: 'btn sm', onclick: async () => {
              selectedId = run.id;
              try { await showDetails(run.id); } catch (err) { toast(err.message, 'error'); }
            } }, 'Inspect'))))))) : h('p', { class: 'muted' }, 'No runs yet.'),
        h('div', { class: 'lab-detail mt16', 'aria-live': 'polite' }));
      if (selectedId) await showDetails(selectedId);
      if (!closed && runs.some(active)) timer = setTimeout(updateRuns, 2500);
      else if (!closed && selectedId) await refreshUser();
    } catch (err) {
      if (closed) return;
      mount(historyBody, h('p', { role: 'alert' }, err.message),
        h('button', { class: 'btn sm', onclick: updateRuns }, 'Try again'));
    }
  }

  try {
    const { labs } = await api('/student-labs');
    if (closed) return;
    mount(catalog, labs.map(makeCard));
    await updateRuns();
  } catch (err) {
    if (!closed) mount(catalog, h('p', { role: 'alert' }, err.message));
  }
}
