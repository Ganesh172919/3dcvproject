// 3DFUN AI — Student 3D Computer Vision Laboratory View
// Course: 3D Computer Vision · IIIT Dharwad
import { api, esc, fmt, h, mount, state, statusBadge, toast } from './lib.js';

export async function renderLabs(main) {
  mount(main, h('div', { class: 'empty' }, h('div', { class: 'spinner' }), h('p', { class: 'mt16' }, 'Loading Student Curriculum Labs…')));

  let catalog = [];
  let runs = [];
  let selectedRun = null;
  let pollInterval = null;

  async function loadData() {
    try {
      const catRes = await api('/student-labs');
      catalog = (catRes && catRes.labs) ? catRes.labs : [];
      if (state.token) {
        const runsRes = await api('/student-labs/runs');
        runs = Array.isArray(runsRes) ? runsRes : [];
      }
    } catch (err) {
      toast(err.message || 'Failed to load student labs', 'error');
    }
  }

  await loadData();

  function startPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
      const hasActive = runs.some((r) => r.status === 'queued' || r.status === 'running');
      if (hasActive && state.token) {
        try {
          const freshRuns = await api('/student-labs/runs');
          runs = Array.isArray(freshRuns) ? freshRuns : [];
          if (selectedRun) {
            const freshSel = await api(`/student-labs/runs/${selectedRun.id}`);
            selectedRun = freshSel;
          }
          render();
        } catch (_) {}
      }
    }, 2500);
  }

  function render() {
    const hasActive = runs.some((r) => r.status === 'queued' || r.status === 'running');
    if (hasActive) startPolling();

    const container = h('div', { class: 'labs-container page-content', style: 'max-width:1200px;margin:0 auto;padding:24px 16px;' });

    // Header section
    const header = h('div', { class: 'labs-header mb24', style: 'border-bottom:1px solid var(--border-soft);padding-bottom:16px;' },
      h('div', { class: 'row wrap', style: 'justify-content:space-between;align-items:flex-start;gap:16px;' },
        h('div', {},
          h('div', { class: 'course-label' }, '3D COMPUTER VISION · EDUCATIONAL LABORATORY'),
          h('h1', { style: 'font-size:28px;margin:4px 0 8px;' }, 'Interactive Student Training Curriculum'),
          h('p', { class: 'muted', style: 'max-width:760px;margin:0;' },
            'Explore, configure, and train custom deep learning models for monocular depth estimation, implicit signed distance fields (SDF), 3D Gaussian Splatting, and geometric metric evaluation. Real PyTorch execution with export to ONNX, OBJ, and PLY.')),
        h('div', { class: 'wallet-badge card tight', style: 'background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);padding:10px 16px;text-align:right;' },
          h('div', { class: 'muted', style: 'font-size:11px;text-transform:uppercase;letter-spacing:0.05em;' }, 'Student Credit Balance'),
          h('div', { style: 'font-size:20px;font-weight:700;color:var(--accent);' }, `◆ ${state.user?.credits ?? 0} Credits`),
          h('div', { class: 'dim', style: 'font-size:11px;' }, '+10 Daily Streak Refill Active'))));

    // Labs catalog grid
    const labCards = h('div', { class: 'grid c2 gap16 mb32' });

    catalog.forEach((lab) => {
      const config = { ...lab.default_config };
      const runBtn = h('button', { class: 'btn primary sm' }, `Launch Lab Run (${lab.cost_credits} Credits)`);

      // Config inputs per lab
      const configInputs = h('div', { class: 'lab-config-row row wrap gap12 mt12', style: 'background:rgba(0,0,0,0.2);padding:10px;border-radius:6px;font-size:12px;' });

      if (lab.id === 'lab1_depth') {
        const epochInput = h('input', { type: 'number', min: '1', max: '20', value: config.epochs, style: 'width:60px;' });
        epochInput.addEventListener('change', () => { config.epochs = parseInt(epochInput.value, 10) || 4; });
        const lrInput = h('input', { type: 'number', step: '0.0005', value: config.lr, style: 'width:80px;' });
        lrInput.addEventListener('change', () => { config.lr = parseFloat(lrInput.value) || 0.001; });
        configInputs.appendChild(h('label', { class: 'row gap4' }, h('span', { class: 'muted' }, 'Epochs:'), epochInput));
        configInputs.appendChild(h('label', { class: 'row gap4' }, h('span', { class: 'muted' }, 'Learning Rate:'), lrInput));
      } else if (lab.id === 'lab2_occupancy') {
        const primSelect = h('select', { style: 'padding:4px 8px;border-radius:4px;' });
        ['torus', 'sphere', 'cylinder', 'cube'].forEach((p) => {
          const opt = h('option', { value: p }, p.toUpperCase());
          if (p === config.primitive) opt.selected = true;
          primSelect.appendChild(opt);
        });
        primSelect.addEventListener('change', () => { config.primitive = primSelect.value; });
        const epochInput = h('input', { type: 'number', min: '2', max: '30', value: config.epochs, style: 'width:60px;' });
        epochInput.addEventListener('change', () => { config.epochs = parseInt(epochInput.value, 10) || 10; });
        configInputs.appendChild(h('label', { class: 'row gap4' }, h('span', { class: 'muted' }, 'Target CAD Shape:'), primSelect));
        configInputs.appendChild(h('label', { class: 'row gap4' }, h('span', { class: 'muted' }, 'Epochs:'), epochInput));
      } else if (lab.id === 'lab3_3dgs') {
        const ptsInput = h('input', { type: 'number', min: '100', max: '3000', step: '100', value: config.num_points, style: 'width:70px;' });
        ptsInput.addEventListener('change', () => { config.num_points = parseInt(ptsInput.value, 10) || 800; });
        const iterInput = h('input', { type: 'number', min: '10', max: '100', value: config.iterations, style: 'width:60px;' });
        iterInput.addEventListener('change', () => { config.iterations = parseInt(iterInput.value, 10) || 25; });
        configInputs.appendChild(h('label', { class: 'row gap4' }, h('span', { class: 'muted' }, 'Gaussians:'), ptsInput));
        configInputs.appendChild(h('label', { class: 'row gap4' }, h('span', { class: 'muted' }, 'Iterations:'), iterInput));
      } else if (lab.id === 'lab4_evaluator') {
        configInputs.appendChild(h('div', { class: 'muted' }, 'Computes symmetric Chamfer Distance (CD), EMD, and F-score @ τ=0.01 & 0.05.'));
      }

      runBtn.addEventListener('click', async () => {
        if (!state.token) {
          toast('Please sign in or create an account to run training labs', 'warn');
          return;
        }
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="spinner"></span> Launching…';
        try {
          const res = await api(`/student-labs/${lab.id}/run`, {
            method: 'POST',
            body: { config },
          });
          toast(res.message || 'Lab training started!', 'success');
          if (state.user) {
            state.user.credits = res.balance;
          }
          await loadData();
          startPolling();
          render();
        } catch (err) {
          toast(err.message || 'Failed to start lab run', 'error');
        } finally {
          runBtn.disabled = false;
          runBtn.textContent = `Launch Lab Run (${lab.cost_credits} Credits)`;
        }
      });

      const card = h('div', { class: 'card lab-card', style: 'border:1px solid var(--border);position:relative;display:flex;flex-direction:column;justify-content:space-between;' },
        h('div', {},
          h('div', { class: 'row wrap gap8', style: 'justify-content:space-between;align-items:center;margin-bottom:8px;' },
            h('span', { class: 'badge', style: 'background:rgba(99,102,241,0.15);color:var(--accent);font-weight:600;' }, lab.id.toUpperCase()),
            h('span', { class: 'mono dim', style: 'font-size:12px;' }, `◆ ${lab.cost_credits} credits`)),
          h('h3', { style: 'font-size:18px;margin:0 0 4px;' }, lab.title),
          h('div', { class: 'muted mb8', style: 'font-size:12.5px;font-style:italic;' }, lab.subtitle),
          h('p', { style: 'font-size:13px;line-height:1.5;color:var(--text-2);margin-bottom:12px;' }, lab.description),
          h('div', { class: 'mono-math-card', style: 'background:rgba(0,0,0,0.3);padding:8px 12px;border-radius:6px;font-family:var(--mono);font-size:11.5px;color:var(--cyan);margin-bottom:12px;overflow-x:auto;' },
            lab.math_formula),
          configInputs),
        h('div', { class: 'row mt16', style: 'justify-content:flex-end;' }, runBtn));

      labCards.appendChild(card);
    });

    // Runs History & Realtime Telemetry Section
    const runsSection = h('div', { class: 'card runs-dashboard mt24' },
      h('div', { class: 'row wrap gap12', style: 'justify-content:space-between;align-items:center;margin-bottom:16px;' },
        h('div', {},
          h('h2', { style: 'font-size:20px;margin:0;' }, 'Student Lab Execution History & Telemetry'),
          h('p', { class: 'muted', style: 'font-size:13px;margin:2px 0 0;' },
            runs.length ? `${runs.length} experiment run(s) recorded. Click any run to inspect loss metrics and artifacts.` : 'No lab runs yet. Launch any lab above to start an experiment.')),
        h('button', { class: 'btn sm', onclick: async () => { await loadData(); render(); } }, 'Refresh')));

    if (runs.length) {
      const runsTable = h('div', { class: 'table-scroll' },
        h('table', { class: 'tbl' },
          h('thead', {},
            h('tr', {},
              h('th', {}, 'Run ID'),
              h('th', {}, 'Curriculum Lab'),
              h('th', {}, 'Status'),
              h('th', {}, 'Started'),
              h('th', {}, 'Key Metrics'),
              h('th', { style: 'text-align:right;' }, 'Actions'))),
          h('tbody', {},
            runs.map((r) => {
              const metricsStr = r.metrics
                ? Object.entries(r.metrics)
                    .map(([k, v]) => `${k}: ${typeof v === 'number' ? v.toFixed(3) : v}`)
                    .slice(0, 3)
                    .join(' · ')
                : (r.status === 'running' ? 'Training in progress…' : '—');

              const isSel = selectedRun && selectedRun.id === r.id;

              return h('tr', { style: isSel ? 'background:rgba(99,102,241,0.08);' : '' },
                h('td', { class: 'mono', style: 'font-weight:600;color:var(--accent);' }, r.id.substring(0, 10)),
                h('td', {}, r.name || r.lab_id),
                h('td', {}, statusBadge(r.status)),
                h('td', { class: 'muted', style: 'font-size:12px;' }, fmt.date(r.created_at)),
                h('td', { class: 'mono', style: 'font-size:12px;' }, metricsStr),
                h('td', { style: 'text-align:right;' },
                  h('button', {
                    class: 'btn sm ghost',
                    onclick: async () => {
                      try {
                        const full = await api(`/student-labs/runs/${r.id}`);
                        selectedRun = full;
                        render();
                      } catch (err) {
                        toast(err.message, 'error');
                      }
                    },
                  }, isSel ? 'Selected' : 'Inspect')));
            }))));

      runsSection.appendChild(runsTable);
    }

    // Selected Run Detail Drawer / Modal
    if (selectedRun) {
      const artMap = selectedRun.artifacts || {};
      const artPills = Object.entries(artMap).map(([k, pathStr]) => {
        const basename = pathStr ? pathStr.split(/[\\/]/).pop() : k;
        const dlUrl = `/api/v1/student-labs/runs/${selectedRun.id}/artifacts/${basename}`;
        return h('a', {
          class: 'btn sm ghost',
          href: dlUrl,
          download: basename,
          target: '_blank',
          style: 'font-size:11.5px;color:var(--cyan);border-color:rgba(6,182,212,0.3);',
        }, `Download ${basename} (${k})`);
      });

      const metricsGrid = selectedRun.metrics
        ? h('div', { class: 'grid c4 gap8 mt8' },
            Object.entries(selectedRun.metrics).map(([k, v]) =>
              h('div', { class: 'card tight', style: 'background:rgba(0,0,0,0.25);padding:8px;' },
                h('div', { class: 'muted', style: 'font-size:11px;text-transform:uppercase;' }, k),
                h('div', { class: 'mono', style: 'font-size:15px;font-weight:600;color:var(--text);' },
                  typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)))))
        : h('div', { class: 'muted' }, 'Metrics will appear when execution finishes.');

      const detailCard = h('div', { class: 'card run-detail-panel mt16', style: 'border:1px solid rgba(99,102,241,0.35);background:rgba(13,17,26,0.9);' },
        h('div', { class: 'row wrap gap12', style: 'justify-content:space-between;align-items:center;margin-bottom:12px;' },
          h('div', {},
            h('h3', { style: 'margin:0;font-size:18px;' }, `Run Telemetry: ${selectedRun.name}`),
            h('div', { class: 'mono muted', style: 'font-size:12px;' }, `ID: ${selectedRun.id} · Status: ${selectedRun.status}`)),
          h('button', { class: 'btn sm ghost', onclick: () => { selectedRun = null; render(); } }, 'Close')),
        h('div', { class: 'mb12' },
          h('h4', { style: 'font-size:13px;margin:0 0 4px;' }, 'Quantitative Metrics'),
          metricsGrid),
        artPills.length ? h('div', { class: 'mb12' },
          h('h4', { style: 'font-size:13px;margin:8px 0 6px;' }, 'Exported Lab Artifacts'),
          h('div', { class: 'row wrap gap8' }, artPills)) : null,
        h('div', { class: 'mb8' },
          h('h4', { style: 'font-size:13px;margin:8px 0 4px;' }, 'Console Output & Training Log'),
          h('pre', {
            class: 'mono',
            style: 'background:#05070a;border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:12px;font-size:12px;line-height:1.45;color:#e2e8f0;max-height:220px;overflow-y:auto;white-space:pre-wrap;',
          }, selectedRun.logs || 'No log output recorded yet.')));

      runsSection.appendChild(detailCard);
    }

    container.appendChild(header);
    container.appendChild(labCards);
    container.appendChild(runsSection);

    mount(main, container);
  }

  render();
}
