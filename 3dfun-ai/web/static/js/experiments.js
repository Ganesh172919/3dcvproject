import { fmt, h, mount, statusBadge } from './lib.js';

export const ENGINE_NAMES = { local: 'Depth Anything V2', midas: 'MiDaS Small', triposr: 'TripoSR INT8' };
export const PROVIDER_NAMES = { CPUExecutionProvider: 'CPU', CUDAExecutionProvider: 'CUDA', DmlExecutionProvider: 'DirectML' };

export function inferenceLabel(hardware) {
  const names = [...new Set(Object.values(hardware?.providers || {}))];
  return names.length ? names.map(name => PROVIDER_NAMES[name] || name).join(' + ')
    : hardware?.device === 'cpu' ? 'CPU' : '—';
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (!sorted.length) return null;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function csvCell(value) {
  let text = String(value ?? '');
  if (/^\s*[=+@-]/.test(text)) text = "'" + text;
  return `"${text.replaceAll('"', '""')}"`;
}

function exportRuns(runs) {
  const rows = [['run_id', 'image', 'image_sha256', 'engine', 'quality', 'status', 'duration_ms', 'triangles',
    'watertight', 'size_bytes', 'inference_provider', 'config_hash'], ...runs.map(run => [
    run.id, run.reference?.name, run.reference?.digest, ENGINE_NAMES[run.provider] || run.provider,
    run.quality, run.status, run.duration_ms, run.measurement?.triangles, run.measurement?.watertight,
    run.measurement?.bytes, inferenceLabel(run.measurement?.hardware), run.measurement?.config_hash,
  ])];
  const url = URL.createObjectURL(new Blob([rows.map(row => row.map(csvCell).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' }));
  h('a', { href: url, download: '3dfun-experiment-runs.csv' }).click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function experimentsPanel(runs, total, inference, scrollTable) {
  const groups = new Map();
  for (const run of runs) {
    const digest = run.reference?.digest;
    if (!digest) continue;
    if (!groups.has(digest)) groups.set(digest, []);
    groups.get(digest).push(run);
  }
  const preferred = PROVIDER_NAMES[inference?.preferred] || 'CPU';
  const engineCards = Object.entries(ENGINE_NAMES).map(([key, label]) => {
    const attempts = runs.filter(run => run.provider === key);
    const completed = attempts.filter(run => run.status === 'COMPLETED');
    const measured = median(completed.map(run => run.duration_ms));
    return h('article', { class: 'card tight experiment-summary' }, h('h3', {}, label),
      h('p', { class: 'mb0' }, `${completed.length} completed · ${attempts.length} total`),
      h('small', {}, measured === null ? 'Median runtime appears after a completed run.' : `Median runtime ${fmt.ms(measured)}`));
  });
  const header = h('div', { class: 'spread wrap' },
    h('div', {}, h('h2', { class: 'mb0' }, 'Experiment notebook'),
      h('p', { class: 'hint mb0' }, total > runs.length ? `Latest ${runs.length} of ${total} runs` : 'Compare measurements from your local runs')),
    h('span', { class: 'runtime-chip' }, `Preferred inference · ${preferred}`));
  const guide = h('details', { class: 'experiment-guide mt16' },
    h('summary', {}, 'How to compare and read your results'),
    h('p', { class: 'mt8' }, 'Hold the photo, quality, preview, and export settings fixed. Change one setting at a time, then rotate the results to the same angle and inspect the outline and hidden surfaces.'),
    h('dl', {},
      h('dt', {}, 'Inference'), h('dd', {}, 'The providers used by the models during this run. The preferred provider is the runtime setting. Meshing and file export run on the CPU.'),
      h('dt', {}, 'Time'), h('dd', {}, 'Elapsed time from worker start through asset creation. Compare repeated runs on the same machine if you want to account for model startup.'),
      h('dt', {}, 'Triangles and watertightness'), h('dd', {}, 'Triangles count mesh faces. Watertightness records whether the surface closes. Inspect the mesh to judge whether its shape is plausible.'),
      h('dt', {}, 'Configuration hash'), h('dd', {}, 'Included in the CSV when available. Matching hashes identify the same recorded configuration. The image SHA-256 identifies the input file.')),
    h('div', { class: 'row wrap gap12' },
      h('a', { href: '/#guide' }, 'Read the lab guide'),
      h('a', { href: '/app/benchmark', 'data-link': '' }, 'View the course benchmarks')));
  const comparison = h('div', { class: 'mt16' });
  if (!groups.size) {
    mount(comparison, h('p', { class: 'mb0' }, 'Run the same photo with different models to compare runtime, mesh size, and watertightness.'),
      h('a', { class: 'btn sm mt8', href: '/app/studio?sample=teapot', 'data-link': '' }, 'Start with the teapot example'));
  } else {
    const choices = [...groups].sort(([, a], [, b]) => b.length - a.length);
    const input = h('select', { id: 'experiment-photo', onchange: showComparison }, choices.map(([digest, related]) =>
      h('option', { value: digest }, `${related[0].reference.name} · ${related.length} ${related.length === 1 ? 'run' : 'runs'}`)));
    const results = h('div', { class: 'col' });
    function showComparison() {
      const related = groups.get(input.value);
      const latest = related[0];
      const params = new URLSearchParams({ upload: latest.upload_id, quality: latest.quality });
      mount(results, h('div', { class: 'spread wrap' },
        h('div', { class: 'experiment-reference' }, h('img', { src: latest.source_image_url, alt: '' }),
          h('span', {}, 'Matching image files are grouped by SHA-256. Compare runs with the same quality setting.')),
        h('div', { class: 'row wrap gap8' },
          h('a', { class: 'btn sm', href: `/app/studio?${params}`, 'data-link': '' }, 'Run this image again'),
          h('button', { class: 'btn sm', type: 'button', onclick: () => exportRuns(related) }, 'Download CSV'))),
        scrollTable('Runs on the selected image',
          h('thead', {}, h('tr', {}, ['Model', 'Quality', 'Status', 'Inference', 'Time', 'Triangles', 'Watertight', ''].map(label => h('th', { scope: 'col' }, label)))),
          h('tbody', {}, related.map(run => h('tr', {},
            h('td', {}, ENGINE_NAMES[run.provider] || run.provider), h('td', {}, run.quality),
            h('td', {}, statusBadge(run.status)), h('td', {}, inferenceLabel(run.measurement?.hardware)),
            h('td', {}, fmt.ms(run.duration_ms)), h('td', {}, run.measurement ? fmt.n(run.measurement.triangles) : '—'),
            h('td', {}, run.measurement ? (run.measurement.watertight ? 'Yes' : 'No') : '—'),
            h('td', {}, h('a', { class: 'btn sm', 'data-link': '', href: run.asset_id
              ? `/app/assets/${run.asset_id}` : `/app/studio?generation=${run.id}` }, run.asset_id ? 'Open' : 'Track')))))));
    }
    mount(comparison, h('label', { class: 'field experiment-photo' }, h('span', {}, 'Compare runs on the same photo'), input), results);
    showComparison();
  }
  return h('section', { class: 'card mt16' }, header, h('div', { class: 'grid c3 mt16' }, engineCards), comparison, guide);
}
