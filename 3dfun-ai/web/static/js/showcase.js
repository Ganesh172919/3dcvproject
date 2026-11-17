import { api, h, mount } from './lib.js';

function link(label, href, primary = false) {
  return h('a', { href, class: primary ? 'btn primary' : 'btn' }, label);
}

export async function renderShowcase(main) {
  const results = h('div', { class: 'table-scroll', 'aria-live': 'polite' }, 'Loading the saved scores…');
  const demo = h('section', { class: 'card mt24', id: 'demo' },
    h('h2', {}, '84-second course demo'),
    h('p', { class: 'muted' }, 'See the sneaker, duck and ring meshes from different angles. The ring shows where the single-view method fills a real opening. The video uses on-screen captions.'),
    h('video', {
      class: 'course-video', controls: true, preload: 'metadata', playsinline: true,
      src: '/demo.mp4', 'aria-label': 'Group 17 reconstruction demo',
    }, 'Open or download the MP4 to view the demo.'),
    h('div', { class: 'row wrap gap8 mt12' }, link('Open MP4', '/demo.mp4'), link('Inspect the ring', '/?example=ring#examples')));
  mount(main, h('div', { class: 'course-results' },
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, 'IIIT DHARWAD · GROUP 17'),
      h('h1', {}, 'Course report and results'),
      h('p', { class: 'lead' }, 'Our mesh pipeline reconstructs a full shape from one photo. In a separate experiment, TinyDepth estimates the visible surface from rendered views with known cameras.'),
      h('div', { class: 'row wrap gap8' },
        link('Read the IEEE report', '/site/report-preview/', true),
        h('a', { class: 'btn', href: '/report.pdf?download=1', download: 'report.pdf' }, 'Download PDF'),
        link('Mesh benchmark', '/app/benchmark'),
        link('Group 17 team', '/app/team'),
        link('Project handbook', '/project-docs'))),
    demo,
    h('section', { class: 'card mt24' },
      h('h2', {}, 'TinyDepth on the held-out cylinder'),
      h('p', { class: 'muted' }, 'Eight rendered views of one mesh. Chamfer measures distance in normalized units. F-score uses a 0.01-unit tolerance.'),
      results,
      h('div', { class: 'row wrap gap8 mt12' },
        link('Plan an experiment', '/#training'), link('Read the experiment guide', '/project-docs/reproducibility.md')))));
  try {
    const { methods } = await api('/results/tinydepth');
    if (!methods.length) throw new Error('Saved scores are missing');
    const score = value => Number(value).toFixed(4);
    mount(results, h('table', { class: 'tbl' },
      h('thead', {}, h('tr', {},
        ...['Method', 'Chamfer ↓', 'F-score ↑'].map(name => h('th', { scope: 'col' }, name)))),
      h('tbody', {}, methods.map(({ method, chamfer, f_score: fScore }) =>
        h('tr', {}, h('th', { scope: 'row' }, method), h('td', {}, score(chamfer)), h('td', {}, score(fScore)))))));
  } catch {
    results.textContent = 'Saved scores are unavailable. See the Results and Discussion section of the course report.';
  }
}
