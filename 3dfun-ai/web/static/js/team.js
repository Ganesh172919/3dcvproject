import { h, mount } from './lib.js';

const TEAM = [
  ['J Ganesh', '23BDS024', 'DSAI', '0:00–2:15', 'Problem and geometric priors'],
  ['M Jagadeshwar', '23BDS033', 'DSAI', '2:15–4:45', 'Architecture and reconstruction'],
  ['B Varshith', '23BDS011', 'DSAI', '4:45–7:30', 'Benchmark protocol and results'],
  ['Pranav Kashyap', '23BCS098', 'CSE', '7:30–10:15', 'Ablations, examples, and Studio demo'],
  ['C Sai Aravind', '23BDS075', 'DSAI', '10:15–12:00', 'Limitations, next steps, and references'],
];

const SLIDES = [
  'Project and speakers', 'The single-view problem', 'CPU evaluation scope',
  'Related work', 'System architecture', 'The reconstruction pipeline',
  'Evaluation set and metrics', 'Results on identical inputs', 'Runtime and trade-offs',
  'Ablations and thickness sweep', 'Qualitative comparison', 'Studio demonstration',
  'Limitations and failure cases', 'Conclusion and next steps', 'Team and references',
];

const QUESTIONS = [
  ['What can one photo reveal?', 'The visible outline and appearance constrain the estimate. The back, cavities, and physical dimensions remain ambiguous.'],
  ['Which methods were benchmarked?', 'The course depth-based method, a single-view TSDF baseline, and a visual-hull baseline. The benchmark page links to the measured results.'],
  ['Why use synthetic scenes?', 'They provide reference geometry for Chamfer distance, F-score, and volumetric IoU. Real photographs without reference meshes are used for appearance, runtime, and topology checks.'],
  ['What does watertightness mean?', 'The topology check measures whether the surface is closed. Rotate the mesh and compare it with the original photo to assess the visible shape.'],
];

export function renderTeam(main) {
  let index = 0;
  const title = h('h3', { class: 'mb0', 'aria-live': 'polite' });
  const photo = h('img', { style: 'display:block;width:100%;height:auto;aspect-ratio:16/9;background:var(--bg-2)' });
  const open = h('a', { class: 'btn sm', target: '_blank', rel: 'noopener' }, 'Open full-size slide');
  const previous = h('button', { class: 'btn sm', type: 'button', onclick: () => show(index - 1) }, 'Previous slide');
  const next = h('button', { class: 'btn sm primary', type: 'button', onclick: () => show(index + 1) }, 'Next slide');
  const jump = h('select', { 'aria-label': 'Choose a slide', onchange: () => show(Number(jump.value)) },
    SLIDES.map((name, i) => h('option', { value: i }, (i + 1) + '. ' + name)));

  function show(position) {
    if (position < 0 || position >= SLIDES.length) return;
    index = position;
    const url = '/presentations/slides_png/Slide' + (index + 1) + '.PNG';
    title.textContent = 'Slide ' + (index + 1) + ' of ' + SLIDES.length + ' · ' + SLIDES[index];
    photo.src = open.href = url;
    photo.alt = SLIDES[index];
    jump.value = index;
    if (index === 0 && document.activeElement === previous) next.focus();
    if (index === SLIDES.length - 1 && document.activeElement === next) previous.focus();
    previous.disabled = index === 0;
    next.disabled = index === SLIDES.length - 1;
  }

  show(0);
  mount(main,
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
      h('h1', {}, 'Project team and presentation'),
      h('p', {}, 'Project guide: Dr. Shirshendu Layek, Department of Data Science and Artificial Intelligence.'),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary', href: '/presentations/3DFUN-AI-Project.pptx', download: '3DFUN-AI-Project.pptx' }, 'Download PowerPoint'),
        h('a', { class: 'btn', href: '/presentations/speaker_notes.md' }, 'Read speaker notes'),
        h('a', { class: 'btn', href: '/paper/main.pdf' }, 'Read the paper'),
        h('a', { class: 'btn', href: '/report.pdf' }, 'Read the report'))),
    h('section', { class: 'card mt16', 'aria-label': 'Presentation preview' },
      h('div', { class: 'spread wrap gap8 mb12' }, title, h('div', { class: 'row wrap gap8' }, previous, next)),
      photo,
      h('div', { class: 'spread wrap gap8 mt12' }, jump, open)),
    h('h2', { class: 'mt24' }, 'Suggested 12-minute defense schedule'),
    h('div', { class: 'table-scroll', role: 'region', 'aria-label': 'Defense schedule', tabindex: '0' },
      h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {}, ['Presenter', 'Roll number', 'Department', 'Time', 'Slides', 'Topic'].map(label => h('th', { scope: 'col' }, label)))),
        h('tbody', {}, TEAM.map(([name, roll, dept, time, topic], i) =>
          h('tr', {}, [name, roll, dept, time, (i * 3 + 1) + '–' + (i * 3 + 3), topic].map(value => h('td', {}, value))))))),
    h('h2', { class: 'mt24' }, 'Viva practice questions'),
    h('div', { class: 'col gap8' }, QUESTIONS.map(([question, answer]) =>
      h('details', { class: 'card tight' }, h('summary', {}, question), h('p', { class: 'mt8 mb0' }, answer)))),
    h('div', { class: 'row wrap gap8 mt16' },
      h('a', { class: 'btn', href: '/app/work', 'data-link': '' }, 'Read the method notes'),
      h('a', { class: 'btn', href: '/app/benchmark', 'data-link': '' }, 'View benchmark results')));
}
