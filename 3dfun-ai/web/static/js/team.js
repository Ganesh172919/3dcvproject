import { h, mount } from './lib.js';

const TEAM = [
  ['J Ganesh', '23BDS024', 'DSAI', '0:00–2:15', 'Problem and geometric priors'],
  ['M Jagadeshwar', '23BDS033', 'DSAI', '2:15–4:45', 'Architecture and reconstruction'],
  ['B Varshith', '23BDS011', 'DSAI', '4:45–7:30', 'Benchmark protocol and results'],
  ['Pranav Kashyap', '23BCS098', 'CSE', '7:30–10:15', 'Ablations, examples, and Studio demo'],
  ['C Sai Aravind', '23BDS075', 'DSAI', '10:15–12:00', 'Limitations, next steps, and references'],
];

const QUESTIONS = [
  ['What can one photo reveal?', 'The visible outline and appearance constrain the estimate. The back, cavities, and physical dimensions remain ambiguous.'],
  ['Which methods were benchmarked?', 'The course depth-based method, a single-view TSDF baseline, and a visual-hull baseline. The benchmark page links to the measured results.'],
  ['How are 3D reconstruction error metrics evaluated?', 'We compute symmetric Chamfer Distance (CD), Earth Mover\'s Distance (EMD), and F-Score at thresholds tau=0.01 and 0.05 against ground-truth meshes, coupled with normal consistency and watertightness checks.'],
  ['What does watertightness mean?', 'The topology check measures whether the surface is closed. Rotate the mesh and compare it with the original photo to assess the visible shape.'],
];

export function renderTeam(main) {
  mount(main,
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
      h('h1', {}, 'Project team and presentation'),
      h('p', {}, 'Project guide: Dr. Shirshendu Layek, Department of Data Science and Artificial Intelligence.'),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary', href: '/demo.mp4' }, 'Watch the course demo'),
        h('a', { class: 'btn', href: '/presentations/3DFUN-AI-Project.pptx', download: '' }, 'Download benchmark slides'),
        h('a', { class: 'btn', href: '/paper/main.pdf' }, 'Read the paper'),
        h('a', { class: 'btn', href: '/report.pdf' }, 'Read the report'))),
    h('h2', { class: 'mt24' }, 'Suggested 12-minute defense schedule'),
    h('div', { class: 'table-scroll', role: 'region', 'aria-label': 'Defense schedule', tabindex: '0' },
      h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {}, ['Presenter', 'Roll number', 'Department', 'Time', 'Topic'].map(label => h('th', { scope: 'col' }, label)))),
        h('tbody', {}, TEAM.map(([name, roll, dept, time, topic]) =>
          h('tr', {}, [name, roll, dept, time, topic].map(value => h('td', {}, value))))))),
    h('h2', { class: 'mt24' }, 'Viva practice questions'),
    h('div', { class: 'col gap8' }, QUESTIONS.map(([question, answer]) =>
      h('details', { class: 'card tight' }, h('summary', {}, question), h('p', { class: 'mt8 mb0' }, answer)))),
    h('div', { class: 'row wrap gap8 mt16' },
      h('a', { class: 'btn', href: '/app/work', 'data-link': '' }, 'Read the method notes'),
      h('a', { class: 'btn', href: '/app/benchmark', 'data-link': '' }, 'View benchmark results')));
}
