import { h, mount } from './lib.js';

const TEAM = [
  ['J Ganesh', '23BDS024', 'DSAI', 'Reconstruction, data preparation and evaluation'],
  ['M Jagadeshwar', '23BDS033', 'DSAI', 'Reconstruction, data preparation and evaluation'],
  ['B Varshith', '23BDS011', 'DSAI', 'Backend, generation jobs and Studio'],
  ['Pranav Kashyap', '23BCS098', 'CSE', 'Backend, authentication and generation jobs'],
  ['C Sai Aravind', '23BDS075', 'DSAI', 'Studio, interactive viewer and gallery'],
];

const QUESTIONS = [
  ['What can one photo reveal?', 'The visible outline and appearance constrain the estimate. The back, cavities, and physical dimensions remain ambiguous.'],
  ['Which methods were benchmarked?', 'The course depth-based method, a single-view TSDF baseline, and a visual-hull baseline. The benchmark page links to the measured results.'],
  ['How do we measure reconstruction quality?', 'The mesh benchmark compares sampled surfaces using Chamfer distance, F-score and surface IoU. The separate metrics lab also demonstrates approximate Earth Mover’s Distance.'],
  ['What does watertightness mean?', 'The topology check measures whether the surface is closed. Rotate the mesh and compare it with the original photo to assess the visible shape.'],
];

export function renderTeam(main) {
  mount(main,
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
      h('h1', {}, 'Group 17 project team'),
      h('p', {}, 'Dr. Shirshendu Layek guided the project in the Department of Data Science and Artificial Intelligence.'),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary', href: '/app/showcase#demo', 'data-link': '' }, 'Watch the course demo'),
        h('a', { class: 'btn', href: '/presentations/3DFUN-AI-Project.pptx', download: '' }, 'Download benchmark slides'),
        h('a', { class: 'btn', href: '/site/report-preview/' }, 'Read the report'))),
    h('h2', { class: 'mt24' }, 'What each member worked on'),
    h('div', { class: 'table-scroll', role: 'region', 'aria-label': 'Author contributions', tabindex: '0' },
      h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {}, ['Name', 'Roll number', 'Department', 'Work'].map(label => h('th', { scope: 'col' }, label)))),
        h('tbody', {}, TEAM.map(([name, roll, dept, work]) =>
          h('tr', {}, h('th', { scope: 'row' }, name), [roll, dept, work].map(value => h('td', {}, value))))))),
    h('p', { class: 'muted mt8' }, 'All five contributed to the report and submission materials.'),
    h('h2', { class: 'mt24' }, 'Viva practice questions'),
    h('div', { class: 'col gap8' }, QUESTIONS.map(([question, answer]) =>
      h('details', { class: 'card tight' }, h('summary', {}, question), h('p', { class: 'mt8 mb0' }, answer)))),
    h('div', { class: 'row wrap gap8 mt16' },
      h('a', { class: 'btn', href: '/app/work', 'data-link': '' }, 'Read the method notes'),
      h('a', { class: 'btn', href: '/app/benchmark', 'data-link': '' }, 'View benchmark results')));
}
