import { h, mount } from './lib.js';

const TEAM = [
  ['J Ganesh', '23BDS024', 'DSAI', 'Reconstruction, data preparation and evaluation'],
  ['M Jagadeshwar', '23BDS033', 'DSAI', 'Reconstruction, data preparation and evaluation'],
  ['B Varshith', '23BDS011', 'DSAI', 'Backend, generation jobs and Studio'],
  ['Pranav Kashyap', '23BCS098', 'CSE', 'Backend, authentication and generation jobs'],
  ['C Sai Aravind', '23BDS075', 'DSAI', 'Studio, interactive viewer and gallery'],
];

export function renderTeam(main) {
  mount(main,
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
      h('h1', {}, 'Group 17 project team'),
      h('p', {}, 'Dr. Shirshendu Layek guided the project in the Department of Data Science and Artificial Intelligence.'),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary', href: '/app/showcase#demo', 'data-link': '' }, 'Watch the course demo'),
        h('a', { class: 'btn', href: '/presentations/3DCV-Lab-Group17-Project.pptx', download: '' }, 'Download benchmark slides'),
        h('a', { class: 'btn', href: '/site/report-preview/' }, 'Read the report'))),
    h('h2', { class: 'mt24' }, 'What each member worked on'),
    h('div', { class: 'table-scroll', role: 'region', 'aria-label': 'Author contributions', tabindex: '0' },
      h('table', { class: 'tbl' },
        h('thead', {}, h('tr', {}, ['Name', 'Roll number', 'Department', 'Work'].map(label => h('th', { scope: 'col' }, label)))),
        h('tbody', {}, TEAM.map(([name, roll, dept, work]) =>
          h('tr', {}, h('th', { scope: 'row' }, name), [roll, dept, work].map(value => h('td', {}, value))))))),
    h('p', { class: 'muted mt8' }, 'All five contributed to the report and submission materials.'),
    h('div', { class: 'row wrap gap8 mt16' },
      h('a', { class: 'btn', href: '/app/work', 'data-link': '' }, 'Read the method notes'),
      h('a', { class: 'btn', href: '/app/benchmark', 'data-link': '' }, 'View benchmark results')));
}
