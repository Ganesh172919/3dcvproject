import { h, mount } from './lib.js';

const stages = [
  ['Isolate the object', 'Use the image alpha channel or estimate a mask with U²-NetP. Crop the foreground and pad it onto a square canvas.'],
  ['Estimate relative depth', 'Run Depth Anything V2 or MiDaS Small. Normalize the depth cue over foreground pixels.'],
  ['Build a volume', 'Give the silhouette a thickness based on its width and relative depth. This guesses the hidden side of the object.'],
  ['Extract and texture the mesh', 'Run Marching Cubes on the distance field, clean and simplify the mesh, then project the source photo into a texture atlas.'],
];

export function renderWork(main) {
  mount(main, h('div', { class: 'course-results' },
    h('header', { class: 'studio-title' },
      h('h1', {}, 'From a photo to a mesh'),
      h('p', { class: 'lead' }, 'This is the course pipeline used by the two depth-based providers in Studio. The optional TripoSR provider generates a mesh from a learned density field.'),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary', href: '/app/studio?sample=sneaker&engine=local&quality=draft' }, 'Try the sneaker'),
        h('a', { class: 'btn', href: '/app/inspector', 'data-link': '' }, 'Inspect the stages'),
        h('a', { class: 'btn', href: '/project-docs/architecture.md' }, 'Read the architecture notes'))),
    h('section', { class: 'card mt24' },
      h('h2', {}, 'Reconstruction steps'),
      h('ol', {}, stages.map(([title, description]) => h('li', {},
        h('h3', {}, title), h('p', {}, description))))),
    h('section', { class: 'card mt24' },
      h('h2', {}, 'Where it breaks'),
      h('p', {}, 'A closed mesh can still be the wrong shape. The ring example loses its opening, and thin structures can become too thick when the depth is ambiguous.'),
      h('a', { class: 'btn', href: '/?example=ring#examples' }, 'Compare the ring with its photo'))));
}
