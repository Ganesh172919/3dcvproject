import { h, mount } from './lib.js';

const STAGES = [
  ['Foreground matting', 'Use an existing alpha channel or estimate the foreground with U²-Net. Crop and centre the object on a square canvas.'],
  ['Relative depth', 'Depth Anything V2 or MiDaS estimates which visible regions are nearer. The pipeline normalizes depth within the object mask.'],
  ['Symmetry prior', 'When enabled, average depth with its horizontal reflection within the mask. This can help symmetric objects and distort asymmetric ones.'],
  ['Volume and distance field', 'Give each masked pixel thickness based on relative depth. Compute a signed distance field from the inside and outside distances on the voxel grid.'],
  ['Surface extraction', 'Marching cubes extracts triangles at the zero crossing. Cleanup removes small components, smooths the surface, and attempts to fill small holes. The mesh is checked before UV seams split its vertices.'],
  ['Texturing and export', 'Project colour from the photo into a planar atlas and write a GLB. Oblique surfaces can show stretching. TripoSR instead predicts a density field and vertex colours.'],
];

const FIGURES = [
  ['Qualitative comparison', 'fig_qualitative.png', 'The original images and reconstructions from the course pipeline, TSDF and visual hull.'],
  ['Surface error', 'fig_error_heatmaps.png', 'Local distance to the reference surfaces for selected synthetic examples, shown in normalized units.'],
  ['Stage timing', 'fig_time_stages.png', 'Elapsed time for each stage in the saved CPU benchmark, including rendering.'],
  ['Chamfer distance', 'fig_chamfer_bar.png', 'Nearest-neighbour surface distance on scenes with reference meshes. Smaller values indicate closer surface agreement.'],
];

export function renderWork(main) {
  mount(main,
    h('header', { class: 'studio-title' },
      h('p', { class: 'course-label' }, '3D COMPUTER VISION · IIIT DHARWAD'),
      h('h1', {}, 'Reconstruction method'),
      h('p', { class: 'muted' }, 'The course method builds a triangle mesh from the visible outline and estimated depth of one object. The shape behind the object is inferred.'),
      h('div', { class: 'row wrap gap8' },
        h('a', { class: 'btn primary', href: '/app/studio?sample=sneaker&engine=local&quality=draft', 'data-link': '' }, 'Try the sneaker'),
        h('a', { class: 'btn', href: '/paper/main.pdf' }, 'Read the paper'),
        h('a', { class: 'btn', href: '/report.pdf' }, 'Read the project report'))),
    h('h2', { class: 'mt24' }, 'From image to mesh'),
    h('ol', { class: 'col gap8', style: 'padding-left:24px' },
      STAGES.map(([title, description]) => h('li', { class: 'card tight' },
        h('h3', {}, title), h('p', { class: 'mb0' }, description)))),
    h('section', { class: 'card mt24' },
      h('h2', {}, 'What the priors constrain'),
      h('p', {}, 'Occupied voxels project within the sampled silhouette. Interpolation and smoothing can shift the final outline. Padding helps keep the extracted surface away from the grid boundary.'),
      h('p', { class: 'mb0' }, 'Watertightness is measured on the output. A closed mesh can still miss cavities or thin parts. Taubin smoothing limits shrinkage, and the enclosed volume can change as vertices move.')),
    h('h2', { class: 'mt24' }, 'Benchmark evidence'),
    h('p', {}, 'The saved CPU experiment compares the course method with TSDF and visual hull. Geometry metrics use the synthetic scenes with reference meshes.'),
    h('div', { class: 'grid c2' },
      FIGURES.map(([title, file, description]) => {
        const url = '/figures/' + file;
        return h('article', { class: 'card col gap8' },
          h('h3', { class: 'mb0' }, title),
          h('p', { class: 'mb0' }, description),
          h('img', { src: url, alt: title, loading: 'lazy', style: 'width:100%;height:auto' }),
          h('a', { href: url }, 'Open full figure'));
      })),
    h('a', { class: 'btn mt16', href: '/app/benchmark', 'data-link': '' }, 'View the benchmark tables'));
}
