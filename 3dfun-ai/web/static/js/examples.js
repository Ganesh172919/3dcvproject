const directory = '/site/examples';
const upstream = 'https://github.com/VAST-AI-Research/TripoSR/blob/107cefdc244c39106fa830359024f6a2f1c78871/examples';

export const EXAMPLES = [
  { id: 'chair', label: 'Chair', engine: 'triposr', method: 'TripoSR', source: `${upstream}/chair.png`,
    observation: 'Rotate under the seat to inspect how the model connects the thin legs and infers the hidden supports.',
    input: `${directory}/triposr-chair.png`, preview: `${directory}/triposr-chair-preview.png`, glb: `${directory}/chair.glb` },
  { id: 'teapot', label: 'Teapot', engine: 'midas', method: 'MiDaS Small', source: `${upstream}/teapot.png`,
    observation: 'Compare the outline of the spout and the opening inside the handle with the original photo.',
    input: `${directory}/triposr-teapot.png`, preview: `${directory}/midas-teapot-preview.png`, glb: `${directory}/teapot.glb` },
  { id: 'robot', label: 'Robot', engine: 'triposr', method: 'TripoSR', source: `${upstream}/robot.png`,
    observation: 'Switch to wireframe to see how the hands, feet, and gaps between parts are represented.',
    input: `${directory}/triposr-robot.png`, preview: `${directory}/triposr-robot-preview.png`, glb: `${directory}/robot.glb` },
  { id: 'sneaker', label: 'Sneaker', engine: 'local', method: 'Depth Anything V2',
    observation: 'Turn the mesh from the photographed side toward the toe to see where the projected texture stretches.',
    input: `${directory}/source-sneaker.png`, preview: `${directory}/depth-sneaker-preview.png`, glb: `${directory}/sneaker.glb` },
  { id: 'rubber-duck', label: 'Rubber duck', engine: 'local', method: 'Depth Anything V2',
    source: 'https://www.rawpixel.com/image/6030802/photo-image-public-domain-kids-cute', credit: 'Rawpixel · CC0',
    input: `${directory}/rubber-duck-input.png`, preview: `${directory}/rubber-duck-preview.png`, glb: `${directory}/rubber-duck.glb`,
    observation: 'Compare the beak and tail in profile, then orbit around the head to see how colour stretches away from the camera.',
    quality: 'draft', prototype: { triangles: 7748, watertight: true } },
  { id: 'toy-car', label: 'Toy car', engine: 'local', method: 'Depth Anything V2',
    source: 'https://commons.wikimedia.org/wiki/File:Red_car_toy.jpg', credit: 'Jebulon · CC0',
    input: `${directory}/toy-car-input.png`, preview: `${directory}/toy-car-preview.png`, glb: `${directory}/toy-car.glb`,
    observation: 'Use wireframe to check whether the wheel arches are cut into the surface or filled by the volume.',
    quality: 'draft', prototype: { triangles: 7324, watertight: true } },
  { id: 'ring', label: 'Ring', engine: 'local', method: 'Depth Anything V2',
    credit: 'Synthetic benchmark scene · CC0',
    input: `${directory}/ring-input.png`, preview: `${directory}/ring-preview.png`, glb: `${directory}/ring.glb`,
    observation: 'Orbit above the ring. The source shows an opening, but the reconstruction gives it a floor. Compare it with the MiDaS run.',
    quality: 'draft', prototype: { triangles: 7748, watertight: true } },
  { id: 'ring-midas', label: 'Ring with MiDaS', engine: 'midas', method: 'MiDaS Small',
    credit: 'Synthetic benchmark scene · CC0',
    input: `${directory}/ring-input.png`, preview: `${directory}/ring-midas-preview.png`, glb: `${directory}/ring-midas.glb`,
    observation: 'This run uses the same ring image with MiDaS. Check the depth of the depression and whether you can see through it.',
    quality: 'draft', prototype: { triangles: 8012, watertight: true } },
];

export function studioExampleUrl(example) {
  const params = new URLSearchParams({ sample: example.id, engine: example.engine });
  if (example.quality) params.set('quality', example.quality);
  return `/app/studio?${params}`;
}

export function homepageExampleUrl(example) {
  return `/?${new URLSearchParams({ example: example.id })}#examples`;
}
