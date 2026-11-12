const directory = '/site/examples';
const upstream = 'https://github.com/VAST-AI-Research/TripoSR/blob/107cefdc244c39106fa830359024f6a2f1c78871/examples';

export const EXAMPLES = [
  { id: 'chair', label: 'Chair', engine: 'triposr', method: 'TripoSR', source: `${upstream}/chair.png`,
    input: `${directory}/triposr-chair.png`, preview: `${directory}/triposr-chair-preview.png`, glb: `${directory}/chair.glb` },
  { id: 'teapot', label: 'Teapot', engine: 'midas', method: 'MiDaS Small', source: `${upstream}/teapot.png`,
    input: `${directory}/triposr-teapot.png`, preview: `${directory}/midas-teapot-preview.png`, glb: `${directory}/teapot.glb` },
  { id: 'robot', label: 'Robot', engine: 'triposr', method: 'TripoSR', source: `${upstream}/robot.png`,
    input: `${directory}/triposr-robot.png`, preview: `${directory}/triposr-robot-preview.png`, glb: `${directory}/robot.glb` },
  { id: 'sneaker', label: 'Sneaker', engine: 'local', method: 'Depth Anything V2',
    input: `${directory}/source-sneaker.png`, preview: `${directory}/depth-sneaker-preview.png`, glb: `${directory}/sneaker.glb` },
];

export function studioExampleUrl(example) {
  return `/app/studio?${new URLSearchParams({ sample: example.id, engine: example.engine })}`;
}
