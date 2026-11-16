"""Specialized 3D Visualizer & Interactive WebGL HTML Viewer Generator.

Generates an HTML5/WebGL viewer for an OBJ or PLY served over HTTP.
"""
from __future__ import annotations

import argparse
from html import escape
from os.path import relpath
from pathlib import Path
from urllib.parse import quote


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>3DCV Interactive 3D Viewer - {title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin: 0;
      background: #0f172a;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
    }}
    #header {{
      position: absolute;
      top: 1rem;
      left: 1rem;
      z-index: 10;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      padding: 0.75rem 1.25rem;
      border-radius: 0.5rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    h1 {{ margin: 0 0 0.25rem 0; font-size: 1.1rem; color: #38bdf8; }}
    p {{ margin: 0; font-size: 0.85rem; color: #94a3b8; }}
    #controls {{
      position: absolute;
      bottom: 1rem;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      padding: 0.5rem 1rem;
      border-radius: 9999px;
      display: flex;
      gap: 0.75rem;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    button {{
      background: #1e293b;
      color: #f1f5f9;
      border: 1px solid #334155;
      padding: 0.35rem 0.85rem;
      border-radius: 9999px;
      cursor: pointer;
      font-size: 0.8rem;
      transition: all 0.2s;
    }}
    button:hover {{ background: #0284c7; border-color: #38bdf8; }}
    #canvas-container {{ width: 100vw; height: 100vh; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/PLYLoader.js"></script>
</head>
<body>
  <div id="header">
    <h1>3DCV Interactive Model Inspector</h1>
    <p>File: <strong>{filename}</strong> | Drag to rotate, scroll to zoom, right-drag to pan</p>
  </div>
  <div id="controls">
    <button onclick="toggleWireframe()">Toggle Wireframe</button>
    <button onclick="resetCamera()">Reset Camera</button>
    <button onclick="toggleAutoRotate()">Auto Rotate</button>
  </div>
  <div id="canvas-container"></div>

  <script>
    let scene, camera, renderer, controls, currentMesh;
    let autoRotate = true;

    function init() {{
      const container = document.getElementById('canvas-container');
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0f172a);

      camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
      camera.position.set(0, 0, 3.5);

      renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      container.appendChild(renderer.domElement);

      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = autoRotate;
      controls.autoRotateSpeed = 2.0;

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);

      const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
      dirLight1.position.set(5, 10, 7);
      scene.add(dirLight1);

      const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.5);
      dirLight2.position.set(-5, -5, -5);
      scene.add(dirLight2);

      // Grid helper
      const grid = new THREE.GridHelper(4, 20, 0x334155, 0x1e293b);
      grid.position.y = -1;
      scene.add(grid);

      loadModel("{model_data_uri}");

      window.addEventListener('resize', onWindowResize);
      animate();
    }}

    function loadModel(uri) {{
      const isPLY = "{is_ply}" === "true";
      const loader = isPLY ? new THREE.PLYLoader() : new THREE.OBJLoader();

      if (isPLY) {{
        loader.load(uri, function(geometry) {{
          geometry.computeVertexNormals();
          geometry.center();
          const material = geometry.hasAttribute('color')
            ? new THREE.PointsMaterial({{ size: 0.015, vertexColors: true }})
            : new THREE.MeshStandardMaterial({{ color: 0x38bdf8, roughness: 0.4, metalness: 0.1 }});
          currentMesh = geometry.hasAttribute('color')
            ? new THREE.Points(geometry, material)
            : new THREE.Mesh(geometry, material);
          scene.add(currentMesh);
        }});
      }} else {{
        loader.load(uri, function(obj) {{
          currentMesh = obj;
          // Center and normalize size
          const box = new THREE.Box3().setFromObject(obj);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          const maxDim = Math.max(size.x, size.y, size.z);
          const scale = 2.0 / (maxDim || 1);
          obj.position.sub(center.multiplyScalar(scale));
          obj.scale.setScalar(scale);

          obj.traverse(child => {{
            if (child.isMesh) {{
              child.material = new THREE.MeshStandardMaterial({{
                color: 0x93c5fd,
                roughness: 0.35,
                metalness: 0.2,
                wireframe: false
              }});
            }}
          }});
          scene.add(obj);
        }});
      }}
    }}

    function toggleWireframe() {{
      if (!currentMesh) return;
      currentMesh.traverse(child => {{
        if (child.isMesh && child.material) {{
          child.material.wireframe = !child.material.wireframe;
        }}
      }});
    }}

    function resetCamera() {{
      camera.position.set(0, 0, 3.5);
      controls.target.set(0, 0, 0);
      controls.update();
    }}

    function toggleAutoRotate() {{
      autoRotate = !autoRotate;
      controls.autoRotate = autoRotate;
    }}

    function onWindowResize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}

    window.onload = init;
  </script>
</body>
</html>
"""


def generate_viewer_html(model_path: Path | str, output_html_path: Path | str | None = None) -> Path:
    """Generate an HTML viewer that loads the model relative to the HTML file."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    suffix = model_path.suffix.lower()
    if suffix not in {".obj", ".ply"}:
        raise ValueError("The viewer requires an OBJ or PLY file.")

    if output_html_path is None:
        output_html_path = model_path.with_suffix(".html")
    else:
        output_html_path = Path(output_html_path)

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    model_rel = Path(relpath(model_path, output_html_path.parent)).as_posix()

    html_content = HTML_TEMPLATE.format(
        title=escape(model_path.stem),
        filename=escape(model_path.name),
        model_data_uri=quote(model_rel),
        is_ply=str(suffix == ".ply").lower(),
    )

    output_html_path.write_text(html_content, encoding="utf-8")
    print(f"[Viewer] Generated interactive WebGL 3D viewer -> {output_html_path.resolve()}")
    return output_html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "-m", type=Path, required=True, help="Input OBJ or PLY file path")
    parser.add_argument("--out", "-o", type=Path, default=None, help="Output HTML file path")
    args = parser.parse_args()

    generate_viewer_html(args.model, args.out)


if __name__ == "__main__":
    main()
