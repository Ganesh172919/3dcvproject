// Reusable Three.js Model Viewer with OrbitControls, PBR/Normals/Wireframe/
// Clay/Depth Cloud modes, camera FOV, lighting environment, and PNG snapshot.
// Course: 3D Computer Vision Course Project · IIIT Dharwad
// Supervisor: Dr. Shirshendu Layek (Dept. of DSAI)
// Authors: J Ganesh (23BDS024), M Jagadeshwar (23BDS033), B Varshith (23BDS011),
//          Pranav Kashyap (23BCS098), C Sai Aravind (23BDS075)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { GLTFLoader } from 'three/addons/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/RoomEnvironment.js';

function disposeModel(root) {
  const geometry = new Set();
  const materials = new Set();
  const textures = new Set();
  root?.traverse(object => {
    if (!object.isMesh) return;
    geometry.add(object.geometry);
    const original = object.userData.originalMaterial || object.material;
    for (const material of [original].flat()) {
      materials.add(material);
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    }
  });
  for (const resource of [...textures, ...materials, ...geometry]) resource.dispose();
}

export function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (c.getContext('webgl2') || c.getContext('webgl')));
  } catch {
    return false;
  }
}

export class ModelViewer {
  constructor(container, { modelUrl, previewUrl, altText = '', autoRotate = true, height = 420 } = {}) {
    this.container = container;
    this.modelUrl = modelUrl;
    this.previewUrl = previewUrl;
    this.altText = altText;
    this.height = height;
    this.mode = 'pbr';
    this.autoRotate = autoRotate;
    this.showGrid = false;
    this.disposed = false;
    this.disposables = [];
    this.needsRender = true;
    this._init();
  }

  _init() {
    const wrap = document.createElement('div');
    wrap.className = 'viewer-wrap';
    wrap.style.height = `${this.height}px`;
    this.wrap = wrap;

    this.hud = document.createElement('div');
    this.hud.className = 'viewer-hud';
    this.hud.setAttribute('aria-hidden', 'true');

    this.tools = document.createElement('div');
    this.tools.className = 'viewer-tools';
    this.toolsRole();

    this.modes = document.createElement('div');
    this.modes.className = 'viewer-modes';
    this.modes.setAttribute('role', 'group');
    this.modes.setAttribute('aria-label', 'Render mode');

    wrap.append(this.hud, this.tools, this.modes);
    this.container.appendChild(wrap);

    if (!webglAvailable()) {
      wrap.appendChild(this._fallback('WebGL is unavailable in this browser. Showing the rendered preview instead.'));
      return;
    }

    try {
      this._initThree();
      this._load();
    } catch (err) {
      console.warn('viewer init failed', err);
      wrap.appendChild(this._fallback('The 3D viewer could not start. Showing the rendered preview.'));
    }
  }

  toolsRole() {
    const mk = (label, title, onClick, pressed = false) => {
      const b = document.createElement('button');
      b.className = 'btn sm';
      b.textContent = label;
      b.title = title;
      b.setAttribute('aria-label', title);
      if (pressed) b.setAttribute('aria-pressed', 'true');
      b.addEventListener('click', onClick);
      this.tools.appendChild(b);
      return b;
    };
    this.btnRotate = mk('⟳', 'Toggle auto-rotate', () => {
      this.autoRotate = !this.autoRotate;
      if (this.controls) this.controls.autoRotate = this.autoRotate;
      this.btnRotate.setAttribute('aria-pressed', String(this.autoRotate));
      this.btnRotate.classList.toggle('on', this.autoRotate);
    }, this.autoRotate);
    this.btnGrid = mk('▦', 'Toggle grid', () => {
      this.showGrid = !this.showGrid;
      if (this.grid) this.grid.visible = this.showGrid;
      this.needsRender = true;
      this.btnGrid.setAttribute('aria-pressed', String(this.showGrid));
      this.btnGrid.classList.toggle('on', this.showGrid);
    });

    // Caliper bounding box toggle
    this.btnCaliper = mk('📏', 'Toggle 3D Caliper Bounding Box', () => {
      this.showCaliper = !this.showCaliper;
      if (this.boxHelper) this.boxHelper.visible = this.showCaliper;
      this.needsRender = true;
      this.btnCaliper.setAttribute('aria-pressed', String(this.showCaliper));
      this.btnCaliper.classList.toggle('on', this.showCaliper);
    });

    // Camera FOV slider
    const fovWrap = document.createElement('label');
    fovWrap.className = 'viewer-fov-control';
    fovWrap.title = 'Camera Field of View (25° - 75°)';
    const fovLabel = document.createElement('span');
    fovLabel.textContent = '38°';
    const fovSlider = document.createElement('input');
    fovSlider.type = 'range';
    fovSlider.min = '25';
    fovSlider.max = '75';
    fovSlider.value = '38';
    fovSlider.addEventListener('input', (e) => {
      const val = Number(e.target.value);
      this.camera.fov = val;
      this.camera.updateProjectionMatrix();
      fovLabel.textContent = `${val}°`;
      this.needsRender = true;
    });
    fovWrap.append(fovLabel, fovSlider);
    this.tools.appendChild(fovWrap);

    // Lighting environment switcher
    const lightSelect = document.createElement('select');
    lightSelect.className = 'viewer-select';
    lightSelect.title = 'Lighting environment';
    lightSelect.setAttribute('aria-label', 'Lighting Environment');
    [
      ['studio', '☀ Studio'],
      ['neutral', '⚖ Neutral'],
      ['daylight', '☼ Daylight'],
      ['rim', '✨ Rim Light'],
    ].forEach(([val, label]) => {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = label;
      lightSelect.appendChild(opt);
    });
    lightSelect.addEventListener('change', (e) => this.setLighting(e.target.value));
    this.tools.appendChild(lightSelect);

    mk('⌂', 'Reset camera', () => this.resetCamera());
    mk('📱', 'Mobile AR QuickLook', () => this.openAR());
    mk('⛶', 'Toggle fullscreen', () => this.fullscreen());
    mk('⤓', 'Download transparent PNG snapshot', () => this.screenshot());
  }

  setLighting(mode) {
    this.lightingMode = mode;
    this.needsRender = true;
    if (!this.key || !this.fill || !this.ambient) return;
    if (mode === 'studio') {
      this.key.color.setHex(0xffffff);
      this.key.intensity = 2.1;
      this.key.position.set(2.4, 3.2, 2.2);
      this.fill.color.setHex(0xbfd8ff);
      this.fill.intensity = 0.75;
      this.fill.position.set(-2.6, 1.2, -1.6);
      this.ambient.intensity = 0.35;
    } else if (mode === 'neutral') {
      this.key.color.setHex(0xffffff);
      this.key.intensity = 1.3;
      this.key.position.set(1.5, 3.0, 2.0);
      this.fill.color.setHex(0xffffff);
      this.fill.intensity = 1.1;
      this.fill.position.set(-2.0, 1.5, -1.0);
      this.ambient.intensity = 0.6;
    } else if (mode === 'daylight') {
      this.key.color.setHex(0xfffaed);
      this.key.intensity = 2.8;
      this.key.position.set(3.5, 4.0, 2.5);
      this.fill.color.setHex(0xa0c4ff);
      this.fill.intensity = 1.2;
      this.fill.position.set(-3.0, 2.0, -2.0);
      this.ambient.intensity = 0.45;
    } else if (mode === 'rim') {
      this.key.color.setHex(0x94a3b8);
      this.key.intensity = 0.6;
      this.key.position.set(0, 1.5, 2.5);
      this.fill.color.setHex(0x38bdf8);
      this.fill.intensity = 3.2;
      this.fill.position.set(-2.5, 2.8, -3.2);
      this.ambient.intensity = 0.15;
    }
  }

  openAR() {
    if (!this.modelUrl) return;
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
    if (isIOS) {
      const usdzUrl = this.modelUrl.replace(/\.glb$/i, '.usdz');
      const a = document.createElement('a');
      a.setAttribute('rel', 'ar');
      a.setAttribute('href', usdzUrl);
      a.appendChild(document.createElement('img'));
      a.click();
    } else {
      const fullUrl = new URL(this.modelUrl, window.location.href).toString();
      const intentUrl = `intent://arvr.google.com/scene-viewer/1.0?file=${encodeURIComponent(fullUrl)}&mode=ar_only#Intent;scheme=https;package=com.google.android.googlequicksearchbox.GSA;action=android.intent.action.VIEW;end;`;
      window.location.href = intentUrl;
    }
  }

  _modeButtons() {
    const modes = [
      ['pbr', 'PBR'],
      ['normals', 'Normals'],
      ['wire', 'Wireframe'],
      ['clay', 'Clay Studio'],
      ['depthcloud', 'Depth Cloud'],
      ['albedo', 'Albedo'],
    ];
    this.modeButtons = modes.map(([key, label]) => {
      const b = document.createElement('button');
      b.className = `btn sm ${key === this.mode ? 'on' : ''}`;
      b.textContent = label;
      b.setAttribute('aria-pressed', String(key === this.mode));
      b.addEventListener('click', () => this.setMode(key));
      this.modes.appendChild(b);
      return { key, el: b };
    });
  }

  setMode(mode) {
    this.mode = mode;
    this.modeButtons.forEach(({ key, el }) => {
      el.classList.toggle('on', key === mode);
      el.setAttribute('aria-pressed', String(key === mode));
    });
    this._applyMode();
  }

  _applyMode() {
    if (!this.root) return;
    this.needsRender = true;
    this._updateHud();

    if (this.mode === 'depthcloud') {
      this.root.visible = false;
      if (this.pointsCloud) this.pointsCloud.visible = true;
      return;
    } else {
      this.root.visible = true;
      if (this.pointsCloud) this.pointsCloud.visible = false;
    }

    this.root.traverse((obj) => {
      if (!obj.isMesh) return;
      if (this.mode === 'wire') {
        obj.material = this.wireMaterial;
      } else if (this.mode === 'normals') {
        obj.material = this.normalMaterial;
      } else if (this.mode === 'clay') {
        obj.material = this.clayMaterial;
      } else {
        obj.material = obj.userData.originalMaterial;
        if (this.mode === 'albedo') {
          if (!obj.userData.albedoMaterial) {
            const map = obj.userData.originalMaterial.map || null;
            const vertexColors = Boolean(obj.geometry.attributes.color);
            obj.userData.albedoMaterial = new THREE.MeshBasicMaterial({
              map, vertexColors, color: map || vertexColors ? 0xffffff : 0x9aa6b5,
            });
            this.disposables.push(obj.userData.albedoMaterial);
          }
          obj.material = obj.userData.albedoMaterial;
        }
      }
    });
  }

  _initThree() {
    const w = this.wrap.clientWidth || 640;
    const h = this.height;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('role', 'img');
    this.renderer.domElement.setAttribute('aria-label', this.altText || 'Interactive 3D model. Drag to orbit, scroll to zoom.');
    this.wrap.insertBefore(this.renderer.domElement, this.wrap.firstChild);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, w / h, 0.01, 100);
    this.camera.position.set(1.35, 0.95, 1.85);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const room = new RoomEnvironment();
    this.scene.environment = pmrem.fromScene(room, 0.04).texture;
    room.dispose();
    pmrem.dispose();

    this.key = new THREE.DirectionalLight(0xffffff, 2.1);
    this.key.position.set(2.4, 3.2, 2.2);
    this.scene.add(this.key);
    this.fill = new THREE.DirectionalLight(0xbfd8ff, 0.75);
    this.fill.position.set(-2.6, 1.2, -1.6);
    this.scene.add(this.fill);
    this.ambient = new THREE.AmbientLight(0xffffff, 0.35);
    this.scene.add(this.ambient);

    this.boxHelper = new THREE.Box3Helper(new THREE.Box3(), 0x6366f1);
    this.boxHelper.visible = false;
    this.scene.add(this.boxHelper);

    this.grid = new THREE.GridHelper(4, 20, 0x9298bb, 0xc2c7dc);
    this.grid.position.y = -0.52;
    this.grid.visible = this.showGrid;
    this.scene.add(this.grid);

    this.wireMaterial = new THREE.MeshBasicMaterial({ color: 0x4338a3, wireframe: true });
    this.normalMaterial = new THREE.MeshNormalMaterial();
    this.clayMaterial = new THREE.MeshStandardMaterial({
      color: 0xb9c2cf, roughness: 0.72, metalness: 0.04, flatShading: false,
    });
    this.disposables.push(this.wireMaterial, this.clayMaterial, this.normalMaterial);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.addEventListener('change', () => { this.needsRender = true; });
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.autoRotate = this.autoRotate;
    this.controls.autoRotateSpeed = 1.5;
    this.controls.minDistance = 0.7;
    this.controls.maxDistance = 8;
    this.controls.target.set(0, 0, 0);

    this._modeButtons();
    this._animate();

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.wrap);
  }

  _onResize() {
    if (!this.renderer) return;
    const w = this.wrap.clientWidth || 640;
    const height = this.wrap.clientHeight || this.height;
    this.camera.aspect = w / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, height);
    this.needsRender = true;
  }

  _animate() {
    if (this.disposed) return;
    this._raf = requestAnimationFrame(() => this._animate());
    this.controls.update();
    if (this.needsRender) {
      this.needsRender = false;
      this.renderer.render(this.scene, this.camera);
    }
  }

  _updateHud() {
    if (!this.stats) return;
    const dim = this.dimensions ? `<span>📏 ${this.dimensions.x.toFixed(2)}×${this.dimensions.y.toFixed(2)}×${this.dimensions.z.toFixed(2)}</span>` : '';
    const density = (this.mode === 'wire' && this.stats.vertices > 0)
      ? `<span title="Topology mesh density">🕸 ${(this.stats.triangles / this.stats.vertices).toFixed(2)} tris/vert</span>`
      : '';
    this.hud.innerHTML = `
      <span>${this.stats.triangles.toLocaleString()} tris</span>
      <span title="Vertices after export, including face corners and texture seams">${this.stats.vertices.toLocaleString()} render verts</span>
      ${density}
      ${dim}
      <span>${this.stats.materials} mat</span>
      <span>${this.stats.textures} tex</span>`;
  }

  _load() {
    const loader = new GLTFLoader();
    const loadingEl = document.createElement('div');
    loadingEl.className = 'viewer-fallback';
    loadingEl.innerHTML = '<span class="spinner"></span><p class="mt8">Loading geometry…</p>';
    this.wrap.appendChild(loadingEl);
    loader.load(this.modelUrl, (gltf) => {
      loadingEl.remove();
      if (this.disposed) { disposeModel(gltf.scene); return; }
      this.root = gltf.scene;
      let triangles = 0; let vertices = 0;
      const materials = new Set();
      const textures = new Set();
      this.root.traverse((obj) => {
        if (!obj.isMesh) return;
        obj.castShadow = true;
        obj.receiveShadow = true;
        const geo = obj.geometry;
        obj.userData.originalMaterial = obj.material;
        if (obj.material && obj.material.map) obj.material.map.colorSpace = THREE.SRGBColorSpace;
        const idx = geo.index ? geo.index.count : (geo.attributes.position?.count || 0);
        triangles += Math.floor(idx / 3);
        vertices += geo.attributes.position?.count || 0;
        for (const material of [obj.material].flat()) {
          materials.add(material);
          for (const value of Object.values(material)) if (value?.isTexture) textures.add(value.source);
        }
      });
      this.stats = { triangles, vertices, materials: materials.size, textures: textures.size };
      this.scene.add(this.root);

      // Synthesize depth point cloud from mesh vertices
      const ptsGroup = new THREE.Group();
      this.root.traverse((obj) => {
        if (!obj.isMesh) return;
        const geo = obj.geometry.clone();
        geo.computeBoundingBox();
        const bbox = geo.boundingBox;
        const minZ = bbox.min.z;
        const rangeZ = Math.max(bbox.max.z - minZ, 0.001);
        const count = geo.attributes.position.count;
        const colors = new Float32Array(count * 3);
        const col = new THREE.Color();
        for (let i = 0; i < count; i++) {
          const z = geo.attributes.position.getZ(i);
          const t = Math.min(Math.max((z - minZ) / rangeZ, 0), 1);
          col.setHSL((1.0 - t) * 0.65, 0.9, 0.55);
          colors[i * 3] = col.r;
          colors[i * 3 + 1] = col.g;
          colors[i * 3 + 2] = col.b;
        }
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const ptsMat = new THREE.PointsMaterial({
          size: 0.016,
          vertexColors: true,
          sizeAttenuation: true,
        });
        this.disposables.push(ptsMat);
        const pts = new THREE.Points(geo, ptsMat);
        ptsGroup.add(pts);
      });
      this.pointsCloud = ptsGroup;
      this.pointsCloud.visible = false;
      this.scene.add(this.pointsCloud);

      this._frameModel();
      this._applyMode();
      this._updateHud();
      if (this.stats.triangles > 120000) {
        this.hud.insertAdjacentHTML('beforeend', '<span title="Large mesh">perf ⚠</span>');
      }
    }, undefined, (err) => {
      loadingEl.remove();
      if (this.disposed) return;
      console.warn('model load failed', err);
      this.wrap.appendChild(this._fallback('The model could not be loaded in the viewer.'));
    });
  }

  _frameModel() {
    this.needsRender = true;
    const box = new THREE.Box3().setFromObject(this.root);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    this.dimensions = { x: size.x, y: size.y, z: size.z };
    this.root.position.sub(center);
    if (this.pointsCloud) this.pointsCloud.position.sub(center);
    if (this.boxHelper) this.boxHelper.box.setFromObject(this.root);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const distance = (maxDim / 2) / Math.tan((this.camera.fov * Math.PI) / 360) * (this.camera.aspect < 1 ? 1.85 : 1.5);
    this.camera.position.set(distance * 0.62, distance * 0.48, distance * 0.85);
    this.camera.near = maxDim / 100;
    this.camera.far = maxDim * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.set(0, 0, 0);
    this.controls.minDistance = maxDim * 0.45;
    this.controls.maxDistance = maxDim * 5;
    this.grid.position.y = -(size.y / 2) - 0.02;
    this.controls.update();
  }

  resetCamera() { if (this.root) this._frameModel(); }

  fullscreen() {
    const el = this.wrap;
    if (document.fullscreenElement) document.exitFullscreen();
    else if (el.requestFullscreen) el.requestFullscreen();
    else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
  }

  screenshot() {
    if (!this.renderer) return;
    this.renderer.render(this.scene, this.camera);
    const url = this.renderer.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `3dfun-view-${Date.now()}.png`;
    a.click();
  }

  _fallback(message) {
    const box = document.createElement('div');
    box.className = 'viewer-fallback';
    box.innerHTML = `<p>${message}</p>`;
    if (this.previewUrl) {
      const img = document.createElement('img');
      img.src = this.previewUrl;
      img.alt = this.altText || 'Rendered preview of the generated 3D model';
      img.style.maxWidth = '100%';
      img.style.borderRadius = '12px';
      box.appendChild(img);
    }
    return box;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this._raf);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    if (this.controls) this.controls.dispose();
    this.disposables.forEach((d) => { try { d.dispose && d.dispose(); } catch { /* noop */ } });
    disposeModel(this.root);
    if (this.boxHelper) {
      this.boxHelper.geometry.dispose();
      this.boxHelper.material.dispose();
    }
    if (this.pointsCloud) {
      this.pointsCloud.traverse((c) => {
        if (c.isPoints) {
          c.geometry.dispose();
          c.material.dispose();
        }
      });
    }
    this.grid?.geometry.dispose();
    this.grid?.material.dispose();
    this.scene?.environment?.dispose();
    if (this.renderer) {
      this.renderer.dispose();
      this.renderer.forceContextLoss && this.renderer.forceContextLoss();
      this.renderer.domElement.remove();
    }
    if (this.wrap && this.wrap.parentNode) this.wrap.remove();
  }
}
