// 3DCV Fun AI Lab · Model Deployment & Runtime Guide · Group 17 · IIIT Dharwad
// Complete visual guide for deploying trained PyTorch/ONNX models across DirectML, CUDA, and CPU

import { copyText, h, mount, renderMathInDOM, toast } from './lib.js';

export function renderDeployment(main) {
  document.title = 'Model Deployment Guide · 3DCV Fun AI Lab · Group 17 · IIIT Dharwad';

  const container = h('div', { class: 'page-container', style: 'max-width: 1300px; margin: 0 auto; padding: 24px;' });

  // Header Banner
  const header = h('header', { class: 'mb24' },
    h('div', { class: 'spread' },
      h('div', {},
        h('span', { class: 'badge', style: 'background: rgba(2, 132, 199, 0.1); color: #0284c7; border: 1px solid rgba(2, 132, 199, 0.2); margin-bottom: 6px; display: inline-block;' }, 'Deployment & Edge Inference Architecture'),
        h('h1', { style: 'font-size: 26px; margin: 4px 0 8px 0;' }, '🚀 Trained Model Deployment & Edge Runtime Guide'),
        h('p', { class: 'muted mb0', style: 'font-size: 14px;' },
          'Step-by-step production deployment: Hardware Execution Providers, PyTorch-to-ONNX Quantization, Docker Containers, and Embedded Edge Runtimes.')),
      h('div', { class: 'row gap8' },
        h('a', { class: 'btn sm', href: '/app/architecture', 'data-link': '' }, '← Architecture Mindmaps'),
        h('a', { class: 'btn sm', href: '/app/optimization', 'data-link': '' }, '← Performance Optimization'))));

  // Section 1: Hardware Execution Providers (Interactive Selector)
  let selectedProvider = 'directml';
  const providerDetailsCard = h('div', { class: 'p16 surface-subtle rounded border mt16' });

  const PROVIDER_INFO = {
    directml: {
      title: 'DirectX 12 DirectML (Universal GPU)',
      tag: 'Windows · Intel Arc / Iris Xe, AMD Radeon, NVIDIA GeForce',
      latency: '2.4 – 6.8 s per generation',
      memory: 'Under 1.2 GB VRAM',
      cmd: 'powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Device directml',
      desc: 'DirectML maps ONNX operators directly to Microsoft DirectX 12 compute shaders. It provides zero-configuration hardware acceleration across all Windows GPUs without requiring proprietary NVIDIA drivers.'
    },
    cuda: {
      title: 'NVIDIA CUDA 12.8 & cuDNN 9.x (Maximum Throughput)',
      tag: 'Linux & Windows · NVIDIA RTX 30xx/40xx & Data Center GPUs',
      latency: '0.8 – 2.1 s per generation',
      memory: 'Under 2.0 GB VRAM (Arena capped)',
      cmd: 'powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Device cuda',
      desc: 'Leverages Tensor Cores for INT8 matrix multiplication (IMMA/DP4A). Memory arena allocator is strictly capped at 2,048 MiB via RECON_CUDA_MEMORY_MB to eliminate out-of-memory spikes.'
    },
    cpu: {
      title: 'Multi-Threaded CPU (Universal Fallback)',
      tag: 'Cross-Platform · x86_64 & ARM64 (Intel, AMD, Apple Silicon)',
      latency: '8.5 – 18.2 s per generation',
      memory: 'Under 450 MB RAM',
      cmd: 'powershell -ExecutionPolicy Bypass -File .\\run.ps1 -Device cpu',
      desc: 'ONNX Runtime CPUExecutionProvider utilizes OpenMP intra-op parallelism scaled to physical core count. Features automatic AVX-512 and AVX2 vector SIMD acceleration with zero external dependencies.'
    }
  };

  function updateProviderView(key) {
    selectedProvider = key;
    const p = PROVIDER_INFO[key];
    providerDetailsCard.innerHTML = '';
    providerDetailsCard.append(
      h('div', { class: 'spread' },
        h('h3', { style: 'color: var(--accent); margin: 0; font-size: 16px;' }, p.title),
        h('span', { class: 'badge', style: 'background: rgba(99, 102, 241, 0.1); color: var(--accent);' }, p.tag)),
      h('p', { class: 'my8', style: 'font-size: 13px; line-height: 1.6; color: var(--text-2);' }, p.desc),
      h('div', { class: 'grid-2 gap12 my12' },
        h('div', { class: 'p12 rounded', style: 'background: #ffffff; border: 1px solid #e2e8f0;' },
          h('strong', { style: 'font-size: 12px; color: var(--text); display: block;' }, 'Average Generation Latency:'),
          h('span', { style: 'font-size: 14px; font-weight: 600; color: #059669;' }, p.latency)),
        h('div', { class: 'p12 rounded', style: 'background: #ffffff; border: 1px solid #e2e8f0;' },
          h('strong', { style: 'font-size: 12px; color: var(--text); display: block;' }, 'VRAM / Memory Allocation:'),
          h('span', { style: 'font-size: 14px; font-weight: 600; color: #0284c7;' }, p.memory))),
      h('div', { class: 'spread p12 rounded mt8', style: 'background: #0f172a; color: #f8fafc; font-family: monospace; font-size: 12px;' },
        h('span', {}, p.cmd),
        h('button', { class: 'btn ghost sm', style: 'color: #94a3b8; border-color: #334155;', onclick: () => { copyText(p.cmd); toast('Launch command copied', 'ok'); } }, 'Copy'))
    );
  }

  const providerTabs = h('div', { class: 'row wrap gap8 mt12' },
    Object.entries(PROVIDER_INFO).map(([k, info]) => {
      const btn = h('button', { class: 'btn sm', type: 'button' }, info.title.split(' (')[0]);
      btn.addEventListener('click', () => {
        providerTabs.querySelectorAll('button').forEach(b => b.classList.remove('primary'));
        btn.classList.add('primary');
        updateProviderView(k);
      });
      if (k === selectedProvider) btn.classList.add('primary');
      return btn;
    })
  );

  const providerSection = h('section', { class: 'card mb24', style: 'background: #ffffff;' },
    h('h3', { style: 'color: var(--accent); margin-top: 0;' }, '1. Hardware Execution Provider Selection'),
    h('p', { class: 'muted', style: 'font-size: 13px;' },
      'Our unified reconstruction engine supports multi-backend execution. Switch runtime hardware targets dynamically based on your available GPU/accelerator:'),
    providerTabs,
    providerDetailsCard
  );
  updateProviderView('directml');

  // Section 2: PyTorch-to-ONNX Export & Quantization Pipeline
  const exportSection = h('section', { class: 'card mb24', style: 'background: #ffffff;' },
    h('h3', { style: 'color: #0284c7; margin-top: 0;' }, '2. PyTorch Checkpoint to INT8 ONNX Export Pipeline'),
    h('p', { class: 'muted', style: 'font-size: 13px;' },
      'How to take a model trained in one of our student labs (e.g., TinyDepth Lab 1) and export it for production inference in the Studio:'),
    
    // Formula
    h('div', { class: 'formula-card my12 p12', style: 'background: #f8fafc; border: 1px solid #e2e8f0; font-family: monospace; font-size: 13px; color: #312e81; border-radius: 6px;' },
      '$$ \\mathcal{M}_{\\text{ONNX}} = \\text{torch.onnx.export}(f_\\theta, \\mathbf{x}, \\dots), \\quad \\mathcal{M}_{\\text{INT8}} = \\text{quantize\\_dynamic}(\\mathcal{M}_{\\text{ONNX}}, \\{ \\text{Linear, Conv} \\}, \\text{qint8}) $$'),
    
    // Code Snippet
    h('div', { class: 'p16 rounded', style: 'background: #0f172a; color: #f8fafc; font-family: monospace; font-size: 12px; line-height: 1.6; overflow-x: auto;' },
      '# 1. Export PyTorch weights to standard ONNX graph\n' +
      'import torch, onnx\n' +
      'from onnxruntime.quantization import quantize_dynamic, QuantType\n\n' +
      'model = TinyDepthStudentCNN().eval()\n' +
      'dummy_input = torch.randn(1, 3, 512, 512)\n' +
      'torch.onnx.export(model, dummy_input, "tinydepth.onnx",\n' +
      '                  input_names=["input_image"], output_names=["inverse_depth"],\n' +
      '                  dynamic_axes={"input_image": {0: "batch_size"}, "inverse_depth": {0: "batch_size"}})\n\n' +
      '# 2. Dynamic Symmetric INT8 Quantization (Reduces weights by 75%)\n' +
      'quantize_dynamic("tinydepth.onnx", "tinydepth_int8.onnx",\n' +
      '                 weight_type=QuantType.QInt8,\n' +
      '                 op_types_to_quantize=["Conv", "MatMul", "Add"])\n' +
      'print("Model ready for 3DCV Studio execution.")')
  );

  // Section 3: Containerized Docker & Systemd Deployment
  const containerSection = h('section', { class: 'card mb24', style: 'background: #ffffff;' },
    h('h3', { style: 'color: #059669; margin-top: 0;' }, '3. Containerized Microservice Deployment (Docker & Systemd)'),
    h('p', { class: 'muted', style: 'font-size: 13px;' },
      'Production deployment scripts for running the 3DCV Fun AI Lab on dedicated campus servers or cloud VMs:'),
    h('div', { class: 'grid-2 gap16 mt12' },
      h('div', { class: 'p16 rounded border', style: 'background: #f8fafc;' },
        h('strong', { style: 'color: #047857; font-size: 13px; display: block; margin-bottom: 6px;' }, 'A. Docker Multi-Stage Container'),
        h('pre', { style: 'font-size: 11px; margin: 0; background: #0f172a; color: #f8fafc; padding: 12px; border-radius: 4px; overflow-x: auto;' },
          'FROM python:3.12-slim AS builder\n' +
          'WORKDIR /app\n' +
          'COPY requirements.txt .\n' +
          'RUN pip install --no-cache-dir -r requirements.txt\n\n' +
          'FROM python:3.12-slim\n' +
          'WORKDIR /app\n' +
          'COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12\n' +
          'COPY . .\n' +
          'EXPOSE 8010\n' +
          'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]')),
      h('div', { class: 'p16 rounded border', style: 'background: #f8fafc;' },
        h('strong', { style: 'color: #0284c7; font-size: 13px; display: block; margin-bottom: 6px;' }, 'B. Linux Systemd Service Unit'),
        h('pre', { style: 'font-size: 11px; margin: 0; background: #0f172a; color: #f8fafc; padding: 12px; border-radius: 4px; overflow-x: auto;' },
          '[Unit]\n' +
          'Description=3DCV Fun AI Lab Course Service\n' +
          'After=network.target\n\n' +
          '[Service]\n' +
          'Type=simple\n' +
          'User=student\n' +
          'WorkingDirectory=/opt/3dfun-ai\n' +
          'ExecStart=/opt/3dfun-ai/.venv/bin/uvicorn app.main:app --port 8010\n' +
          'Restart=always\n' +
          'RestartSec=3\n\n' +
          '[Install]\n' +
          'WantedBy=multi-user.target')))
  );

  mount(container, header, providerSection, exportSection, containerSection);
  renderMathInDOM(container);
  mount(main, container);
}
