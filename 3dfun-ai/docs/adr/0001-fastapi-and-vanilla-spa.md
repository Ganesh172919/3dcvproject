# ADR 0001: FastAPI Backend and Dependency-Free Vanilla JS Web Studio

## Status
Accepted

## Context
3DFUN AI requires an interactive web interface allowing students and researchers to upload photographs, trigger 3D reconstructions, inspect geometric progress, and visualize 3D assets in a real-time WebGL canvas. Many contemporary research projects adopt heavy frontend frameworks (React, Next.js, Vue) requiring Node.js build pipelines (`npm install`, Webpack/Vite, node_modules containing thousands of dependencies).

## Decision
We chose a high-performance **FastAPI** Python service paired with a **zero-dependency vanilla JavaScript SPA** utilizing standard Three.js via CDN:
1. **Backend:** FastAPI provides asynchronous request handling, automatic OpenAPI/Swagger documentation, native type validation via Pydantic, and direct access to native NumPy/SciPy/ONNX memory buffers without inter-process communication overhead.
2. **Frontend:** Vanilla JavaScript (ES6+), semantic HTML5, and pure CSS. Three.js is loaded as an ES module directly from CDN or vendored static files.

## Consequences
### Positive:
- **Zero Frontend Build Step:** Running the web studio requires zero Node.js or npm tooling. Running `uvicorn app.main:app` serves both the API and client studio immediately.
- **Minimal Footprint:** Total frontend assets (CSS + JS) occupy $<65$ KB, loading in milliseconds even on throttled mobile connections.
- **Portability:** Can be packaged cleanly into a single lightweight Python wheel or minimal Docker container.

### Negative:
- Complex UI state management must be handled with explicit DOM event listeners rather than reactive framework data binding.
