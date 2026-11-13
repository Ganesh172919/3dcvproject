# Production Container Deployment & Orchestration Guide (Docker & Docker Compose)

**Target System**: Linux (Ubuntu 22.04+ / Debian 12 / RHEL 9), Windows WSL2, Cloud VMs  
**Components**: 3DFUN AI Microservice, Nginx Reverse Proxy, SQLite WAL Storage

---

## 1. Overview & Architecture

Deploying 3DFUN AI into production requires:
1. Packaging the FastAPI application and ONNX Runtime CPU dependencies into an isolated container.
2. Managing persistent storage volumes for uploaded photographs and generated 3D assets (`.glb`, `.obj`, `.ply`, `.stl`).
3. Configuring a production reverse proxy (Nginx) to handle SSL/TLS termination, static asset caching, and **disabling proxy buffering for real-time Server-Sent Events (SSE)**.

```
+--------------------------------------------------------------------------------------------------+
|                                    DOCKER DEPLOYMENT TOPOLOGY                                    |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   [Client Browser / Studio]                                                                      |
|               |                                                                                  |
|               | HTTPS (Port 443) / HTTP (Port 80)                                                |
|               v                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   |  Nginx Reverse Proxy Container (`nginx:alpine`)                                          |   |
|   |  * TLS Termination (Certbot / Let's Encrypt)                                             |   |
|   |  * Static Asset Caching (`/static/`)                                                     |   |
|   |  * SSE Buffer Disabling (`proxy_buffering off;`)                                         |   |
|   +------------------------------------------------------------------------------------------+   |
|               |                                                                                  |
|               | Forward to Internal Network (Port 8000)                                          |
|               v                                                                                  |
|   +------------------------------------------------------------------------------------------+   |
|   |  3DFUN AI Application Container (`3dfun-ai:latest`)                                      |   |
|   |  * Python 3.12 Non-Root User (`appuser`, UID 1001)                                       |   |
|   |  * FastAPI Uvicorn Server (Workers: 2-4)                                                 |   |
|   |  * ONNX Runtime INT8 Inference Engine                                                    |   |
|   +------------------------------------------------------------------------------------------+   |
|               |                                                                                  |
|               +-----------------------+-----------------------+                                  |
|               |                       |                       |                                  |
|               v                       v                       v                                  |
|       [Database Volume]        [Uploads Volume]        [Artifacts Volume]                        |
|       `/app/data/app.db`       `/app/data/uploads`     `/app/data/artifacts`                     |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Multi-Stage Production `Dockerfile`

Save the following file as `Dockerfile` in the root of the project:

```dockerfile
# ==============================================================================
# Stage 1: Build Dependencies and Python Wheels
# ==============================================================================
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements-common.txt requirements.txt ./

# Install wheels into user local directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# Stage 2: Final Lean Production Image
# ==============================================================================
FROM python:3.12-slim-bookworm AS runner

# Create non-root application user for container security
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Install runtime system libraries (OpenCV, FFmpeg for turntable rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Ensure persistent directories exist with correct permissions
RUN mkdir -p /app/data/uploads /app/data/artifacts /app/engine/models && \
    chown -R appuser:appgroup /app/data /app/engine/models

# Set environment variables
ENV PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RECON_HOST=0.0.0.0 \
    RECON_PORT=8000 \
    RECON_STORAGE_DIR=/app/data/uploads \
    RECON_OUTPUT_DIR=/app/data/artifacts \
    DATABASE_URL=sqlite:////app/data/app.db

# Switch to non-privileged user
USER appuser

# Health check verifies API responsiveness
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/api/v1/system/health || exit 1

EXPOSE 8000

# Start Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## 3. Docker Compose Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: 3dfun-ai-app
    restart: unless-stopped
    environment:
      - RECON_HOST=0.0.0.0
      - RECON_PORT=8000
      - RECON_DEVICE=cpu
      - RECON_THREADS=2
      - GEMINI_API_KEY=${GEMINI_API_KEY:-}
      - GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.5-flash-lite}
    volumes:
      - app-data:/app/data
      - app-models:/app/engine/models
    networks:
      - 3dfun-network
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4096M
        reservations:
          cpus: '1.0'
          memory: 1024M

  nginx:
    image: nginx:alpine
    container_name: 3dfun-ai-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - certs-data:/etc/letsencrypt:ro
    depends_on:
      - app
    networks:
      - 3dfun-network

volumes:
  app-data:
    driver: local
  app-models:
    driver: local
  certs-data:
    driver: local

networks:
  3dfun-network:
    driver: bridge
```

---

## 4. Production Nginx Configuration (`nginx.conf`)

> [!IMPORTANT]
> The `/api/v1/generations/.*/events` block **must** contain `proxy_buffering off;` and `chunked_transfer_encoding off;`. If omitted, Nginx will buffer Server-Sent Events, causing the web studio's progress rail to stall completely until the entire 30-second job finishes!

```nginx
upstream backend_service {
    server app:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name _;
    client_max_body_size 50M;

    # General API and Web UI routes
    location / {
        proxy_pass http://backend_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # CRITICAL: Server-Sent Events (SSE) Unbuffered Route
    location ~ ^/api/v1/generations/[^/]+/events$ {
        proxy_pass http://backend_service;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Disable all buffering for real-time streaming
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }

    # Static Assets Fast Caching
    location /static/ {
        proxy_pass http://backend_service;
        expires 7d;
        add_header Cache-Control "public, no-transform";
    }
}
```

---

## 5. Deployment Commands & Verification

### Initial Build & Model Seeding
```bash
# 1. Download models into the models volume prior to first boot
docker compose run --rm app python scripts/fetch_models.py --all

# 2. Launch services in daemon mode
docker compose up -d

# 3. Check health and inspect live logs
docker compose ps
docker compose logs -f app
```

### Verification Health Check
```bash
curl -I http://127.0.0.1/api/v1/system/health
```
Expected output:
```http
HTTP/1.1 200 OK
content-type: application/json
```
Response body:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "providers": ["CPUExecutionProvider"],
  "database": "connected"
}
```
