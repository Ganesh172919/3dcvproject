# How-To Guide: Containerized Deployment via Docker

This guide details how to build, test, and deploy the 3DFUN AI service using Docker and Docker Compose for production environments.

---

## 1. Dockerfile Architecture

3DFUN AI uses a multi-stage Docker build to keep the production runtime image under 280 MB:
- **Builder Stage (`python:3.11-slim`):** Compiles native C++ extensions for marching cubes and installs Python wheels.
- **Runtime Stage:** Copies the clean virtual environment and pre-quantized ONNX models, running as an unprivileged system user (`appuser`).

---

## 2. Building the Production Container

Run the build command from the project root:

```bash
docker build -t 3dfun-ai:2.0.0 -f deploy/Dockerfile .
```

Verify that the image builds without embedding secrets or development caches:

```bash
docker images 3dfun-ai:2.0.0
```

---

## 3. Running with Docker Compose

A production-ready `docker-compose.yml` is provided in `deploy/docker-compose.yml`:

```yaml
version: "3.8"

services:
  web:
    image: 3dfun-ai:2.0.0
    container_name: 3dfun_service
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - NUM_WORKERS=4
      - MAX_UPLOAD_MB=25
      - DATABASE_URL=sqlite:////data/3dfun.db
    volumes:
      - 3dfun_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 4096M
        reservations:
          cpus: "1.0"
          memory: 1024M

volumes:
  3dfun_data:
```

### Launch the Service:
```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Check Logs and Service Health:
```bash
docker compose -f deploy/docker-compose.yml logs -f web
curl -s http://localhost:8000/api/health | jq .
```

Expected health status:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "models_loaded": {
    "u2netp": true,
    "depth_anything_v2": true
  },
  "database": "connected"
}
```

---

## 4. Production Hardening Checklist

- **Non-root Execution:** The container automatically switches to `UID 10001 (appuser)`.
- **Read-Only Root Filesystem:** Set `read_only: true` and mount `/tmp` and `/data` as explicit writable volumes.
- **Resource Constraints:** Ensure memory ceiling (`4096M`) prevents out-of-memory cascading in high-concurrency batch environments.
- **Reverse Proxy:** Terminate TLS via Nginx, Traefik, or Caddy in front of port 8000.

---
*Evidence: Docker configuration in `deploy/Dockerfile` and `deploy/docker-compose.yml`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
