# Reference: REST API Specification

3DFUN AI exposes a high-performance RESTful API powered by FastAPI, featuring OAuth2 password bearer authentication, job state machines, and streaming asset delivery.

Base URL: `http://localhost:8000/api`

---

## 1. Authentication Endpoints

### `POST /api/auth/token`
Authenticate using username/password credentials to receive a signed JWT access token.

- **Request Format:** `application/x-www-form-urlencoded`
  - `username` (string, required): Student email or username.
  - `password` (string, required): Account password.
- **Response Format:** `application/json` (Status `200 OK`)
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "bearer",
    "user": {
      "id": 1,
      "email": "student@iiitdwd.ac.in",
      "credits": 95
    }
  }
  ```

---

## 2. Reconstruction Job Endpoints

### `POST /api/recon/create`
Submits a single image for 3D reconstruction.

- **Headers:** `Authorization: Bearer <token>`
- **Request Format:** `multipart/form-data`
  - `file` (binary, required): PNG, JPEG, or WEBP image ($\le 25$ MB).
  - `triangles` (integer, optional, default: 8000): Target polygon count.
  - `symmetry` (boolean, optional, default: true): Horizontal symmetry prior.
  - `thickness` (float, optional, default: 0.70): Maximum volumetric slab factor.
- **Response Format:** `application/json` (Status `202 Accepted`)
  ```json
  {
    "job_id": "job_e7a2b91c",
    "status": "processing",
    "created_at": "2026-11-12T10:15:30Z",
    "progress": 0.10,
    "current_stage": "matting"
  }
  ```

---

### `GET /api/recon/status/{job_id}`
Polls status and stage-by-stage progress for an active reconstruction.

- **Parameters:** `job_id` (string, required in path)
- **Response Format:** `application/json` (Status `200 OK`)
  ```json
  {
    "job_id": "job_e7a2b91c",
    "status": "completed",
    "stages": {
      "matting_sec": 0.124,
      "depth_sec": 0.352,
      "inflation_sec": 0.181,
      "meshing_sec": 0.218,
      "texturing_sec": 0.449,
      "total_sec": 1.324
    },
    "artifacts": {
      "glb": "/api/recon/download/job_e7a2b91c/model.glb",
      "obj": "/api/recon/download/job_e7a2b91c/model.obj",
      "ply": "/api/recon/download/job_e7a2b91c/model.ply",
      "preview": "/api/recon/download/job_e7a2b91c/preview.png"
    }
  }
  ```

---

### `GET /api/recon/download/{job_id}/{filename}`
Streams generated 3D assets (`.glb`, `.obj`, `.mtl`, `.ply`, `.png`).

- **Parameters:**
  - `job_id`: Unique job identifier.
  - `filename`: Requested asset filename.
- **Response Format:** Binary octet-stream with appropriate `Content-Type` headers (`model/gltf-binary`, `model/obj`, `image/png`).

---

## 3. System & Health Endpoints

### `GET /api/health`
Service health probe reporting loaded model states, database connectivity, and available hardware memory.

- **Response Format:** `application/json` (Status `200 OK`)
  ```json
  {
    "status": "healthy",
    "version": "2.0.0",
    "uptime_seconds": 14205,
    "device": "cpu",
    "active_threads": 4
  }
  ```

---
*Evidence: FastAPI router definitions in `app/api/recon.py`, `app/api/auth.py`, and `app/main.py`.*  
*Authors: J Ganesh, M Jagadeshwar, B Varshith, Pranav Kashyap, C Sai Aravind (Supervisor: Dr. Shirshendu Layek, Dept. of DSAI, IIIT Dharwad).*
