# REST API reference

The API is rooted at `/api/v1`. Start the local app and open [Swagger UI](http://127.0.0.1:8010/docs) or download the [OpenAPI schema](http://127.0.0.1:8010/api/v1/openapi.json). Use the port from your launcher.

| Method and path | Request or response |
| --- | --- |
| `POST /auth/register` | JSON with `email`, `password`, `display_name`, and `accept_terms`. |
| `POST /auth/login` | JSON with `email` and `password`, plus `mfa_code` when enabled. Read `data.access_token`. |
| `GET /auth/me` | Bearer-authenticated profile. |
| `POST /uploads/` | Send one multipart `file`. Read `data.id` for its upload ID. |
| `POST /generations/estimate` | Estimate costs for a `quality` and `options`. |
| `POST /generations` | Queue a job using `upload_id`. Optional `Idempotency-Key` lets a client retry creation. Read `data.id`. |
| `GET /generations/{id}` | Job stage, integer percent, status, and `asset_id` when completed. |
| `GET /generations/{id}/events` | Server-Sent Events for the job owner. |
| `POST /generations/{id}/cancel` | Request cancellation where the job state permits it. |
| `POST /generations/{id}/retry` | Retry a failed or cancelled job. |
| `GET /assets/{id}` | Asset metadata and available formats. |
| `GET /assets/{id}/download?format=glb` | Download binary GLB. Use `format=obj&bundle=true` or `format=gltf&bundle=true` for a ZIP with companion files. |
| `GET /health` | Database, engine, storage and queue status. |
| `GET /health/ready` | Database readiness. |
| `GET /meta` | Installed providers, policy limits and pricing. |
| `GET /results/summary` | Recorded benchmark summary and dataset metadata. |

For authenticated requests, send `Authorization: Bearer <access_token>`. Standard JSON responses put payloads under `data` and domain errors under `error`. File and event streams use their own content types.

Example generation request after uploading a photo:

```json
{
  "upload_id": "<id from upload>",
  "quality": "draft",
  "privacy_mode": true,
  "options": {
    "engine": "local",
    "turntable": false,
    "formats": ["glb"]
  }
}
```

Poll `GET /generations/{id}` until the status is `COMPLETED`, `FAILED` or `CANCELLED`. Use the returned `asset_id` to download a successful result. The [operations guide](../handbook/operations.md) covers running the local service.


---

## Detailed REST API Specification by Domain

All endpoints are hosted under `http://127.0.0.1:8010/api/v1`.

### 1. Authentication & User Profiles

#### `POST /auth/register`
Creates a new student account, initializes a free trial credit wallet, and automatically grants 50 credits.
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "email": "student@iiitdwd.ac.in",
    "password": "StrongPassword2026!",
    "display_name": "Student Researcher",
    "accept_terms": true
  }
  ```
- **Response `201 Created`**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsIn...",
    "token_type": "bearer",
    "user": {
      "id": "usr_78a19bc3",
      "email": "student@iiitdwd.ac.in",
      "display_name": "Student Researcher",
      "role": "student",
      "credits": 50,
      "created_at": "2026-11-17T12:00:00Z"
    }
  }
  ```

#### `POST /auth/login`
Authenticates user and returns JWT session token in body and `Set-Cookie`.
- **Request Body**:
  ```json
  {
    "email": "student@iiitdwd.ac.in",
    "password": "StrongPassword2026!"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1NiIsIn...",
    "token_type": "bearer",
    "user": {
      "id": "usr_78a19bc3",
      "email": "student@iiitdwd.ac.in",
      "credits": 50
    }
  }
  ```

#### `POST /auth/demo`
Instantly provisions or re-uses a zero-friction student sandbox account with 50 credits.

---

### 2. Free Trial Credit Ledger

#### `GET /credits/wallet`
Retrieves current student wallet status.
- **Response `200 OK`**:
  ```json
  {
    "balance": 50,
    "reserved": 0,
    "available": 50,
    "lifetime_granted": 50,
    "lifetime_spent": 0
  }
  ```

#### `POST /credits/daily-streak`
Claims +10 free trial credits once every 24 hours.
- **Response `200 OK`**:
  ```json
  {
    "status": "success",
    "granted": 10,
    "new_balance": 60,
    "next_claim_available_at": "2026-11-18T12:00:00Z"
  }
  ```

#### `GET /credits/transactions`
Lists immutable ledger entries for auditing.
- **Query Parameters**: `limit=20`, `offset=0`
- **Response `200 OK`**:
  ```json
  [
    {
      "id": "ctx_01j7abc...",
      "reason": "SIGNUP_GRANT",
      "amount": 50,
      "balance_after": 50,
      "created_at": "2026-11-17T12:00:00Z"
    }
  ]
  ```

---

### 3. Image Uploads

#### `POST /uploads`
Uploads a reference photograph for 3D reconstruction.
- **Request Headers**: `Content-Type: multipart/form-data`
- **Form Fields**: `file=@reference_image.png`
- **Response `201 Created`**:
  ```json
  {
    "id": "upl_49c0d12e",
    "filename": "reference_image.png",
    "byte_size": 284192,
    "width": 512,
    "height": 512,
    "created_at": "2026-11-17T12:05:00Z"
  }
  ```

---

### 4. 3D Generation Pipeline

#### `POST /generations`
Queues a new 3D reconstruction job with two-phase credit reservation.
- **Request Body**:
  ```json
  {
    "upload_id": "upl_49c0d12e",
    "engine": "depth_anything_v2",
    "quality": "standard",
    "options": {
      "taubin_iterations": 4,
      "resolution": 128
    }
  }
  ```
- **Response `202 Accepted`**:
  ```json
  {
    "id": "gen_830dfa21",
    "status": "QUEUED",
    "stage": "QUEUED",
    "progress": 0,
    "credits_reserved": 10,
    "created_at": "2026-11-17T12:06:00Z"
  }
  ```

#### `GET /generations/{id}`
Polls generation progress and returns intermediate telemetry.
- **Response `200 OK`**:
  ```json
  {
    "id": "gen_830dfa21",
    "status": "COMPLETED",
    "stage": "COMPLETED",
    "progress": 100,
    "asset_id": "ast_39a92c7f",
    "timings": {
      "preprocessing": 0.34,
      "depth": 0.82,
      "volume": 0.45,
      "geometry": 1.12,
      "uv": 0.28,
      "texture": 0.94
    }
  }
  ```

#### `GET /generations/{id}/events`
Opens a Server-Sent Events (SSE) telemetry connection.
- **Response Stream (`Content-Type: text/event-stream`)**:
  ```text
  event: progress
  data: {"stage": "DEPTH", "progress": 35, "message": "Estimating monocular depth"}

  event: progress
  data: {"stage": "GEOMETRY", "progress": 85, "message": "Extracting watertight isosurface"}

  event: completed
  data: {"status": "COMPLETED", "asset_id": "ast_39a92c7f"}
  ```

---

### 5. Zero-Copy Asset Streaming

#### `GET /assets/{id}/stream`
Streams the 3D model asset directly from memory over HTTP/2 without intermediate disk writes.
- **Response Headers**:
  - `Content-Type: model/gltf-binary`
  - `Content-Disposition: inline; filename="model.glb"`
  - `Content-Length: 1845920`
  - `Cache-Control: public, max-age=86400, immutable`
- **Response Body**: Direct binary glTF 2.0 (`.glb`) byte stream.

---

### 6. Autonomous Multimodal AI Co-Pilot

#### `POST /chat/message`
Sends a prompt to the student AI co-pilot with autonomous tool calling.
- **Request Body**:
  ```json
  {
    "model": "nvidia/nemotron-3-ultra-free",
    "prompt": "Decimate this mesh to 5000 triangles and inspect surface normals",
    "active_mesh_id": "ast_39a92c7f"
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "reply": "I have executed mesh decimation using Quadric Error Metrics (QEM). The mesh has been simplified to 5,000 triangles while maintaining an Euler characteristic of chi = 2 (watertight genus-0 manifold).",
    "tools_executed": [
      {
        "tool": "tool_decimate_mesh",
        "parameters": {"target_triangles": 5000},
        "result": {"status": "success", "new_face_count": 5000, "euler_characteristic": 2}
      }
    ]
  }
  ```


---

### 7. Student Laboratory Execution API

#### `POST /student-labs/run`
Executes an interactive PyTorch student lab notebook run and returns performance metrics.
- **Request Headers**: `Authorization: Bearer <access_token>`, `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "lab_id": "lab1_depth_finetuning",
    "epochs": 5,
    "batch_size": 4,
    "learning_rate": 0.0001
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "status": "COMPLETED",
    "lab_id": "lab1_depth_finetuning",
    "final_loss": 0.0421,
    "silog_metric": 0.118,
    "checkpoint_path": "models/checkpoints/lab1_best.pt",
    "execution_time_seconds": 12.4
  }
  ```

#### `GET /student-labs/curriculum`
Lists all 6 student laboratory modules with their mathematical descriptions and prerequisites.

---

### 8. System Health & Observability Metrics API

#### `GET /system/health`
Returns granular health telemetry across database, inference hardware, memory, and queue.
- **Response `200 OK`**:
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "timestamp": "2026-11-18T04:45:00Z",
    "database": {
      "status": "connected",
      "mode": "WAL",
      "open_connections": 1
    },
    "inference": {
      "preferred_provider": "DirectML",
      "available_providers": ["DirectML", "CPUExecutionProvider"]
    },
    "queue": {
      "active_jobs": 0,
      "max_workers": 2
    },
    "memory": {
      "rss_mb": 142.5,
      "vram_allocated_mb": 0.0
    }
  }
  ```

#### `GET /system/metrics`
Exposes Prometheus-formatted metrics for scrape targets:
```
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/api/v1/health",status="200"} 42
http_requests_total{method="POST",path="/api/v1/generations",status="202"} 14
# HELP generation_duration_seconds Latency of 3D reconstructions
# TYPE generation_duration_seconds histogram
generation_duration_seconds_bucket{le="15.0"} 3
generation_duration_seconds_bucket{le="30.0"} 12
generation_duration_seconds_bucket{le="60.0"} 14
```

---

### 9. Python Client SDK Snippet (`requests`)

```python
import requests

BASE_URL = "http://127.0.0.1:8010/api/v1"

# 1. Authenticate
auth_resp = requests.post(f"{BASE_URL}/auth/login", json={
    "email": "student@iiitdwd.ac.in",
    "password": "StrongPassword2026!"
}).json()
token = auth_resp["data"]["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Upload Photograph
with open("test_image.png", "rb") as f:
    upload_resp = requests.post(f"{BASE_URL}/uploads/", files={"file": f}, headers=headers).json()
upload_id = upload_resp["data"]["id"]

# 3. Request 3D Reconstruction
gen_resp = requests.post(f"{BASE_URL}/generations", json={
    "upload_id": upload_id,
    "quality": "standard",
    "options": {"engine": "local", "formats": ["glb"]}
}, headers=headers).json()
job_id = gen_resp["data"]["id"]

# 4. Stream Result Directly to Memory
mesh_resp = requests.get(f"{BASE_URL}/assets/{job_id}/stream", headers=headers, stream=True)
with open("reconstructed_mesh.glb", "wb") as f:
    for chunk in mesh_resp.iter_content(chunk_size=65536):
        f.write(chunk)
print("Mesh successfully reconstructed and saved via Zero-Copy RAM stream.")
```


---

## 10. Complete REST & WebSocket API Endpoint Matrix

| URI Path | Method | Auth | Payload Schema | Success Response | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/api/v1/auth/register` | `POST` | None | `{"email": "...", "password": "..."}` | `{"user_id": "...", "balance": 50}` | `201 Created` |
| `/api/v1/auth/token` | `POST` | None | `{"email": "...", "password": "..."}` | `{"access_token": "...", "type": "bearer"}` | `200 OK` |
| `/api/v1/auth/me` | `GET` | Bearer | Empty | `{"id": "...", "balance": 50, "role": "student"}` | `200 OK` |
| `/api/v1/uploads/` | `POST` | Bearer | `multipart/form-data; file=@image.png` | `{"id": "upl_...", "sha256": "..."}` | `201 Created` |
| `/api/v1/generations/` | `POST` | Bearer | `{"upload_id": "...", "quality": "standard"}` | `{"id": "gen_...", "status": "pending"}` | `201 Created` |
| `/api/v1/generations/{id}` | `GET` | Bearer | Empty | `{"id": "gen_...", "progress": 0.85, ...}` | `200 OK` |
| `/api/v1/assets/{id}/stream`| `GET` | Bearer | Empty (Query: `?format=glb`) | `model/gltf-binary` (Chunked Stream) | `200 OK` |
| `/api/v1/chat/message` | `POST` | Bearer | `{"message": "Decimate mesh to 5k tris"}` | `{"reply": "...", "tools_executed": [...]}` | `200 OK` |
| `/api/v1/student-labs` | `GET` | Bearer | Empty | `{"labs": [{"id": "lab1_depth", ...}, ...]}` | `200 OK` |
| `/api/v1/student-labs/run` | `POST` | Bearer | `{"lab_id": "lab2_occupancy", "config": {...}}`| `{"run_id": "run_...", "status": "queued"}` | `200 OK` |
| `/api/v1/system/health` | `GET` | None | Empty | `{"status": "ok", "checks": {...}}` | `200 OK` |

---

## 11. Real-Time WebSocket Progress Protocol & Event Schema

Clients can subscribe to generation progress via WebSocket connection at `/ws/events?token={access_token}`:

```json
{
  "type": "generation_progress",
  "data": {
    "generation_id": "gen_88f9a2",
    "stage_index": 4,
    "stage_name": "Volumetric SDF Inflation",
    "progress": 0.65,
    "elapsed_seconds": 1.84,
    "message": "Voxelizing silhouette into 128^3 signed distance field..."
  }
}
```

When completed:
```json
{
  "type": "generation_complete",
  "data": {
    "generation_id": "gen_88f9a2",
    "asset_id": "ast_39a1b0",
    "vertex_count": 14820,
    "triangle_count": 29636,
    "stream_url": "/api/v1/assets/ast_39a1b0/stream"
  }
}
```

---

## 12. Two-Phase Credit Accounting Error Responses & Rollbacks

If a compute worker encounters an out-of-memory or timeout exception, the transaction is reversed and the user receives:

```json
{
  "success": false,
  "error": {
    "code": "COMPUTE_ERROR_REFUNDED",
    "message": "Reconstruction encountered a compute exception. 100% of reserved credits (10 credits) have been refunded to your wallet.",
    "wallet": {
      "available_balance": 50,
      "reserved_balance": 0
    }
  }
}
```

---

## 13. Zero-Copy Binary Streaming Protocol & MIME Types

The streaming endpoint returns binary payloads with strict headers:
- `Content-Type`: `model/gltf-binary` (for `.glb`) or `application/octet-stream` (for `.stl` / `.obj`).
- `Content-Disposition`: `inline; filename="model.glb"`
- `Cache-Control`: `public, max-age=86400, immutable`
- `X-Zero-Copy-Stream`: `true`

---

## 14. Comprehensive TypeScript SDK Definitions for Frontend Integration

```typescript
export interface GenerationResponse {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stage?: string;
  asset_id?: string;
  error?: string;
}

export interface CreditWallet {
  available: number;
  reserved: number;
  lifetime_granted: number;
  lifetime_spent: number;
}
```

*API Specification documented by Group 17 · 3D Computer Vision Course Project · Department of Data Science & Artificial Intelligence · IIIT Dharwad.*
