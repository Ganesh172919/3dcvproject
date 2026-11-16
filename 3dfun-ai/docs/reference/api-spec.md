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
