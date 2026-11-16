# Serving the application

The course submission uses FastAPI with SQLite, local storage, and one in-process generation worker. Start with the [local setup guide](../handbook/local-setup.md).

For a hosted evaluation instance, use one application worker, persistent storage, and HTTPS. Give the process environment explicit `APP_URL`, `JWT_ACCESS_SECRET`, and `JWT_REFRESH_SECRET` values. Use `COOKIE_SECURE=true` with HTTPS and keep environment files and user assets outside the served static directories. Seed development credentials only for a local demo. See [deployment configuration](../DEPLOYMENT.md).

Check `GET /api/v1/health/ready` for database readiness. Inspect `GET /api/v1/health` for model availability and queue state. For Server-Sent Events, configure a reverse proxy to stream responses as they arrive. Back up the database and its matching storage together, and test restoration on a separate copy.
