#!/usr/bin/env bash
# Start the 3DFUN AI development server.
#   bash run.sh         # http://127.0.0.1:8010
#   ./run.sh --seed     # seed dev data first
#   PORT=9000 ./run.sh  # custom port
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8010}"
HOST="${HOST:-127.0.0.1}"
export APP_URL="${APP_URL:-http://127.0.0.1:${PORT}}"

if [ "$#" -gt 0 ] && [ "$1" = "--seed" ]; then
  echo "==> seeding development data"
  python -m app.seed --with-sample
  shift
fi

echo "==> 3DFUN AI on http://localhost:${PORT}  (Ctrl-C to stop)"
exec python -m uvicorn app.main:app --host "$HOST" --port "$PORT"
