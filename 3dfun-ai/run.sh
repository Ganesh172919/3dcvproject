#!/usr/bin/env bash
# Start the 3DFUN AI development server.
#   ./run.sh            # http://localhost:8000
#   ./run.sh --seed     # seed dev data first
#   PORT=9000 ./run.sh  # custom port
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

if [ "$#" -gt 0 ] && [ "$1" = "--seed" ]; then
  echo "==> seeding development data"
  python3 -m app.seed --with-sample
  shift
fi

echo "==> 3DFUN AI on http://localhost:${PORT}  (Ctrl-C to stop)"
exec python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
