#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f .env ]; then
  echo "ERROR: No .env file found. Copy .env.example to .env and fill in your keys."
  exit 1
fi

if [ -d .venv ]; then
  source .venv/bin/activate
else
  echo "Creating virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
fi

echo "Starting server on http://localhost:8000"
echo "To expose via HTTPS: npx cloudflared tunnel --url http://localhost:8000"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
