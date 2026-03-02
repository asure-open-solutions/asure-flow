#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Load .env if it exists
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

echo "=== Asuré Flow Dev ==="

# Auto-setup on first run
if [ ! -d "$ROOT/server/.venv" ]; then
  echo "→ First run — installing dependencies…"
  bash "$ROOT/scripts/setup.sh"
fi

# Start server in background
echo "→ Starting server on ${HOST:-0.0.0.0}:${PORT:-8000}…"
cd "$ROOT/server"
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
export PYTHONPATH="$ROOT/server/src:${PYTHONPATH:-}"

python -m uvicorn asure_flow.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --reload \
  --reload-exclude '.venv' \
  --ws-max-size 1048576 &
SERVER_PID=$!

# Start client
echo "→ Starting Electron client…"
cd "$ROOT/client"
npm run dev &
CLIENT_PID=$!

# Trap to kill both on exit
cleanup() {
  echo ""
  echo "Shutting down…"
  kill "$SERVER_PID" 2>/dev/null || true
  kill "$CLIENT_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

# Detect LAN IP
get_lan_ip() {
    case "$(uname -s)" in
        Darwin)
            ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "YOUR_IP"
            ;;
        *)
            hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_IP"
            ;;
    esac
}
LAN_IP=$(get_lan_ip)

echo ""
echo "  Local:    http://localhost:${PORT:-8000}"
echo "  Network:  http://${LAN_IP}:${PORT:-8000}"
echo "  API docs: http://localhost:${PORT:-8000}/docs"
echo ""
echo "  Remote client:"
echo "    ASUREFLOW_SERVER=http://${LAN_IP}:${PORT:-8000} ./scripts/start-client.sh"
echo ""
echo "Press Ctrl+C to stop both."
echo ""

wait
