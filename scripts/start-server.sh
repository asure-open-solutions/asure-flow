#!/usr/bin/env bash
## Asuré Flow — Start server only (macOS / Linux)
## Usage: ./scripts/start-server.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "=== Asuré Flow — Server ==="
echo ""

# Auto-setup on first run
if [ ! -d "$ROOT/server/.venv" ]; then
    echo "→ First run — installing dependencies…"
    bash "$ROOT/scripts/setup.sh"
fi

# Load .env
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

HOST_="${HOST:-0.0.0.0}"
PORT_="${PORT:-8000}"

# Detect LAN IP
get_lan_ip() {
    case "$(uname -s)" in
        Darwin)
            ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "YOUR_IP"
            ;;
        *)
            hostname -I 2>/dev/null | awk '{print $1}' || ip -4 addr show scope global 2>/dev/null | grep -oP '(?<=inet )[\d.]+' | head -1 || echo "YOUR_IP"
            ;;
    esac
}

LAN_IP=$(get_lan_ip)

echo "  Local:    http://localhost:$PORT_"
echo "  Network:  http://${LAN_IP}:$PORT_"
echo "  API docs: http://localhost:$PORT_/docs"
echo ""
echo "  To connect from another machine:"
echo "    ASUREFLOW_SERVER=http://${LAN_IP}:$PORT_ ./scripts/start-client.sh"
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Activate venv and start server
cd "$ROOT/server"
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
export PYTHONPATH="$ROOT/server/src:${PYTHONPATH:-}"

exec python -m uvicorn asure_flow.main:app \
    --host "$HOST_" \
    --port "$PORT_" \
    --reload \
    --reload-exclude '.venv' \
    --ws-max-size 1048576
