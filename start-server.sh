#!/usr/bin/env bash
## AsuréFlow - Start Server
## Usage: ./start-server.sh
set -euo pipefail
cd "$(dirname "$0")"

echo ""
echo "  AsuréFlow - Server"
echo "  =================="
echo ""

# ---- Check Python ----
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
    if command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "  ERROR: Python is not installed."
        echo "  Download: https://www.python.org/downloads/"
        echo ""
        read -rp "  Press Enter to exit..."
        exit 1
    fi
fi

# ---- Auto-setup ----
if [ ! -d "server/.venv" ]; then
    echo "  [setup] First run - installing server dependencies..."
    echo ""
    cd server
    if command -v uv &>/dev/null; then
        echo "  Using uv for fast install..."
        uv venv .venv
        source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
        uv pip install -e ".[dev]"
    else
        echo "  Creating virtual environment..."
        "$PYTHON_CMD" -m venv .venv
        source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
        echo "  Installing dependencies (this may take a few minutes)..."
        pip install -e ".[dev]"
    fi
    cd ..
    echo ""
    echo "  [setup] Server ready."
    echo ""
fi

# ---- Load .env ----
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

HOST_="${HOST:-0.0.0.0}"
PORT_="${PORT:-8000}"

# ---- Detect LAN IP ----
get_lan_ip() {
    case "$(uname -s)" in
        Darwin)
            ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "your-ip"
            ;;
        *)
            hostname -I 2>/dev/null | awk '{print $1}' || echo "your-ip"
            ;;
    esac
}
LAN_IP=$(get_lan_ip)

echo "  Local:   http://localhost:$PORT_"
echo "  Network: http://${LAN_IP}:$PORT_"
echo "  Docs:    http://localhost:$PORT_/docs"
echo ""
echo "  Remote client:"
echo "    ./start-client.sh http://${LAN_IP}:$PORT_"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

# ---- Run ----
cd server
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

exec python -m uvicorn asure_flow.main:app \
    --host "$HOST_" \
    --port "$PORT_" \
    --reload \
    --reload-exclude '.venv' \
    --ws-max-size 1048576
