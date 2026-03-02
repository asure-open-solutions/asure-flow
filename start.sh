#!/usr/bin/env bash
## AsuréFlow - Start Server + Client
## Usage: ./start.sh
set -euo pipefail
cd "$(dirname "$0")"

echo ""
echo "  AsuréFlow"
echo "  ========="
echo ""

# ---- Check prerequisites ----
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

if ! command -v node &>/dev/null; then
    echo "  ERROR: Node.js is not installed."
    echo "  Download: https://nodejs.org/"
    echo ""
    read -rp "  Press Enter to exit..."
    exit 1
fi

# ---- Auto-setup server ----
if [ ! -d "server/.venv" ]; then
    echo "  [setup] Installing server dependencies..."
    cd server
    if command -v uv &>/dev/null; then
        echo "  Using uv for fast install..."
        uv venv .venv
        source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
        uv pip install -e ".[dev]"
    else
        "$PYTHON_CMD" -m venv .venv
        source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
        echo "  Installing dependencies (this may take a few minutes)..."
        pip install -e ".[dev]"
    fi
    cd ..
    echo "  [setup] Server ready."
    echo ""
fi

# ---- Auto-setup client ----
if [ ! -d "client/node_modules" ]; then
    echo "  [setup] Installing client dependencies..."
    cd client
    npm install
    cd ..
    echo "  [setup] Client ready."
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

# ---- Start server in background ----
echo "  Starting server..."
cd server
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python -m uvicorn asure_flow.main:app \
    --host "$HOST_" \
    --port "$PORT_" \
    --reload \
    --reload-exclude '.venv' \
    --ws-max-size 1048576 &
SERVER_PID=$!
cd ..

# ---- Start client in background ----
echo "  Starting client..."
cd client
npm run dev &
CLIENT_PID=$!
cd ..

# ---- Cleanup on exit ----
cleanup() {
    echo ""
    echo "  Shutting down..."
    kill "$SERVER_PID" 2>/dev/null || true
    kill "$CLIENT_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

echo ""
echo "  Local:   http://localhost:$PORT_"
echo "  Network: http://${LAN_IP}:$PORT_"
echo ""
echo "  Remote client:"
echo "    ./start-client.sh http://${LAN_IP}:$PORT_"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

wait
