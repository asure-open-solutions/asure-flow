#!/usr/bin/env bash
## AsuréFlow - Start Client
## Usage: ./start-client.sh [server-url]
## Example: ./start-client.sh http://192.168.1.50:8000
set -euo pipefail
cd "$(dirname "$0")"

echo ""
echo "  AsuréFlow - Client"
echo "  =================="
echo ""

# ---- Check Node.js ----
if ! command -v node &>/dev/null; then
    echo "  ERROR: Node.js is not installed."
    echo "  Download: https://nodejs.org/"
    echo ""
    read -rp "  Press Enter to exit..."
    exit 1
fi

# ---- Server URL ----
SERVER_URL="${1:-${ASUREFLOW_SERVER:-http://localhost:8000}}"
export ASUREFLOW_SERVER="$SERVER_URL"

# ---- Auto-setup ----
if [ ! -d "client/node_modules" ]; then
    echo "  [setup] First run - installing client dependencies..."
    cd client
    npm install
    cd ..
    echo ""
    echo "  [setup] Client ready."
    echo ""
fi

echo "  Server: $SERVER_URL"

# ---- Health check ----
if command -v curl &>/dev/null; then
    echo "  Waiting for server..."
    SERVER_READY=0
    ATTEMPT=0
    while [ "$ATTEMPT" -lt 30 ]; do
        ATTEMPT=$((ATTEMPT + 1))
        if curl -s --max-time 2 "$SERVER_URL/api/health" >/dev/null 2>&1; then
            SERVER_READY=1
            break
        fi
        sleep 1
    done
    if [ "$SERVER_READY" -eq 1 ]; then
        echo "  Status: online"
    else
        echo "  WARNING: Server is not reachable."
        echo "  Check its LAN bind, Windows Firewall, IP address, and Wi-Fi network."
    fi
fi

echo ""
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "  macOS: allow Microphone access, plus Screen Recording for system audio."
fi
echo "  Tip: Change server URL in Settings inside the app."
echo "  Press Ctrl+C to stop."
echo ""

# ---- Run ----
cd client
exec npm run dev
