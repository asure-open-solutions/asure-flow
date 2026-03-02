#!/usr/bin/env bash
## Asuré Flow — Start client only (macOS / Linux)
## Usage: ASUREFLOW_SERVER=http://192.168.1.50:8000 ./scripts/start-client.sh
##    or: ./scripts/start-client.sh http://192.168.1.50:8000
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "=== Asuré Flow — Client ==="
echo ""

# Accept server URL from: argument > env var > default
SERVER_URL="${1:-${ASUREFLOW_SERVER:-http://localhost:8000}}"
export ASUREFLOW_SERVER="$SERVER_URL"

# Auto-install on first run
if [ ! -d "$ROOT/client/node_modules" ]; then
    echo "→ First run — installing client dependencies…"
    cd "$ROOT/client"
    npm install
    echo ""
fi

echo "  Connecting to server: $SERVER_URL"
echo ""

# Verify server is reachable (non-blocking — just a warning)
if command -v curl &>/dev/null; then
    if curl -s --max-time 3 "$SERVER_URL/api/health" >/dev/null 2>&1; then
        echo "  Server status: online ✓"
    else
        echo "  ⚠  Server at $SERVER_URL is not reachable yet."
        echo "     Make sure the server is running and the URL is correct."
    fi
    echo ""
fi

echo "Press Ctrl+C to stop."
echo ""

# Start Electron client
cd "$ROOT/client"
exec npm run dev
