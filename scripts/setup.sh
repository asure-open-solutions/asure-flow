#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Asuré Flow Setup ==="
echo ""

# ── Check prerequisites ──

check_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: $1 is not installed. Please install it first."
    echo "  $2"
    exit 1
  fi
}

# python3 on Linux/Mac, python on Windows
PYTHON_CMD="python3"
if ! command -v python3 &>/dev/null; then
  PYTHON_CMD="python"
fi
check_cmd "$PYTHON_CMD" "https://www.python.org/downloads/"
check_cmd node          "https://nodejs.org/"

# ── Python server setup ──

echo "→ Setting up Python server…"
cd "$ROOT/server"

if command -v uv &>/dev/null; then
  echo "  Using uv for Python dependencies"
  if [ ! -d ".venv" ]; then
    uv venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
  uv pip install -e ".[dev]"
else
  echo "  Using pip (install 'uv' for faster installs: pip install uv)"
  if [ ! -d ".venv" ]; then
    "$PYTHON_CMD" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null
  pip install -e ".[dev]"
fi

echo "  Server dependencies installed."

# ── Client setup ──

echo "→ Setting up Electron client…"
cd "$ROOT/client"
npm install
echo "  Client dependencies installed."

# ── Environment file ──

if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo ""
  echo "→ Created .env from .env.example"
  echo "  Edit .env and add at least one LLM API key to enable AI features."
else
  echo "→ .env already exists, skipping."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start:"
echo "  Double-click:  start.command  (macOS)"
echo "  Terminal:      ./start.sh     (Linux)"
echo "  Double-click:  start.bat      (Windows)"
echo ""
