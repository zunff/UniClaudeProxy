#!/usr/bin/env bash
# Start the UniClaudeProxy admin dashboard.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ADMIN_DIR="$SCRIPT_DIR/admin-dashboard"
cd "$ADMIN_DIR"

if [ ! -d "node_modules" ]; then
  echo "[admin] Installing dependencies..."
  if command -v pnpm >/dev/null 2>&1; then
    pnpm install
  elif command -v npm >/dev/null 2>&1; then
    npm install
  else
    echo "Error: neither pnpm nor npm is available in PATH."
    exit 1
  fi
fi

echo ""
echo "==========================================="
echo "  UniClaudeProxy Admin Dashboard"
echo "  Frontend: http://127.0.0.1:5173"
echo "  Proxy API (must be running): 127.0.0.1:10388"
echo "==========================================="
echo ""

if command -v pnpm >/dev/null 2>&1; then
  pnpm dev
else
  npm run dev
fi
