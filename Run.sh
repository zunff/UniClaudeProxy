#!/bin/bash
cd "$(dirname "$0")"
echo "Starting UniClaudeProxy..."

HOST=127.0.0.1
PORT=9223

if [ -f global.json ]; then
  HOST=$(python3 -c "import json; print(json.load(open('global.json', encoding='utf-8')).get('server', {}).get('host') or '127.0.0.1')")
  PORT=$(python3 -c "import json; print(json.load(open('global.json', encoding='utf-8')).get('server', {}).get('port') or 9223)")
fi

echo "Using host=${HOST} port=${PORT}"
python3 -m uvicorn app.main:app --host "$HOST" --port "$PORT"
