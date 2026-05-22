#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PID_FILE="$ROOT_DIR/.case_study_servers.pid"
export PYTHONDONTWRITEBYTECODE=1

cleanup_servers() {
  if [[ -f "$PID_FILE" ]]; then
    while read -r pid_to_stop; do
      if [[ -n "$pid_to_stop" ]] && kill -0 "$pid_to_stop" 2>/dev/null; then
        kill "$pid_to_stop" 2>/dev/null || true
      fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
  fi
}

if [[ -f "$PID_FILE" ]]; then
  cleanup_servers
fi

if [[ ! -f "$ROOT_DIR/cit-HepPh.txt" ]]; then
  echo "Missing $ROOT_DIR/cit-HepPh.txt"
  echo "Place the SNAP cit-HepPh edge list at the repository root before running Case 1."
  exit 1
fi

echo "Generating Case 1: cit-HepPh macro-topology validation"
"$PYTHON_BIN" "$ROOT_DIR/case_1_citation/pipeline.py" \
  --input "$ROOT_DIR/cit-HepPh.txt" \
  --output "$ROOT_DIR/case_1_citation/graph_data.json"

echo "Generating Case 2: live Transformer layer execution graph"
"$PYTHON_BIN" "$ROOT_DIR/case_2_transformer/pipeline.py" \
  --output "$ROOT_DIR/case_2_transformer/graph_data.json" \
  --onnx "$ROOT_DIR/case_2_transformer/transformer_layer.onnx"

start_server() {
  local port="$1"
  local directory="$2"
  local log_file="/tmp/topolayout-dg-case-${port}.log"
  "$PYTHON_BIN" -m http.server "$port" --bind 127.0.0.1 --directory "$directory" > "$log_file" 2>&1 &
  local pid="$!"
  echo "$pid" >> "$PID_FILE"
  sleep 0.5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Failed to start server on port $port. See $log_file"
    exit 1
  fi
}

start_server 8001 "$ROOT_DIR/case_1_citation"
start_server 8002 "$ROOT_DIR/case_2_transformer"

CASE_1_URL="http://127.0.0.1:8001/index.html"
CASE_2_URL="http://127.0.0.1:8002/index.html"

echo "Case 1: $CASE_1_URL"
echo "Case 2: $CASE_2_URL"
echo "Server PIDs written to $PID_FILE"

if [[ "${NO_OPEN:-0}" != "1" ]] && command -v open >/dev/null 2>&1; then
  open "$CASE_1_URL"
  open "$CASE_2_URL"
fi

trap cleanup_servers EXIT INT TERM
echo "Press Ctrl-C to stop both local servers."
wait
