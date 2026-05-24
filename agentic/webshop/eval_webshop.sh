#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# WebShop Standalone Evaluation Launcher
#
# Prerequisites:
#   1. SGLang server running with the model to evaluate, OR use --auto-start
#   2. WebShop data available (default: cloned at ../../webshop/)
#
# Usage:
#   # Auto-start server and run eval
#   bash eval_webshop.sh --auto-start
#
#   # Connect to existing server
#   bash eval_webshop.sh
#
# Environment variables (with defaults):
#   MODEL_PATH    - HF model path        (default: /data/qwen25_7b/)
#   SERVER_URL    - SGLang /generate URL  (default: http://127.0.0.1:30000/generate)
#   TP            - Tensor parallel       (default: 4)
#   CONCURRENCY   - Max concurrent sessions (default: 16)
#   MAX_STEPS     - Max steps per session (default: 20)
#   OUTPUT        - Output JSON path      (default: eval_webshop_results.json)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTIC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$AGENTIC_DIR")"

# ── Defaults ───────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/data/qwen25_7b/}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:30000/generate}"
TP="${TP:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.8}"
CTX_LEN="${CTX_LEN:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e5m2}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_STEPS="${MAX_STEPS:-20}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_SESSIONS="${NUM_SESSIONS:-}"
OBSERVATION_MODE="${OBSERVATION_MODE:-text_rich}"
OUTPUT="${OUTPUT:-eval_webshop_results.json}"
PORT="${PORT:-30000}"

AUTO_START=0

# ── Parse flags ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --auto-start) AUTO_START=1; shift ;;
        --model)      MODEL_PATH="$2"; TOKENIZER_PATH="$2"; shift 2 ;;
        --tokenizer)  TOKENIZER_PATH="$2"; shift 2 ;;
        --tp)         TP="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --max-steps)  MAX_STEPS="$2"; shift 2 ;;
        --output)     OUTPUT="$2"; shift 2 ;;
        --num-sessions) NUM_SESSIONS="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── PYTHONPATH ─────────────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/slime:${SCRIPT_DIR}:${AGENTIC_DIR}/agentflow:${PROJECT_ROOT}/webshop:${PYTHONPATH:-}"

# ── Server management ──────────────────────────────────────────────────────────
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "[eval_webshop] Stopping SGLang server (PID=$SERVER_PID)…"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$AUTO_START" -eq 1 ]]; then
    echo "[eval_webshop] Auto-starting SGLang server (port=$PORT, tp=$TP)…"
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --port "$PORT" \
        --tp "$TP" \
        --mem-fraction-static "$MEM_FRACTION" \
        --context-length "$CTX_LEN" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --trust-remote-code &
    SERVER_PID=$!

    # Wait for server
    echo "[eval_webshop] Waiting for server to be ready…"
    for _ in $(seq 1 60); do
        if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            echo "[eval_webshop] Server is ready."
            break
        fi
        sleep 5
    done

    SERVER_URL="http://127.0.0.1:$PORT/generate"
fi

# ── Build command ──────────────────────────────────────────────────────────────
CMD=(
    python "${SCRIPT_DIR}/eval_webshop.py"
    --tokenizer "$TOKENIZER_PATH"
    --server-url "$SERVER_URL"
    --concurrency "$CONCURRENCY"
    --max-steps "$MAX_STEPS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --observation-mode "$OBSERVATION_MODE"
    --output "$OUTPUT"
)

if [[ -n "$NUM_SESSIONS" ]]; then
    CMD+=(--num-sessions "$NUM_SESSIONS")
fi

echo "[eval_webshop] Running: ${CMD[*]}"
echo "[eval_webshop] PYTHONPATH=$PYTHONPATH"

"${CMD[@]}"
