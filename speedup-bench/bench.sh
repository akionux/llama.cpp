#!/usr/bin/env bash
# speedup-bench: measure llama-server prompt(pp)/decode(tg) throughput for one binary + config.
#
# It starts llama-server on a free port, sends one deterministic chat request with a long
# repeated prefix, and prints the prompt/decode throughput reported in the response timings.
#
# usage:
#   ./bench.sh -b BIN -l LABEL [options]
#
#   -b BIN     llama-server binary to test                       (required)
#   -l LABEL   label used in log/output file names               (required)
#   -m MODEL   target model (default: $MODEL_DEFAULT)
#   -d DRAFT   draft/MTP model; enables --spec-type draft-mtp     (default: none)
#   -n N       --spec-draft-n-max                                 (default: 3)
#   -c CTX     context size                                       (default: 16384)
#   -r REP     repeat count of the filler sentence (~10 tok each)(default: 200)
#   -t TOKENS  max_tokens for the answer                          (default: 200)
#   -f FLAGS   extra llama-server flags (quoted string)
#   -e ENVV    extra environment, e.g. "GGML_CUDA_TOPK_RADIX_MIN_ROWS=1" (quoted string)
#
# environment overrides:
#   MODEL_DEFAULT  default target model path (set it instead of -m to avoid the long default)
#   WORKDIR        directory llama-server runs in (default: the repo root next to this script)
#   PORT_BASE      first port to try (default: 19800)
#   TIMEOUT_S      max seconds for the request (default: 1800)
#
# example (matched unsloth target + its shared MTP draft, 16k context):
#   ./bench.sh -b build/bin/llama-server -l mtp16k -c 16384 -r 100 -d /path/mtp-shared-Q8_0.gguf
set -euo pipefail

BIN=""; LABEL=""; MODEL="${MODEL_DEFAULT:-}"; DRAFT=""; NMAX=3
CTX=16384; REP=200; TOKENS=200; FLAGS=""; ENVV=""
PORT_BASE="${PORT_BASE:-19800}"; TIMEOUT_S="${TIMEOUT_S:-1800}"

while getopts "b:l:m:d:n:c:r:t:f:e:" opt; do
    case "$opt" in
        b) BIN="$OPTARG" ;;
        l) LABEL="$OPTARG" ;;
        m) MODEL="$OPTARG" ;;
        d) DRAFT="$OPTARG" ;;
        n) NMAX="$OPTARG" ;;
        c) CTX="$OPTARG" ;;
        r) REP="$OPTARG" ;;
        t) TOKENS="$OPTARG" ;;
        f) FLAGS="$OPTARG" ;;
        e) ENVV="$OPTARG" ;;
        *) sed -n '2,30p' "$0"; exit 2 ;;
    esac
done

[ -n "$BIN" ]   || { echo "error: -b BIN is required" >&2; exit 2; }
[ -n "$LABEL" ] || { echo "error: -l LABEL is required" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "error: -m MODEL (or \$MODEL_DEFAULT) is required" >&2; exit 2; }
[ -x "$BIN" ]   || { echo "error: $BIN is not executable" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG="/tmp/speedupbench_${LABEL}.log"
OUT="/tmp/speedupbench_${LABEL}.json"
REQ="/tmp/speedupbench_${LABEL}_req.json"

PORT=$((PORT_BASE + RANDOM % 250))

python3 - "$REQ" "$REP" "$TOKENS" <<'PY'
import json, sys
req, rep, tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# ~10 tokens per filler sentence; the question forces a long, deterministic answer
filler = "The history of the Roman Empire spans many centuries. " * rep
prompt = filler + "\nSummarize the above in one sentence, then write a 200-word essay about Roman roads."
json.dump({"model": "x", "messages": [{"role": "user", "content": prompt}],
           "max_tokens": tokens, "temperature": 0}, open(req, "w"))
PY

SPEC=""
if [ -n "$DRAFT" ]; then
    SPEC="--spec-type draft-mtp --spec-draft-n-max $NMAX"
fi

cd "$WORKDIR"
# shellcheck disable=SC2086
env $ENVV "$BIN" -m "$MODEL" $SPEC ${DRAFT:+-md "$DRAFT"} \
    --host 127.0.0.1 --port "$PORT" -c "$CTX" --no-webui -fa on \
    --cache-type-k q8_0 --cache-type-v q8_0 $FLAGS > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
    grep -q "listening on" "$LOG" 2>/dev/null && break
    kill -0 $SRV 2>/dev/null || { echo "[$LABEL] FAILED TO START (binary exited)"; tail -n 6 "$LOG"; exit 1; }
    sleep 2
done
if ! grep -q "listening on" "$LOG" 2>/dev/null; then
    echo "[$LABEL] FAILED TO START (timeout)"; tail -n 6 "$LOG"; exit 1
fi

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
curl -s -m "$TIMEOUT_S" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"$REQ" -o "$OUT"

python3 - "$OUT" "$LABEL" "$CTX" "$VRAM" <<'PY'
import json, sys
path, label, ctx, vram = sys.argv[1:5]
try:
    t = json.load(open(path)).get("timings", {})
except Exception as e:                                    # noqa: BLE001
    print(f"[{label}] no timings ({e}) - see /tmp/speedupbench_{label}.log"); sys.exit(0)
print("[%s] ctx=%s vram=%sMB | prompt=%d tok @ %.2f t/s | decode=%d tok @ %.2f t/s"
      % (label, ctx, vram, t.get("prompt_n", 0), t.get("prompt_per_second", 0),
         t.get("predicted_n", 0), t.get("predicted_per_second", 0)))
PY

grep -iE "draft acceptance|mean len" "$LOG" | tail -n 2 || true
