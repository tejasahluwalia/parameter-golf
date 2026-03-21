#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod Pod Entrypoint
# Runs on the pod — sets up env, trains, serves web, saves results
# ============================================================

WORKDIR="/workspace/parameter-golf/code"
RESULTS_DIR="$WORKDIR/results"
LOG_DIR="$WORKDIR/logs"
POLL_INTERVAL=30
TRAINING_TIMEOUT=1800

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [POD] $*"; }
die() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [POD] ERROR: $*" >&2; exit 1; }

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

cd "$WORKDIR" || die "Cannot cd to $WORKDIR"

log "========================================"
log "  Pod Entrypoint Starting"
log "  Hostname: $(hostname)"
log "  Workdir: $WORKDIR"
log "========================================"

log "Pulling latest changes from GitHub..."
git fetch origin main
git checkout main
git pull origin main

COMMIT_ID=$(git rev-parse --short HEAD)
log "Git commit: $COMMIT_ID"

log "Installing Python dependencies..."
if command -v uv &>/dev/null; then
    uv sync --frozen
elif command -v pip &>/dev/null; then
    pip install -e . --quiet
else
    die "Neither uv nor pip found"
fi

log "Downloading dataset if not present..."
DATA_DIR="./data/datasets/fineweb10B_sp1024"
TOKENIZER_DIR="./data/tokenizers"
if [[ ! -d "$DATA_DIR" ]] || [[ ! -d "$TOKENIZER_DIR" ]]; then
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
else
    log "Dataset already present, skipping download"
fi

log "Starting web server on port 3000..."
nohup python -m http.server 3000 --directory "$WORKDIR" > "$WORKDIR/web.log" 2>&1 &
WEB_PID=$!
log "Web server PID: $WEB_PID"

log "Starting training..."
nohup torchrun --standalone --nproc_per_node=1 train_gpt.py > "$WORKDIR/train.log" 2>&1 &
TRAIN_PID=$!
log "Training PID: $TRAIN_PID"

log "Waiting for training to complete (timeout: ${TRAINING_TIMEOUT}s)..."
elapsed=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
    if (( elapsed >= TRAINING_TIMEOUT )); then
        log "Training timeout reached, killing..."
        kill "$TRAIN_PID" 2>/dev/null || true
        break
    fi
    log "  Training still running... (${elapsed}s elapsed)"
done
wait "$TRAIN_PID" 2>/dev/null || true
log "Training process finished"

log "Waiting briefly for logs to flush..."
sleep 3

log "Extracting results..."
RUN_ID=$(grep -m1 'logs/.*\.txt' "$WORKDIR/train.log" 2>/dev/null | grep -oP 'logs/\K[^/]+(?=\.txt)' | head -1 || true)
if [[ -z "$RUN_ID" ]]; then
    log "Could not detect RUN_ID from train.log, scanning logs dir..."
    RUN_ID=$(ls -t "$LOG_DIR"/ 2>/dev/null | grep -E '^[^.]+\.txt$' | head -1 | sed 's/\.txt$//' || true)
fi

if [[ -z "$RUN_ID" ]]; then
    log "WARNING: Could not find run log file"
    VAL_BPB="unknown"
    FINAL_LOG_FILE=""
else
    TRAIN_LOG="$LOG_DIR/${RUN_ID}.txt"
    if [[ ! -f "$TRAIN_LOG" ]]; then
        log "WARNING: Train log not found at $TRAIN_LOG"
        VAL_BPB="unknown"
    else
        VAL_BPB=$(grep -oP 'val_bpb:\K[0-9.]+' "$TRAIN_LOG" | tail -1)
        log "Best val_bpb: $VAL_BPB"
    fi
fi

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
if [[ -f "$TRAIN_LOG" ]]; then
    FINAL_NAME="${TIMESTAMP}_${COMMIT_ID}_${VAL_BPB}.txt"
    cp "$TRAIN_LOG" "$RESULTS_DIR/$FINAL_NAME"
    log "Saved log as: $FINAL_NAME"
fi

if [[ -f "$WORKDIR/final_model.int8.ptz" ]]; then
    MODEL_NAME="${TIMESTAMP}_${COMMIT_ID}_${VAL_BPB}.ptz"
    cp "$WORKDIR/final_model.int8.ptz" "$RESULTS_DIR/$MODEL_NAME"
    log "Saved model as: $MODEL_NAME"
fi

if [[ -n "$RUN_ID" && -f "$WORKDIR/logs/${RUN_ID}_loss.html" ]]; then
    CHART_NAME="${TIMESTAMP}_${COMMIT_ID}_${VAL_BPB}_loss.html"
    cp "$WORKDIR/logs/${RUN_ID}_loss.html" "$RESULTS_DIR/$CHART_NAME"
    log "Saved chart as: $CHART_NAME"
else
    log "Chart not found (RUN_ID=$RUN_ID, path=$WORKDIR/logs/${RUN_ID}_loss.html)"
fi

if kill -0 "$WEB_PID" 2>/dev/null; then
    log "Stopping web server..."
    kill "$WEB_PID" 2>/dev/null || true
fi

log "========================================"
log "  Entrypoint Complete"
log "  Commit: $COMMIT_ID"
log "  val_bpb: $VAL_BPB"
log "  Results dir: $RESULTS_DIR"
log "========================================"
