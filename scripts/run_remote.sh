#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod Training Orchestrator
# ============================================================
# Usage: ./run_remote.sh [--pod-id <id>]
# Environment: Reads RUNPOD_POD_ID from .env in project root
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

SSH_HOST="87.120.211.209"
SSH_PORT="19145"
SSH_KEY="~/.ssh/id_ed25519"
SSH_USER="root"
REMOTE_WORKDIR="/workspace/parameter-golf/code"

SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST"
SCP_CMD="scp -o StrictHostKeyChecking=no -P $SSH_PORT -i $SSH_KEY"
ENTRYPOINT_FILE="$SCRIPT_DIR/pod_entrypoint.sh"

POD_ID=""
POLL_INTERVAL=15
STARTUP_TIMEOUT=300
TRAINING_TIMEOUT=1800

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $*" >&2; }
die() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; exit 1; }

load_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        die ".env file not found at $ENV_FILE"
    fi
    POD_ID=$(grep "^RUNPOD_POD_ID=" "$ENV_FILE" | cut -d'=' -f2 | tr -d '[:space:]')
    if [[ -z "$POD_ID" ]]; then
        die "RUNPOD_POD_ID not found in $ENV_FILE"
    fi
    log "Using pod ID: $POD_ID"
}

wait_for_pod_running() {
    log "Waiting for pod to be RUNNING (timeout: ${STARTUP_TIMEOUT}s)..."
    local elapsed=0
    while true; do
        local status
        status=$(runpodctl get pod "$POD_ID" 2>/dev/null | grep -i "RUNNING\|ACTIVE\|RUN" || true)
        if echo "$status" | grep -qi "RUNNING\|ACTIVE"; then
            log "Pod is RUNNING: $status"
            sleep 5
            return 0
        fi
        if echo "$status" | grep -qi "STOPPED\|TERMINATED\|FAILED"; then
            die "Pod entered unexpected state: $status"
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
        if (( elapsed >= STARTUP_TIMEOUT )); then
            die "Timeout waiting for pod to start (${STARTUP_TIMEOUT}s)"
        fi
        log "  Still waiting... (${elapsed}s elapsed)"
    done
}

stop_pod() {
    log "Stopping pod $POD_ID..."
    runpodctl stop pod "$POD_ID" 2>/dev/null || warn "Failed to stop pod (may already be stopped)"
}

cleanup_on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        warn "Script exited with error code $exit_code"
        warn "NOT stopping pod — inspect state manually with: runpodctl get pod $POD_ID"
        warn "To stop manually: runpodctl stop pod $POD_ID"
    fi
}
trap cleanup_on_exit EXIT

main() {
    load_env

    log "========================================"
    log "  RunPod Training Orchestrator"
    log "  Pod: $POD_ID"
    log "  Remote dir: $REMOTE_WORKDIR"
    log "========================================"

    log "Starting pod..."
    runpodctl start pod "$POD_ID" 2>/dev/null || true
    wait_for_pod_running

    log "Transferring entrypoint script to pod..."
    $SCP_CMD "$ENTRYPOINT_FILE" "$SSH_USER@$SSH_HOST:$REMOTE_WORKDIR/pod_entrypoint.sh" || die "Failed to upload entrypoint script"

    log "Executing training workflow on pod..."
    log "(This will take up to ${TRAINING_TIMEOUT}s — training + 10min wallclock)"

    set +e
    $SSH_CMD "cd $REMOTE_WORKDIR && chmod +x pod_entrypoint.sh && bash pod_entrypoint.sh" 2>&1
    SSH_EXIT=$?
    set -e

    if (( SSH_EXIT != 0 )); then
        warn "SSH command exited with code $SSH_EXIT"
    fi

    log "Downloading results from pod..."
    mkdir -p "$PROJECT_ROOT/logs"
    $SCP_CMD "$SSH_USER@$SSH_HOST:$REMOTE_WORKDIR/results/"* "$PROJECT_ROOT/logs/" 2>/dev/null || warn "No results files found to download"

    log "Stopping pod to halt billing..."
    stop_pod

    log ""
    log "========================================"
    log "  Run Complete"
    log "  Results saved to: $PROJECT_ROOT/logs/"
    log "  List results: ls $PROJECT_ROOT/logs/"
    log "========================================"
}

main "$@"
