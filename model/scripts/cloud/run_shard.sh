#!/usr/bin/env bash
# Usage: scripts/cloud/run_shard.sh <run_name> <shard_index|parity> [workers]
# Resumable: re-run the same command after any interruption. Logs to data/landmarks/runs/<run>/logs/.
set -euo pipefail
RUN=$1; SHARD=$2; WORKERS=${3:-$(( $(nproc) - 1 ))}
LOG=data/landmarks/runs/$RUN/logs; mkdir -p "$LOG"
if [[ "$SHARD" == "parity" ]]; then ARGS=(--parity); else ARGS=(--shard "$SHARD"); fi
PYTHONPATH=src nohup .venv/bin/python -W ignore scripts/s2_08_extract_shard.py --run-name "$RUN" "${ARGS[@]}" \
  --workers "$WORKERS" --approved-large-run >> "$LOG/$SHARD.log" 2>&1 &
echo "started shard $SHARD with $WORKERS workers; log: $LOG/$SHARD.log (pid $!)"
