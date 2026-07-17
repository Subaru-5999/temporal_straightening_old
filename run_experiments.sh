#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_experiments.sh
# Runs a QUEUE of training jobs back-to-back on the single B200 MIG slice
# (one at a time), detached so it survives disconnects. Each job trains one
# Table-1 cell (encoder x straighten) for a given env and seed.
#
# Reproduction note: the paper reports mean +/- std over 3 seeds. Because the
# Hydra run dir is NOT keyed on seed, we isolate each seed by pointing
# ckpt_base_path at ./checkpoints/seed<N> so seeds never collide.
#
# Usage:
#   bash run_experiments.sh                       # Phase 1 defaults below
#   ENV=wall SEEDS="0 1 2" bash run_experiments.sh
#
# Edit the JOBS list to change which cells run. Format: "encoder|straighten".
# ---------------------------------------------------------------------------
set -euo pipefail

export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export WANDB_MODE="${WANDB_MODE:-offline}"

ENV="${ENV:-point_maze}"        # point_maze (UMaze) | wall | pusht
SEEDS="${SEEDS:-0}"             # e.g. "0 1 2" for error bars

# Phase 1: the core straightening claim on UMaze (channel projector).
# Add/remove lines to run other Table-1 cells. Format: "encoder|straighten".
JOBS=(
  "dino_channel|False"
  "dino_channel|aggcos1e-1"
)

# --- Self-detach: re-exec this script in the background (SIGHUP-proof) --------
if [ "${DETACHED:-0}" != "1" ]; then
  LOG="experiments_$(date +%Y%m%d_%H%M%S).log"
  DETACHED=1 setsid nohup bash "$0" "$@" > "${LOG}" 2>&1 < /dev/null &
  echo "$!" > .experiments_pid
  cat <<EOF
Experiment queue started (PID $!), logging to: ${LOG}
  env=${ENV}  seeds=[${SEEDS}]  jobs=[${JOBS[*]}]

  Watch:   tail -f ${LOG}
  Alive?:  ps -p $! -o pid,etime,cmd
  Stop:    kill $!

Checkpoints per seed: ./checkpoints/seed<N>/test/<run_name>/checkpoints/
EOF
  exit 0
fi

# --- Detached child: actually run the queue ----------------------------------
for seed in ${SEEDS}; do
  for job in "${JOBS[@]}"; do
    enc="${job%%|*}"; str="${job##*|}"
    echo "==================================================================="
    echo "=== $(date '+%F %T')  env=${ENV}  encoder=${enc}  straighten=${str}  seed=${seed}"
    echo "==================================================================="
    python train.py --config-name train.yaml \
      env="${ENV}" \
      encoder="${enc}" \
      training.straighten="${str}" \
      training.seed="${seed}" \
      ckpt_base_path="./checkpoints/seed${seed}" \
      || echo "!!! job FAILED: env=${ENV} encoder=${enc} straighten=${str} seed=${seed} (continuing queue)"
  done
done
echo "=== QUEUE COMPLETE $(date '+%F %T') ==="
