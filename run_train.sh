#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train.sh
# Launch training so it SURVIVES laptop/browser disconnects.
#
# When you run a long job in a Jupyter terminal/cell, the process is a child of
# that session. If your laptop's internet drops, the session can send SIGHUP and
# kill training. `nohup` + background detaches the process so it keeps running on
# the pod regardless of your connection. Reconnect later and tail the log.
#
# Usage (from the project root, inside the conda env):
#     bash run_train.sh                         # default point_maze run
#     bash run_train.sh env=pusht               # pass any hydra overrides
#     bash run_train.sh training.resume_from=/abs/path/model_10.pth
#
# Even better for interactive monitoring: use tmux (see README notes below).
# ---------------------------------------------------------------------------
set -euo pipefail

export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
# Offline by default so a dropped connection never blocks wandb logging.
export WANDB_MODE="${WANDB_MODE:-offline}"

LOG="train_$(date +%Y%m%d_%H%M%S).log"

echo "DATASET_DIR = ${DATASET_DIR}"
echo "WANDB_MODE  = ${WANDB_MODE}"
echo "Launching training detached (SIGHUP-proof); logging to: ${LOG}"

# nohup + setsid: fully detach from the controlling terminal/session.
setsid nohup python train.py --config-name train.yaml env=point_maze "$@" \
  > "${LOG}" 2>&1 < /dev/null &

PID=$!
echo "${PID}" > .train_pid
cat <<EOF

Training started with PID ${PID} (also saved to .train_pid).
It will keep running even if you close the browser or lose internet.

  Watch live:      tail -f ${LOG}
  Check running:   ps -p ${PID} -o pid,etime,cmd
  Stop it:         kill ${PID}

If the pod itself restarts, just rerun this script -- it auto-resumes from the
latest checkpoint (or pass training.resume_from=<path> for a specific one).
EOF
