#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# evaluate.sh  <run_dir>  [model_epoch]
# Runs both planners (open-loop GD + MPC) on ONE trained run and prints the two
# success rates that fill a Table-1 row.
#
# <run_dir> = the folder that directly contains hydra.yaml and checkpoints/
#   e.g. /workspace/arun/temporal-straightening/checkpoints/test/umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05
#
# plan.py treats an ABSOLUTE ckpt_base_path as the model folder directly
# (model_name is only used for output naming), so we pass the run dir as-is.
#
# Example:
#   bash evaluate.sh /workspace/arun/temporal-straightening/checkpoints/test/umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05
#
# For PushT add: EXTRA="objective.alpha=1"  (MPC also wants objective.mode=staged)
# ---------------------------------------------------------------------------
set -uo pipefail

RUN_DIR="${1:?usage: evaluate.sh <run_dir> [model_epoch]}"
RUN_DIR="$(readlink -f "$RUN_DIR")"
MODEL_EPOCH="${2:-latest}"
MODEL_NAME="$(basename "$RUN_DIR")"
EXTRA="${EXTRA:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
# Headless GL for the sim's offscreen rendering.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
# MIG fix: torch 2.7's expandable_segments allocator uses NVML/VMM APIs that fail
# on MIG instances (NVML_SUCCESS assert). Disable it to use the classic allocator.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
# MIG fix: run planning envs serially (no subprocess fork). Forking after CUDA/NVML
# init on a MIG slice trips the allocator NVML assert during backward.
export PLAN_SERIAL_ENV="${PLAN_SERIAL_ENV:-1}"

if [ ! -f "${RUN_DIR}/hydra.yaml" ]; then
  echo "ERROR: ${RUN_DIR}/hydra.yaml not found. Pass the run dir that contains hydra.yaml + checkpoints/."
  exit 1
fi

for cfg in plan_gd plan_gd_mpc; do
  echo "=== ${cfg} ==="
  # decode_for_viz=false: skips decoding the full (growing) rollout into images +
  # video each MPC iter. This does NOT change success_rate (computed from env state),
  # but massively cuts GPU memory growth -> avoids the MIG NVML-under-pressure assert.
  python plan.py --config-name "${cfg}.yaml" \
    ckpt_base_path="${RUN_DIR}" model_name="${MODEL_NAME}" model_epoch="${MODEL_EPOCH}" \
    decode_for_viz=false ${EXTRA} \
    || echo "!!! ${cfg} failed for ${MODEL_NAME}"
done

echo ""
echo "=== success_rate values (x100 = %) ==="
grep -rh "success_rate" plan_outputs_gd/ 2>/dev/null | tail -n 1 || echo "  (open-loop logs.json not found)"
grep -rh "success_rate" plan_outputs_gd_mpc/ 2>/dev/null | tail -n 1 || echo "  (mpc logs.json not found)"
echo "Tip: 'python collect_results.py' aggregates everything into a table."
