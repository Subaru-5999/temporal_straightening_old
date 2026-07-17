#!/usr/bin/env bash
# ===========================================================================
# reproduce_table1.sh
# Paper-faithful reproduction of the 5 Table-1 cells we track, using the trained
# runs stored in checkpoints/test/. Follows the paper exactly:
#   - Planner: GD (open-loop = plan_gd; MPC = plan_gd_mpc with GD subplanner)
#   - 50 test samples, goal_H=25 env-steps -> H=5 model steps (frameskip 5)
#   - 3 data seeds: 100 / 200 / 300  (eval_seed = seed*n+1 in plan.py)
#   - Planning hyperparams (Table 4): horizon 25, zero init, Adam, lr 0.1, 100 steps;
#     open-loop executes 25 actions (max_iter 1), MPC executes 5 (max_iter 20).
#   - Objectives (Sec 5.3):
#       * UMaze (target images only): open-loop mode=last, MPC mode=all, alpha=0
#       * PushT (target images + proprio): open-loop mode=last, MPC mode=staged, alpha=1
#
# NO RESULT MIXING:
#   * Before each run, ONLY that run's plan_outputs are removed (basename-scoped),
#     so its logs.json holds exactly its 3 seeds.
#   * After each run, summarize_run.py reads ONLY that run's logs.json and stores
#     results/<run>.json + prints a run-scoped block (ours vs paper).
#   * The master table (results/table1_reproduction.{md,csv}) is rebuilt from the
#     per-run json files, keyed by unique run name.
#
# Usage:
#   bash reproduce_table1.sh                       # all 5 runs
#   bash reproduce_table1.sh <run_basename> ...    # only the named run(s)
#   BASE=/abs/path/to/runs bash reproduce_table1.sh
#
# Detached (survives disconnects):
#   setsid nohup bash reproduce_table1.sh > eval_all.log 2>&1 < /dev/null &
#   tail -f eval_all.log
# ===========================================================================
set -uo pipefail

BASE="${BASE:-$PWD/checkpoints/test}"
SEEDS=(100 200 300)

export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
# MIG allocator fix + serial envs (do not change success_rate).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export PLAN_SERIAL_ENV="${PLAN_SERIAL_ENV:-1}"

# name -> "alpha mpc_mode"  (alpha 0 = images only; 1 = images + proprio)
declare -A CFG=(
  [umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05]="0 all"
  [umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06]="0 all"
  [umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05]="0 all"
  [pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06]="1 staged"
  [pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05]="1 staged"
)

# Default order = the 5 tracked runs; or take run names from argv.
DEFAULT_ORDER=(
  umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05
  umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
  umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
  pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
  pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
)
if [ "$#" -gt 0 ]; then RUNS=("$@"); else RUNS=("${DEFAULT_ORDER[@]}"); fi

run_eval () {
  local name="$1"
  local cfg="${CFG[$name]:-}"
  if [ -z "$cfg" ]; then echo "!!! SKIP $name : not a known run (no alpha/mode mapping)"; return; fi
  local alpha; local mpc_mode
  alpha="$(echo "$cfg" | cut -d' ' -f1)"
  mpc_mode="$(echo "$cfg" | cut -d' ' -f2)"
  local RUN="$BASE/$name"

  echo ""
  echo "############################################################"
  echo "# RUN: $name"
  echo "#   alpha=$alpha  open-loop=last  mpc=$mpc_mode"
  echo "############################################################"
  if [ ! -f "$RUN/hydra.yaml" ] || [ ! -f "$RUN/checkpoints/model_latest.pth" ]; then
    echo "!!! SKIP $name : missing hydra.yaml or checkpoints/model_latest.pth under $RUN"
    return
  fi

  # RUN-SCOPED clean so this run's logs.json holds exactly its own 3 seeds.
  rm -rf plan_outputs_gd/${name}_*/ plan_outputs_gd_mpc/${name}_*/ 2>/dev/null

  echo ">>> OPEN-LOOP  (plan_gd.yaml, objective.mode=last, execute 25 actions)"
  for s in "${SEEDS[@]}"; do
    python plan.py --config-name plan_gd.yaml \
      ckpt_base_path="$RUN" model_name="$name" model_epoch=latest \
      decode_for_viz=false objective.alpha="$alpha" seed="$s" \
      || echo "FAIL OL $name seed=$s"
  done

  echo ">>> MPC  (plan_gd_mpc.yaml, objective.mode=$mpc_mode, execute 5 actions)"
  for s in "${SEEDS[@]}"; do
    python plan.py --config-name plan_gd_mpc.yaml \
      ckpt_base_path="$RUN" model_name="$name" model_epoch=latest \
      decode_for_viz=false objective.alpha="$alpha" objective.mode="$mpc_mode" seed="$s" \
      || echo "FAIL MPC $name seed=$s"
  done

  # Immediate, run-scoped summary (reads ONLY this run's logs.json) + store.
  python summarize_run.py "$name"
}

for name in "${RUNS[@]}"; do
  run_eval "$name"
done

echo ""
echo "############### FINAL TABLE-1 REPRODUCTION (all recorded runs) ###############"
python summarize_run.py --all
echo ""
echo "ALL EVALS DONE"
