#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# eval_pusht_3seeds.sh  <run_dir>
# Reproduces a Table-1 PushT cell the way the paper does it: mean +/- std over
# THREE data-sampling seeds (100/200/300). The `seed` arg controls which 50 test
# start/goal pairs are drawn (plan.py line 134: eval_seed = seed*n + 1).
#
# RUN-SPECIFIC: only removes/aggregates the plan outputs for THIS run's basename,
# so evaluating the straightened (✓) run does NOT touch the ✗ run's results and
# the two never get mixed in the aggregation.
#
# Open-loop uses objective.mode=last (terminal MSE within H).
# MPC uses objective.mode=staged (terminal within H, weighted beyond H).
# Both use objective.alpha=1 (PushT plans on target images AND proprio).
#
# Usage (✗): bash eval_pusht_3seeds.sh .../pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
# Usage (✓): bash eval_pusht_3seeds.sh .../pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05
# ---------------------------------------------------------------------------
set -uo pipefail

RUN_DIR="${1:?usage: eval_pusht_3seeds.sh <run_dir>}"
RUN_DIR="$(readlink -f "$RUN_DIR")"
NAME="$(basename "$RUN_DIR")"
SEEDS=(100 200 300)

export DATASET_DIR="${DATASET_DIR:-/workspace/arun/data}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export PLAN_SERIAL_ENV="${PLAN_SERIAL_ENV:-1}"

# Start clean for THIS run only (basename-scoped) so each logs.json holds exactly
# one entry per seed. Other runs' outputs (e.g. the ✗ run, UMaze) are untouched.
echo ">>> Removing previous plan outputs for run '${NAME}' only..."
rm -rf plan_outputs_gd/${NAME}_*/ plan_outputs_gd_mpc/${NAME}_*/ 2>/dev/null

for s in "${SEEDS[@]}"; do
  echo ""
  echo "=================== OPEN-LOOP  seed=$s ==================="
  python plan.py --config-name plan_gd.yaml \
    ckpt_base_path="$RUN_DIR" model_name="$NAME" model_epoch=latest \
    decode_for_viz=false objective.alpha=1 seed=$s \
    || echo "!!! open-loop seed $s failed"
done

for s in "${SEEDS[@]}"; do
  echo ""
  echo "=================== MPC  seed=$s ==================="
  python plan.py --config-name plan_gd_mpc.yaml \
    ckpt_base_path="$RUN_DIR" model_name="$NAME" model_epoch=latest \
    decode_for_viz=false objective.alpha=1 objective.mode=staged seed=$s \
    || echo "!!! MPC seed $s failed"
done

echo ""
echo "############### RESULTS for ${NAME} (mean +/- std over 3 seeds) ###############"
NAME="$NAME" python - <<'PY'
import glob, json, os, statistics as st
name=os.environ["NAME"]
def collect(root):
    vals=[]
    for f in glob.glob(f"{root}/{name}_*/**/logs.json", recursive=True):
        for line in open(f):
            line=line.strip()
            if not line: continue
            try: vals.append(json.loads(line)["final_eval/success_rate"])
            except Exception: pass
    return vals
for label, root in [("OPEN-LOOP","plan_outputs_gd"),("MPC","plan_outputs_gd_mpc")]:
    v=collect(root)
    if v:
        m=st.mean(v); s=st.pstdev(v) if len(v)>1 else 0.0
        print(f"{label:10s} seeds={[round(x,4) for x in v]}  ->  mean {m*100:.2f} +/- {s*100:.2f} %")
    else:
        print(f"{label:10s} no results found")
PY
