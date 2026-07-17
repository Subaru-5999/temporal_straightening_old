#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# all_results.sh
# Evaluate EVERY finished training run (both planners) and print the results
# table. Run this ONLY after setup_planning.sh has succeeded.
#
# It finds every run dir containing checkpoints/model_latest.pth under
# checkpoints/test/ and checkpoints/seed*/test/, runs open-loop + MPC planning
# on each, then aggregates with collect_results.py.
#
# Usage:
#   bash all_results.sh                 # evaluates model_epoch=latest
#   EPOCH=20 bash all_results.sh        # evaluate a specific epoch
# ---------------------------------------------------------------------------
set -uo pipefail
export WANDB_MODE="${WANDB_MODE:-offline}"
EPOCH="${EPOCH:-latest}"
shopt -s nullglob

runs=()
for run in checkpoints/test/*/ checkpoints/seed*/test/*/; do
  [ -f "${run}checkpoints/model_latest.pth" ] && runs+=("${run%/}")
done

if [ ${#runs[@]} -eq 0 ]; then
  echo "No finished runs found (no checkpoints/*/test/*/checkpoints/model_latest.pth)."
  exit 1
fi

echo "Found ${#runs[@]} run(s) to evaluate:"
printf '  %s\n' "${runs[@]}"

for run in "${runs[@]}"; do
  run_dir="$(readlink -f "$run")"                  # the run folder itself
  model_name="$(basename "$run")"
  extra=""
  case "$model_name" in *pusht*|*PushT*) extra="objective.alpha=1";; esac
  echo "==================================================================="
  echo "### Evaluating ${model_name}"
  echo "==================================================================="
  EXTRA="$extra" bash evaluate.sh "$run_dir" "$EPOCH" \
    || echo "!!! evaluation failed for ${model_name} (continuing)"
done

echo ""
echo "=== Aggregated results ==="
python collect_results.py
