#!/usr/bin/env python3
"""
reproduce_table1.py  --  pure-Python (no bash) driver for the 5 Table-1 cells.

Runs the paper's exact GD protocol for each tracked run, with NO result mixing:
  - Planner: GD (open-loop = plan_gd; MPC = plan_gd_mpc, GD subplanner)
  - 50 samples, goal_H=25 env-steps -> H=5 model steps (frameskip 5)
  - 3 data seeds: 100 / 200 / 300
  - Table 4 hyperparams come from the plan configs (horizon 25, zero init, Adam,
    lr 0.1, 100 steps; open-loop executes 25, MPC executes 5)
  - Objectives (Sec 5.3):
      UMaze (images only)          -> open-loop mode=last, MPC mode=all,    alpha=0
      PushT (images + proprio)     -> open-loop mode=last, MPC mode=staged, alpha=1

No mixing: before each run, ONLY that run's plan_outputs are removed (basename-
scoped), so its logs.json holds exactly its 3 seeds; summarize_run reads ONLY that
run's logs.json and stores results/<run>.json + rebuilds results/table1_reproduction.*

Usage:
    python reproduce_table1.py                       # all 5 runs
    python reproduce_table1.py <run> [<run> ...]     # only the named run(s)
    python reproduce_table1.py --base /abs/checkpoints/test
Detached (survives disconnects; nohup is POSIX, not bash-specific):
    nohup python reproduce_table1.py > eval_all.log 2>&1 &
    tail -f eval_all.log
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess

# ---- env defaults so NO shell exports are needed (export beforehand to override) ----
os.environ.setdefault("DATASET_DIR", "/workspace/arun/data")
# Fully disable wandb for headless eval: no login, no background service, no repo
# scanning. Results come from logs.json, not wandb. (offline still does work/threads.)
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
# On this B200 MIG slice, torch 2.7's default caching allocator NVML-asserts;
# cudaMallocAsync avoids that NVML path and works. (Do NOT set CUDA_VISIBLE_DEVICES
# to the MIG UUID -- mujoco-py int()-parses it and crashes; leave it unset.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PLAN_SERIAL_ENV", "1")
# Cap CPU threads: on many-core nodes torch's default thread pool makes tiny CPU
# ops (e.g. DINOv2 trunc_normal_ weight init) pathologically slow due to
# thread-launch/sync overhead. 8 is plenty for dataloading/env; GPU does the math.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

# mujoco-py needs MuJoCo 210 + nvidia libs on LD_LIBRARY_PATH at import time.
# Set it here so the plan.py subprocess (which inherits os.environ) can import gym/env.
_ld = os.environ.get("LD_LIBRARY_PATH", "")
for _p in (os.path.expanduser("~/.mujoco/mujoco210/bin"), "/usr/lib/nvidia"):
    if _p not in _ld.split(":"):
        _ld = (_ld + ":" + _p) if _ld else _p
os.environ["LD_LIBRARY_PATH"] = _ld

import summarize_run  # reuse the run-scoped summarizer (must sit next to this file)

SEEDS = [100, 200, 300]

# name -> (alpha, mpc_mode). alpha 0 = target images only; 1 = images + proprio.
CFG = {
 "umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05":          (0, "all"),
 "umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06":         (0, "all"),
 "umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05": (0, "all"),
 "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06":         (1, "staged"),
 "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05": (1, "staged"),
}
ORDER = list(CFG)


def clean_scoped(name):
    """Remove ONLY this run's plan_outputs so its logs.json holds exactly its 3 seeds."""
    for root in ("plan_outputs_gd", "plan_outputs_gd_mpc"):
        for d in glob.glob(os.path.join(root, f"{name}_*")):
            shutil.rmtree(d, ignore_errors=True)


def run_plan(cfg_name, run_dir, name, extra):
    cmd = [sys.executable, "plan.py", "--config-name", cfg_name,
           f"ckpt_base_path={run_dir}", f"model_name={name}", "model_epoch=latest",
           "decode_for_viz=false"] + extra
    print("   $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=os.environ).returncode


def run_eval(name, base):
    if name not in CFG:
        print(f"!!! SKIP {name}: unknown run (no alpha/mode mapping)", flush=True)
        return
    alpha, mpc_mode = CFG[name]
    run_dir = os.path.join(base, name)
    print("\n" + "#" * 60, flush=True)
    print(f"# RUN: {name}\n#   alpha={alpha}  open-loop=last  mpc={mpc_mode}", flush=True)
    print("#" * 60, flush=True)
    if not (os.path.isfile(os.path.join(run_dir, "hydra.yaml"))
            and os.path.isfile(os.path.join(run_dir, "checkpoints", "model_latest.pth"))):
        print(f"!!! SKIP {name}: missing hydra.yaml or checkpoints/model_latest.pth under {run_dir}", flush=True)
        return

    clean_scoped(name)

    print(">>> OPEN-LOOP (plan_gd.yaml, objective.mode=last, execute 25)", flush=True)
    for s in SEEDS:
        if run_plan("plan_gd.yaml", run_dir, name, [f"objective.alpha={alpha}", f"seed={s}"]):
            print(f"FAIL OL {name} seed={s}", flush=True)

    print(f">>> MPC (plan_gd_mpc.yaml, objective.mode={mpc_mode}, execute 5)", flush=True)
    for s in SEEDS:
        if run_plan("plan_gd_mpc.yaml", run_dir, name,
                    [f"objective.alpha={alpha}", f"objective.mode={mpc_mode}", f"seed={s}"]):
            print(f"FAIL MPC {name} seed={s}", flush=True)

    # Immediate, run-scoped summary + results/<name>.json
    summarize_run.summarize_one(name)


def main():
    ap = argparse.ArgumentParser(description="Pure-Python Table-1 reproduction driver (no bash).")
    ap.add_argument("runs", nargs="*", help="run basenames to evaluate (default: all 5 tracked runs)")
    ap.add_argument("--base", default=os.path.join(os.getcwd(), "checkpoints", "test"),
                    help="folder containing the run dirs (default ./checkpoints/test)")
    args = ap.parse_args()
    runs = args.runs if args.runs else ORDER
    print(f"BASE={args.base}  DATASET_DIR={os.environ['DATASET_DIR']}  runs={len(runs)}", flush=True)
    for name in runs:
        run_eval(name, args.base)
    print("\n############### FINAL TABLE-1 REPRODUCTION ###############", flush=True)
    summarize_run.rebuild_master()
    print("\nALL EVALS DONE", flush=True)


if __name__ == "__main__":
    main()
