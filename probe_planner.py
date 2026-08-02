#!/usr/bin/env python3
"""
probe_planner.py -- run planner probes on an EXISTING checkpoint and print one clean table.
No training. Wraps plan.py as a subprocess, exactly like reproduce_table1.py does.

WHY THIS EXISTS
    Two practical problems made hand-typed probes unreliable:
      1. `ckpt_base_path` is an ABSOLUTE path and Hydra embeds it verbatim in the output dir
         template, so a run's folder ends up ~6 levels deep
         (plan_outputs_gd/<model>_gH25_dset/workspace/arun/.../test/<model>_gd_..._objlast_initzero).
         A one-level shell glob never finds its logs.json. This script parses the exact directory
         out of plan.py's own "Planning result saved dir:" line instead of guessing.
      2. Multi-line shell commands with backslash continuations break on paste; the pieces run as
         separate commands, `model_name` stays null, and Hydra dies in `replace_slash(None)`.
         One `python probe_planner.py ...` invocation has nothing to mangle.

WHAT IT PROBES
    --steps A B C ...   success rate vs the number of Adam steps on the SAME task set.
        Motivated by the measured result that optimizing this objective can DESTROY working plans
        (oracle-initialised: 1.00 at 0 steps -> 0.86 at 100 steps). If success is non-monotone in
        opt_steps, the deployed planner over-optimizes its own cost. NOTE: this establishes the
        SHAPE of the curve. Choosing a stopping point by test success would be eval tuning and
        must not be reported as a result.
    --gt-init           initialise the planner at the dataset's ground-truth actions.
        PRIVILEGED INFORMATION -- diagnosis only, never a reportable configuration. Runs land in
        a `_gtinit`-tagged folder so they cannot be confused with reportable numbers.

    For every probe it also pulls `obj_init` / `obj_final` / `pathdev_init` / `pathdev_final` out
    of that run's logs.json, so the model's own cost before and after optimization is visible next
    to the success rate. With --gt-init, obj_init is the cost of a plan that solves the task
    exactly: obj_final < obj_init together with a fall in success is direct evidence that the
    objective's minimizer is not the task solution.

USAGE
    python probe_planner.py --run $PWD/checkpoints_baseline_matched/test/<NAME> \
        --steps 0 10 25 50 100 200 --alpha 1 --seed 100
    python probe_planner.py --run <same> --steps 0 100 --alpha 1 --seed 100 --gt-init
"""
import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path

# ---- env defaults so no shell exports are needed (export beforehand to override) ----
os.environ.setdefault("DATASET_DIR", "/workspace/arun/data")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PLAN_SERIAL_ENV", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
# mujoco-py needs MuJoCo 210 + nvidia libs on LD_LIBRARY_PATH at import time; the plan.py
# subprocess inherits os.environ at exec, so setting it here works (it would not in-process).
_ld = os.environ.get("LD_LIBRARY_PATH", "")
for _p in (os.path.expanduser("~/.mujoco/mujoco210/bin"), "/usr/lib/nvidia"):
    if _p not in _ld:
        _ld = f"{_ld}:{_p}" if _ld else _p
os.environ["LD_LIBRARY_PATH"] = _ld
# A MIG UUID breaks mujoco-py's int() parse of CUDA_VISIBLE_DEVICES.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cvd and not all(p.strip().isdigit() for p in _cvd.split(",") if p.strip()):
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

RE_SUCCESS = re.compile(r"Success rate:\s*([0-9.]+)")
RE_SAVEDIR = re.compile(r"Planning result saved dir:\s*(\S+)")
LOG_KEYS = ("obj_init", "obj_final", "pathdev_init", "pathdev_final")
# GDPlanner prints this; it does NOT reliably reach logs.json, because MPCPlanner builds its
# sub-planner with log_filename=None and BasePlanner.dump_logs is then a no-op. Every config here
# plans through MPCPlanner, so stdout is the only dependable source.
RE_PROBE = re.compile(r"\[probe\]\s+(\S+)\s+"
                      r"obj_init=(\S+)\s+obj_final=(\S+)\s+"
                      r"pathdev_init=(\S+)\s+pathdev_final=(\S+)")


def parse_probe(text):
    """First [probe] line = MPC iteration 0, the plan made from the true initial observation.
    That is the one the diagnostic is about; later iterations replan from executed states."""
    m = RE_PROBE.search(text)
    if not m:
        return {}
    return {
        "obj_init": float(m.group(2)),
        "obj_final": float(m.group(3)),
        "pathdev_init": float(m.group(4)),
        "pathdev_final": float(m.group(5)),
    }


def read_probe_logs(save_dir):
    """Pull the last recorded obj_*/pathdev_* values from a run's logs.json.

    logs.json is APPEND-only and plan.py chdirs into the output dir, so the file may hold
    several runs that shared a folder. Taking the last occurrence of each key gives the run that
    just finished.
    """
    out = {}
    p = Path(save_dir) / "logs.json"
    if not p.is_file():
        return out
    for line in p.read_text(errors="ignore").splitlines():
        try:
            entry = json.loads(line)
        except Exception:
            continue
        for k, v in entry.items():
            short = k.split("/")[-1]
            if short in LOG_KEYS:
                out[short] = v
    return out


def run_probe(cfg_name, run_dir, name, steps, alpha, seed, gt_init, extra):
    cmd = [sys.executable, "plan.py", "--config-name", cfg_name,
           f"ckpt_base_path={run_dir}", f"model_name={name}", "model_epoch=latest",
           f"objective.alpha={alpha}", f"seed={seed}", "decode_for_viz=false",
           f"planner.sub_planner.opt_steps={steps}"]
    if gt_init:
        cmd.append("debug_dset_init=true")
    cmd += extra
    print(f"\n$ {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, env=os.environ, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    text = proc.stdout or ""
    if proc.returncode != 0:
        print(text[-4000:], flush=True)
        return {"steps": steps, "success": None, "error": f"exit {proc.returncode}"}

    succ = RE_SUCCESS.findall(text)
    save = RE_SAVEDIR.findall(text)
    row = {
        "steps": steps,
        "success": float(succ[-1]) if succ else None,
        "save_dir": save[-1] if save else None,
    }
    row.update(parse_probe(text))                    # stdout: always present
    if row["save_dir"]:
        for k, v in read_probe_logs(row["save_dir"]).items():
            row.setdefault(k, v)                     # logs.json: only if the sub-planner logs
    print(f"   -> success={row['success']}"
          + (f"  obj {row['obj_init']:.6g} -> {row['obj_final']:.6g}"
             if "obj_init" in row else ""), flush=True)
    return row


def report(rows, gt_init):
    print("\n" + "=" * 92)
    print("PLANNER PROBE" + ("  [gt-init: PRIVILEGED, diagnosis only]" if gt_init else ""))
    print("=" * 92)
    head = (f"{'opt_steps':>10}{'success':>10}{'obj_init':>13}{'obj_final':>13}"
            f"{'obj_drop%':>11}{'pathdev_i':>11}{'pathdev_f':>11}")
    print(head)
    print("-" * len(head))
    for r in rows:
        s = "ERR" if r.get("success") is None else f"{r['success']:.2f}"
        oi, of = r.get("obj_init"), r.get("obj_final")
        drop = ("" if (oi in (None, 0) or of is None) else f"{100.0*(oi-of)/abs(oi):.1f}")
        fmt = lambda v, w, p=6: (" " * (w - 1) + "-") if v is None else f"{v:>{w}.{p}f}"
        print(f"{r['steps']:>10}{s:>10}{fmt(oi,13)}{fmt(of,13)}{drop:>11}"
              + fmt(r.get('pathdev_init'), 11, 4) + fmt(r.get('pathdev_final'), 11, 4))

    ok = [r for r in rows if r.get("success") is not None]
    if len(ok) > 1:
        best = max(ok, key=lambda r: r["success"])
        last = ok[-1]
        print(f"\npeak success {best['success']:.2f} at opt_steps={best['steps']}; "
              f"final probe {last['success']:.2f} at opt_steps={last['steps']}")
        # n=50 tasks: one task = 2 pp, so treat small gaps as noise, not structure.
        if best["steps"] != last["steps"] and (best["success"] - last["success"]) > 0.04:
            print("NON-MONOTONE: success falls with more optimization of the SAME objective.")
            print("That is a statement about the objective, not a tuned hyperparameter -- do NOT")
            print("report the peak as a result. Confirm across seeds 100/200/300 before claiming")
            print("anything, and remember the ~11.5 pp detection floor for this project.")
        else:
            print("Monotone within noise (1 task = 2 pp at n=50): no over-optimization evidence.")

    # The measurement the whole probe exists for.
    misspec = [r for r in rows
               if r.get("obj_init") is not None and r.get("obj_final") is not None
               and r["obj_final"] < r["obj_init"] and r["steps"] > 0]
    if misspec and gt_init:
        print("\nMISSPECIFICATION CHECK (gt-init): GD reached a LOWER model cost than a plan that")
        print("solves the task exactly, at these step counts:")
        for r in misspec:
            print(f"   steps={r['steps']:>4}  obj {r['obj_init']:.6g} -> {r['obj_final']:.6g}"
                  f"  ({100.0*(r['obj_init']-r['obj_final'])/abs(r['obj_init']):.1f}% lower)"
                  f"  success {r['success']:.2f}")
        print("If success is below the opt_steps=0 row, the true goal is not the argmin of the")
        print("latent cost. That is a property of the objective; no encoder regularizer fixes it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="ABSOLUTE path to the run dir, i.e. <ckpt_root>/test/<model_name>")
    ap.add_argument("--config", default="plan_gd.yaml", help="plan config (default plan_gd.yaml)")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 10, 25, 50, 100, 200])
    ap.add_argument("--alpha", type=float, default=1, help="objective.alpha (PushT: 1, UMaze: 0)")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--gt-init", dest="gt_init", action="store_true",
                    help="initialise at gt_actions (PRIVILEGED -- diagnosis only)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra plan.py overrides, verbatim")
    ap.add_argument("--out", default="results/planner_probe.json")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run)
    if not os.path.isfile(os.path.join(run_dir, "hydra.yaml")):
        ap.error(f"{run_dir} has no hydra.yaml -- pass the RUN dir (<root>/test/<model_name>), "
                 f"not the checkpoint root")
    name = os.path.basename(run_dir.rstrip("/"))
    print(f"run  = {run_dir}\nname = {name}\nsteps= {args.steps}  alpha={args.alpha}  "
          f"seed={args.seed}  gt_init={args.gt_init}", flush=True)

    rows = [run_probe(args.config, run_dir, name, s, args.alpha, args.seed, args.gt_init,
                      args.extra) for s in args.steps]
    report(rows, args.gt_init)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"run": name, "seed": args.seed, "alpha": args.alpha,
                   "gt_init": args.gt_init, "rows": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
