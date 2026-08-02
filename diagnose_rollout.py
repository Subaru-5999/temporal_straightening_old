#!/usr/bin/env python3
"""
diagnose_rollout.py -- forward-only, cross-run diagnostic for autoregressive error growth
and latent curvature, measured on a FIXED window so every checkpoint is comparable.

WHY THIS EXISTS
    The per-k rollout error and the per-scale curvature are logged DURING training, but the
    measurement window there is whatever that run happened to train on (4 frames for the
    paper baseline, 7 for rollout_steps=4, 9 for multi-scale s=4). Curvature and error both
    grow with temporal extent, so those logged numbers are NOT comparable across runs, and
    runs trained before the logging commit have no numbers at all.

    This script re-measures both quantities OFFLINE, on the validation split, with a window
    size WE choose. Same data, same window, same code path for every checkpoint -> the
    comparison is clean.

WHAT IT MEASURES
    rollout_err_k1..kK : MSE between the k-step autoregressive prediction and the real latent,
                         feeding REAL dataset actions (exactly VWorldModel.rollout_consistency_loss).
                         The shape of this curve is the exposure-bias diagnostic: steep growth
                         means the predictor amplifies its own error, flat means it does not.
    curv_s<S>          : the paper's curvature 1 - mean_t cos(v_t^(s), v_{t+s}^(s)) per scale.

CAVEAT ON CURVATURE (read before comparing)
    mode 'aggcos' measures curvature on the encoder's pooled feature, produced by encoder.agg.
    In a run trained WITHOUT straightening, no gradient ever reaches agg, so it stays at its
    random init and the number is meaningless. Only compare curvature between runs that
    trained with aggcos straightening. Rollout error has no such caveat -- it is well defined
    for every checkpoint.

USAGE
    # one run
    python diagnose_rollout.py --base $PWD/checkpoints_rollout/test <run_name>

    # several runs living under different ckpt roots: pass full paths
    python diagnose_rollout.py --runs \
        $PWD/checkpoints/test/pusht_..._ms1-4_lam0.1-0_ep3 \
        $PWD/checkpoints_rollout/test/pusht_..._roll4g0.9_ep3

    # options
    --frames N    measurement window in latent frames (default 7 = num_hist + 4)
    --K N         rollout depth to probe (default: frames - num_hist)
    --batches N   cap the number of val batches (default 0 = all; ~57 for PushT)
    --scales ...  curvature scales to measure (default 1)
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Same env defaults as the eval driver, so this runs on the MIG slice without extra exports.
os.environ.setdefault("DATASET_DIR", "/workspace/arun/data")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("PLAN_SERIAL_ENV", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
# We import plan.py (for load_model), which pulls in env.venv -> gym/d4rl -> mujoco-py.
# mujoco-py needs MuJoCo 210 + the nvidia libs on LD_LIBRARY_PATH at IMPORT time, so set it
# here exactly as reproduce_table1.py does, before the import below.
_ld = os.environ.get("LD_LIBRARY_PATH", "")
for _p in (os.path.expanduser("~/.mujoco/mujoco210/bin"), "/usr/lib/nvidia"):
    if _p not in _ld.split(":"):
        _ld = (_ld + ":" + _p) if _ld else _p
os.environ["LD_LIBRARY_PATH"] = _ld
# A MIG UUID here breaks mujoco-py's int() parse; PushT needs no MuJoCo but stay consistent.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cvd and not all(p.strip().isdigit() for p in _cvd.split(",") if p.strip()):
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import torch
import hydra
from omegaconf import OmegaConf

import custom_resolvers  # noqa: F401  -- registers the OmegaConf resolvers used by hydra.yaml
from plan import load_model
from utils import seed


def build_val_loader(train_cfg, num_frames, batch_size):
    """Validation dataloader for THIS run's env, sliced to a fixed `num_frames` window."""
    datasets, _ = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
        num_frames=num_frames,
    )
    return torch.utils.data.DataLoader(
        datasets["valid"],
        batch_size=batch_size,
        shuffle=False,          # TrajSlicerDataset is already shuffled deterministically
        num_workers=4,
        pin_memory=True,
    )


def configure_probes(model, K, gamma, scales):
    """Turn on the two diagnostics on an already-loaded model.

    load_model() instantiates train_cfg.model, whose yaml block carries none of the
    straighten/rollout kwargs (train.py passes those explicitly), so a freshly loaded model
    has rollout_steps=1 and straightening off. Both losses read these attributes at CALL
    time, so setting them here is enough -- no re-instantiation, and the loaded weights are
    untouched.
    """
    model.rollout_steps = int(K)
    model.rollout_gamma = float(gamma)
    model.rollout_batch_frac = 1.0
    model.rollout_checkpoint = False      # no grad -> nothing to checkpoint
    model.straighten_scales = [int(s) for s in scales]
    model.straighten_scale_weights = [1.0] * len(scales)   # raw curvature, unweighted
    model.curvature_mode = "aggcos"
    model.stop_grad = True


@torch.no_grad()
def measure(run_dir, frames, K, gamma, scales, max_batches, device):
    run_dir = Path(run_dir).resolve()
    cfg_path = run_dir / "hydra.yaml"
    ckpt_path = run_dir / "checkpoints" / "model_latest.pth"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        print(f"!!! SKIP {run_dir.name}: missing hydra.yaml or checkpoints/model_latest.pth")
        return None

    train_cfg = OmegaConf.load(cfg_path)
    seed(0)   # fixed: the val split and its slice order must be identical for every run

    nh = train_cfg.num_hist
    if K is None:
        K = frames - nh
    if K < 1:
        raise ValueError(f"--frames {frames} leaves no rollout steps for num_hist={nh}")
    if frames < nh + K:
        raise ValueError(f"--frames {frames} too small for num_hist={nh} + K={K}")
    for s in scales:
        if frames < 2 * s + 1:
            raise ValueError(f"scale {s} needs >= {2*s+1} frames, got --frames {frames}")

    loader = build_val_loader(train_cfg, frames, train_cfg.training.batch_size)
    model = load_model(ckpt_path, train_cfg, train_cfg.num_action_repeat, device=device)
    configure_probes(model, K, gamma, scales)
    model.eval()

    sums, n = {}, 0
    for i, (obs, act, _state) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs = {k: v.to(device) for k, v in obs.items()}
        act = act.to(device)
        z = model.encode(obs, act)

        _, logs = model.rollout_consistency_loss(z, act)
        model.total_curvature(model.visual_only(z), mode="aggcos")
        logs.update(model._last_scale_curvatures)

        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1

    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["_batches"] = n
    out["_frames"] = frames
    out["_num_hist"] = int(nh)
    out["_run"] = run_dir.name
    del model
    torch.cuda.empty_cache()
    return out


def report(results, K, scales):
    err_keys = [f"rollout_err_k{k}" for k in range(1, K + 1)]
    curv_keys = [f"curv_s{s}" for s in scales]

    print("\n" + "=" * 78)
    print(f"ROLLOUT ERROR vs k  (validation, real actions, fixed window)")
    print("=" * 78)
    head = "run".ljust(46) + "".join(f"{k:>11}" for k in err_keys)
    print(head)
    print("-" * len(head))
    for r in results:
        short = r["_run"][:44]
        print(short.ljust(46) + "".join(f"{r.get(k, float('nan')):>11.6f}" for k in err_keys))

    print("\nsame, normalised to k=1 (this is the shape that matters):")
    print("run".ljust(46) + "".join(f"{k:>11}" for k in err_keys))
    for r in results:
        base = r.get("rollout_err_k1", float("nan"))
        print(r["_run"][:44].ljust(46)
              + "".join(f"{r.get(k, float('nan')) / base:>11.3f}" for k in err_keys))
    print("\nReference shapes for K=4: flat 1/1/1/1 = no accumulation;")
    print("linear 1/2/3/4 = independent errors, unit gain; 1/4/9/16 = aligned errors, unit gain.")
    print("Anything steeper than linear indicates amplification.")

    if curv_keys:
        print("\n" + "=" * 78)
        print("CURVATURE (aggcos, raw/unweighted) -- only meaningful for aggcos-trained runs")
        print("=" * 78)
        print("run".ljust(46) + "".join(f"{k:>11}" for k in curv_keys))
        for r in results:
            print(r["_run"][:44].ljust(46)
                  + "".join(f"{r.get(k, float('nan')):>11.6f}" for k in curv_keys))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run basenames under --base")
    ap.add_argument("--base", default=os.path.join(os.getcwd(), "checkpoints", "test"),
                    help="folder containing the run dirs (default ./checkpoints/test)")
    ap.add_argument("--runs-abs", dest="runs_abs", nargs="*", default=[],
                    help="absolute run dir paths (use instead of --base for mixed roots)")
    ap.add_argument("--frames", type=int, default=7, help="measurement window (default 7)")
    ap.add_argument("--K", type=int, default=None, help="rollout depth (default frames-num_hist)")
    ap.add_argument("--gamma", type=float, default=0.9, help="unused for the per-k table")
    ap.add_argument("--scales", type=int, nargs="*", default=[1], help="curvature scales")
    ap.add_argument("--batches", type=int, default=0, help="cap val batches (0 = all)")
    ap.add_argument("--out", default="results/rollout_diagnostic.json")
    args = ap.parse_args()

    dirs = list(args.runs_abs) + [os.path.join(args.base, r) for r in args.runs]
    if not dirs:
        ap.error("give at least one run (positional under --base, or --runs-abs paths)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  frames={args.frames}  scales={args.scales}  "
          f"batches={args.batches or 'all'}", flush=True)

    results = []
    for d in dirs:
        print(f"\n>>> {d}", flush=True)
        r = measure(d, args.frames, args.K, args.gamma, args.scales, args.batches, device)
        if r:
            results.append(r)

    if not results:
        print("no runs measured")
        return
    K = args.K if args.K is not None else args.frames - results[0]["_num_hist"]
    report(results, K, args.scales)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
