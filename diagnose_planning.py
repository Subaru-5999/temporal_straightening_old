#!/usr/bin/env python3
"""
diagnose_planning.py -- forward-only GO/NO-GO gate for the LATENT GEODESIC CORRIDOR planning
objective. No training, no simulator, no success labels. Runs in minutes on existing checkpoints.

WHAT IS BEING TESTED
    Open-loop planning minimises the TERMINAL latent cost only, so the interior of the plan is
    unconstrained and 100 Adam steps can walk into action sequences the model wrongly believes
    reach the goal. The proposed fix is a trust region in the paper's own geometry: charge the
    predicted latent path for bowing SIDEWAYS off the straight segment z_0 -> z_g, leaving
    progress ALONG the segment free.

    That fix is only worth anything if the premise holds: REAL behaviour must actually stay near
    the straight segment. This script measures that, on real validation trajectories, with the
    exact 25-env-step (H=5 model step) window the Table-1 protocol plans over.

WHAT IT MEASURES  (all quantities are dimensionless: perpendicular deviation divided by the
segment length ||z_g - z_0||, so they are comparable across checkpoints and scales)
    real_dev_*      : deviation of REAL interior latents from the segment joining a real
                      start to a real 25-step-later goal. This is the corridor real behaviour
                      occupies. Its high quantile is the dead zone `corridor_rho` -- MEASURED,
                      not tuned against success.
    mismatch_dev_*  : the same statistic with the goal taken from a DIFFERENT trajectory in the
                      batch. This is the deviation scale of a path that is NOT heading to the
                      goal, i.e. the control condition. It is the reference the real corridor has
                      to beat for the term to carry information.
    separation      : mismatch_dev_p50 / real_dev_p90. How much room the corridor has to
                      discriminate a genuine approach from a path that merely ends up nearby.

    DECISION RULE (pre-registered, before looking at numbers):
      GO    if separation >= 1.5 AND real_dev_p90 <= 0.5. Real approaches then occupy a
            corridor that is both narrow in absolute terms and clearly narrower than the
            control, so penalising sideways deviation carries real information.
      NO-GO otherwise. Real latent paths wander as much as unrelated ones, the straight-segment
            prior is false in this latent space, and the objective term would just add bias.

    SECONDARY, MECHANISM TEST: run this on the straightened (aggcos) checkpoint AND on the
    no-straightening checkpoint. The corridor prior is a consequence of straightening, so
    real_dev should be SMALLER and separation LARGER for the straightened model. If it is not,
    the story is wrong even if the gate passes, and that must be reported.

CAVEAT
    Unlike the aggcos curvature diagnostics, everything here is computed on the PLANNING
    representation (`z_obs["visual"]`, the projected patch latents the objective actually uses),
    NOT on the encoder's agg head. So it is meaningful for straightened and non-straightened
    checkpoints alike, and directly comparable between them.

USAGE
    python diagnose_planning.py --frames 6 --runs-abs \
        $PWD/checkpoints_baseline_matched/test/pusht_..._ms1-4_lam0.1-0_ep3 \
        $PWD/checkpoints/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
    options: --frames N (start + H, default 6 = the eval's 25 env steps)  --batches N (0 = all)
"""
import os
import json
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Reuse the checkpoint/dataloader machinery and the MIG env defaults verbatim.
from diagnose_rollout import load_model, build_val_loader
from utils import seed


def corridor_dev(z0, zk, zg, eps=1e-8):
    """Normalised perpendicular deviation of interior latents from the segment z0 -> zg.

    Mirrors planning/objectives.py::corridor_penalty exactly, so the gate measures the very
    quantity the objective would charge. z0/zg: (b,1,F); zk: (b,K,F). Returns (b,K).
    """
    u = zg - z0
    u_sq = u.pow(2).sum(-1, keepdim=True).clamp_min(eps)
    d = zk - z0
    perp = d - ((d * u).sum(-1, keepdim=True) / u_sq) * u
    return (perp.pow(2).sum(-1) / u_sq.squeeze(-1)).clamp_min(0).sqrt()


def quantiles(x):
    q = torch.tensor([0.5, 0.75, 0.9, 0.95], device=x.device)
    v = torch.quantile(x.float(), q)
    return {
        "mean": x.mean().item(),
        "p50": v[0].item(),
        "p75": v[1].item(),
        "p90": v[2].item(),
        "p95": v[3].item(),
        "max": x.max().item(),
    }


@torch.no_grad()
def measure(run_dir, frames, max_batches, device):
    run_dir = Path(run_dir).resolve()
    cfg_path = run_dir / "hydra.yaml"
    ckpt_path = run_dir / "checkpoints" / "model_latest.pth"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        print(f"!!! SKIP {run_dir.name}: missing hydra.yaml or checkpoints/model_latest.pth")
        return None
    if frames < 3:
        raise ValueError("--frames must be >= 3 (start, >=1 interior, goal)")

    train_cfg = OmegaConf.load(cfg_path)
    seed(0)   # identical val split / slice order for every run
    loader = build_val_loader(train_cfg, frames, train_cfg.training.batch_size)
    model = load_model(ckpt_path, train_cfg, train_cfg.num_action_repeat, device=device)
    model.eval()

    real, mismatch = [], []
    for i, (obs, act, _state) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs = {k: v.to(device) for k, v in obs.items()}
        act = act.to(device)
        z_obs, _ = model.separate_emb(model.encode(obs, act))
        v = z_obs["visual"]                       # (b, T, P, D) -- the PLANNING representation
        b, t = v.shape[0], v.shape[1]
        if b < 2:
            continue
        f = v.reshape(b, t, -1).float()
        z0, zk, zg = f[:, 0:1], f[:, 1 : t - 1], f[:, t - 1 : t]
        real.append(corridor_dev(z0, zk, zg).flatten())
        # control: same path, but the goal belongs to a DIFFERENT trajectory (roll the batch)
        mismatch.append(corridor_dev(z0, zk, zg.roll(1, dims=0)).flatten())

    if not real:
        print(f"!!! SKIP {run_dir.name}: no usable batches")
        return None

    real = torch.cat(real)
    mismatch = torch.cat(mismatch)
    out = {
        "_run": run_dir.name,
        "_frames": frames,
        "_n_samples": int(real.numel()),
        "_straighten": str(train_cfg.training.get("straighten", False)),
        "real": quantiles(real),
        "mismatch": quantiles(mismatch),
    }
    out["separation"] = out["mismatch"]["p50"] / max(out["real"]["p90"], 1e-8)
    del model
    torch.cuda.empty_cache()
    return out


def report(results, sep_thresh=1.5, dev_thresh=0.5):
    print("\n" + "=" * 100)
    print("LATENT GEODESIC CORRIDOR  --  how far real latent paths stray off the z_0 -> z_g line")
    print("=" * 100)
    print("all values are ||perp|| / ||z_g - z_0||, dimensionless. 'mismatch' = goal swapped to")
    print("another trajectory (the control: a path NOT heading to that goal).\n")
    head = "run".ljust(40) + f"{'which':>10}{'mean':>9}{'p50':>9}{'p75':>9}{'p90':>9}{'p95':>9}"
    print(head)
    print("-" * len(head))
    for r in results:
        for which in ("real", "mismatch"):
            d = r[which]
            print(r["_run"][:38].ljust(40) + f"{which:>10}"
                  + f"{d['mean']:>9.4f}{d['p50']:>9.4f}{d['p75']:>9.4f}"
                  + f"{d['p90']:>9.4f}{d['p95']:>9.4f}")
        print()

    print("=" * 100)
    print(f"VERDICT (pre-registered: GO iff separation >= {sep_thresh} and real p90 <= {dev_thresh})")
    print("=" * 100)
    for r in results:
        sep = r["separation"]
        p90 = r["real"]["p90"]
        verdict = "GO" if (sep >= sep_thresh and p90 <= dev_thresh) else "NO-GO"
        print(f"{r['_run'][:70]}")
        print(f"    straighten={r['_straighten']}  real_p90={p90:.4f}  "
              f"mismatch_p50={r['mismatch']['p50']:.4f}  separation={sep:.2f}  ->  {verdict}")
        print(f"    suggested corridor_rho = {p90:.3f}   (the corridor real behaviour occupies; "
              f"do NOT tune this against success)")
    if len(results) > 1:
        print("\nMECHANISM CHECK: the corridor prior is a consequence of straightening, so the")
        print("straightened run should show the SMALLER real deviation and the LARGER separation.")
        print("If the ordering is reversed, report it -- the mechanism story is then wrong even")
        print("if the gate passes.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run basenames under --base")
    ap.add_argument("--base", default=os.path.join(os.getcwd(), "checkpoints", "test"))
    ap.add_argument("--runs-abs", dest="runs_abs", nargs="*", default=[])
    ap.add_argument("--frames", type=int, default=6,
                    help="start + H frames; 6 matches the eval's goal_H=25 at frameskip 5")
    ap.add_argument("--batches", type=int, default=0, help="cap val batches (0 = all)")
    ap.add_argument("--out", default="results/planning_diagnostic.json")
    args = ap.parse_args()

    dirs = list(args.runs_abs) + [os.path.join(args.base, r) for r in args.runs]
    if not dirs:
        ap.error("give at least one run (positional under --base, or --runs-abs paths)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  frames={args.frames}  batches={args.batches or 'all'}", flush=True)

    results = []
    for d in dirs:
        print(f"\n>>> {d}", flush=True)
        r = measure(d, args.frames, args.batches, device)
        if r:
            results.append(r)

    if not results:
        print("no runs measured")
        return
    report(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
