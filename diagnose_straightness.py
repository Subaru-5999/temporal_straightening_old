#!/usr/bin/env python3
"""
diagnose_straightness.py -- forward-only GO/NO-GO gate for the arc-length (constant-speed)
term, measured on checkpoints that ALREADY exist. No training. Runs in a few minutes.

WHY THIS EXISTS
    Every previous extension was launched on a plausible story and cost 20-40 h before the
    story could be checked. Two of them (multi-scale, rollout) were then falsified by a
    measurement that could have been taken FIRST. This script takes that measurement first.

THE QUESTION IT ANSWERS
    The appendix proposition that licenses the cosine proxy assumes ||v_t|| = c for all t, and
    uses the assumption exactly once, to rewrite ||v_{t+1} - v_t||^2 as 2c^2(1 - C_t). Without
    the assumption the same algebra gives the EXACT identity

        ||v_{t+1} - v_t||^2 / (||v_t|| ||v_{t+1}||)  =  (r_t + 1/r_t - 2)  +  2 (1 - C_t)
                                                        \___ speed half __/   \_ direction _/
        r_t = ||v_{t+1}|| / ||v_t||

    The deployed loss regularizes ONLY the direction half (cosine is scale-invariant, so it
    never sees ||v_t||). The bound that motivates the whole method is on

        ||(A - I) v_hat_t||  <=  sqrt( (r_t - 1)^2 + 2 r_t (1 - C_t) )  +  sigma_max(B) Da / ||v_t||

    (put r_t = 1 to recover the paper's Eq. app_dir_point_const verbatim). So:

        eps_now          = current directional residual, both halves as they are
        eps_if_straight  = what survives if the cosine term reached its OPTIMUM, C_t -> 1.
                           This is |r_t - 1| -- pure speed mismatch, invisible to the deployed
                           loss, and eps is amplified as ((1+eps)/(1-eps))^(2(H-1)).
        eps_if_const_speed = what survives if speeds were already equal (r_t -> 1)

    DECISION RULE (fix it before looking at the numbers):
      GO      if speed_share >= 0.15, i.e. the speed half is at least ~15% of the total. Then
              straightening is leaving a real, un-attacked residual and the new term has
              somewhere to go.
      NO-GO   if speed_share < 0.15. Then r_t is already ~1, the paper's assumption holds in
              practice, the term is a no-op dressed up as a fix, and we do NOT spend 20 h on it.

    Also reported: whether eps < 1 at all. If eps >= 1 the appendix's epsilon-specialization
    (which needs ||A - I|| < 1) is VACUOUS at these curvature levels. That does not invalidate
    the identity above -- it is exact -- but it does mean the exponential conditioning bound is
    design guidance, not a proof of anything about success rate. Report it either way.

CAVEAT (same as diagnose_rollout.py)
    mode 'aggcos' measures the pooled feature from encoder.agg. In a run trained WITHOUT
    straightening no gradient ever reaches agg, so it sits at random init and every number here
    is meaningless. ONLY point this at aggcos-trained checkpoints.

USAGE
    python diagnose_straightness.py --runs-abs \
        $PWD/checkpoints_baseline_matched/test/pusht_..._ms1-4_lam0.1-0_ep3
    options: --frames N (window, default 7)  --scales ... (default 1)  --batches N (0 = all)
"""
import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# Reuse the checkpoint/dataloader machinery and the MIG env defaults verbatim.
from diagnose_rollout import load_model, build_val_loader
from utils import seed


def agg_latents(model, z):
    """The exact representation the straightening loss is computed on (mode 'aggcos')."""
    feats = model.visual_only(z)
    b, t, p, d = feats.shape
    tokens = feats.reshape(b * t, p, d)
    return model.encoder.agg(tokens).reshape(b, t, -1).float()


def scale_stats(zagg, s, step_thresh=1e-6, eps=1e-6):
    """Per-scale direction/speed decomposition of the paper's own directional residual."""
    T = zagg.shape[1]
    if T < 2 * s + 1:
        return None
    va = zagg[:, s:] - zagg[:, :-s]
    v1, v2 = va[:, :-s], va[:, s:]
    n1, n2 = v1.norm(dim=-1), v2.norm(dim=-1)
    mask = (n1 > step_thresh) & (n2 > step_thresh)
    if mask.sum() == 0:
        return None

    cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)[mask]
    r = (n2.clamp_min(eps) / n1.clamp_min(eps))[mask]
    one_minus_c = 1.0 - cos

    direction_half = 2.0 * one_minus_c            # 2(1 - C_t)
    speed_half = r + 1.0 / r - 2.0                # (sqrt r - 1/sqrt r)^2

    # the exact bound's first term, and the two counterfactuals
    eps_now = torch.sqrt(((r - 1.0) ** 2 + 2.0 * r * one_minus_c).clamp_min(0))
    eps_if_straight = (r - 1.0).abs()             # cosine term at its optimum
    eps_if_const_speed = torch.sqrt((2.0 * one_minus_c).clamp_min(0))

    return {
        f"curv_s{s}": one_minus_c.mean(),
        f"speed_s{s}": speed_half.mean(),
        f"direction_half_s{s}": direction_half.mean(),
        f"eps_now_s{s}": eps_now.mean(),
        f"eps_if_straight_s{s}": eps_if_straight.mean(),
        f"eps_if_const_speed_s{s}": eps_if_const_speed.mean(),
        f"logr_rms_s{s}": r.log().pow(2).mean().sqrt(),
        f"r_mean_s{s}": r.mean(),
        f"speed_cv_s{s}": n1[mask].std() / n1[mask].mean().clamp_min(eps),
        f"_n_s{s}": mask.sum().float(),
    }


@torch.no_grad()
def measure(run_dir, frames, scales, max_batches, device):
    run_dir = Path(run_dir).resolve()
    cfg_path = run_dir / "hydra.yaml"
    ckpt_path = run_dir / "checkpoints" / "model_latest.pth"
    if not cfg_path.is_file() or not ckpt_path.is_file():
        print(f"!!! SKIP {run_dir.name}: missing hydra.yaml or checkpoints/model_latest.pth")
        return None

    train_cfg = OmegaConf.load(cfg_path)
    seed(0)   # identical val split / slice order for every run
    for s in scales:
        if frames < 2 * s + 1:
            raise ValueError(f"scale {s} needs >= {2*s+1} frames, got --frames {frames}")

    trained_straight = bool(train_cfg.training.get("straighten", False))
    loader = build_val_loader(train_cfg, frames, train_cfg.training.batch_size)
    model = load_model(ckpt_path, train_cfg, train_cfg.num_action_repeat, device=device)
    model.eval()

    sums, n = {}, 0
    for i, (obs, act, _state) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs = {k: v.to(device) for k, v in obs.items()}
        act = act.to(device)
        zagg = agg_latents(model, model.encode(obs, act))
        for s in scales:
            st = scale_stats(zagg, s)
            if st is None:
                continue
            for k, v in st.items():
                sums[k] = sums.get(k, 0.0) + float(v)
        n += 1

    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["_batches"] = n
    out["_frames"] = frames
    out["_run"] = run_dir.name
    out["_straighten"] = str(train_cfg.training.get("straighten", False))
    out["_agg_is_trained"] = trained_straight
    del model
    torch.cuda.empty_cache()
    return out


def report(results, scales, threshold=0.15):
    print("\n" + "=" * 96)
    print("DIRECTION vs SPEED  --  the two exact halves of the paper's directional residual")
    print("=" * 96)
    print("identity:  ||v_{t+1}-v_t||^2 / (||v_t|| ||v_{t+1}||)  =  (r+1/r-2)  +  2(1-C)")
    print("           the deployed loss regularizes the SECOND half only.\n")
    for s in scales:
        print(f"--- scale s={s} " + "-" * 78)
        head = ("run".ljust(38) + f"{'1-C':>9}{'2(1-C)':>9}{'r+1/r-2':>10}"
                f"{'speed_sh':>10}{'rms logr':>10}{'speed_cv':>10}")
        print(head)
        for r in results:
            d = r.get(f"direction_half_s{s}")
            sp = r.get(f"speed_s{s}")
            if d is None or sp is None:
                continue
            share = sp / (sp + d) if (sp + d) > 0 else float("nan")
            print(r["_run"][:36].ljust(38)
                  + f"{r[f'curv_s{s}']:>9.4f}{d:>9.4f}{sp:>10.4f}"
                  + f"{share:>10.3f}{r[f'logr_rms_s{s}']:>10.4f}{r[f'speed_cv_s{s}']:>10.4f}")

        print("\nresidual eps = ||(A-I) v_hat|| bound term, and what each half would leave:")
        print("run".ljust(38) + f"{'eps_now':>10}{'if C->1':>10}{'if r->1':>10}{'eps<1?':>9}")
        for r in results:
            if f"eps_now_s{s}" not in r:
                continue
            e = r[f"eps_now_s{s}"]
            print(r["_run"][:36].ljust(38)
                  + f"{e:>10.4f}{r[f'eps_if_straight_s{s}']:>10.4f}"
                  + f"{r[f'eps_if_const_speed_s{s}']:>10.4f}{('yes' if e < 1 else 'NO'):>9}")
        print()

    print("=" * 96)
    print(f"VERDICT (pre-registered rule: GO iff speed_share >= {threshold} at s=1)")
    print("=" * 96)
    for r in results:
        if not r.get("_agg_is_trained", False):
            print(f"{r['_run'][:60]}: agg head UNTRAINED (straighten={r['_straighten']}) "
                  f"-> numbers meaningless, ignore")
            continue
        d, sp = r.get("direction_half_s1"), r.get("speed_s1")
        if d is None or sp is None:
            print(f"{r['_run'][:60]}: s=1 not measurable")
            continue
        share = sp / (sp + d) if (sp + d) > 0 else 0.0
        verdict = "GO" if share >= threshold else "NO-GO"
        print(f"{r['_run'][:60]}")
        print(f"    speed_share = {share:.3f}  ->  {verdict}")
        print(f"    if the cosine term were PERFECT, eps would still be "
              f"{r['eps_if_straight_s1']:.4f} (pure speed mismatch)")
        if r["eps_now_s1"] >= 1.0:
            print("    NOTE eps >= 1: the appendix's exponential specialization needs "
                  "||A-I|| < 1, so it is VACUOUS here.")
            print("         The decomposition identity is still exact; treat the bound as "
                  "design guidance, not proof.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run basenames under --base")
    ap.add_argument("--base", default=os.path.join(os.getcwd(), "checkpoints", "test"))
    ap.add_argument("--runs-abs", dest="runs_abs", nargs="*", default=[])
    ap.add_argument("--frames", type=int, default=7, help="measurement window (default 7)")
    ap.add_argument("--scales", type=int, nargs="*", default=[1])
    ap.add_argument("--batches", type=int, default=0, help="cap val batches (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="pre-registered GO threshold on speed_share (default 0.15)")
    ap.add_argument("--out", default="results/straightness_diagnostic.json")
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
        r = measure(d, args.frames, args.scales, args.batches, device)
        if r:
            results.append(r)

    if not results:
        print("no runs measured")
        return
    report(results, args.scales, args.threshold)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
