#!/usr/bin/env python3
"""
summarize_run.py  --  RUN-SCOPED Table-1 summarizer (no result mixing).

The paper's Table 1 uses the GD planner, 50 test samples, mean +/- std over 3 data
seeds (100/200/300), open-loop and MPC. Because plan.py APPENDS one line to
logs.json per seed and the output path has no seed in it, one run's
(open-loop | MPC) logs.json holds exactly its 3 seed lines.

This tool reads ONLY the given run's logs.json (basename-scoped globs), so results
for different runs / planners can never be mixed. It stores results/<run>.json and
rebuilds a master table (results/table1_reproduction.{md,csv}) keyed by run name.

Usage:
    python summarize_run.py <run_basename>   # summarize one run, store, print, refresh master table
    python summarize_run.py --all            # only rebuild + print the master table from results/*.json
"""
import os
import sys
import glob
import json
import argparse
import statistics

RESULTS_DIR = "results"

# Paper Table 1 (GD planner) targets for the exact 5 cells we reproduce: (mean, std) %.
PAPER = {
 "umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05":
    {"label": "UMaze | DINOv2 patch 14x14x384 | no-straighten", "ol": (35.33, 4.11), "mpc": (80.67, 6.18)},
 "umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06":
    {"label": "UMaze | +proj 14x14x8 | no-straighten",          "ol": (44.00, 7.12), "mpc": (81.33, 6.80)},
 "umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05":
    {"label": "UMaze | +proj 14x14x8 | straighten",             "ol": (94.00, 1.63), "mpc": (100.00, 0.00)},
 "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06":
    {"label": "PushT | +proj 14x14x8 | no-straighten",          "ol": (70.00, 1.63), "mpc": (78.67, 0.94)},
 "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05":
    {"label": "PushT | +proj 14x14x8 | straighten",             "ol": (77.33, 6.18), "mpc": (85.33, 4.99)},
}

# tolerance (percentage points) added to the paper's std when judging "within band"
BAND_TOL = 3.0


def read_success_rates(root, name):
    """All final_eval/success_rate (%) values from THIS run's logs.json under `root`."""
    vals = []
    pattern = os.path.join(root, f"{name}_*", "**", "logs.json")
    for f in sorted(glob.glob(pattern, recursive=True)):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in d.items():
                # ONLY the final planning result per seed. Planners also log a
                # success_rate every iteration (MPC replans ~20x/seed), so matching
                # "success_rate in k" would average intermediate climbing values and
                # inflate n. perform_planning() writes exactly "final_eval/success_rate".
                if k == "final_eval/success_rate":
                    vals.append(round(100.0 * float(v), 2))
    return vals


def stats(vals):
    if not vals:
        return (None, None, 0)
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return (round(m, 2), round(s, 2), len(vals))


def fmt_cell(d):
    if d["mean"] is None:
        return "n/a"
    tag = "" if d["n"] == 3 else f" (!!n={d['n']})"
    return f"{d['mean']:.2f}+/-{d['std']:.2f}{tag}"


def within_band(d):
    if d["mean"] is None or not d["paper"]:
        return "?"
    pm, ps = d["paper"]
    return "OK" if abs(d["mean"] - pm) <= ps + BAND_TOL else f"OFF {d['mean']-pm:+.2f}"


def summarize_one(name):
    ol = read_success_rates("plan_outputs_gd", name)      # open-loop GD
    mpc = read_success_rates("plan_outputs_gd_mpc", name)  # MPC (GD subplanner)
    ol_m, ol_s, ol_n = stats(ol)
    mpc_m, mpc_s, mpc_n = stats(mpc)
    paper = PAPER.get(name, {"label": name, "ol": None, "mpc": None})
    if ol_n == 0 and mpc_n == 0:
        # No logs.json for this run yet -- don't clobber any existing results/<name>.json
        print(f"  (no logs found for {name}; skipping)")
        return None
    rec = {
        "run": name, "label": paper["label"],
        "open_loop": {"seeds": ol, "mean": ol_m, "std": ol_s, "n": ol_n, "paper": paper["ol"]},
        "mpc":       {"seeds": mpc, "mean": mpc_m, "std": mpc_s, "n": mpc_n, "paper": paper["mpc"]},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print_block(rec)
    return rec


def print_block(rec):
    print("\n" + "=" * 76)
    print(f"RESULT  {rec['run']}")
    print(f"        {rec['label']}")
    print("=" * 76)
    for key, tag in [("open_loop", "Open-loop"), ("mpc", "MPC")]:
        d = rec[key]
        paper = f"{d['paper'][0]:.2f}+/-{d['paper'][1]:.2f}" if d["paper"] else "n/a"
        seeds = ", ".join(f"{x:.0f}" for x in d["seeds"]) if d["seeds"] else "none"
        print(f"  {tag:10s} ours {fmt_cell(d):24s} paper {paper:14s} [{within_band(d)}]   seeds: {seeds}")
    for key in ("open_loop", "mpc"):
        if rec[key]["n"] != 3:
            print(f"  WARNING: {key} has {rec[key]['n']} seed result(s), expected 3 -- rerun the missing seed(s).")


def rebuild_master():
    recs = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        try:
            recs.append(json.load(open(f)))
        except Exception:
            pass
    order = {n: i for i, n in enumerate(PAPER)}
    recs.sort(key=lambda r: order.get(r["run"], 999))

    md = ["# Table 1 reproduction (GD planner, 3 data seeds 100/200/300, 50 samples)",
          "",
          "| Run | Setting | Ours Open-loop | Paper OL | Ours MPC | Paper MPC |",
          "|-----|---------|----------------|----------|----------|-----------|"]
    csv = ["run,setting,ol_mean,ol_std,ol_n,ol_seeds,paper_ol_mean,paper_ol_std,"
           "mpc_mean,mpc_std,mpc_n,mpc_seeds,paper_mpc_mean,paper_mpc_std"]
    for r in recs:
        ol, mpc = r["open_loop"], r["mpc"]
        pol = f"{ol['paper'][0]:.2f}+/-{ol['paper'][1]:.2f}" if ol["paper"] else ""
        pmpc = f"{mpc['paper'][0]:.2f}+/-{mpc['paper'][1]:.2f}" if mpc["paper"] else ""
        md.append(f"| {r['run']} | {r['label']} | {fmt_cell(ol)} | {pol} | {fmt_cell(mpc)} | {pmpc} |")
        csv.append(",".join(str(x) for x in [
            r["run"], r["label"].replace(",", " "),
            ol["mean"], ol["std"], ol["n"], " ".join(map(str, ol["seeds"])),
            ol["paper"][0] if ol["paper"] else "", ol["paper"][1] if ol["paper"] else "",
            mpc["mean"], mpc["std"], mpc["n"], " ".join(map(str, mpc["seeds"])),
            mpc["paper"][0] if mpc["paper"] else "", mpc["paper"][1] if mpc["paper"] else "",
        ]))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "table1_reproduction.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS_DIR, "table1_reproduction.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")
    print("\n" + "\n".join(md))
    print(f"\n(master table -> {RESULTS_DIR}/table1_reproduction.md and .csv; {len(recs)} run(s) recorded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="run basename to summarize (omit with --all)")
    ap.add_argument("--all", action="store_true", help="rebuild + print master table only")
    args = ap.parse_args()
    if args.run and not args.all:
        summarize_one(args.run)
    elif args.all:
        # Re-scan every known run's logs.json from scratch (recomputes results/<name>.json
        # with the current parser -- use this to re-derive results without re-running evals).
        for name in PAPER:
            summarize_one(name)
    rebuild_master()


if __name__ == "__main__":
    main()
