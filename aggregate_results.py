#!/usr/bin/env python3
"""
aggregate_results.py

Correct Table-1 aggregator for temporal-straightening planning evals.

Each plan.py run APPENDS one line to logs.json, and the Hydra output path does
NOT include the seed. So for a given (run, planner, objective-mode) the 3 data
seeds (100/200/300) land as 3 lines in ONE logs.json = exactly one Table-1 cell.

This script reads EVERY line of EVERY logs.json (unlike collect_results.py, which
kept only the last line), groups by the cell folder, and reports mean +/- std over
seeds. It writes:
  - results_per_seed.csv : one row per (cell, seed-line)
  - results_cells.csv    : one row per cell with mean/std/n

Usage:
    python aggregate_results.py                 # scans ./plan_outputs_*
    python aggregate_results.py --root /path
"""
import os
import re
import csv
import json
import glob
import argparse
import statistics
from collections import defaultdict


def find_logs(root):
    return sorted(set(glob.glob(os.path.join(root, "plan_outputs_*", "**", "logs.json"), recursive=True)))


def read_all_success_rates(logs_path):
    """Return the list of final_eval/success_rate values (one per appended line)."""
    vals = []
    try:
        with open(logs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k, v in d.items():
                    # Only the final result per seed (planners also log per-iteration
                    # success_rate; matching substring would average intermediates).
                    if k == "final_eval/success_rate":
                        vals.append(float(v))
    except OSError:
        pass
    return vals


def parse_meta(logs_path):
    full = logs_path.replace("\\", "/")
    parts = full.split("/")
    top = next((p for p in parts if p.startswith("plan_outputs_")), "")
    planner = top.replace("plan_outputs_", "") or "unknown"      # gd | gd_mpc | cem
    setting = "open-loop" if planner in ("gd", "cem") else "MPC"

    env = next((e for e in ("umaze", "medium", "wall", "pusht") if e in full), "?")
    if "aggcos" in full or "aggmlpcos" in full:
        curv = "curv(agg)"
    elif re.search(r"(?<!agg)cos1e", full):
        curv = "curv"
    elif "False" in full:
        curv = "none"
    else:
        curv = "?"
    m = re.search(r"obj([a-z]+)_init", full)
    mode = m.group(1) if m else "?"
    # cell key = the folder that holds this logs.json
    cell = os.path.dirname(full)
    return dict(planner=planner, setting=setting, env=env, curv=curv, mode=mode, cell=cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    logs = find_logs(args.root)
    if not logs:
        print(f"No logs.json found under {os.path.abspath(args.root)}/plan_outputs_*")
        print("Run a plan.py evaluation first (see the command list).")
        return

    per_seed_rows = []
    cell_rows = []
    for lp in logs:
        vals = read_all_success_rates(lp)
        if not vals:
            continue
        meta = parse_meta(lp)
        for i, v in enumerate(vals):
            per_seed_rows.append(dict(env=meta["env"], curv=meta["curv"], setting=meta["setting"],
                                      planner=meta["planner"], mode=meta["mode"], line=i,
                                      success_pct=round(100.0 * v, 2), logs=lp))
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        cell_rows.append(dict(env=meta["env"], curv=meta["curv"], setting=meta["setting"],
                              planner=meta["planner"], mode=meta["mode"], n=len(vals),
                              mean_pct=round(100.0 * mean, 2), std_pct=round(100.0 * std, 2),
                              seeds_pct=[round(100.0 * x, 2) for x in vals], logs=lp))

    with open("results_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "curv", "setting", "planner", "mode", "line", "success_pct", "logs"])
        w.writeheader()
        w.writerows(sorted(per_seed_rows, key=lambda r: (r["env"], r["curv"], r["setting"], r["line"])))

    with open("results_cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "curv", "setting", "planner", "mode", "n", "mean_pct", "std_pct", "seeds_pct", "logs"])
        w.writeheader()
        for r in sorted(cell_rows, key=lambda r: (r["env"], r["curv"], r["setting"])):
            r = dict(r); r["seeds_pct"] = " ".join(str(x) for x in r["seeds_pct"])
            w.writerow(r)

    print("\n| Env | Curv | Setting | n | Success % (mean +/- std) | per-seed |")
    print("|-----|------|---------|---|--------------------------|----------|")
    for r in sorted(cell_rows, key=lambda r: (r["env"], r["curv"], r["setting"])):
        seeds = ", ".join(f"{x:.0f}" for x in r["seeds_pct"])
        flag = "" if r["n"] == 3 else f"  (!! n={r['n']}, expected 3)"
        print(f"| {r['env']} | {r['curv']} | {r['setting']} | {r['n']} | {r['mean_pct']:.2f} +/- {r['std_pct']:.2f} | {seeds}{flag} |")

    print(f"\nParsed {len(cell_rows)} cell(s) from {len(logs)} logs.json file(s).")
    print("Wrote results_per_seed.csv and results_cells.csv")


if __name__ == "__main__":
    main()
