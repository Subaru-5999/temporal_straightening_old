#!/usr/bin/env python3
"""
collect_results.py

Scrape every planning run's logs.json under plan_outputs_*/ , pull the final
success_rate, group by (config, planner) across seeds, and print a Table-1-style
markdown table (+ write results.csv).

Usage:
    python collect_results.py                 # scans ./plan_outputs_*
    python collect_results.py --root /path    # scan elsewhere

NOTE: The folder-name pattern is inferred from the Hydra run.dir templates in
conf/plan_*.yaml. If your first real logs.json lives somewhere slightly
different, tweak find_logs()/parse_meta() -- the parsing is intentionally simple.
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
    # plan_outputs_gd/, plan_outputs_gd_mpc/, plan_outputs_cem/ ...
    pats = [os.path.join(root, "plan_outputs_*", "**", "logs.json")]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    return sorted(set(files))


def read_success_rate(logs_path):
    """logs.json holds one JSON object per line; take the last final_eval success_rate."""
    val = None
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
                    if "success_rate" in k:
                        val = float(v)
    except OSError:
        return None
    return val


def parse_meta(logs_path):
    """Infer planner / env / straighten / seed from the path."""
    parts = logs_path.replace("\\", "/").split("/")
    top = next((p for p in parts if p.startswith("plan_outputs_")), "")
    planner = top.replace("plan_outputs_", "") or "unknown"   # gd | gd_mpc | cem

    full = logs_path.replace("\\", "/")
    # env from the save_name embedded in model_name
    env = next((e for e in ("umaze", "medium", "wall", "pusht") if e in full), "?")
    # straightening flag
    if "aggcos" in full:
        curv = "agg-curv"
    elif "cos1e" in full:
        curv = "curv"
    elif "False" in full:
        curv = "none"
    else:
        curv = "?"
    m = re.search(r"seed(\d+)", full)
    seed = m.group(1) if m else "?"
    return planner, env, curv, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    logs = find_logs(args.root)
    if not logs:
        print(f"No logs.json found under {os.path.abspath(args.root)}/plan_outputs_*")
        print("Run a plan.py evaluation first (see evaluate.sh).")
        return

    rows = []
    for lp in logs:
        sr = read_success_rate(lp)
        if sr is None:
            continue
        planner, env, curv, seed = parse_meta(lp)
        rows.append(dict(env=env, curv=curv, planner=planner, seed=seed,
                         success_pct=100.0 * sr, path=lp))

    # aggregate across seeds
    groups = defaultdict(list)
    for r in rows:
        groups[(r["env"], r["curv"], r["planner"])].append(r["success_pct"])

    # write raw csv
    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["env", "curv", "planner", "seed", "success_pct", "path"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["env"], r["curv"], r["planner"], r["seed"])):
            w.writerow(r)

    # markdown summary
    print("\n| Env | L_curv | Planner | n | Success % (mean +/- std) |")
    print("|-----|--------|---------|---|--------------------------|")
    for (env, curv, planner), vals in sorted(groups.items()):
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"| {env} | {curv} | {planner} | {len(vals)} | {mean:.2f} +/- {std:.2f} |")

    print(f"\nParsed {len(rows)} run(s) from {len(logs)} logs.json file(s). Raw rows -> results.csv")


if __name__ == "__main__":
    main()
