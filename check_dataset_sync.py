#!/usr/bin/env python3
"""
check_dataset_sync.py

Verify that the datasets on disk (DATASET_DIR) and the dataset-loading code are
EXACTLY in sync with what trained the checkpoints, before re-running evaluation.

It performs three independent checks and prints a PASS/FAIL report:

  [1] REPO STRUCTURE
      The dataset loader modules (datasets/*.py) and env configs
      (conf/env/*.yaml) that the training pipeline imports are present.

  [2] CONFIG DRIFT  ("in sync with the training")
      For every trained run (each run dir has a resolved hydra.yaml saved AT
      TRAIN TIME), compare the dataset block the model was actually trained with
      against the CURRENT repo conf/env/<name>.yaml. Any difference means the
      code drifted from what produced the checkpoint.

  [3] DATA ON DISK
      For each env actually used by the runs, resolve its data_path and verify
      the folder contains exactly the files the loader will try to read
      (states.pth, actions.pth, seq_lengths, obses/..., pusht train/val split,
      .mp4 vs .pth, etc.). With torch/pickle available it also checks that the
      episode count matches len(states)/len(seq_lengths).

Usage:
    export DATASET_DIR=/workspace/arun/data
    python check_dataset_sync.py                         # repo=., runs=./eval_runs
    python check_dataset_sync.py --runs /workspace/arun/eval_runs
    python check_dataset_sync.py --repo . --dataset-dir /workspace/arun/data --deep

Exit code is non-zero if any FAIL is found (handy in CI / a pre-eval gate).
"""
import os
import sys
import glob
import argparse

try:
    import yaml
except Exception:
    print("ERROR: pyyaml is required (pip install pyyaml).")
    sys.exit(2)

# torch/pickle are only needed for the optional deep count checks.
try:
    import torch
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False
import pickle

GREEN, RED, YEL, RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
def ok(m):   print(f"  {GREEN}[OK]{RST}   {m}")
def bad(m):  print(f"  {RED}[FAIL]{RST} {m}")
def warn(m): print(f"  {YEL}[WARN]{RST} {m}")

FAILS = []
def fail(m):
    FAILS.append(m); bad(m)

# ---------------------------------------------------------------------------
# Ground truth: what each dataset loader reads from its data_path.
# Derived directly from datasets/{point_maze,wall,pusht}_dset.py.
# 'layout=split' means the loader appends /train and /val to data_path.
# ---------------------------------------------------------------------------
EXPECTED = {
    "point_maze": {
        "subdir": "point_maze", "layout": "flat", "obs_ext": ".pth",
        "files": ["states.pth", "actions.pth", "seq_lengths.pth"],
        "count_from": ("states.pth", "torch"),
    },
    "point_maze_medium": {
        "subdir": "point_maze_medium", "layout": "flat", "obs_ext": ".pth",
        "files": ["states.pth", "actions.pth", "seq_lengths.pth"],
        "count_from": ("states.pth", "torch"),
    },
    "wall": {
        "subdir": "wall_single", "layout": "flat", "obs_ext": ".pth",
        "files": ["states.pth", "actions.pth", "door_locations.pth", "wall_locations.pth"],
        "count_from": ("states.pth", "torch"),
    },
    "pusht": {
        "subdir": "pusht_noise", "layout": "split", "obs_ext": ".mp4",
        "files": ["states.pth", "rel_actions.pth", "seq_lengths.pkl", "velocities.pth"],
        "optional": ["shapes.pkl", "abs_actions.pth"],
        "count_from": ("seq_lengths.pkl", "pickle"),
    },
}

REQUIRED_LOADERS = [
    "datasets/__init__.py", "datasets/traj_dset.py", "datasets/img_transforms.py",
    "datasets/point_maze_dset.py", "datasets/pusht_dset.py", "datasets/wall_dset.py",
]


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# [1] Repo structure
# ---------------------------------------------------------------------------
def check_repo(repo):
    section("[1] REPO STRUCTURE (loader code + env configs present)")
    for rel in REQUIRED_LOADERS:
        p = os.path.join(repo, rel)
        ok(rel) if os.path.isfile(p) else fail(f"missing loader module: {rel}")
    for env in EXPECTED:
        p = os.path.join(repo, "conf", "env", f"{env}.yaml")
        ok(f"conf/env/{env}.yaml") if os.path.isfile(p) else fail(f"missing env config: conf/env/{env}.yaml")


def repo_env_dataset_block(repo, env_name):
    """Return the raw dataset block from conf/env/<env_name>.yaml (interpolations intact)."""
    p = os.path.join(repo, "conf", "env", f"{env_name}.yaml")
    if not os.path.isfile(p):
        return None
    return load_yaml(p).get("dataset", {})


def subdir_from_datapath(data_path):
    """Last path component of a data_path (e.g. .../pusht_noise -> pusht_noise)."""
    return os.path.basename(str(data_path).rstrip("/\\"))


# ---------------------------------------------------------------------------
# [2] Config drift vs training
# ---------------------------------------------------------------------------
DRIFT_FIELDS = ["_target_", "split_ratio", "use_preprocessed", "use_frame_files",
                "with_velocity", "split_mode", "normalize_action"]

def find_runs(runs_root):
    """A run dir is any folder that directly contains hydra.yaml."""
    hits = glob.glob(os.path.join(runs_root, "**", "hydra.yaml"), recursive=True)
    return sorted({os.path.dirname(h) for h in hits})


def check_config_drift(repo, runs_root):
    section("[2] CONFIG DRIFT  (trained-with config  vs  current repo env config)")
    runs = find_runs(runs_root)
    if not runs:
        warn(f"no run dirs (with hydra.yaml) found under {runs_root} -- skipping drift check.")
        warn("download the runs first (snapshot_download) so we can compare against training.")
        return set()
    used_envs = set()
    for run in runs:
        name = os.path.basename(run)
        try:
            cfg = load_yaml(os.path.join(run, "hydra.yaml"))
        except Exception as e:
            fail(f"{name}: cannot read hydra.yaml ({e})"); continue
        env = cfg.get("env", {})
        env_name = env.get("name")
        trained_ds = env.get("dataset", {})
        # medium maze is saved as name=point_maze_medium; map save_name too
        key = env_name if env_name in EXPECTED else None
        if key is None:
            fail(f"{name}: unknown env.name='{env_name}' (not in EXPECTED)"); continue
        used_envs.add(key)
        repo_ds = repo_env_dataset_block(repo, key)
        if repo_ds is None:
            fail(f"{name}: repo has no conf/env/{key}.yaml to compare"); continue

        print(f"\n  -- run: {name}  (env={env_name}) --")
        # data_path: compare the subdir only (DATASET_DIR differs per machine)
        trained_sub = subdir_from_datapath(trained_ds.get("data_path", ""))
        repo_sub = subdir_from_datapath(repo_ds.get("data_path", ""))
        if trained_sub == repo_sub and trained_sub == EXPECTED[key]["subdir"]:
            ok(f"data_path subdir = '{trained_sub}'")
        else:
            fail(f"data_path subdir mismatch: trained='{trained_sub}' repo='{repo_sub}' expected='{EXPECTED[key]['subdir']}'")
        # the drift-sensitive fields
        for fld in DRIFT_FIELDS:
            if fld not in trained_ds and fld not in repo_ds:
                continue
            tv = trained_ds.get(fld, "<absent>")
            rv = repo_ds.get(fld, "<absent>")
            # repo value may be an interpolation like ${normalize_action}; resolve the common ones
            if isinstance(rv, str) and rv.startswith("${"):
                rv_resolved = {"${normalize_action}": True}.get(rv, rv)
            else:
                rv_resolved = rv
            if tv == rv_resolved:
                ok(f"{fld} = {tv}")
            else:
                fail(f"{fld} mismatch: trained={tv!r}  repo={rv!r}")
    return used_envs


# ---------------------------------------------------------------------------
# [3] Data on disk
# ---------------------------------------------------------------------------
def count_episodes(obs_dir, ext):
    if not os.path.isdir(obs_dir):
        return None
    return len(glob.glob(os.path.join(obs_dir, f"episode_*{ext}")))


def check_one_data_folder(folder, spec, deep):
    """Check a single flat data folder (or a pusht train/ or val/ subfolder)."""
    if not os.path.isdir(folder):
        fail(f"data folder not found: {folder}"); return
    print(f"\n  -- data folder: {folder} --")
    for fn in spec["files"]:
        p = os.path.join(folder, fn)
        ok(f"{fn}") if os.path.isfile(p) else fail(f"missing required file: {fn}")
    for fn in spec.get("optional", []):
        p = os.path.join(folder, fn)
        if os.path.isfile(p): ok(f"{fn} (optional, present)")
        else: warn(f"{fn} (optional) absent -- loader falls back to defaults")

    obs_dir = os.path.join(folder, "obses")
    n_eps = count_episodes(obs_dir, spec["obs_ext"])
    if n_eps is None:
        fail(f"missing obses/ dir (expected episode_*{spec['obs_ext']} files)")
    elif n_eps == 0:
        fail(f"obses/ has no episode_*{spec['obs_ext']} files")
    else:
        ok(f"obses/ has {n_eps} episode_*{spec['obs_ext']} files")

    if deep:
        fn, kind = spec["count_from"]
        p = os.path.join(folder, fn)
        n_meta = None
        try:
            if kind == "torch" and _HAVE_TORCH:
                n_meta = len(torch.load(p, map_location="cpu"))
            elif kind == "pickle":
                with open(p, "rb") as f:
                    n_meta = len(pickle.load(f))
        except Exception as e:
            warn(f"deep check: could not load {fn} ({e})")
        if n_meta is not None:
            if n_eps is not None and n_eps >= n_meta:
                ok(f"episode count {n_eps} >= trajectories in {fn} ({n_meta})")
            elif n_eps is not None:
                fail(f"only {n_eps} obs episodes but {fn} lists {n_meta} trajectories")


def check_data(dataset_dir, used_envs, deep):
    section("[3] DATA ON DISK (files the loaders will actually read)")
    if not dataset_dir:
        fail("DATASET_DIR not set (export DATASET_DIR=/path/to/data). Skipping data checks.")
        return
    if not os.path.isdir(dataset_dir):
        fail(f"DATASET_DIR does not exist: {dataset_dir}"); return
    ok(f"DATASET_DIR = {dataset_dir}")
    if not used_envs:
        warn("no envs discovered from runs; checking all envs the repo knows about.")
        used_envs = set(EXPECTED)
    for env in sorted(used_envs):
        spec = EXPECTED[env]
        base = os.path.join(dataset_dir, spec["subdir"])
        print(f"\n### env '{env}'  (expects {spec['subdir']}/, layout={spec['layout']}, obs={spec['obs_ext']})")
        if spec["layout"] == "split":
            for sub in ("train", "val"):
                check_one_data_folder(os.path.join(base, sub), spec, deep)
        else:
            check_one_data_folder(base, spec, deep)


def main():
    ap = argparse.ArgumentParser(description="Verify dataset<->training sync before evaluation.")
    ap.add_argument("--repo", default=".", help="repo root (default .)")
    ap.add_argument("--dataset-dir", default=os.environ.get("DATASET_DIR"),
                    help="data root (default $DATASET_DIR)")
    ap.add_argument("--runs", default="./eval_runs",
                    help="folder holding downloaded trained runs (default ./eval_runs)")
    ap.add_argument("--deep", action="store_true",
                    help="also load states/seq_lengths to verify episode counts")
    args = ap.parse_args()

    print(f"repo        = {os.path.abspath(args.repo)}")
    print(f"dataset_dir = {args.dataset_dir}")
    print(f"runs        = {os.path.abspath(args.runs)}")
    print(f"torch       = {'available' if _HAVE_TORCH else 'NOT available (deep count checks limited)'}")

    check_repo(args.repo)
    used_envs = check_config_drift(args.repo, args.runs)
    check_data(args.dataset_dir, used_envs, args.deep)

    section("SUMMARY")
    if FAILS:
        print(f"{RED}{len(FAILS)} problem(s) found:{RST}")
        for m in FAILS:
            print(f"  - {m}")
        sys.exit(1)
    print(f"{GREEN}All checks passed: datasets are in sync with the trained runs.{RST}")


if __name__ == "__main__":
    main()
