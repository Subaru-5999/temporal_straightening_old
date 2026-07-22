#!/usr/bin/env python3
"""
hf_sync_pusht.py -- replace the old PushT world models on the Hugging Face repo
                    with the freshly-trained local checkpoints.

Repo (model): gravycrazy/temporal_straightening

AUTH: the token is read ONLY from an environment variable (never hardcoded / logged).
      Set one of: HF_TOKEN, HUGGINGFACE_TOKEN, HF_API_KEY  (in that order of precedence).

WHAT IT DOES (per PushT run):
  Uploads the local run folder to the repo under <repo_prefix>/<run_name> and, in the
  SAME commit, deletes every pre-existing file under that prefix (delete_patterns="*").
  => the old training for that run is removed and replaced atomically -- there is never
     a moment where the repo has neither the old nor the new model.

SAFETY:
  * Dry-run by default: prints exactly what would be uploaded/deleted. Nothing changes
    until you pass --yes.
  * Refuses to touch the remote for a run whose local model_latest.pth is missing
    (so we never delete a remote model without a replacement in hand).
  * Use --list first to see how the runs are currently stored in the repo.

USAGE (on the pod):
  export HF_TOKEN=hf_xxx            # your write token, in the env only
  python hf_sync_pusht.py --list                 # inspect current repo layout
  python hf_sync_pusht.py                         # DRY RUN (shows the plan)
  python hf_sync_pusht.py --yes                    # actually replace
  # options:
  #   --ckpt-root checkpoints/test   (local run dirs live here: <root>/<run_name>)
  #   --repo-prefix ""               (path in repo under which runs are stored; default: repo root)
  #   --full                          (upload ALL files incl. per-epoch ckpts, plots, videos)
  #   --runs <name> [<name> ...]      (override which runs to sync)
"""
import os
import sys
import argparse

REPO_ID_DEFAULT = "gravycrazy/temporal_straightening"

# The two PushT cells we retrained (✗ no-straighten, ✓ straighten).
DEFAULT_RUNS = [
    "pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06",
    "pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05",
]

# Minimal set that fully defines a trained model: weights + config + log.
ESSENTIAL_PATTERNS = [
    "checkpoints/model_latest.pth",
    "hydra.yaml",
    ".hydra/**",
    "*.log",
]


def get_token():
    for var in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_KEY"):
        tok = os.environ.get(var)
        if tok:
            return tok, var
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Replace old PushT models on Hugging Face with freshly trained ones.")
    ap.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    ap.add_argument("--ckpt-root", default=os.path.join("checkpoints", "test"),
                    help="local dir containing the run folders (default: checkpoints/test)")
    ap.add_argument("--repo-prefix", default="",
                    help="path in the repo under which runs are stored (default: repo root)")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="run folder names to sync")
    ap.add_argument("--full", action="store_true",
                    help="upload ALL files in the run dir (default: only model_latest.pth + config + logs)")
    ap.add_argument("--list", action="store_true", help="just list current repo files and exit")
    ap.add_argument("--yes", action="store_true", help="actually perform the replace (otherwise dry-run)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub not installed. Run:  pip install -U huggingface_hub")

    token, tok_var = get_token()
    if not token:
        sys.exit("No token found. Set HF_TOKEN (or HUGGINGFACE_TOKEN / HF_API_KEY) in the environment.")
    print(f"[auth] using token from ${tok_var}  (repo: {args.repo_id})", flush=True)

    api = HfApi(token=token)

    # ---- list mode: show what's currently in the repo ----
    if args.list:
        try:
            files = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
        except Exception as e:
            sys.exit(f"Could not list repo files: {e}")
        print(f"\n=== {args.repo_id} currently contains {len(files)} file(s) ===")
        for f in sorted(files):
            print("  " + f)
        return

    allow = None if args.full else ESSENTIAL_PATTERNS

    # ---- validate local checkpoints BEFORE touching the remote ----
    plan = []
    for name in args.runs:
        local_dir = os.path.join(args.ckpt_root, name)
        model_pth = os.path.join(local_dir, "checkpoints", "model_latest.pth")
        path_in_repo = f"{args.repo_prefix.rstrip('/')}/{name}" if args.repo_prefix else name
        if not os.path.isdir(local_dir):
            print(f"[SKIP] {name}: local dir not found ({local_dir})")
            continue
        if not os.path.isfile(model_pth):
            print(f"[SKIP] {name}: no trained model at {model_pth} (won't delete remote without a replacement)")
            continue
        size_gb = os.path.getsize(model_pth) / (1024**3)
        plan.append((name, local_dir, path_in_repo, size_gb))

    if not plan:
        sys.exit("Nothing to do: no runs with a local model_latest.pth were found.")

    print("\n=== PLAN (old repo files under each prefix are DELETED and replaced, atomically) ===")
    for name, local_dir, path_in_repo, size_gb in plan:
        print(f"  {name}")
        print(f"     local : {local_dir}  (model_latest.pth ~{size_gb:.2f} GB)")
        print(f"     repo  : {args.repo_id}:/{path_in_repo}")
        print(f"     files : {'ALL' if args.full else ', '.join(ESSENTIAL_PATTERNS)}")
    if not args.yes:
        print("\n[DRY RUN] nothing changed. Re-run with --yes to perform the replace.")
        return

    # ---- execute: atomic replace per run (upload new + delete old in one commit) ----
    for name, local_dir, path_in_repo, _ in plan:
        print(f"\n>>> replacing {path_in_repo} ...", flush=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=local_dir,
            path_in_repo=path_in_repo,
            allow_patterns=allow,
            delete_patterns="*",  # scoped to path_in_repo: clears the old run, atomic replace
            commit_message=f"Replace {name} with freshly trained PushT model (B200 reproduction)",
        )
        print(f"    done: {args.repo_id}:/{path_in_repo}", flush=True)

    print("\nAll PushT runs replaced.")


if __name__ == "__main__":
    main()
