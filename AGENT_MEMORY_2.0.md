# AGENT MEMORY 2.0 — Reproducing Table 1 on the NVIDIA DGX (B200 MIG) pod

This is the running log/playbook for bringing up the *temporal-straightening* Table-1
reproduction on a **fresh DGX pod** (`nvidiadgx`), distinct from the original B200 pod
that `REPRODUCTION.md` / `POD_SETUP_LOG.md` were written on. It records every issue we
hit, the root cause, and the exact fix — in the order they surfaced — plus the final
working recipe.

> TL;DR: A brand-new pod had **none** of the validated environment. We rebuilt it layer
> by layer; each fix revealed the next gate. All environment errors are resolved; the
> pipeline is verified (smoke test → `Success rate: 0.40` on UMaze DINOv2-patch ✗, which
> matches the paper's `35.33 ± 4.11`). The remaining work is the full 30-eval run.

---

## 0. What we're reproducing (scope)

Exactly the **5 Table-1 cells** we've tracked all along (GD planner, 50 samples,
mean±std over 3 data seeds 100/200/300):

| Run dir (`checkpoints/test/<name>`) | Env / config | Paper OL / MPC |
|---|---|---|
| `umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05` | UMaze DINOv2 patch 14×14×384, ✗ | 35.33 / 80.67 |
| `umaze_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06` | UMaze +proj 14×14×8, ✗ | 44.00 / 81.33 |
| `umaze_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` | UMaze +proj 14×14×8, ✓ | 94.00 / 100.00 |
| `pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06` | PushT +proj 14×14×8, ✗ | 70.00 / 78.67 |
| `pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05` | PushT +proj 14×14×8, ✓ | 77.33 / 85.33 |

Paper protocol (verified in `reproduce_table1.py`): Table 4 planning hyperparams
(horizon 25, zero init, Adam, lr 0.1, 100 steps; OL executes 25 / MPC executes 5),
`goal_H=25` ÷ frameskip 5 → H=5 model steps, §5.3 objectives (UMaze images-only
`alpha=0`, OL `mode=last`, MPC `mode=all`; PushT images+proprio `alpha=1`, OL
`mode=last`, MPC `mode=staged`).

---

## 1. Hardware / platform facts (this pod)

- **GPU**: NVIDIA **B200**, **MIG enabled**, one `1g.45gb` slice (~45 GB).
  MIG UUID: `MIG-90532e6e-2246-5f8b-84eb-cefedb38f2c1`.
- Driver 570.124.06 / CUDA 12.8. System Python 3.10 (`/usr/bin/python`), no conda.
- torch already present and **correct**: `2.7.0+cu128`, `cuda 12.8`, capability `(10, 0)`
  (native Blackwell — NOT the source of the slowness; see Issue 5).
- Data at `/workspace/arun/data`; project at `/workspace/arun/temporal_straightening_old`.
- **The 45 GB slice holds exactly one job and fills instantly from a single stray
  process.** `nvidia-smi` often shows *no processes* on MIG even when one is using it —
  use `ps`, not `nvidia-smi`, to find GPU-memory holders.

---

## 2. Issues faced (in order) → root cause → fix

Each error only appears after the previous one is fixed (imports/GPU init are sequential
gates), so this is monotonic progress, not a loop.

### Issue 1 — `ModuleNotFoundError: No module named 'gym'`
- **Cause**: fresh pod had no simulator/planning stack.
- **Fix**: `python -m pip install -r requirements-plan.txt`; MuJoCo 210 to `~/.mujoco`;
  apt libs (`libgl1-mesa-dev libglew-dev libosmesa6-dev libglfw3 patchelf gcc build-essential`);
  d4rl from git (fallback: tarball `--no-deps`); `pip install h5py`.

### Issue 2 — `ModuleNotFoundError: No module named 'hydra'`
- **Cause**: core/training-tier deps (shared by `plan.py`) not installed.
- **Fix**: `python -m pip install -r requirements-train.txt`
  (hydra-core 1.2.0, omegaconf, einops, accelerate, decord, wandb, submitit).
  Note: this file does **not** pin torch, so it won't disturb the cu128 build.

### Issue 3 — `wandb`: `ImportError: cannot import name 'TypeIs' from 'typing_extensions'`
- **Cause**: latest `wandb` needs `typing_extensions >= 4.10`; pod had an older one.
- **Fix**: `python -m pip install -U "typing_extensions>=4.12"`. Also set
  `WANDB_MODE=disabled` (headless eval needs no wandb; results come from `logs.json`).

### Issue 4 — gym: "does not support NumPy 2.0" (and downstream breakage)
- **Cause**: gym 0.23.1 / d4rl / mujoco-py predate NumPy 2 (removed aliases).
- **Fix**: `python -m pip install "numpy<2"` (paper env used 1.26.x).

### Issue 5 — ~250 s "hang" at `setup_model_s` (looked frozen at 92% one CPU core)
- **Symptom**: `[timing] setup_model_s=243–269` (vs ~9 s on the reference pod), then
  appears stuck.
- **Debug**: `py-spy` blocked (ptrace disabled, `/proc` read-only). Used in-process
  **faulthandler** (`faulthandler.dump_traceback_later(..., file=...)` via a `runpy`
  wrapper) to dump the stack to a file. Stack showed the main thread in
  `torch.nn.init.trunc_normal_` → DINOv2 `init_weights` (building the throwaway encoder
  in `load_ckpt`).
- **Root cause**: **CPU thread oversubscription** on a many-core node — thousands of tiny
  per-layer init ops each paying huge thread-launch/sync overhead.
- **Fix**: cap threads → `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=8`.
  `setup_model_s` dropped 250 s → ~5.5 s. (Baked into `reproduce_table1.py`.)

### Issue 6 — `FileNotFoundError: 'plan_targets.pkl'` in `dump_targets()`
- **Cause**: `plan.py` writes `plan_targets.pkl`/`logs.json` relative to cwd, relying on
  Hydra having `chdir`'d into a created run dir; on this pod that cwd wasn't reliably
  present.
- **Fix**: patched `planning_main` in `plan.py` to
  `os.makedirs(output_dir, exist_ok=True); os.chdir(output_dir)` before any writes.

### Issue 7 — `RuntimeError: NVML_SUCCESS == r ... CUDACachingAllocator.cpp:1016`
- **Cause**: torch 2.7's default caching allocator makes an NVML query that **fails on a
  MIG slice** (fires during the first GD backward). `expandable_segments:False` (the fix
  documented for the original pod) did **not** help here.
- **Fix**: `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` — uses CUDA's async allocator,
  bypassing the NVML-asserting caching allocator. (Baked into the driver.)

### Issue 8 — `ValueError: invalid literal for int() with base 10: 'MIG-...'`
- **Cause**: **mujoco-py** does `int(os.environ['CUDA_VISIBLE_DEVICES'])` to pick its
  render device (`maze_model.py → sim.render → _setup_opengl_context`). We had set
  `CUDA_VISIBLE_DEVICES` to the MIG **UUID** (attempting to fix Issue 7) — not an integer.
  `MUJOCO_GL=osmesa` did NOT help (mujoco-py ignores it).
- **Fix**: **leave `CUDA_VISIBLE_DEVICES` unset** (torch still sees the MIG device via the
  container). The driver now auto-unsets any non-integer `CUDA_VISIBLE_DEVICES` defensively.

### Issue 9 — `torch.OutOfMemoryError` in the ViT predictor attention (`vit.py:71`)
- **Symptom**: OOM at planning step 0 on a 45 GB slice.
- **Debug**: `nvidia-smi` showed `41544MiB / 45312MiB` used but **empty** Processes table
  (MIG can't enumerate processes). `ps -eo pid,ppid,etime,rss,cmd | grep python` revealed
  a **live stopped** process **PID 1407 `python -`** (state `Tl`) holding the 41.5 GB — a
  leftover heredoc that had been Ctrl-Z'd/suspended and never released its CUDA context.
- **Root cause**: **leaked GPU memory from a stray process**, NOT a workload-size problem.
  (The reference pod fit this exact workload in the same 45 GB.)
- **Fix**: `kill -9 1407` → slice freed to 16 MiB. Re-ran → `Success rate: 0.40` ✓.

---

## 3. Ruled-out / dead-end hypotheses (don't chase these again)

- **Wrong/old torch build** → ruled out: `2.7.0+cu128 (10,0)` is native Blackwell.
- **Disk I/O slow** → ruled out: `time cat model_latest.pth` = 0.09 s (445 MB, cached).
- **wandb causing the hang** → ruled out: `setup_model` logs *after* `wandb.init`; the
  hang was DINOv2 init (Issue 5). (wandb still disabled for cleanliness.)
- **`MUJOCO_GL=osmesa` to dodge EGL** → ineffective: mujoco-py ignores `MUJOCO_GL`.
- **`CUDA_VISIBLE_DEVICES=MIG-uuid` to fix NVML** → backfired (Issue 8); use
  `cudaMallocAsync` instead and keep it unset.

---

## 4. Final working environment recipe (copy/paste)

```bash
cd /workspace/arun/temporal_straightening_old
unset CUDA_VISIBLE_DEVICES                          # MIG UUID breaks mujoco-py; leave unset
export DATASET_DIR=/workspace/arun/data
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled WANDB_SILENT=true
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync   # MIG NVML fix
export PLAN_SERIAL_ENV=1                                 # MIG fork-safety
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
```

`reproduce_table1.py` sets all of the above as defaults itself (and auto-unsets a
non-integer `CUDA_VISIBLE_DEVICES`), so `python reproduce_table1.py` works in a bare shell.

One-time installs (already done on this pod): torch stays `2.7.0+cu128`; then
`requirements-plan.txt`, MuJoCo 210 + apt libs, d4rl, h5py, `requirements-train.txt`,
`typing_extensions>=4.12`, `numpy<2`.

---

## 5. Operational lessons (this MIG slice)

- **One job at a time.** 45 GB fills instantly; a single leftover process = OOM.
- **Before every run**: `ps -eo pid,etime,rss,cmd | grep -i python | grep -v grep` and
  `kill -9` any stray/stopped (`Tl`) python. Do NOT trust `nvidia-smi`'s process list on MIG.
- **Never Ctrl-Z a GPU python job** — that's how PID 1407 leaked 41.5 GB.
- **Debugging a "hang"**: `ps` state tells a lot (`R`=busy, `S`=blocked/idle, `Tl`=stopped).
  For stacks without ptrace: in-process `faulthandler.dump_traceback_later(..., file=open(...))`.

---

## 6. Files created/changed during this effort

- `reproduce_table1.py` — pure-Python driver: 5 runs × (OL ×3 + MPC ×3), paper objectives,
  run-scoped (no mixing), env defaults baked in.
- `summarize_run.py` — run-scoped aggregator: reads only one run's `logs.json`, writes
  `results/<run>.json`, rebuilds `results/table1_reproduction.{md,csv}` (ours vs paper).
- `check_dataset_sync.py` — verifies DATASET_DIR data + trained-run configs are in sync
  with the loaders.
- `aggregate_results.py` — global multi-seed aggregator (correct mean±std over appended
  `logs.json` lines; fixes `collect_results.py`'s last-line-only bug).
- `plan.py` — patched `planning_main` to ensure the Hydra run dir exists and is cwd
  (Issue 6).

---

## 7. Status & next step

- **Verified**: env built, data in sync, checkpoints load (epoch 20/2), smoke test
  `Success rate: 0.40` (UMaze DINOv2-patch ✗ OL, seed 100) — matches paper `35.33 ± 4.11`.
- **Next**: launch the full run (detached), then compare `results/table1_reproduction.md`
  to the paper targets in §0:
  ```bash
  nohup python reproduce_table1.py > eval_all.log 2>&1 &
  grep -aE "RUN:|Success rate|RESULT|FAIL|ALL EVALS DONE" eval_all.log
  ```
- **Residual risk**: PushT MPC memory (only untested path). If a seed OOMs → add
  sample-chunking (process the 50 test samples in sub-batches; identical results, slower).

---

## 8. Deep audit — why B200 drifts from the paper's Table 1 (Task 6)

**Question:** why do our numbers drift from the paper, especially several points HIGH on the
✗ (no-straightening) cells while ✓ cells match?

**Drift is structured, not random.** From REPRODUCTION.md §4/§7: every ✗ cell runs +3 to +11
above the paper band (UMaze +proj ✗ OL 52 vs 44, MPC 92 vs 81.3; PushT ✗ OL 76 vs 70), while
both ✓ cells land inside the band (UMaze ✓ 90.7/100 vs 94/100; PushT ✓ 75.3/82 vs 77.3/85.3).

**Evaluation is NOT the source (proven).** Re-evaluating a fixed checkpoint reproduces identical
success rates, and a full retrain reproduced identical loss AND identical eval. `plan.py` runs
fp32/no-autocast, the planner is deterministic given weights+seed, success is computed from
CPU-deterministic env state, and `cudaMallocAsync` only affects memory management. ⇒ 100% of the
drift is baked into the TRAINED WEIGHTS.

**Root causes (ranked), all in the training run:**
1. **Single training seed** (`conf/train.yaml training.seed=0`). Table 1's band folds in training
   variability; the encoder is trained once here. Planning's 3 seeds (100/200/300) only vary the
   50 test start/goal pairs, NOT the weights. Biggest lever.
2. **bf16 mixed precision on Blackwell tensor cores** (`train.yaml mixed_precision=bf16` via
   accelerate). bf16 (8 mantissa bits) tensor-core kernels/accumulation order differ on sm_100 vs
   the paper's GPUs (unspecified; likely Ampere/Hopper) → different-but-valid local minimum over
   20 epochs.
3. **No determinism/precision controls.** `utils.seed()` sets RNG only; grep confirms the repo
   never calls `use_deterministic_algorithms`, `cudnn.deterministic`, or
   `set_float32_matmul_precision`/TF32 flags.
4. **Different torch/CUDA/cuDNN** (2.7.0+cu128, forced for Blackwell) → different kernel
   autotuning/fusion vs the authors' stack.
5. Ruled out: `models/vit.py` uses manual `nn.Softmax` attention (no SDPA backend variance).

**Why ✗ drifts and ✓ doesn't — corroborates the paper's thesis.** The method improves the
CONDITIONING of the planning objective. ✓ cells are well-conditioned/near-saturated → insensitive
to weight perturbations → reproduce tightly (paper's own ✓ stds are smallest). ✗ cells are
ill-conditioned → GD success swings with tiny weight changes → most sensitive to seed/bf16/arch
noise (paper's own ✗ stds are largest, ±6–7). So the drift concentrates exactly where the paper
predicts sensitivity; it validates the mechanism rather than contradicting it.

**Verdict:** expected single-seed + Blackwell-bf16 + torch-2.7 variance on the sensitive ✗ cells.
Not a bug. Core ✗→✓ claim reproduces (UMaze OL 52→91, MPC 92→100; PushT lift present); all ✓ in band.

**To shrink drift if desired (none required for correctness):**
- Train 3 seeds (`training.seed=0,1,2`) per ✗ cell, report mean±std (matches paper variance model).
- Attribution experiment: retrain one ✗ cell with `mixed_precision=no` (fp32) +
  `torch.set_float32_matmul_precision("high")` + `matmul.allow_tf32=False` to isolate bf16 vs seed.
- Pin torch version if non-Blackwell hardware becomes available.

---

## 9. Original-code diff + paper protocol confirmation (Task: exact PushT reproduction)

**Diffed the authors' original code (`temporal_straightening_original.zip`) against our repo.**
Functionally IDENTICAL in every result-affecting path. The only differences:
- `conf/train.yaml`, `conf/plan_gd.yaml`, `conf/plan_gd_mpc.yaml`, `conf/env/*.yaml`: launcher
  `submitit_slurm` (+ `gres: "gpu:h100:1"`, `mem_gb 512/256`) → our `submitit_local` + smaller
  mem. **The paper trained/evaluated on H100 (Hopper, sm_90).** We run B200 (Blackwell, sm_100).
- `planning/mpc.py`: we added `torch.cuda.empty_cache()` (MIG memory only, no math).
- `train.py`: `weights_only=False` + offline `resume_from` logic (load/resume only, no math).
- `utils.py`, `models/visual_world_model.py`, `plan.py` core, `datasets/*`, `models/*`,
  `conf/encoder/*`: byte-identical (seed helper, bf16, straightening loss, planner, objectives).

⇒ **No code bug/discrepancy causes the ✗ drift.** Our repo faithfully reproduces the original.

**Paper protocol confirmed from `_paper.txt`:**
- Table 1 caption (L655): "mean ± std over three **data sampling seeds**."
- `plan.py`: `eval_seed = [cfg_dict["seed"] * n + 1 for n in range(n_evals)]` → the `seed` arg
  only selects which 50 TEST samples are drawn. Training uses fixed `training.seed=0`.
- ⇒ "three data sampling seeds" = three draws of the 50 test samples on ONE trained model =
  exactly our (train-once, plan seeds 100/200/300) protocol.

**CORRECTION to earlier note (§8 recommendation):** using 3 TRAINING seeds would DEVIATE from
the paper (paper = 1 training seed + 3 data-sampling/planning seeds). Do NOT multi-train-seed if
the goal is "exactly per paper." Our current PushT numbers (✗ 76/82, ✓ 75/82) were produced by
the exact paper protocol on B200.

**Consequence for exact reproducibility:** bit-exact reproduction is impossible across H100→B200
because the shared code trains in bf16 with NO determinism controls (true in the ORIGINAL too).
Re-running the exact protocol on B200 reproduces the SAME numbers (training is deterministic
run-to-run on the same slice — proven). The ✗ upward drift is a pure H100→B200 + torch-2.7
artifact, not fixable in code. ✓ rows land in band (method validated); ✗ rows sit high (hardware).
Only H100 hardware (or a bf16→fp32/determinism change, which deviates from the paper) would move ✗.

---

## 10. NOVEL EXTENSION — Multi-Scale (Hierarchical) Straightening (ICRA experiment)

This section is the **standalone reference** for the multi-scale straightening extension we
designed, implemented, committed, and launched. It is a *novel research extension*, cleanly
separated from and config-gated off of the faithful paper reproduction (§0–§9). The faithful
path is untouched: with no multi-scale flags the code is **bit-identical** to the original
(verified curvature value `1.49174845` and study-guide worked example `L^(1)=0.3333`,
`L^(2)=1.0000`).

### 10.1 The idea (what & why)
The paper straightens the latent trajectory **only at the finest scale** — consecutive-frame
velocity cosine (`z_t, z_{t+1}, z_{t+2}`). Multi-scale straightening adds curvature penalties at
**coarser temporal scales** so latent trajectories are straight at multiple resolutions, which
targets long-horizon drift and hierarchical abstraction (robotics-relevant: manipulation,
navigation, model-based RL).

### 10.2 Math (grounded in paper Eqs 3–7 + user's ICRA spec)
For scale `s`:
- scale-`s` velocity:  `v_t^(s) = z_{t+s} − z_t`
- scale-`s` curvature term: `C_t^(s) = cos( v_t^(s), v_{t+s}^(s) )`, then `L^(s) = 1 − mean_t C_t^(s)`
- multi-scale loss: `L_multi = Σ_s w_s · L^(s)`
- total objective (extends paper Eq 7): `L_total = L_pred + λ · Σ_s w_s L^(s)`
  where `λ` = the existing `training.straighten` strength (e.g. `aggcos1e-1` → 0.1).
- optional directional/goal term (off by default): `L_goal = Σ_s μ_s (1 − cos(v_t^(s), z_g − z_t))`.

**Theory link (paper Theorem 4.4):** coarse-scale velocities regularize higher powers `A^s ≈ I`
of the transition matrix, tightening the bound on the planning-Hessian condition number
`κ_eff` for large horizon `K` → more stable GD/MPC at `H ≫ 5`.

### 10.3 Two documented deviations from the spec's literal notation
1. **Objective re-parameterization.** The spec writes `L_pred + λ_local L^(1) + Σ_s λ_s L^(s)`,
   which double-counts `s=1`. We implement the clean, equivalent `L_pred + λ · Σ_s w_s L^(s)`
   (single `λ` = `training.straighten`, per-scale `w_s` = `straighten_scale_weights`). No literal
   double-count.
2. **Goal term uses a pseudo-goal** = the window's last latent `z[:, -1]` (there is no true goal
   at training time). It is **off by default** (`straighten_goal_weight=0.0`).

### 10.4 Implementation (files & exact changes) — committed
- **`models/visual_world_model.py`**
  - New constructor args: `straighten_scales`, `straighten_scale_weights`, `straighten_goal_weight`.
  - Backward compat: `straighten_scales` falsy → `[1]` (== paper single-scale). `scales=[1]` is
    bit-identical to the original.
  - `self.straighten_min_frames = 2·max(scales) + 1` (min latent frames a window must hold).
  - New `_scale_velocity_curvature(z, s)`: builds `v^(s)=z[:,s:]−z[:,:-s]`, cosine of `va[:,:-s]`
    vs `va[:,s:]`; returns `None` if the window is too short for scale `s`.
  - New `_scale_goal_alignment(z, s, z_goal)`: the optional directional term.
  - `total_curvature` rewritten as the **weighted sum over scales** (raises if no scale fits).
  - `forward`: prediction target **capped** to `z[:, num_pred : num_pred+num_hist]` so a widened
    window (T can exceed `num_hist+num_pred`) does NOT change the prediction loss.
- **`train.py`**
  - Passes the 3 new params to the model (`self.cfg.training.get(...)`).
  - **Auto-widens the dataset window** to `num_frames = 2·max_scale+1` when scales are set,
    then passes `num_frames` to the loader. Logs: `Multi-scale straightening: dataset window
    num_frames=...`.
- **`conf/train.yaml`**: added `straighten_scales: null`, `straighten_scale_weights: null`,
  `straighten_goal_weight: 0.0` (documented; defaults reproduce the paper).
- **`datasets/pusht_dset.py`, `datasets/point_maze_dset.py`**: accept a `num_frames` override.

### 10.5 Git state
- Code committed as **`66e4b28`** ("Add multi-scale (hierarchical) straightening loss …"),
  pushed to `main` (`github.com/Subaru-5999/temporal_straightening_old`).
- Follow-up **`0593d77`** committed this memory file + backward-compatible `_seed` handling in
  `reproduce_table1.py`/`summarize_run.py`.
- **Deliberately NOT on the hub** (per user): `STUDY_GUIDE_temporal_straightening.md`, `.kiro/`,
  `temporal_straightening_original.zip`, `arXiv-2603.12231v2.tar.gz`. They remain untracked locally.

### 10.6 Exact run command (multi-scale PushT ✓, B200 pod) — VERIFIED launched
Uses a **separate ckpt path** (`checkpoints_multiscale`) so it can never overwrite the faithful ✓.
```bash
cd /workspace/arun/temporal_straightening_old && git pull
unset CUDA_VISIBLE_DEVICES
export DATASET_DIR=/workspace/arun/data D4RL_SUPPRESS_IMPORT_ERROR=1 WANDB_MODE=disabled WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
setsid nohup python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.epochs=2 env.num_workers=4 \
  'training.straighten_scales=[1,3,5,10]' 'training.straighten_scale_weights=[1,1,2,4]' \
  has_decoder=false ckpt_base_path=$PWD/checkpoints_multiscale \
  > train_pusht_multiscale.log 2>&1 < /dev/null &
```
**Verify it engaged** (must print all three):
```bash
grep -aE "Multi-scale straightening|Straightening enabled|dataset window num_frames" train_pusht_multiscale.log
```
**Evaluate after training:**
```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PLAN_SERIAL_ENV=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
python reproduce_table1.py pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
  --base $PWD/checkpoints_multiscale/test
```

### 10.7 Monitoring (progress snapshot)
```bash
ps -eo pid,etime,rss,cmd | grep "[t]rain.py"     # alive? etime = wall-clock elapsed
tr '\r' '\n' < train_pusht_multiscale.log | \
  grep -aE "Multi-scale straightening|Straightening enabled|Epoch [0-9]+ (Train|Valid)|Training loss|Saved model" | tail -n 15
```

### 10.8 Understanding the epoch length (IMPORTANT — asked & answered)
- The tqdm total (e.g. **15798**) is the **number of mini-batches per epoch** =
  `ceil(num_train_windows / batch_size)` (batch_size=32) → here ~505k training windows.
  It is **the same every epoch** (slices are computed once at dataset init and reused).
- **Why multi-scale epochs are shorter than a single-scale run:** the slicer
  (`datasets/traj_dset.py TrajSlicerDataset`) cuts `max(0, T − num_frames·frameskip + 1)` windows
  per trajectory of length `T`.
  - Multi-scale `scales=[1,3,5,10]` → `num_frames=21`, `frameskip=5` → window spans **105 env-steps**
    → `T−104` windows/traj, and **any trajectory < 105 env-steps yields ZERO windows**.
  - Faithful single-scale → `num_frames=4` → 20-env-step window → `T−19` windows/traj.
  - ⇒ multi-scale drops ~85 windows/traj + discards short trajectories → far fewer iterations.
    This is expected arithmetic, not a bug. `scales=[1,3,5]` (`num_frames=11`, 55-step window)
    keeps many more windows and yields a longer epoch.

### 10.9 Expectations (honest, for next time)
- **Exploratory research, not reproduction** — outcome uncertain (help / neutral / regress).
- **Compare against single-scale ✓ on the SAME B200 = 75.33 OL / 82.00 MPC**, NOT the paper's
  77.33/85.33 (comparing to the paper would conflate the extension with the H100→B200 bf16 drift
  documented in §8–§9).
- The standard eval uses `goal_H=25` → **H≈5** (short horizon). Multi-scale's claimed +10–25% is a
  **long-horizon (50+ step) claim**, so on the standard eval expect a **small / within-noise**
  change. A real test needs a long-horizon eval harness (**NOT built yet** — future work).
- **Memory / OOM**: 21-frame windows ≈ 5× encoder memory of single-scale. If it OOMs at batch 32:
  fall back to `training.batch_size=8` OR `'training.straighten_scales=[1,3,5]'` (11-frame windows).
- **Slower** than the ~12 h single-scale run.

### 10.10 Runtime warnings seen (triage)
- `tail: inotify cannot be used, reverting to polling` — harmless (network FS + `tail -f`). Ignore.
- `Too many open files` — real FD-limit pressure (DINO backbone + `num_workers`). Kept running, but
  if it hard-crashes workers (`OSError: [Errno 24]`): `ulimit -n 65535` before relaunch and/or
  lower `env.num_workers=2`.

### 10.11 Future work / TODO for the ICRA writeup
- Build a **long-horizon eval** (50+ steps) — the setting where multi-scale should actually help;
  the standard `goal_H=25`/H=5 eval cannot show the claimed gain.
- **Per-scale curvature logging** to training logs (spec §6) — not yet added.
- Ablations: scale sets `{1,3,5}` vs `{1,3,5,10}`, weighting schemes, global vs spatial features,
  GD vs CEM (show reduced reliance on samplers), Hessian condition-number analysis.
- Run the same extension on **PointMaze/UMaze** (note len-100 trajectories only support small
  scales; `max_scale=10` needs ≥105-step trajectories, so UMaze can only use smaller scales).

---

## 11. DEFINITIVE drift conclusion (canonical — supersedes scattered notes in §8–§9)

Full reproduction, all 4 tracked PushT/UMaze cells, 3 data-sampling seeds each, single training
seed 0, on the B200 MIG pod. This is the authoritative drift table + explanation.

| Env | Cell | Metric | Ours | Paper | Δ | Verdict |
|---|---|---|---|---|---|---|
| UMaze | patch 14×14×384, ✗ (frozen, no projector) | OL | 38.00±3.46 | 35.33±4.11 | +2.67 | in band |
| UMaze | patch 14×14×384, ✗ (frozen) | MPC | 87.33±2.31 | 80.67±6.18 | +6.66 | in band |
| UMaze | +proj 14×14×8, ✗ (trainable projector) | OL | 58.00±5.29 | 44.00±7.12 | +14.0 | OUT (high) |
| UMaze | +proj 14×14×8, ✗ (trainable) | MPC | 92.67±1.15 | 81.33±6.80 | +11.3 | OUT (high) |
| UMaze | +proj 14×14×8, ✓ (straighten) | OL | 90.67±1.15 | 94.00±1.63 | −3.33 | in band |
| UMaze | +proj 14×14×8, ✓ | MPC | 100.0±0.0 | 100.0±0.0 | 0 | exact |
| PushT | +proj 14×14×8, ✗ (trainable) | OL | 76.00±4.00 | 70.00±1.63 | +6.0 | OUT (high) |
| PushT | +proj 14×14×8, ✗ (trainable) | MPC | 82.00±5.29 | 78.67±0.94 | +3.33 | overlaps |
| PushT | +proj 14×14×8, ✓ (straighten) | OL | 75.33±6.11 | 77.33±6.18 | −2.0 | in band |
| PushT | +proj 14×14×8, ✓ (straighten) | MPC | 82.00±2.00 | 85.33±4.99 | −3.33 | in band |

**Drift ordering (the key clue):** frozen ✗ drifts least (+2.67 OL) → trainable-projector ✗
drifts most (+14, +6, OUT) → trainable-projector ✓ back in band. Drift concentrates exactly on
the trainable-representation, no-straightening cells.

**Root cause (by elimination):** the only uncontrolled variable is the GPU's bf16 arithmetic.
B200 (Blackwell, 5th-gen TC) + torch 2.7 / cuDNN 9.7 vs paper's H100 (Hopper, 4th-gen) + torch
2.3 / cuDNN 8.9 → same code lands on a slightly different trained model. Amplified into a large
success swing only on the ✗-trainable cells by two paper mechanisms:
1. **Implicit straightening (§5.2):** ✗-trainable cells get performance from a training-dynamics-
   dependent implicit straightening — the exact thing GPU arithmetic perturbs. Frozen ✗ has
   nothing trainable to shape → least drift. ✓ cells force straightening → don't rely on the
   fragile implicit effect → stable.
2. **Conditioning (Theorem 4.4):** without straightening the planning objective is ill-conditioned
   → tiny weight change → large success swing. With straightening it's well-conditioned → stable.

**Ruled out (do not re-chase):** TF32 (inert under bf16), data/data-order (deterministic, split
seed 42 + shuffle after seed 0), evaluation (deterministic, identical on re-run, same 50 tasks),
code/protocol/hyperparams (byte-identical to authors' zip), the method itself (✓ cells reproduce
in band = the paper's actual claim). ⇒ drift is baked into the TRAINED WEIGHTS, hardware-origin.

**One-line:** B200/torch-2.7 vs H100/torch-2.3 → slightly different model → paper's own math
(implicit straightening + ill-conditioned planning) amplifies it into a large swing on exactly the
no-straightening cells; straightening cells reproduce faithfully. Not a misconfiguration.

**Newly pinned baseline (for the multi-scale comparison):** single-scale ✓ PushT on this B200 =
**OL 75.33±6.11, MPC 82.00±2.00**. Multi-scale s=4 (λ₂=0.2, 3 ep) = OL 76.67±6.43, MPC 88.00±3.46.
MPC gain +6.0 → t≈2.6, p≈0.06 two-sided; borderline significant, still confounded by epochs (3 vs
2) and single training seed. Matched-epoch single-scale run + a 2nd training seed would lock it.

---

## 12. Independent per-scale λ, per-setting folders, and the has_decoder+multiscale crash fix

Work done after §10/§11, all config-gated and paper-faithful (verified line-by-line against the
paper's LaTeX `sec/1_main.tex`: Eqs 5–7, stop-grad, agg head, detached decoder, OL/MPC objectives).

### 12.1 Independent per-scale λ (`straighten_lambdas`) — commit `86c585c`
- Added `straighten_lambdas` (absolute λ per scale) to `VWorldModel` + `conf/train.yaml` + `train.py`.
  `loss = MSE + Σ_s λ_s·L_curv^(s)`. Takes precedence over the legacy `straighten_scale_weights`
  (which was over-parameterized: only the product `straighten_scale·w_s` was identifiable).
- **λ_s = 0 fully disables scale s**: `total_curvature` skips `w==0`; the training window is widened
  only for **active** scales (λ>0) — so `straighten_scales=[1,4] straighten_lambdas=[0.1,0]` reverts
  to the paper's loss AND 4-frame window (bit-identical). Fixes the earlier "λ=0 doesn't shrink the
  data window" redundancy.
- Backward compatible: `straighten_lambdas=null` → falls back to `straighten_scale × weights`;
  default single-scale `[1]` = paper. Verified with 7 scenarios.

### 12.2 Per-setting checkpoint folders — commits `cd1af2a`, `ec5d65a`
- `custom_resolvers.py`: `straighten_tag` (appends `_ms1-4_lam0.1-0.2` etc.; empty for paper default)
  and `run_variant_tag` (appends `_ep<N>` always; `_seed<N>` only if seed≠0).
- `hydra.run.dir` now encodes the full setting → each config lands in its OWN folder (no collisions;
  self-documenting; directly reusable as the HF sub-path). Paper default folder name unchanged.
- `reproduce_table1.py` + `summarize_run.py`: `base_cell` now strips `_ms/_lam/_w/_ep/_seed` tags so
  variants still map to their Table-1 cell for the (alpha, mpc_mode) + paper-target lookup. Planning
  outputs inherit the tag via `model_name` → results are distinguishable per setting.
- Example folders: paper 2ep `..._lr1e-05_ep2`; multi-scale `..._ms1-4_lam0.1-0.2_ep3`; seed variant
  `..._ep3_seed1`. Fixes the earlier 2-epoch-vs-3-epoch collision.

### 12.3 The `has_decoder=true` + multi-scale crash — commit `149382f` (ROOT CAUSE + FIX)
- **Symptom**: `RuntimeError: size of tensor a (3) must match b (8) at dim 1` in `mse_loss`, on the
  first batch, only with `has_decoder=true` + a widened multi-scale window (9-frame). All prior
  multi-scale runs used `has_decoder=false`, which skipped the failing block → never seen before.
- **Root cause (from traceback)**: NOT the loss/decoder. It's `train.py err_eval` — a *logging-only*
  reconstruction-error metric (feeds wandb, disabled), run only when `decoder_active and plot`.
  It compares the predictor output `z_obs_out` (num_hist=3 frames) against
  `z_tgt = slice_trajdict_with_t(z_gt, start_idx=num_pred)` = `z_gt[1:]` = **8 frames** for the
  9-frame window. In the paper's 4-frame window `z_gt[1:]` = 3 frames, so it always matched before.
- **Fix**: inside `err_eval`, cap `z_tgt` to `z_out`'s frame count (`slice[0:num_hist]`) — the same
  frames the loss path uses. **No-op for the paper window**; logging-only; does NOT touch the loss,
  the model, or the eval. Two call sites (`train()` + `val()`) fixed centrally.
- **Not a methodology change**: `err_eval` is a diagnostic, never added to `loss` / never backprops.

### 12.4 Paper-faithfulness re-verification (against `sec/1_main.tex`)
All recent edits are no-ops on the paper path or are logging/naming/loader plumbing:
- `forward` z_tgt/visual_tgt cap → for T=4, `z[1:4]` == original `z[1:]` (bit-identical).
- `err_eval` cap → logging-only, no-op for T=4.
- `straighten_lambdas` → off by default = paper single-scale.
- folder tags + `base_cell` → cosmetic/eval-mapping.
- `plan.py` None-skip → loader robustness for `has_decoder=false`.
- Decoder is detached (`decode(z.detach())`) — matches paper L382 "decoder detached, interpretability only".

### 12.5 The matched-comparison run plan (paper-faithful except epochs + multi-scale)
Both PushT ✓, 3 epochs, decoder ON, B200/MIG recipe; distinct auto-tagged folders:
- **Run 1 (multi-scale s=4)**: `env=pusht encoder=dino_channel training.straighten=aggcos1e-1
  training.encoder_lr=1e-5 training.epochs=3 'training.straighten_scales=[1,4]'
  'training.straighten_lambdas=[0.1,0.2]' ckpt_base_path=$PWD/checkpoints`
  → `..._ms1-4_lam0.1-0.2_ep3`.
- **Run 2 (single-scale paper)**: same minus the scales/lambdas → `..._ep3`.
- Eval each with `reproduce_table1.py <folder> --base $PWD/checkpoints/test`.
- **Caveats**: epoch-matched (not update-matched: single-scale ~186k vs multi-scale ~142k updates,
  since multi-scale windows are fewer/longer); judge on MPC (OL is short-horizon insensitive);
  single training seed (confirm any MPC gain with `training.seed=1`); watch B200 45GB OOM with
  decoder ON + 9-frame window (fallback `has_decoder=false` on BOTH, inert for results).
- **Prior s=4 result (has_decoder=false, λ₂=0.2, 3ep)**: OL 76.67±6.43, MPC 88.00±3.46 vs single-scale
  ✓ 2ep 75.33/82.00 and matched 3ep 70.67/84.67 — MPC gain shrinks to +3.33 (n.s.) once epoch-matched.

---

## 13. Seed methodology — SETTLED (see `PAPER_TABLE1_METHODOLOGY.md`)

**Canonical reference file: `PAPER_TABLE1_METHODOLOGY.md`** (grounded in `sec/1_main.tex`,
`sec/2_appendix.tex`, and `plan.py`, verified line-by-line). Consult it whenever the "how many
seeds / do we need more training seeds?" question comes up.

One-line: **the paper trains ONE model per Table-1 cell and reports mean ± std over THREE
DATA-SAMPLING seeds** (the `plan.py` `eval_seed = seed*n+1` that redraws the 50 test start/goal
pairs) — **NOT multiple training seeds.** No `training seed` / `retrain` / `independent runs`
language exists anywhere in the tex.

**Exact values (verified in code):** the 3 data seeds are **100, 200, 300** (`reproduce_table1.py`
L73 `SEEDS = [100, 200, 300]`); test tasks per seed = **50** (`n_evals: 50` in all `conf/plan_*.yaml`;
`plan.py` L287 loops 50) — matches the paper's "50 test samples". NOT 5 (the 5s are H=goal_H/frameskip
=25/5 and MPC's 5 executed actions, not the test-task count).

Consequences (do not re-litigate):
- Our protocol (train once at `training.seed=0`; eval data seeds 100/200/300; mean±std) **matches
  the paper exactly** → our multi-scale vs single-scale numbers (both 3 data seeds) are already a
  paper-consistent comparison. Multi-scale s=4 3ep = OL 76.67±6.43 / MPC 88.00±3.46 vs single-scale
  75.33/82.00 (2ep) and 70.67/84.67 (3ep matched).
- **Multiple training seeds are NOT required to be paper-faithful** (that's a higher bar than the
  paper). They're optional extra rigor only if a reviewer challenges the novel multi-scale claim.
- Corrects earlier repeated advice in this log that implied training-seed averaging was needed for
  "significance" — it is beyond the paper's own standard.

## 14. Iteration-matched paper baseline — `training.max_train_steps` knob (commit 209cafd)

**Goal.** Build the *iteration-matched* paper baseline for the multi-scale comparison: run the
paper's single-scale straightening (its exact 4-frame window + loss) for the **same number of
optimizer steps** as the multi-scale s=4 3-epoch run, so any success-rate difference is
attributable to the coarse-scale term, not to training length. Everything except the iteration
budget must match the paper.

### 14.1 The exact iteration count (derived + cross-checked)
- Window formula (`datasets/traj_dset.py`, §10.8): `windows/traj = max(0, T − num_frames·frameskip + 1)`.
- PushT training set = **18,685 rollouts** (confirmed in eval log: "Loaded 18685 rollouts"); all
  trajectories are ≥45 env-steps, so none are dropped at either window size.
- **Single-scale / paper** (`num_frames = num_hist+num_pred = 4`, frameskip 5 → 20-step window):
  `T−19` windows/traj → **61,929 iters/epoch** (authoritative, `REPRODUCTION.md`).
- **Multi-scale s=4** (`num_frames = 2·4+1 = 9`, frameskip 5 → 45-step window): `T−44` windows/traj.
  Loses exactly `44−19 = 25` windows/traj vs single-scale:
  - single total windows ≈ 61,929×32 = 1,981,728
  - multi  total windows ≈ 1,981,728 − 25×18,685 = 1,514,603 → /32 = **47,332 iters/epoch** (matches
    the figure recorded earlier — cross-check passes).
- ⇒ **Multi-scale 3 epochs = 47,332×3 = 141,996 steps** (this is "our loss in 3 epochs").
  Single-scale: 2 ep = 123,858; 3 ep = 185,787. So 141,996 = **2 full paper epochs + 18,138 steps
  into epoch 3** (= 2.293 epochs) → NOT an integer epoch count.

### 14.2 The trilemma (why a code knob was unavoidable)
You cannot have all three: (a) paper 4-frame window, (b) exactly 141,996 steps, (c) no code change —
because 141,996 is not a multiple of 61,929, so hitting it exactly means stopping **mid-epoch**,
which the epoch-driven loop can't do without a stop-at-N-steps knob. User chose to add the knob
(Option B). (Option A = bracket with 2ep/3ep baselines, already have it; Option C = force 9-frame
window on single-scale via `+env.dataset.num_frames=9` with epochs=3, no code but not the paper's
4-frame window. Both rejected in favor of the exact paper-window match.)

### 14.3 What was implemented — `training.max_train_steps` (commit 209cafd, pushed to main)
Minimal, backward-compatible, **default `null` = paper behavior unchanged** (2 files, +39 lines):
- `conf/train.yaml`: new `max_train_steps: null` under `training:` (with comment block).
- `train.py` constructor (~L75): `self.max_train_steps = int(_mts) if _mts is not None else None`,
  `self.global_step = 0`, `self._stop_training = False`.
- `train.py` `train()` batch loop end (~L744): after each optimizer step `self.global_step += 1`;
  if `max_train_steps is not None and global_step >= max_train_steps` → set `_stop_training=True`,
  log `Reached max_train_steps=... at epoch N (batch i)`, `break`.
- `train.py` `run()` epoch loop (~L554, L589): after the normal end-of-epoch `val()`+`save_ckpt()`,
  a force-save guard (only fires if `save_every_x_epoch>1`) + `if self._stop_training: break`.
- Flow when cap hits: finish current batch → break batch loop → run `val()` → `save_ckpt()` →
  break epoch loop. Checkpoint saved is the model at exactly 141,996 steps.
- `global_step` counts tqdm mini-batches (= optimizer steps; single GPU, gpu_batch_size=32 → 1
  batch = 1 step), the same unit as 47,332 / 61,929, so the match is exact in those units.
- NOTE: this is a DIFFERENT change from the earlier `straighten_window` edits, which were fully
  reverted and never committed. §14 is only the step-cap knob.

### 14.4 The run command (paper baseline via λ₂=0, matched to 141,996)
```bash
cd /workspace/arun/temporal_straightening_old
export DATASET_DIR=/workspace/arun/data
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
setsid nohup python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.epochs=3 env.num_workers=4 \
  'training.straighten_scales=[1,4]' 'training.straighten_lambdas=[0.1,0]' \
  training.max_train_steps=141996 \
  ckpt_base_path=$PWD/checkpoints_baseline_matched \
  > train_pusht_baseline_matched_141996.log 2>&1 < /dev/null &
tail -f train_pusht_baseline_matched_141996.log
```
- `straighten_lambdas=[0.1,0]` on `scales=[1,4]` = **bit-identical to paper single-scale**: scale 4
  (λ=0) is skipped in `total_curvature` AND doesn't widen the window (`needed=3 < base=4`), so window
  stays 4 frames and loss = `MSE + 0.1·L1`. Same as plain `straighten=aggcos1e-1` with no scales.
- All paper variables verified against `REPRODUCTION.md`/Table 3: encoder_lr 1e-5 (✓ straightening),
  predictor/proprio/action lr 5e-4, batch 32, num_hist 3, num_pred 1, frameskip 5, stop_grad True,
  bf16, seed 0, vcreg off, goal_weight 0. **Only epochs/iterations deviate (intended).**
- `env.num_workers=4` = non-paper DataLoader knob, results-neutral (`REPRODUCTION.md`).
- Shell exports matter: `DATASET_DIR` required; `WANDB_MODE`/`PYTORCH_CUDA_ALLOC_CONF` are
  logging/MIG-allocator only (no effect on numerics). MUJOCO_GL/EGL/PLAN_SERIAL_ENV are
  planning-only, not needed for training.

### 14.5 Save location (Hydra run dir from the tag resolvers)
```
$PWD/checkpoints_baseline_matched/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05_ms1-4_lam0.1-0_ep3/checkpoints/model_latest.pth
```
- `aggmlpcos1e-1` (agg_type=mlp folded into the straighten string), `_ms1-4_lam0.1-0` (straighten_tag),
  `_ep3` (run_variant_tag). `max_train_steps` is NOT encoded in the folder name → isolate it via its
  own `ckpt_base_path` root (done). `base_cell` strips `_ms/_lam/_ep` → maps to the PushT straighten
  cell (alpha=1, staged) for eval.
- Training log lands in the shell cwd: `.../temporal_straightening_old/train_pusht_baseline_matched_141996.log`.

### 14.6 Eval command
```bash
python reproduce_table1.py \
  pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05_ms1-4_lam0.1-0_ep3 \
  --base $PWD/checkpoints_baseline_matched/test
```

### 14.7 Verification checks (in the training log)
- Epoch-1 tqdm total ≈ **`/61929`** (paper 4-frame window; NOT 47,332 — confirms λ₂=0 reverted the window).
- Stop line: **`Reached max_train_steps=141996 at epoch 3 (batch 18137)`** (i is 0-based:
  2×61,929=123,858 done; epoch-3 stops when global_step=141,996 → i+1=18,138 → i=18137).

### 14.8 PITFALL hit on the pod (RESOLVED) — pull before running
Symptom: `Could not override 'training.max_train_steps'. Key 'max_train_steps' is not in struct`.
Cause: the pod was on old code (hadn't pulled 209cafd), so the yaml key + train.py logic weren't
present. Fix: `git pull` on the pod (`grep -n max_train_steps conf/train.yaml` must show
`max_train_steps: null`; `git log --oneline -1` = 209cafd), then the plain
`training.max_train_steps=141996` works. **Do NOT use the `+training.max_train_steps=...` append
workaround** — it lets the override through but the enforcement lives in `train.py`, so without the
pull the cap is silently ignored and it trains all 3 full epochs (185,787 steps). Both files must
update together via the pull.

## 15. DECISIVE RESULT — multi-scale gain does NOT survive iteration matching (PushT)

**The iteration-matched experiment from §14 is complete. It overturns the earlier apparent MPC gain.**

### 15.1 The numbers (all PushT ✓, single training run, mean±std over 3 data seeds 100/200/300, 50 tasks)
| Run | Steps | Window | Loss | Open-loop | MPC |
|---|---|---|---|---|---|
| Multi-scale s=4 3ep | 141,996 | 9-frame | MSE+0.1·L1+0.2·L4 | 76.67±6.43 (72,84,74) | 88.00±3.46 |
| **Iter-matched baseline** (`ms1-4_lam0.1-0_ep3`, cap 141996) | 141,996 | 4-frame (paper) | MSE+0.1·L1 | 74.67±6.43 (72,82,70) | **88.67±6.11 (94,82,90)** |
| Single-scale 2ep (paper setting) | ~124k | 4-frame | MSE+0.1·L1 | 75.33±6.11 | 82.00±2.00 |
| Single-scale 3ep | ~186k | 4-frame | MSE+0.1·L1 | 70.67±1.15 | 84.67±4.62 |
| Paper ✓ reported | ~124k | 4-frame | MSE+0.1·L1 | 77.33±6.18 | 85.33±4.99 |

### 15.2 Verdict — NULL result at matched compute
- **Both runs at EXACTLY 141,996 steps:** OL 76.67 vs 74.67 (+2.0, bands overlap → tie);
  MPC 88.00 vs **88.67 (baseline nominally HIGHER by 0.67)**, bands overlap → tie.
- ⇒ **The multi-scale coarse-scale (L4) term provides NO measurable benefit on PushT at matched
  training budget.** The earlier "+3.33 MPC gain" was an artifact of comparing against baselines at
  DIFFERENT (non-matched) budgets.

### 15.3 Why the earlier "gain" was not real (correction to §10/§14 optimism)
- Single-scale MPC across budgets is non-monotonic and noisy: 124k→82.00, **142k→88.67**, 186k→84.67.
  No trend; bounces ±3–4 pts. The matched baseline's own MPC seeds (94/82/90, ±6.11) span 12 points.
- ⇒ **single-training-run variance is LARGER than the effect we were chasing.** The bracketing
  argument (§ earlier: "88 beats 82 and 84.67") was inside that noise; the iteration-matched control
  exposed it. This is the value of having run the control — a trustworthy negative.

### 15.4 Implications
- As a POSITIVE claim ("multi-scale straightening helps planning"), **NOT supported** by PushT at
  matched compute. Do not frame it as an improvement; a reviewer running this exact control reaches
  the same null. Be honest about this.
- Reproduction itself is solid: BOTH runs land within the paper band (`[OK]` on OL and MPC vs
  77.33±6.18 / 85.33±4.99). It's the EXTENSION that shows no value, not the reproduction.
- One task, single training run each, high variance → a null here, not proof multi-scale can't help.
  Only credible next steps: (1) multiple TRAINING seeds (3+) per arm to actually measure variance
  (now clearly the bottleneck, not the loss term); (2) other envs (PointMaze/Wall, longer/curvier
  trajectories). Manage expectations: the clean single-task result is null.
- Master table: `results/table1_reproduction.md`/`.csv` now holds all 3 multi-scale/baseline rows.

## 16. Long-horizon evaluation of multi-scale (DESIGN + derivation; results pending)

**Goal.** Test whether the multi-scale coarse term helps where the theory says it should — LONG planning
horizons — using the already-trained checkpoints (NO retraining). At H=5 the two tied (§15); the
hypothesis is a gap opens as H grows.

### 16.1 Sweet-spot H derivation (from the loss + Thm 1)
- frameskip=5 → 1 model step = 5 env-steps. Loss scales s=1,4 (latent-frame gaps) → fine scale = 1
  model step (5 env-steps), coarse scale s=4 = **4 model steps (20 env-steps)**.
- Paper conditioning law κ_eff(H) ~ ρ^{2(H-1)}, ρ=(1+ε)/(1-ε). Single-scale shrinks ε over 1-step
  spans; the coarse term shrinks effective curvature over 4-step spans, so it only ACTS once the
  horizon traverses its reach:
  - H≤4: coarse term irrelevant → tie (matches observed H=5 tie).
  - H≈8–12 (=2–3× coarse reach): coarse term fully engaged → best chance of a gap. **Sweet spot.**
  - H≳15: model trained num_pred=1 → autoregressive rollout error dominates, washes out signal.
- **Sweet spot H* ≈ 8–12 → goal_H ≈ 40–60; primary bet H=10 (goal_H=50, the paper's long-horizon
  point).** Sweep H=5(anchor)/8/10/12/15 → goal_H 25/40/50/60/75 (all divisible by frameskip 5).
- CAVEAT: Thm 1 is proven for s=1 only; coarse-scale benefit is a heuristic extension, not guaranteed.

### 16.2 KEY metric choice — OPEN-LOOP, not MPC
The conditioning theorem is about optimizing a FULL length-H action sequence in one shot = exactly
open-loop GD (`GDPlanner` builds a length-`horizon` action tensor). MPC deliberately uses a short
5-step receding lookahead + replans, so it NEVER sees the long-horizon conditioning problem the loss
targets. ⇒ the effect (if real) shows up in OPEN-LOOP. Open-loop is also the cheap one (~1.5 min/seed).

### 16.3 Command mechanics (verified in plan.py / gd.py / mpc.py)
- `plan.py`: goal_H_model_steps = goal_H//frameskip (L180); n_taken_actions//=frameskip (L184);
  sub_planner.horizon//=frameskip (L186); self.planner.horizon set to goal_H_model_steps (L204).
- For a longer OPEN-LOOP horizon, scale all three together (env-step units): set
  `goal_H=G planner.sub_planner.horizon=G planner.n_taken_actions=G` (open-loop executes the full
  horizon). Output auto-separates per H: `plan_outputs_gd/<model>_gH<G>_dset/...` (goal_H in folder).
- Sweep = 2 models × {25,40,50,60,75} × seeds {100,200,300}. ~45–60 min for the open-loop pass.
- Watch the TREND: (multi-scale − baseline) success vs H. A monotone widening gap = real signal;
  flat gap = coarse term adds nothing. Single numbers are noise (§ power analysis).
- Checkpoints: baseline = checkpoints_baseline_matched/test/pusht_...ms1-4_lam0.1-0_ep3;
  multi-scale = checkpoints_s4/test/pusht_...ms1-4_lam0.1-0.2_ep3 (CONFIRM exact folder via
  `ls -d checkpoints*/test/pusht_*ep3`; the two model_names MUST differ or plan_outputs collide).
- MPC (staged) is the expensive optional follow-up (~12h full) ONLY if open-loop shows a widening gap.

### 16.4 Status: commands issued to user; awaiting open-loop sweep results.

## 17. Long-horizon boundary probe — FIRST on-hypothesis signal (open-loop, H=6/8)

**Setup.** Open-loop GD, PushT, pure spatial cost (mode=last, alpha=1), **n_evals=50 kept via new
`plan_chunk_size=10` chunking** (commit 4097eb7; each eval independent → identical to single batch,
just memory-bounded — fixed the 45GB MIG OOM at long horizon). Multi-scale (`ms1-4_lam0.1-0.2_ep3`,
in `checkpoints/`) vs iteration-matched single-scale baseline (`ms1-4_lam0.1-0_ep3`, in
`checkpoints_baseline_matched/`). 3 data seeds 100/200/300. Both models, both horizons complete.

### 17.1 Results (open-loop success %, mean±std over 3 data seeds)
| H (goal_H) | Multi-scale (+L4) | Baseline (L1 only) | gap (multi−base) |
|---|---|---|---|
| 5 (25) | 76.67±6.43 | 74.67±6.43 | +2.00 (tie) |
| 6 (30) | 56.67±5.77 (50/60/60) | 60.67±6.11 (66/62/54) | −4.00 (tie, within noise) |
| 8 (40) | **32.00±2.00 (34/32/30)** | **22.67±2.31 (24/20/24)** | **+9.33 (bands DON'T overlap)** |

### 17.2 Read
- **H=8 is the first result that is both on-hypothesis AND outside data-seed noise:** multi-scale
  degrades more gracefully at the horizon that stresses the model (multi [30,34] vs base [20,25],
  no overlap). Pooled 2-prop z≈1.8, p≈0.07. Consistent with "coarse-scale straightening helps
  long-horizon" (paper Thm-1 conditioning benefit compounds with horizon).
- **H=5, H=6 are ties** → NOT a clean monotone trend; signal appears only at H=8.
- **Collapse regime** (both <35% at H=8): the claim is *relative robustness / graceful
  degradation*, a modest on-theory finding, not "multi-scale wins planning".

### 17.3 Caveats (unchanged bottleneck)
- Still **n=1 training run** each; the tight ±2 bands are data-seed only, do NOT capture training
  variance. The +9.33 could be a lucky training draw.
- Horizons H≥8 are past the model's reliable rollout range (num_pred=1) → both collapse; this
  boundary (H=6–8) is the only place the coarse scale is active AND the model still half-works.
- Pure spatial cost (no `L_agg`); paper's long-horizon used `L_spatial+0.1·L_agg` (not implemented).

### 17.4 DECISIVE next step (now justified — there is a candidate signal)
Retrain BOTH models with **2–3 training seeds** (`training.seed=0,1,2`), evaluate at **H=8**
(+ H=5 control). If the +9 gap at H=8 holds across independent training runs → real finding:
*multi-scale straightening improves long-horizon planning robustness*. If it washes out → single-run
fluke, stop. (Optional: add `L_agg` long-horizon planning cost to lift both out of the collapse
regime for a healthier-absolute comparison; but the training-seed test is the decider.)

### 17.5 Tooling added
- `plan_chunk_size` (conf/plan_gd.yaml null default; plan.py wires onto planner+sub_planner;
  planning/gd.py `plan()` chunks then delegates to `_plan_batch()`). Commit 4097eb7, pushed.
- Peak mem at H=8 with chunk=10, n_evals=50 ≈ 27GB/45GB (was OOM). NVML-assert fix remains
  `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` on this DGX/MIG pod.

## 18. MULTI-SCALE CLOSED — dense 4-scale variant is WORSE (exit criterion met)

**Run:** `pusht_..._ms1-2-3-4_lam0.1-0.1-0.075-0.05_ep3` (own root `checkpoints_ms_4scale`).
scales=[1,2,3,4], lambdas=[0.1,0.1,0.075,0.05]; max scale 4 -> 9-frame window -> 47,332 iters/epoch
x 3 = **141,996 steps**, i.e. iteration-matched to §15/§17 runs. Eval: OL+MPC, H=5, 3 data seeds, 50 tasks.

### 18.1 Results at matched 141,996 steps (PushT ✓)
| Run | Loss | Open-loop | MPC |
|---|---|---|---|
| Baseline (paper loss) | MSE+0.1·L1 | **74.67±6.43** (72,82,70) | **88.67±6.11** (94,82,90) |
| s=4 multi-scale | +0.2·L4 | 76.67±6.43 (72,84,74) | 88.00±3.46 |
| **4-scale (new)** | +0.1·L2+0.075·L3+0.05·L4 | **68.67±7.02** (76,68,62) | **82.67±6.43** (90,80,78) |

Deltas of 4-scale: **vs baseline −6.00 OL / −6.00 MPC**; vs s=4 run −8.00 OL / −5.33 MPC.
→ **the dense 4-scale variant is the WORST configuration tested.**

### 18.2 Why this is more informative than the earlier ties (SIGN CONSISTENCY)
- s=4 run: +2.00 OL, −0.67 MPC → signs DISAGREE → noise scatter about zero (a tie).
- 4-scale run: −6.00 OL, −6.00 MPC → **both negative, equal magnitude** → directional evidence of
  mild real HARM. Individually z=1.15 (OL, p≈0.25) and z=1.48 (MPC, p≈0.14) — not individually
  significant, but sign-consistency across two metrics on the same model is hard to get from noise.

### 18.3 Two hypotheses REFUTED, one prediction CONFIRMED
- **CONFIRMED (over-regularization):** total λ = 0.1+0.1+0.075+0.05 = **0.325 = 3.25× the paper's
  validated 0.1**. Paper's App. ablation (smoothness/temporal-contrastive) states larger weights on
  temporal regularizers HURT. Exactly what happened — extra curvature pressure starves L_pred.
- **REFUTED (scale/horizon mismatch):** user's hypothesis was that s=4 failed because its 8-step
  reach overshoots H=5, and adding s=2 (4-step reach, INSIDE H=5, 5 triplets/window) would fix MPC.
  s=2 was included and MPC went DOWN 6 points. Horizon-matching did not rescue it.
- **REFUTED ("more scales = better global straightening"):** 2 scales → 4 scales made it worse.
  Consistent with telescoping redundancy (§ below): correlated constraints, no new information,
  plus added pressure and gradient noise.

### 18.4 The mechanism (canonical explanation for the whole null)
Telescoping identity (exact algebra): `v_t^(s) = v_t + v_{t+1} + ... + v_{t+s-1}`. The s=1 term already
aligns consecutive fine velocities; aligned vectors have aligned partial sums, so `L_curv^(s) ≈ 0`
for s>1 BEFORE λ_s acts. Coarse scales are **spanned by** the fine scale → little independent
gradient. Cross-check with paper Thm 1: the fine scale alone drives ε=||A−I||→0, so coarse terms
don't lower ε further → same conditioning → same success. Theory, algebra, and measurement agree.

### 18.5 STATUS: MULTI-SCALE ROUTE CLOSED (user's pre-committed exit criterion)
User's note: "If we don't see any improvement, we will completely abandon the multiscale route and
move to something more concrete and sensible." Scoreboard at matched compute: best case a TIE
(s=[1,4]), worst case mild HARM (s=[1,2,3,4]). ⇒ **Closed.** Outcome = a rigorous negative result
WITH a mechanism, which is a legitimate contribution (not a wasted effort).

Pending (optional, cheap) mechanistic confirmation: `grep -a "per-scale curvature" train_ms_s1234_ep3.log`
— if curv_s2/s3/s4 ≈ 0 while curv_s1 > 0, redundancy is confirmed empirically, not just algebraically.

### 18.6 NEXT DIRECTION (design rule derived from the failure)
A new term must attack a factor of the conditioning bound that straightening does NOT already
control, else it is redundant by construction. From Thm 1:
`kappa_eff(H) <= kappa(B)^2 · ((1+ε)/(1−ε))^(2(H−1))` — straightening only touches ε; **`kappa(B)^2`
(action→latent map conditioning) is UNTOUCHED and horizon-INDEPENDENT**, so improving it can help even
at H=5 where multi-scale had no room. Proposal (untested): action-isometry regularizer
`L_iso = ||J_a^T J_a − cI||_F^2`, `J_a = d(z_{t+1}−z_t)/da_t`, with a cheap 2-perturbation estimator
(no explicit Jacobian). Grounding: Isometric Autoencoders (arXiv 2006.09289), Kato et al.
(1910.04329), dynamical isometry (1711.04735). CAUTION cite: arXiv 2603.03238 reports geometry
regularizers can make latent-dynamics training harder for long rollouts. Full write-up:
`STUDY_PACKAGE/03_PROJECT_LATEST_MATH_AND_CODE.md` §8.

### 18.7 Deliverable created this session
`STUDY_PACKAGE/` (+ `STUDY_PACKAGE.zip`, 76KB): hallucination-guarded LLM tutor kit —
`00_TUTOR_SYSTEM_PROMPT.md`, `01_WORLD_MODELS.md`, `02_TEMPORAL_STRAIGHTENING.md`,
`03_PROJECT_LATEST_MATH_AND_CODE.md`, `04_ARCHITECTURE_DIAGRAMS.md`, `05_REFERENCES_VERIFIED.md`
(refs tagged [BIB]/[TEX]/[CODE]/[WEB]), plus bundled `paper_source/` LaTeX as ground truth.

## 19. CORRECTION — the REDUNDANCY hypothesis is EMPIRICALLY REFUTED (supersedes §18.4)

**Measured per-scale curvature** (4-scale run `ms1-2-3-4_lam0.1-0.1-0.075-0.05_ep3`, end of epoch 3,
from the logging added in commit 5fbc03c):

| scale | train L_curv^(s) | val L_curv^(s) | train ratio to s=1 |
|---|---|---|---|
| s=1 | 0.200932 | 0.335766 | 1.00x |
| s=2 | 0.202086 | 0.386792 | 1.01x |
| s=3 | 0.255287 | 0.494453 | **1.27x** |
| s=4 | 0.321961 | 0.597842 | **1.60x** (val 1.78x) |

**⇒ Coarse curvature is NOT ~0. It is MONOTONICALLY LARGER than fine curvature, on both train and
val.** The §18.4 / §3 prediction (`L_curv^(s) ≈ 0` for s>1 once s=1 is minimized) is **FALSE**.
**RETRACT the telescoping-redundancy explanation for the multi-scale null.**

In angles: cos = 1 − L, so 1-step velocities still turn ~37 deg (arccos 0.799) and 4-step block
velocities ~47 deg (arccos 0.678). The latent path stays substantially bent at EVERY scale.

### 19.1 Why the telescoping argument failed (keep this reasoning)
The identity `v_t^(s) = v_t + ... + v_{t+s-1}` is correct. The bad inference was "consecutive
velocities aligned ⇒ block sums aligned." That needs NEAR-PERFECT alignment; with a residual ~37 deg
turn per step, small turns **accumulate** across a block.
**Killer counterexample: a CIRCLE.** Every small arc looks nearly straight (small fine curvature) yet
a quarter circle rotates direction by 90 deg (large coarse curvature). Locally straight does NOT imply
globally straight. PushT (contact + T-block rotation) is exactly such a gently-but-persistently
curving regime. ⇒ **coarse scales DO carry independent information.**

### 19.2 The reopened question: if not redundant, why did the extra terms HURT (-6.00 OL / -6.00 MPC)?
Three candidates; with n=1 training run per config they CANNOT be separated:
1. **(Leading) The coarse curvature is genuine, irreducible structure.** If PushT trajectories really
   curve over 8 steps, forcing them straight fights the true dynamics -> capacity spent on a
   geometrically wrong constraint -> prediction degrades -> planning degrades. Echoes the paper's
   App. finding that over-strong smoothness / temporal-contrastive terms hurt.
2. **Over-regularization, WORSE than previously estimated.** The earlier claim "redundant terms sit
   near floor so gradients are small, so risk is modest" is ALSO retracted — the terms are at
   0.20-0.32 and exert REAL pressure. Total lambda 0.325 = 3.25x paper of genuinely-active pressure
   starves L_pred.
   COUNTER-EVIDENCE: the s=4-only run had total lambda 0.3 (nearly the same) yet TIED instead of
   dropping 6 pts -> total lambda alone does not explain the difference between the two runs.
3. **Unmeasured training-run variance.** Neither drop is individually significant (z=1.15 OL,
   z=1.48 MPC). Training variance is unestimated (n=1, bf16, no determinism flags) — a live candidate.

### 19.3 Net status after the correction
- **STANDS:** multi-scale does not help empirically (2 configs at matched 141,996 steps: one tie, one
  6 pts worse). The route remains CLOSED per the user's exit criterion.
- **NEW POSITIVE FINDING (measured, novel):** latent curvature **increases with temporal scale**, and
  fine-scale straightening does **not** remove coarse-scale curvature (train 0.20 -> 0.32; val
  0.34 -> 0.60). This is a genuine quantitative statement about the learned geometry and is the
  OPPOSITE of the redundancy prediction. Worth reporting in any writeup.
- **RETRACTED:** redundancy/telescoping as the mechanism (§18.4, and §3 of
  `STUDY_PACKAGE/03_PROJECT_LATEST_MATH_AND_CODE.md` — both need correcting).
- **OPEN:** the true mechanism for the harm.

### 19.4 Optional follow-up (NOT recommended without appetite; user exit criterion already met)
Because harm may be over-regularization rather than redundancy, a gentle multi-scale run keeping the
PAPER's total pressure (e.g. lambdas summing to ~0.1, say [0.05,0.025,0.015,0.01]) is the one
untested variant that could distinguish hypothesis 1 from 2. If coarse curvature is irreducible
structure (h1), gentle pressure still shouldn't help; if it was pure over-regularization (h2), the
tie/harm should disappear. Cost: ~14h train + ~1.5h eval, still n=1.

### 19.5 Lesson for the loop
The ~10-line per-scale logging diagnostic (commit 5fbc03c) **overturned a mechanistic story that had
been asserted confidently across several turns.** Measure the intermediate quantity before building an
explanation on it. Where a hypothesis is cheap to instrument, instrument it FIRST.
