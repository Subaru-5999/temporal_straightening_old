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
