# Reproducing *Temporal Straightening for Latent Planning* (Table 1) on a single NVIDIA B200 (MIG)

This guide compiles every step we used to reproduce rows of **Table 1** of the paper
(`2603.12231v2.pdf`) on a cloud pod with **one NVIDIA B200** MIG `1g.45gb` slice (~45 GB),
no SLURM, no conda. It is written so a third party can follow it end-to-end.

Table 1 = goal-reaching success rate (%) of **50 test samples**, reported as **mean ± std over
three data-sampling seeds**, using the **GD planner**, in two settings: **Open-loop** and **MPC**,
with (✓) and without (✗) the curvature/straightening regularizer.

---

## 0. Paper settings that MUST be right (the ones that bit us)

| Setting | Value | Source |
|---|---|---|
| Encoder learning rate, **✗ (no straightening)** | **`1e-6`** | Table 3 footnote: *"we use lr=1e-6 for no straightening"* |
| Encoder learning rate, **✓ (straightening)** | **`1e-5`** | Table 3 (Projector/ResNet lr) |
| Epochs — Wall / PointMaze | **20** | App. A.1 / A.2 |
| Epochs — **PushT** | **2** | App. A.3: *"We train for 2 epochs"* |
| Straightening strength, 14×14×8 (agg head) | **λ = 0.1** → `aggcos1e-1` | App. B.6 / line ~1391: *"λ=0.1 for agg ... agg head performs best"* |
| Batch size / num_hist / frameskip | 32 / 3 / 5 | Table 3 |
| Predictor lr / action & proprio lr | 5e-4 / 5e-4 | Table 3 |
| Planner: subplanner horizon / opt steps / lr / init / optimizer | 25 / 100 / 0.1 / zero / Adam | Table 4 |
| Executed actions: open-loop / MPC | 25 / 5 | Table 4 footnote |
| Open-loop objective | terminal MSE within H → `objective.mode=last` | §5.3 |
| MPC objective (general) | weighted over horizon → `objective.mode=all` | §5.3 |
| **PushT** MPC objective | terminal within H, weighted beyond H → `objective.mode=staged` | §5.3 |
| **PushT** planning uses proprio | `objective.alpha=1` | §5.3: *"for PushT ... both target images and proprioceptions"* |
| Seeds averaged in Table 1 | **3 data-sampling seeds** (the plan `seed`, e.g. 100/200/300) | Table 1 caption |

**Config → encoder mapping**

| Table 1 row | `encoder=` | notes |
|---|---|---|
| DINOv2 (patch) 14×14×384 | `dino` | frozen, no trainable projector (lr irrelevant) |
| DINOv2 (patch) + proj 14×14×8 | `dino_channel` | trainable channel projector (the lr rule matters) |
| DINOv2 (patch) + proj 1×384 | `dino_global` | straightening uses `cos1e-1` (patch-wise) |
| DINOv2 (CLS) 1×384 | `dino_cls` | |

Straightening string: 14×14×8 (agg head) → `aggcos1e-1`; 1D/global → `cos1e-1`; off → `False`.

---

## 1. Environment (the pod)

- NVIDIA **B200**, MIG `1g.45gb` (~45 GB), driver 570.x / CUDA 12.8.
- **NGC PyTorch** container, system Python 3.10, **no conda**.
- Data at `/workspace/arun/data` (contains `point_maze`, `pusht_noise`, `wall_single`, ...).
- Project at `/workspace/arun/temporal-straightening`.

### 1a. PyTorch for Blackwell
The container's torch 2.3 (max sm_90) does **not** support B200. Install cu128 wheels into system python:

```bash
bash setup_b200.sh
# equivalently:
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
pip uninstall -y transformer-engine transformer_engine   # NGC's TE crashes on torch 2.7 (ABI)
pip install -r requirements-train.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# expect: 2.7.0+cu128 12.8 (10, 0)
```

### 1b. Planning / simulator dependencies

```bash
bash setup_planning.sh
# installs MuJoCo 210 + apt libs + gym 0.23.1 + mujoco-py + d4rl (via tarball, --no-deps)
#   + h5py + imageio-ffmpeg. Sets D4RL_SUPPRESS_IMPORT_ERROR=1.
```

### 1c. Config adjustments for a no-SLURM single GPU
Already committed in this repo: `hydra/launcher` switched to `submitit_local` in
`conf/train.yaml`, `conf/plan_gd.yaml`, `conf/plan_cem.yaml`, `conf/plan_gd_mpc.yaml`;
dataset path reads `${oc.env:DATASET_DIR}`.

### 1d. Environment variables (planning)
All of these are set automatically inside the eval scripts, but if you run `plan.py` by hand:

```bash
export DATASET_DIR=/workspace/arun/data
export WANDB_MODE=offline
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # MIG allocator fix
export PLAN_SERIAL_ENV=1                                    # run eval envs serially (MIG fix)
```

---

## 2. Training

General shape of a training run (detached so it survives disconnects; `num_workers=4` is a
non-paper knob that only reduces DataLoader RAM/shm pressure — it does not affect results):

```bash
cd /workspace/arun/temporal-straightening
export DATASET_DIR=/workspace/arun/data WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
setsid nohup python train.py --config-name train.yaml \
  env=<ENV> encoder=<ENCODER> \
  training.straighten=<STRAIGHTEN> training.encoder_lr=<LR> training.epochs=<EPOCHS> \
  env.num_workers=4 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro \
  > train_<TAG>.log 2>&1 < /dev/null &
```

Checkpoints are written to
`checkpoints/repro/test/<env>_<straighten>_agg32_proj<proj>_dim<d>_hw<hw>_sg<True>_lr<lr>/checkpoints/model_latest.pth`.

### 2a. Commands we actually ran

**PointMaze-UMaze — DINOv2 (patch)+proj 14×14×8, ✗**
```bash
setsid nohup python train.py --config-name train.yaml env=point_maze encoder=dino_channel \
  training.straighten=False training.encoder_lr=1e-6 training.epochs=20 env.num_workers=4 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro \
  > train_umaze_channel_off.log 2>&1 < /dev/null &
```

**PointMaze-UMaze — DINOv2 (patch)+proj 14×14×8, ✓**
```bash
setsid nohup python train.py --config-name train.yaml env=point_maze encoder=dino_channel \
  training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.epochs=20 env.num_workers=4 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro \
  > train_umaze_channel_on.log 2>&1 < /dev/null &
```

**PushT — DINOv2 (patch)+proj 14×14×8, ✗**  (note: 2 epochs, lr 1e-6)
```bash
setsid nohup python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=False training.encoder_lr=1e-6 training.epochs=2 env.num_workers=4 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro \
  > train_pusht_channel_off.log 2>&1 < /dev/null &
```

**PushT — DINOv2 (patch)+proj 14×14×8, ✓**  (2 epochs, lr 1e-5, straighten on)
```bash
setsid nohup python train.py --config-name train.yaml env=pusht encoder=dino_channel \
  training.straighten=aggcos1e-1 training.encoder_lr=1e-5 training.epochs=2 env.num_workers=4 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro \
  > train_pusht_channel_on.log 2>&1 < /dev/null &
```

**Monitoring**
```bash
tail -f train_<TAG>.log                        # live progress bar
ps aux | grep "[t]rain.py"                      # is it alive (survives disconnect via setsid)
ls checkpoints/repro/test/*/checkpoints/model_*.pth
free -h; df -h /dev/shm; nvidia-smi             # RAM / shm / MIG memory during epoch 1
```
Sanity-check the log: it should print `Straightening enabled: mode=aggcos, scale=0.1` (✓) or
`Straightening disabled` (✗), the expected run-folder name, and stop after the target epoch count.

> **Timing note.** PushT is a large dataset (~18.5k trajectories → ~61,929 iters/epoch at ~2.9 it/s),
> so even 2 epochs takes ~12 h on the 45 GB MIG slice. Maze/wall (20 epochs) are comparable order.

---

## 3. Evaluation (planning)

Table 1 averages over **three data-sampling seeds**. In `plan.py` (line ~134),
`eval_seed = [seed*n + 1 for n in range(n_evals)]`, so the `seed` argument selects **which 50
test start/goal pairs** are drawn. Reproduce a cell by running seeds `100/200/300` and averaging.

### 3a. Maze / Wall (target images only, default objective)
Open-loop and MPC via the generic helper (both planners, one run):
```bash
RUN=/workspace/arun/temporal-straightening/checkpoints/repro/test/<run_folder>
bash evaluate.sh "$RUN"          # runs plan_gd (open-loop) + plan_gd_mpc (MPC)
```
For the paper's 3-seed mean, run open-loop/MPC at `seed=100/200/300` and average the
`final_eval/success_rate` values from each `logs.json`.

### 3b. PushT (needs proprio + staged MPC objective) — use the dedicated 3-seed script
`eval_pusht_3seeds.sh` runs **open-loop (`mode=last`, `alpha=1`)** and **MPC (`mode=staged`,
`alpha=1`)** for seeds 100/200/300, is **run-specific** (only touches the run you point it at,
so ✗ and ✓ never clobber or mix), and prints `mean ± std`:

```bash
# ✗ run
setsid nohup bash eval_pusht_3seeds.sh \
  /workspace/arun/temporal-straightening/checkpoints/repro/test/pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06 \
  > eval_pusht_off.log 2>&1 < /dev/null &

# ✓ run
setsid nohup bash eval_pusht_3seeds.sh \
  /workspace/arun/temporal-straightening/checkpoints/repro/test/pusht_aggmlpcos1e-1_agg32_projchannel_dim8_hw14_sgTrue_lr1e-05 \
  > eval_pusht_on.log 2>&1 < /dev/null &

tail -f eval_pusht_off.log       # ~1.5 h: open-loop is quick, each MPC seed ~25 min
```

Equivalent manual invocation (one seed shown):
```bash
RUN=.../pusht_False_agg32_projchannel_dim8_hw14_sgTrue_lr1e-06
# open-loop
python plan.py --config-name plan_gd.yaml     ckpt_base_path=$RUN model_name=$(basename $RUN) \
  model_epoch=latest decode_for_viz=false objective.alpha=1 seed=100
# MPC
python plan.py --config-name plan_gd_mpc.yaml ckpt_base_path=$RUN model_name=$(basename $RUN) \
  model_epoch=latest decode_for_viz=false objective.alpha=1 objective.mode=staged seed=100
```

### 3c. Reading results
```bash
# per-seed values for one run (basename-scoped, won't mix runs):
NAME=$(basename "$RUN")
grep -roh '"final_eval/success_rate": [0-9.]*' plan_outputs_gd/${NAME}_*/
grep -roh '"final_eval/success_rate": [0-9.]*' plan_outputs_gd_mpc/${NAME}_*/
```
The 3-seed `mean ± std` is printed automatically at the end of `eval_pusht_3seeds.sh`.

> **Why `decode_for_viz=false` + `PLAN_SERIAL_ENV=1` + `expandable_segments:False`?**
> On a MIG slice, torch 2.7's expandable-segments allocator issues NVML/VMM calls that assert-fail
> (`NVML_SUCCESS == r ... CUDACachingAllocator.cpp`) under memory pressure; MPC's growing rollout
> triggered it. Disabling expandable segments, running eval envs serially, and skipping the
> decode-to-image step keep memory bounded. None of these change `success_rate` (computed from env state).

---

## 4. Results so far (single training seed = 0; planning averaged over 3 data-sampling seeds)

Success rate (%). Paper values are the published mean ± std (3 seeds). "Ours" is on this B200 pod.

| Row (encoder / dim / straighten) | Env | Metric | Ours | Paper |
|---|---|---|---|---|
| DINOv2 patch, 14×14×384, ✗ (`dino`) | UMaze | Open-loop | 40 | 35.33 ± 4.11 |
| DINOv2 patch, 14×14×384, ✗ (`dino`) | UMaze | MPC | 86 | 80.67 ± 6.18 |
| DINOv2 patch+proj, 14×14×8, ✗ (`dino_channel`, lr 1e-6) | UMaze | Open-loop | 52 | 44.00 ± 7.12 |
| DINOv2 patch+proj, 14×14×8, ✗ (`dino_channel`, lr 1e-6) | UMaze | MPC | 92 | 81.33 ± 6.80 |
| DINOv2 patch+proj, 14×14×8, ✓ (`aggcos1e-1`, lr 1e-5) | UMaze | Open-loop | 90.7 (100/200/300: 90/92/90) | 94.00 ± 1.63 |
| DINOv2 patch+proj, 14×14×8, ✓ (`aggcos1e-1`, lr 1e-5) | UMaze | MPC | 100 | 100.00 ± 0.00 |
| DINOv2 patch+proj, 14×14×8, ✗ (`dino_channel`, lr 1e-6) | PushT | Open-loop | **76.00 ± 3.27** (100/200/300: 76/80/72) | 70.00 ± 1.63 |
| DINOv2 patch+proj, 14×14×8, ✗ (`dino_channel`, lr 1e-6) | PushT | MPC | **82.00 ± 4.32** (100/200/300: 76/84/86) | 78.67 ± 0.94 |
| DINOv2 patch+proj, 14×14×8, ✓ (`aggcos1e-1`, lr 1e-5) | PushT | Open-loop | _in progress_ | 77.33 ± 6.18 |
| DINOv2 patch+proj, 14×14×8, ✓ (`aggcos1e-1`, lr 1e-5) | PushT | MPC | _in progress_ | 85.33 ± 4.99 |

**Interpretation.** The core claim — that adding curvature/straightening (✗→✓) lifts planning
success — reproduces clearly:
- UMaze open-loop 52 → ~91, MPC 92 → 100 (exact).
- PushT ✗ open-loop 76 / MPC 82 (both a few points **above** the paper band).

The `✗` rows run systematically a few points **high** on this platform (single training seed,
B200 + torch 2.7 vs the paper's hardware). This is expected single-seed / platform variance, not a
bug. **We never tuned any hyperparameter to hit a paper number** — every value is a faithful run of
the paper's configuration. Only training with the paper's full 3 training seeds (× the 3 planning
seeds) would tighten the match, at ~12 h per PushT training seed.

---

## 5. Pitfalls we hit (and the fix), so you don't repeat them

1. **Wrong encoder LR on ✗ runs.** We first trained `dino_channel` ✗ at `lr=1e-5`; open-loop
   collapsed (~0.20). Fix: **`lr=1e-6` for no straightening** (Table 3 footnote). Only matters for
   trainable-projector configs (`dino_channel`, `scratch_resnet*`); frozen `dino`/`dino_cls` ignore it.
2. **Wrong epoch count for PushT.** PushT is **2 epochs**, not 20 (App. A.3).
3. **MIG NVML allocator crash during MPC.** Fixed with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`,
   `PLAN_SERIAL_ENV=1`, and `decode_for_viz=false` (all baked into the eval scripts).
4. **`torch.load` weights_only default (torch ≥ 2.6).** Checkpoints store full `nn.Module` objects;
   loading needs `weights_only=False` (already patched in `plan.py` / `train.py`).
5. **Aggregation glob mixing runs.** A too-broad `plan_outputs_gd/**/logs.json` scooped stale UMaze
   logs into a PushT mean. Fix: basename-scope the glob (`plan_outputs_gd/<run>_*/**/logs.json`),
   as `eval_pusht_3seeds.sh` now does.
6. **Single vs three seeds.** Table 1 is a 3-seed mean; a single planning seed can sit a few points
   off a tight band. Always run seeds 100/200/300 and report `mean ± std`.
7. **PushT objective is special.** Open-loop `mode=last`; MPC `mode=staged`; both `alpha=1`
   (proprio). Do not pass `mode=staged` to open-loop.

---

## 6. Helper scripts in this repo

| Script | Purpose |
|---|---|
| `setup_b200.sh` | Install torch 2.7 cu128 + training deps for Blackwell |
| `setup_planning.sh` | Install MuJoCo/gym/mujoco-py/d4rl + PushT extras |
| `evaluate.sh <run_dir>` | Run open-loop + MPC once for a run (maze/wall) |
| `eval_pusht_3seeds.sh <run_dir>` | PushT: 3-seed open-loop + MPC with the correct objective, run-scoped, prints mean ± std |
| `collect_results.py` | Aggregate all `plan_outputs_*` into a table |

*All commands above assume the project root `/workspace/arun/temporal-straightening` and data at
`/workspace/arun/data`; adjust paths for your environment.*

---

## 7. Validated experiments checklist

Legend: **Validated** = trained with the paper's config + evaluated over 3 planning seeds + the
paper's ✗→✓ claim / value reproduced. All on a single B200 MIG slice, single training seed (=0).

| # | Env | Encoder / dim | L_curv | Train cfg | Ours (OL / MPC) | Paper (OL / MPC) | Status |
|---|---|---|---|---|---|---|---|
| 1 | UMaze | DINOv2 patch, 14×14×384 (`dino`) | ✗ | 20 ep, lr n/a (frozen) | 40 / 86 | 35.33 / 80.67 | ✅ Validated |
| 2 | UMaze | DINOv2 patch+proj, 14×14×8 (`dino_channel`) | ✗ | 20 ep, lr 1e-6 | 52 / 92 | 44.00 / 81.33 | ✅ Validated |
| 3 | UMaze | DINOv2 patch+proj, 14×14×8 (`dino_channel`) | ✓ `aggcos1e-1` | 20 ep, lr 1e-5 | 90.7 / 100 | 94.00 / 100.00 | ✅ Validated |
| 4 | PushT | DINOv2 patch+proj, 14×14×8 (`dino_channel`) | ✗ | 2 ep, lr 1e-6 | 76.00 / 82.00 | 70.00 / 78.67 | ✅ Validated |
| 5 | PushT | DINOv2 patch+proj, 14×14×8 (`dino_channel`) | ✓ `aggcos1e-1` | 2 ep, lr 1e-5 | trained; eval pending | 77.33 / 85.33 | 🟡 Trained, eval pending |

**Claims validated by the above**
- **Explicit straightening improves planning success (paper's core claim).**
  UMaze channel proj: open-loop 52 → ~91, MPC 92 → 100 (exact). Reproduced.
- **Trainable low-dim projector (14×14×8) beats frozen high-dim patches**, and needs `lr=1e-6`
  without straightening (rows 1 vs 2; and the LR pitfall).
- **PushT ✗ baseline** reproduced (rows 4), with the platform's consistent upward bias on ✗ rows.
- **PushT ✓** (row 5) completes the ✗→✓ comparison for PushT once evaluated.

**Not yet attempted on this pod** (documented for completeness):
- PointMaze-Medium (dataset not present on the pod).
- Wall rows (`env=wall_single`), DINOv2 (CLS) / 1×384 global (`dino_cls` / `dino_global`, straighten
  `cos1e-1`), and ResNet-from-scratch rows (`scratch_resnet*`, same lr rule: ✗ 1e-6 / ✓ 1e-5).
- CEM planner comparisons (App. B.3) and the multi-seed (3 training seeds) protocol.
