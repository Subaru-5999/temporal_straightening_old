# B200 Pod Setup & Reproduction Log

Chronological record of the commands run on the NVIDIA B200 pod to get
`temporal-straightening` training + planning working, and to reproduce the
PointMaze-UMaze baseline cell of Table 1.

**Environment:** NGC PyTorch container (system Python 3.10, **no conda**),
NVIDIA B200 MIG `1g.45gb` slice, driver 570.124.06 / CUDA 12.8.
**Project:** `/workspace/arun/temporal-straightening` · **Data:** `/workspace/arun/data`

---

## Phase 0 — Diagnostics

```bash
# hardware / limits
nproc                          # 224 cores
free -h                        # 2.0 TiB RAM
df -h /workspace               # overlay, ~535 GB free (ephemeral container FS)
df -h /dev/shm                 # 32 GB
nvidia-smi                     # B200, MIG GI/CI, ~45 GB, no processes
ulimit -a                      # open files 1M, memlock 8 MB

# what Python tooling exists (found: NO conda, system python 3.10, NGC torch 2.3)
which conda mamba micromamba python python3 pip pip3
ls -la /opt ~
find / -maxdepth 6 -type f \( -name conda -o -name mamba -o -name micromamba \) 2>/dev/null

# does the container's torch support the B200? (NO -> torch 2.3, max sm_90)
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))"
```

Key findings: 224 cores / 2 TB RAM (host OOM not a concern), 32 GB `/dev/shm`,
`/workspace` on an **ephemeral overlay**, and the NGC-bundled **torch 2.3 does not
support Blackwell (sm_100)**.

---

## Phase 1 — Install Blackwell-capable PyTorch

Installed straight into the container's system Python (note: not isolated in a venv).

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0

# verify a real kernel launches on the B200
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0)); x=torch.randn(1024,device='cuda'); print('kernel ok', float((x+1).sum()))"
# -> 2.7.0+cu128 12.8 (10, 0)  kernel ok
```

---

## Phase 2 — Project code + training dependencies

```bash
cd /workspace/arun/temporal-straightening
git pull                                   # bring in setup scripts, configs, requirements

pip install -r requirements-train.txt      # hydra, omegaconf, accelerate, einops, decord, wandb, submitit

# accelerate crashed importing NGC's transformer_engine (built for torch 2.3 -> ABI break).
# We don't use it; remove it.
pip uninstall -y transformer-engine transformer_engine
python3 -c "from accelerate import Accelerator; print('accelerate ok')"
```

---

## Phase 3 — Train the baseline (DINOv2 patch, straighten=False, UMaze)

```bash
export DATASET_DIR=/workspace/arun/data
export WANDB_MODE=offline

# detached launch (survives disconnects); runs the full 20-epoch baseline
bash run_train.sh
tail -f train_*.log

# (optional) chain the next experiment queue to auto-start after the baseline.
# NOTE: this PID-watch approach later proved unreliable (PID reuse) -- prefer just
# running `bash run_experiments.sh` directly once the GPU is free.
nohup bash -c 'while kill -0 $(cat .train_pid) 2>/dev/null; do sleep 60; done; bash run_experiments.sh' > chain.log 2>&1 &
```

Result: baseline trained all 20 epochs ->
`checkpoints/test/umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05/checkpoints/model_20.pth`.

---

## Phase 4 — Planning / simulator setup (the build-risky part)

```bash
# MuJoCo 210 + apt libs + gym/mujoco-py + PushT extras
bash setup_planning.sh
# -> mujoco-py compiled OK, gym installed. The d4rl step (git clone) STALLED on this
#    pod's GitHub connection (GnuTLS/pack disconnect).

# kill the stuck experiment chain (PID reuse left it waiting forever)
kill 2047

# install d4rl via TARBALL instead of git (avoids the stalling pack protocol),
# without its heavy optional deps, and verify the whole planning import chain:
cd /workspace/arun/temporal-straightening \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia \
 && export D4RL_SUPPRESS_IMPORT_ERROR=1 \
 && rm -rf /tmp/d4rl /tmp/d4rl.tar.gz \
 && wget -O /tmp/d4rl.tar.gz https://github.com/Farama-Foundation/d4rl/archive/refs/heads/master.tar.gz \
 && mkdir -p /tmp/d4rl && tar -xzf /tmp/d4rl.tar.gz -C /tmp/d4rl --strip-components=1 \
 && pip install --no-deps -e /tmp/d4rl

# d4rl needed h5py (skipped by --no-deps); install it, then full verify
pip install h5py
python -c "import gym, mujoco_py, d4rl, env; print('PLANNING IMPORTS OK')"

# persist the d4rl flag for future shells
grep -qsF 'export D4RL_SUPPRESS_IMPORT_ERROR=1' ~/.bashrc || echo 'export D4RL_SUPPRESS_IMPORT_ERROR=1' >> ~/.bashrc

# ffmpeg backend so the planner can write .mp4 rollout videos
pip install imageio-ffmpeg
```

Also required in each planning shell (added into `evaluate.sh`):
```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl        # headless GL for offscreen rendering
```

---

## Phase 5 — Evaluate the baseline (get Table-1 numbers)

```bash
git pull        # pull code fixes (see below)

bash evaluate.sh /workspace/arun/temporal-straightening/checkpoints/test/umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05
```

### Code fixes made during this phase (already in the repo via `git pull`)
- **plan path:** `evaluate.sh` now passes the run dir as an absolute `ckpt_base_path`
  (plan.py uses an absolute path as the model folder directly).
- **torch>=2.6 checkpoint load:** `plan.py` / `train.py` `torch.load(..., weights_only=False)`
  (checkpoints store full nn.Module objects, not just state-dicts).
- **video writing:** added `imageio-ffmpeg`.

### Results so far
- **Open-loop (plan_gd): success_rate = 0.40 (40%)** — reproduces the paper baseline
  (DINOv2 patch, UMaze, open-loop = 35.33 ± 4.11). ✅
- **MPC (plan_gd_mpc):** was climbing (0.48 at step 4 of 20) then crashed with a
  **MIG-specific PyTorch allocator error** (`NVML_SUCCESS == r ... CUDACachingAllocator`).

---

## Pending next step (NOT yet run)

Fix the MPC crash — torch 2.7's `expandable_segments` allocator uses NVML/VMM APIs
that fail on MIG. `evaluate.sh` now sets this automatically; to re-run MPC:

```bash
git pull
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
bash evaluate.sh /workspace/arun/temporal-straightening/checkpoints/test/umaze_False_agg32_projnone_dim384_hw14_sgTrue_lr1e-05

# then aggregate into the results table
python collect_results.py
```

After the baseline is fully evaluated, launch the straightening comparison:
```bash
bash run_experiments.sh      # dino_channel  straighten=False  vs  aggcos1e-1  (UMaze)
```


---

## CORRECTION (from paper 2603.12231v2, Table 3 footnote)

**Mistake found:** the ✗ (no-straightening) `dino_channel` run was trained with
`encoder_lr=1e-5`. The paper's Table 3 footnote says trainable projectors/ResNets
**"use lr = 1e-6 for no straightening"** (higher lr severely degrades ✗ models).

**Symptom:** ✗ open-loop reproduced at only ~20% (3 planning seeds: 0.14/0.30/0.16)
vs paper's 44.00 ± 7.12. MPC still matched (80% vs 81.33) because MPC feedback masks
the degraded representation; open-loop is sensitive and exposed it. The plain
`encoder=dino` baseline was unaffected (no trainable projector -> encoder_lr irrelevant),
which is why it reproduced fine (40% vs 35.33).

**Correct learning rates (trainable-projector configs: dino_channel, scratch_resnet*):**
| straighten | encoder_lr |
|---|---|
| False (✗) | **1e-6** |
| cos1e-1 / aggcos1e-1 (✓) | 1e-5 |

**Fix — retrain the ✗ row at 1e-6 (new folder, no clash with the wrong lr1e-05 run):**
```bash
export DATASET_DIR=/workspace/arun/data WANDB_MODE=offline
nohup python train.py --config-name train.yaml env=point_maze \
  encoder=dino_channel training.straighten=False training.encoder_lr=1e-6 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro \
  > train_dino_channel_off_lr1e6.log 2>&1 &
```
**✓ counterpart (keeps 1e-5):**
```bash
python train.py --config-name train.yaml env=point_maze \
  encoder=dino_channel training.straighten=aggcos1e-1 training.encoder_lr=1e-5 \
  ckpt_base_path=/workspace/arun/temporal-straightening/checkpoints/repro
```

**Verified matches the paper (no change needed):** planning hyperparams (Table 4:
horizon 25, zero init, lr 0.1, 100 opt steps), open-loop = terminal MSE, batch 32,
history 3, frameskip 5, 20 epochs, predictor/action lr 5e-4, UMaze = 2000-traj
`point_maze` dataset, H = 5 model steps.
