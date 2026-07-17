#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_b200.sh
# One-shot bootstrap for this project on the NVIDIA B200 pod.
#
#   GPU     : NVIDIA B200 (Blackwell, sm_100), MIG ~45 GB
#   Driver  : 570.124.06  /  CUDA driver 12.8
#   Base    : NGC PyTorch container, system Python 3.10 (NO conda)
#   Project : /workspace/arun/temporal-straightening
#   Data    : /workspace/arun/data   (point_maze at /workspace/arun/data/point_maze)
#
# The container ships torch 2.3 (CUDA 12.3, max sm_90) which does NOT support the
# B200. We run inside a deletable venv and install Blackwell torch (cu128) there.
#
# Usage:
#   python3 -m venv --system-site-packages /workspace/arun/envs/ts_b200
#   source /workspace/arun/envs/ts_b200/bin/activate
#   cd /workspace/arun/temporal-straightening
#   bash setup_b200.sh
#
# Override the data root if yours differs:
#   DATASET_ROOT=/some/other/data bash setup_b200.sh
# ---------------------------------------------------------------------------
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/workspace/arun/data}"

# Safety: make sure we're inside a venv so installs don't touch the container's
# system packages (keeps this fully deletable via `rm -rf` of the venv folder).
if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "WARNING: no active venv detected (\$VIRTUAL_ENV empty)."
  echo "Recommended: python3 -m venv --system-site-packages /workspace/arun/envs/ts_b200 && source .../bin/activate"
  echo "Continuing in 5s (Ctrl-C to abort)..."; sleep 5
fi

echo "==> [1/6] Clearing any torch already in the venv (clean slate)..."
pip uninstall -y torch torchvision triton || true

echo "==> [2/6] Installing Blackwell-capable PyTorch (CUDA 12.8 build)..."
# torch 2.7.x is the first release with official Blackwell (sm_100) support.
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0

echo "==> [3/6] Installing training-tier project dependencies..."
pip install -r requirements-train.txt

echo "==> [4/6] Configuring DATASET_DIR=${DATASET_ROOT} ..."
export DATASET_DIR="${DATASET_ROOT}"
if ! grep -qs "export DATASET_DIR=${DATASET_ROOT}" "${HOME}/.bashrc" 2>/dev/null; then
  echo "export DATASET_DIR=${DATASET_ROOT}" >> "${HOME}/.bashrc"
  echo "    added DATASET_DIR to ~/.bashrc (persists across sessions)"
fi
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "    ERROR: DATASET_ROOT '${DATASET_ROOT}' does not exist."
  echo "    Set the correct path, e.g.: DATASET_ROOT=/workspace/arun/data bash setup_b200.sh"
  exit 1
fi
if [ ! -d "${DATASET_ROOT}/point_maze" ]; then
  echo "    WARNING: ${DATASET_ROOT}/point_maze not found -- check the dataset path before training."
fi

echo "==> [5/6] Pre-caching the DINOv2 backbone (so fresh runs work if internet later drops)..."
# All dino* encoder configs use dinov2_vits14. Downloading it now (internet available
# during setup) caches it under ~/.cache/torch/hub so training won't need to fetch it.
python - <<'PY' || echo "    WARNING: DINOv2 pre-cache failed (offline?). First training run will need internet once."
import torch
torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
print("    cached dinov2_vits14 (num_features =", m.num_features, ")")
PY

echo "==> [6/6] Verifying the B200 is usable by PyTorch (real kernel launch)..."
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # expect (10, 0) on B200
    x = torch.randn(1024, device="cuda")
    print("kernel launch ok, sum =", float((x + 1).sum()))
PY

cat <<'EOF'

==> Setup complete. If capability printed (10, 0) and "kernel launch ok", you're good.

Next steps:
  export DATASET_DIR=/workspace/arun/data   # already added to ~/.bashrc

  # quick smoke test (short run) to confirm data loads and the 45 GB slice holds:
  python train.py --config-name train.yaml env=point_maze \
      training.epochs=1 training.save_every_x_iterations=50 debug=True

  # full run, detached so it survives disconnects (see run_train.sh):
  bash run_train.sh

  # ---- Offline / interrupted-run resume ----
  # First run downloads the DINOv2 backbone once (needs internet). After a checkpoint
  # exists, resume needs NO internet: the encoder + DINOv2 weights are in the ckpt.
  #   export WANDB_MODE=offline                       # logging never blocks on network
  #   bash run_train.sh                               # auto-resume from model_latest.pth
  #   python train.py ... training.resume_from=/abs/path/model_10.pth   # specific ckpt
EOF
