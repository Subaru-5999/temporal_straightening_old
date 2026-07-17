#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_planning.sh
# Installs the PLANNING-tier dependencies so plan.py can roll out in the
# simulators. Training does NOT need any of this.
#
# IMPORTANT:
#   * Run this AFTER training finishes (installing packages while training runs
#     in the same Python can crash the live job), OR inside a separate venv.
#   * This is the build-risky part (mujoco-py compiles; d4rl is from source).
#     Run it interactively and watch for errors -- we may need to iterate.
#
# Covers PointMaze (UMaze/Medium). Wall needs nothing extra; PushT extras are
# in requirements-plan.txt.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> [1/5] System libraries for MuJoCo / mujoco-py (needs root)..."
apt-get update -y
apt-get install -y \
  libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
  libglfw3 patchelf gcc build-essential

echo "==> [2/5] Downloading + extracting MuJoCo 210..."
mkdir -p "${HOME}/.mujoco"
if [ ! -d "${HOME}/.mujoco/mujoco210" ]; then
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P "${HOME}/.mujoco/"
  tar -xzf "${HOME}/.mujoco/mujoco210-linux-x86_64.tar.gz" -C "${HOME}/.mujoco/"
fi
# mujoco-py needs these on the library path at import time:
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
for line in \
  "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin" \
  "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia"; do
  grep -qsF "$line" "${HOME}/.bashrc" || echo "$line" >> "${HOME}/.bashrc"
done

echo "==> [3/5] Installing planning Python deps..."
pip install -r requirements-plan.txt

echo "==> [4/5] Installing d4rl from source (provides the PointMaze MazeEnv)..."
pip install "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl" || \
  echo "    WARNING: d4rl install failed -- PointMaze planning needs it. Check the error."

echo "==> [5/5] Verifying imports (mujoco-py compiles on first import; this can take a minute)..."
python - <<'PY' || echo "    Import check failed -- see error above."
import gym, mujoco_py
print("gym:", gym.__version__)
import env  # registers point_maze / pusht / wall gym envs
print("env package imported; gym envs registered OK")
PY

cat <<'EOF'

==> Planning setup attempted. If the import check passed, you can evaluate:
  bash evaluate.sh <ckpt_base_path> <model_name>
EOF
