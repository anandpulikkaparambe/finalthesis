#!/bin/bash
# Headless training entrypoint for a vast.ai GPU instance, after setup_vastai.sh has
# run once. Mirrors the sibling Gazebo repo's vastai_train_entrypoint.sh in spirit
# (env-var-driven config, one command starts everything) but there's no per-instance
# Gazebo/MoveIt boot to stagger here -- SubprocVecEnv launches NUM_ENVS Isaac Sim
# processes directly, each importing its own USD stage.
#
# Usage:
#   NUM_ENVS=4 TOTAL_TIMESTEPS=200000 DEVICE=cuda ./vastai/train_entrypoint.sh
set -euo pipefail

# EULA acceptance is per-process, not cached to disk -- this is a separate script/process from
# setup_vastai.sh, so it needs its own export or the first Isaac Sim boot hangs on an interactive
# prompt with no stdin (confirmed live: setup_vastai.sh's own export does not carry over here).
export OMNI_KIT_ACCEPT_EULA=YES

NUM_ENVS="${NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-./rl_logs}"
CONFIG="${CONFIG:-configs/default.json}"
SEED="${SEED:-0}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"

cd "$(dirname "$0")/.."
source /venv/isaac51/bin/activate

if [ "$DEVICE" = "cuda" ]; then
  # confirmed live 2026-09-22: a later `uv pip install` during setup can silently swap in a torch
  # build the driver can't run (torch.cuda.is_available() -> False), and train.py itself does not
  # check this -- SB3 just prints "Using cpu device" and trains for hours at a small fraction of
  # the speed this instance was rented for, with no error. Fail fast here instead.
  python -c 'import torch; assert torch.cuda.is_available(), f"torch {torch.__version__} (cuda {torch.version.cuda}) cannot see the GPU -- re-run: uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall"'
fi

mkdir -p "$LOG_DIR"

RESUME_ARG=()
if [ -n "$RESUME" ]; then
  RESUME_ARG=(--resume "$RESUME")
fi

echo "Starting training: NUM_ENVS=$NUM_ENVS TOTAL_TIMESTEPS=$TOTAL_TIMESTEPS DEVICE=$DEVICE"
python scripts/train.py \
  --num-envs "$NUM_ENVS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --device "$DEVICE" \
  --log-dir "$LOG_DIR" \
  --headless \
  --config "$CONFIG" \
  --seed "$SEED" \
  --buffer-size "$BUFFER_SIZE" \
  "${RESUME_ARG[@]}" \
  2>&1 | tee "$LOG_DIR/train.log"
