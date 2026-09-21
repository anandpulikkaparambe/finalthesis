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

NUM_ENVS="${NUM_ENVS:-1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-./rl_logs}"

cd "$(dirname "$0")/.."
source /venv/isaac51/bin/activate

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
  "${RESUME_ARG[@]}" \
  2>&1 | tee "$LOG_DIR/train.log"
