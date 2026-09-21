#!/bin/bash
# Train the same configuration with several seeds, one after another (each run gets its own
# log directory). Seeds make runs repeatable and give a spread instead of a single result.
#
#   SEEDS="0 1 2" NUM_ENVS=8 TOTAL_TIMESTEPS=1000000 CONFIG=configs/default.json ./scripts/run_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/isaac51/bin/activate

SEEDS="${SEEDS:-0 1 2}"
NUM_ENVS="${NUM_ENVS:-8}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
CONFIG="${CONFIG:-configs/default.json}"
OUT="${OUT:-./runs}"

for SEED in $SEEDS; do
  LOG_DIR="$OUT/seed_$SEED"
  mkdir -p "$LOG_DIR"
  echo "=== seed $SEED -> $LOG_DIR ==="
  python scripts/train.py --num-envs "$NUM_ENVS" --total-timesteps "$TOTAL_TIMESTEPS" \
      --config "$CONFIG" --seed "$SEED" --log-dir "$LOG_DIR" --device cuda --headless \
      2>&1 | tee "$LOG_DIR/train.log"
  python scripts/evaluate.py --checkpoint "$LOG_DIR/ur5e_isaac_sac_final.zip" --config "$CONFIG" \
      --episodes 50 --out-dir "$LOG_DIR/eval" 2>&1 | tee "$LOG_DIR/eval.log"
done