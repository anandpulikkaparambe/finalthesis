#!/bin/bash
# One-time setup on a freshly rented vast.ai GPU instance. Verified end-to-end
# 2026-09-10 against a "PyTorch (Vast)" template instance (Ubuntu 24.04, image
# vastai/pytorch:cuda-12.8.1-auto, RTX 3070, host 93197/machine 19028) -- these
# exact steps got a stable, working env; earlier attempts and why they failed are
# recorded below since they're easy to reintroduce by "simplifying" this script.
#
# Isaac Sim version is pinned to 5.1.0 (Python 3.11), matching the exact version
# this project was developed and locally verified against on Windows. This
# matters more than it looks: pip-installing "latest" instead resolved to 6.0.1.0
# (Python 3.12) on 2026-09-10 and produced a *working but physically unstable*
# environment -- it boots, converts assets, and steps without crashing, but the
# imported robot's joints have no PD gains by default (the 6.0.1.0 importer
# doesn't set them, unlike 5.1.0) and even after fixing that, holding a fixed
# target with zero policy action still diverged into a velocity explosion within
# ~10 steps. Root cause not found after substantial debugging (isolated
# single-joint tests were stable; the full scene wasn't) -- switching back to
# 5.1.0 fixed it immediately with no other changes. Don't casually bump this
# version without re-verifying stability the same way (scripts/smoke_test.py
# with --episodes 3+, watching for early kill-switch terminations).
#
# The base image's system Python is often NOT 3.11 (the rented instance this was
# verified on shipped Python 3.12) -- use `uv python install 3.11` (works on any
# base image with uv preinstalled, which vast.ai's own templates ship with; no
# apt/sudo needed) rather than assuming a matching system Python exists.
set -euo pipefail

echo "=== Sanity checks ==="
ls /.dockerenv && echo "container confirmed"
cat /etc/os-release
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== System libraries Isaac Sim's renderer needs but this base image doesn't ship ==="
# libxt6: MaterialX render plugins fail to load without it ("libXt.so.6: cannot
# open shared object file"). libglu1-mesa: Iray/MDL plugin fails the same way
# without libGLU.so.1. Neither failure is immediately fatal on its own (Isaac Sim
# logs [Error] and keeps going), but both showed up during the 6.0.1.0 debugging
# above and are cheap to just install upfront.
apt-get update -qq
apt-get install -y -qq libxt6 libglu1-mesa

echo "=== Creating Python 3.11 venv (uv manages the Python version itself) ==="
uv python install 3.11
uv venv /venv/isaac51 --python 3.11
source /venv/isaac51/bin/activate

echo "=== Installing PyTorch (CUDA) ==="
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing Isaac Sim 5.1.0 ==="
export OMNI_KIT_ACCEPT_EULA=YES
# --index-strategy unsafe-best-match: without it, uv's dependency resolver only
# considers versions from the FIRST index that has a given package name (a
# dependency-confusion safety default) -- isaacsim-core depends on
# mujoco-usd-converter==0.2.0, which pypi.nvidia.com has under a different
# version, and the default strategy refuses to fall back to pypi.org for it,
# failing the whole install with "No solution found when resolving dependencies".
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match

echo "=== Installing training deps (without letting them touch the pinned torch build) ==="
uv pip install stable_baselines3 gymnasium pytest pandas matplotlib
echo "torch after stable_baselines3 install (should be unchanged from above):"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'

echo "=== Cloning finalthesis (public repo, no key needed) ==="
if [ ! -d /workspace/finalthesis ]; then
  git clone https://github.com/anandpulikkaparambe/finalthesis.git /workspace/finalthesis
fi
cd /workspace/finalthesis/isaac_sim

echo "=== Converting URDF -> USD (first run only; cached after) ==="
python assets/convert_urdf_to_usd.py

echo "=== Smoke test (3 episodes -- 1 isn't enough to catch an early-kill-switch instability) ==="
python scripts/smoke_test.py --episodes 3 --steps-per-episode 100

echo "=== Unit tests (no Isaac Sim needed) ==="
python -m pytest tests -q

echo "=== Physics validation (collisions, contact sensor, demo grasp) -- READ THE OUTPUT before training ==="
python scripts/validate_collisions.py --config configs/default.json

echo "Setup complete. Run training with: source /venv/isaac51/bin/activate && ./vastai/train_entrypoint.sh"
