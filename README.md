# finalthesis: UR5e sim-to-real reinforcement learning

Code, data and documentation for a master's thesis on sim-to-real transfer of reinforcement
learning for a UR5e arm with a Robotiq 2F-140 gripper, trained in NVIDIA Isaac Sim and deployed
on the real robot.

## Status, read this first

- `isaac_sim_headline_run/` is the code that produced the thesis' headline Isaac Sim result
  (commit `cc9ae661` of [isaacsimtraining](https://github.com/anandpulikkaparambe/isaacsimtraining)).
  It had **no robot collision geometry, self-collision switched off, c_v = 0.05, and success defined
  by a finger-gap proxy**. See `docs/PROVENANCE.md`.
- `isaac_sim/` is a **new version with the thesis-driven changes** (see `docs/CHANGES.md`).
  It is implemented and unit-tested (49 tests on a fake physics backend) but has
  **not been run inside Isaac Sim and has not been trained**. The first thing to do on a GPU
  machine is `scripts/validate_collisions.py` (see `docs/VALIDATION_CHECKLIST.md`).
- Nothing in `isaac_sim/` produced any result reported in the thesis.

## Layout

| Path | Contents |
|---|---|
| `isaac_sim/` | Updated environment, training, evaluation, demonstration and validation scripts, configs, tests, Vast.ai setup |
| `isaac_sim_headline_run/` | The environment and training script exactly as used for the thesis' headline run (verbatim, do not edit) |
| `deployment/` | Real-robot deployment script (ROS 2) with an opt-in collision-check safety filter |
| `data/` | Headline-run telemetry and checkpoint, and the 16 real-hardware deployment logs |
| `analysis/` | Kinematic calibration fit and the collision-proxy analysis of the headline run |
| `docs/` | What changed and why, provenance of every result, validation checklist, real-trial protocol |

## Quick start on Vast.ai

```bash
# on a fresh "PyTorch (Vast)" instance with an RTX 3070 or better
git clone https://github.com/anandpulikkaparambe/finalthesis.git /workspace/finalthesis
bash /workspace/finalthesis/isaac_sim/vastai/setup_vastai.sh   # installs Isaac Sim 5.1.0, converts the asset,
                                                               # runs unit tests and the physics validation
source /venv/isaac51/bin/activate
cd /workspace/finalthesis/isaac_sim
# only after validate_collisions.py looks right:
NUM_ENVS=8 TOTAL_TIMESTEPS=1000000 CONFIG=configs/default.json ./vastai/train_entrypoint.sh
```

Compare against the old behaviour inside the new code with `configs/headline_like_baseline.json`
(gap proxy, self-collision off, c_v = 0.05, no randomization), so changes can be attributed.

## Not included

Visual perception (the target pose is still read from the simulator). The Gazebo pilot (its code
is in [ur3e_working_reference](https://github.com/anandpulikkaparambe/ur3e_working_reference) and
[remoteservertraining-UR5e](https://github.com/anandpulikkaparambe/remoteservertraining-UR5e)). The raw
70 MB per-step telemetry CSVs of the headline run (in the original `isaacsimtraining` repository;
this repo keeps every 100th step).

The thesis document itself is not in this repository.
