# isaacsimtraining

UR5e + Robotiq-140 reach-and-grasp, trained with RL directly in Isaac Sim. This is
the Isaac Sim replacement for [remoteservertraining-UR5e](https://github.com/anandpulikkaparambe/remoteservertraining-UR5e)'s
ROS2 + Gazebo + MoveIt2 stack -- same robot, same task, same reward/curriculum
design, different simulator.

## What changed vs. the Gazebo project, and why

| | Gazebo project | This project |
|---|---|---|
| Simulator | Gazebo (Ignition/Fortress) via ROS2 | Isaac Sim, native Python API |
| Robot control | ROS2 JointTrajectory + MoveIt2/OMPL handoff | Isaac Sim `Articulation` API, direct joint targets, no MoveIt |
| Compute | CPU-only (Gazebo physics + tiny policy net never touched the GPU) | GPU-bound (Isaac Sim needs an NVIDIA RTX GPU) |
| vast.ai sizing | CPU cores / RAM, GPU ignored entirely | GPU + VRAM is now the binding constraint |
| Parallelism | N separate Gazebo+MoveIt Docker instances | N separate Isaac Sim processes (`SubprocVecEnv`) |

The observation space (23-dim), action space (7-dim: 6 joint deltas + 1 gripper
command), reward shaping (potential-based distance shaping, velocity penalty,
episode shaping floor, terminal success/failure bonuses), and curriculum (spawn
radius scaling with rolling success rate) are ported term-for-term from
`ur3e_env.py`/`train_sac.py` in the Gazebo repo -- see `source/ur5e_grasp/ur5e_grasp_env.py`'s
module docstring for the exact mapping and the handful of deliberate simplifications
(most notably: collision detection is a height-only proxy for now, not a full
arm-mesh contact query -- see that file for the details).

## Status (2026-09-09)

- Robot asset: real UR5e + Robotiq-140 geometry, xacro-expanded from the Gazebo
  project's own `ur_description`/`robotiq_2f_gripper_description` packages with
  `ur_type:=ur5e`, converted to USD. **Verified loading and stepping correctly** in
  Isaac Sim 5.1.0 (all 6 arm joints + gripper mimic joints present, physics steps
  without error).
- `Ur5eGraspEnv` (Gymnasium env): smoke-tested (`scripts/smoke_test.py`) and now
  **run through a real (if short) local SAC training session end-to-end**:
  3000 timesteps, 4 episodes, healthy losses (critic_loss 0.0865, no NaN/explosion),
  a checkpoint and the final model both saved successfully. Not yet trained long
  enough for a competent policy -- that's what vast.ai is for.
- **TensorBoard logging is deliberately disabled** (`tensorboard_log` never passed
  to SAC) -- confirmed live that enabling it crashes the process silently the
  instant `.learn()` starts (no Python exception, no traceback, exit code 0),
  almost certainly a protobuf ABI conflict between TensorBoard's and Isaac
  Sim/USD's bundled protobuf in the same process. See `scripts/train.py`'s module
  docstring for the full diagnostic trail (GPU-memory and import-order were both
  ruled out first). Training progress is still fully visible via each env's own
  `hardware_log_env*.csv` and SB3's own console table.
- Local machine: Isaac Sim 5.1.0 confirmed installed and working headless
  (`isaac-sim.compatibility_check.bat` reports VRAM/RAM below NVIDIA's stated
  minimum -- RTX 3050 6GB vs. a stated 16GB minimum, 16GB system RAM vs. 32GB -- but
  it boots, steps physics, and trains correctly anyway; treat local runs as
  toolchain verification, not real training throughput -- 3000 steps took ~2 minutes
  at 16 fps).
- vast.ai: **run against a real rented instance** (2026-09-10, RTX 3070, 16 cores,
  86GB RAM, `vastai/pytorch:cuda-12.8.1-auto` template) -- `scripts/smoke_test.py`
  passes cleanly (3 episodes x 100 steps, no early terminations). Getting here
  required pinning Isaac Sim to **5.1.0 specifically, in its own Python 3.11 venv**
  (the base image ships Python 3.12): pip-installing "latest" resolved to 6.0.1.0,
  which installed and ran without crashing but was **physically unstable** --
  missing default PD gains (the 6.0.1.0 importer doesn't set them) and, even after
  fixing that plus PhysX solver iteration counts, the robot still diverged into a
  velocity explosion within ~10 steps holding a fixed pose with zero policy input.
  Root cause not found despite real effort; switching to 5.1.0 fixed it immediately
  with no other changes. `assets/convert_urdf_to_usd.py` still supports both
  importer APIs (useful if a future Isaac Sim upgrade is worth re-attempting), but
  **5.1.0 is the version this project actually trains on** until 6.x's stability
  gap is understood. Real multi-hour/`--num-envs > 1` training has not been run
  yet -- see `vastai/setup_vastai.sh`'s own header for the full account.

## Running locally (verification, not real training)

```
C:\isaacsim\python.bat assets\convert_urdf_to_usd.py     # one-time, see assets/README.md
C:\isaacsim\python.bat scripts\smoke_test.py
C:\isaacsim\python.bat scripts\train.py --num-envs 1 --total-timesteps 2000
```

## Running on vast.ai

See `vastai/setup_vastai.sh` and `vastai/train_entrypoint.sh`. Unlike the Gazebo
project, **GPU/VRAM matters here** -- rent an instance with a real NVIDIA GPU (this
was developed and verified against Isaac Sim 5.1.0's own stated minimum of an RTX
GPU with >=10GB VRAM; more is better for `--num-envs` > 1, since each is a separate
Isaac Sim process with its own GPU footprint).

```bash
# On the rented instance, after cloning:
./vastai/setup_vastai.sh
NUM_ENVS=4 TOTAL_TIMESTEPS=200000 DEVICE=cuda ./vastai/train_entrypoint.sh
```

## Repo layout

```
assets/            URDF (from the Gazebo project's own robot description) + meshes + USD conversion scripts
source/ur5e_grasp/  Ur5eGraspEnv -- the Gymnasium env, Isaac Sim backend
scripts/            smoke_test.py, train.py (SAC), play.py (load a checkpoint, roll it out)
vastai/             setup_vastai.sh, train_entrypoint.sh
```

## Known gaps (flagged, not silently assumed done)

- Collision detection is a height-only proxy (`ee_z < 0.02`), not a full PhysX
  contact-report-based arm/table or self-collision check.
- No perception/YOLO target-source mode yet (the Gazebo project's ground-truth mode
  only, for now) -- the wrist camera isn't wired into the USD import
  (`use_camera:=false`).
- `scripts/train.py` has only been run for a short local session (3000 steps) to
  confirm the pipeline works -- not yet run long enough, locally or on vast.ai, to
  produce a policy that can actually grasp anything.
- No TensorBoard visualization (see Status above) -- CSV + console table only.
