# Provenance: which result came from which code and data

## The thesis' headline Isaac Sim result

| Item | Source |
|---|---|
| Code that trained it | `isaac_sim_headline_run/`, verbatim from commit `cc9ae661` of [isaacsimtraining](https://github.com/anandpulikkaparambe/isaacsimtraining) (the commit that added the checkpoint and telemetry) |
| Velocity-penalty coefficient | 0.05 (`velocity_penalty = 0.05 * ...` in that file). It was reduced to 0.001 in a later commit (`e01ebc6f`) |
| Robot collision geometry | **None.** Collider wiring was added in a later commit (`84c77675`), after this run |
| Self-collision | Off (`self_collision=False`, docstring lines 24 to 25) |
| Success criterion | `gripper_closed and (commanded - achieved finger position) > 0.13`, i.e. a finger-gap proxy, in a scene without robot collision geometry |
| Domain randomization | None (only the target spawn position is random within the curriculum radius) |
| Checkpoint | `data/headline_run/ur5e_isaac_sac_final.zip`, sha256 `68A2FB52CC5D1F47F4BC9011FA64A6A712347ED6285AC0B7E1AE5E69074E7CA7` (identical to the copy in the original repository) |
| Training console log | `data/headline_run/train_console.log` (the 600,000-step resumed session; last line: 629,668 cumulative steps, rollout success rate 0.69) |
| Telemetry | `data/headline_run/telemetry_env{0,1}_every100th_step.csv`; the full per-step CSVs (about 70 MB each) are in the original repository under `rl_logs_7hr_final/` |
| Hardware | one rented RTX 3070 on Vast.ai, Ubuntu 24.04, Isaac Sim 5.1.0, two parallel environments |

What the telemetry shows (the "resumed session" is the 600,000-step continuation):
success stayed at 0 until roughly 300k cumulative steps; the curriculum level started moving after
roughly 360k and reached 1.0; 1,364 episode terminations were logged, 1,331 with a reward above 10.

`analysis/headline_collision_proxy.py` applies a capsule collision proxy to the logged poses (6,000 poses, every 100th step of both environments). The proxy is approximate (hand-set radii, centimetre-level model error), so read the numbers as an indication:

- **Self-overlap:** the arm overlaps itself in about 46 % of the poses, and it is not marginal: 35 % overlap by more than 5 cm (the gripper through the upper arm, the forearm through the shoulder). The policy visited many configurations the real arm cannot take.
- **Table:** the logged fingertip was never below the table top (median 3.6 cm above it). Arm links overlap the table plane in about 80 % of the poses, but by at most 2.6 cm, which is within the proxy's error, so this is weak evidence: it suggests the policy held the gripper low and roughly horizontal, a pose the real arm can barely take.

## The real-hardware deployment

| Item | Source |
|---|---|
| 16 runs of the headline checkpoint on the real UR5e with a Robotiq 2F-140 | `data/real_deployment_16_trials/` (`REPORT.md` is the session report, `run_01` to `run_16` are the raw ROS logs) |
| Best approach | 0.407 to 0.412 m against a 0.05 m threshold, about 66 % of the starting gap; `object_grasped` never fired at a target-proximal state |
| Script | `deployment/real_phase2_unclamped_run.py` (opt-in `--safety-filter` added in this repository; the rest is unchanged) |

## What this repository adds

`isaac_sim/` is derived from the latest `main` of the original repository (commit `e01ebc6f`,
which already has the collision geometry and c_v = 0.001) plus the changes in `docs/CHANGES.md`.
It has produced no results. Exploratory trial runs made after the headline run under later
versions of the original repository (three PPO trials and a lift-task SAC variant) are not part
of this repository and are not results.

## Where the rest lives

The Gazebo pilot (code and its training export) is not in this repository: see
[ur3e_working_reference](https://github.com/anandpulikkaparambe/ur3e_working_reference) and
[remoteservertraining-UR5e](https://github.com/anandpulikkaparambe/remoteservertraining-UR5e).
The Gazebo campaign's step counters sum to about 1.43M only because restarts that resumed from a
checkpoint inherit its earlier step count; about 0.63M new steps were simulated.