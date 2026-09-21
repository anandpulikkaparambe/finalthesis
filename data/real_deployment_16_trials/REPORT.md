# Real-Hardware SAC Reach-and-Grasp Testing — Session Report
**Date:** 2026-09-11
**Checkpoint:** `rl_logs_7hr_final/ur5e_isaac_sac_final.zip` (Isaac Sim, SAC, 629,668 training steps, curriculum maxed to 1.0, peak sim success 97%)
**Robot:** Real UR5e + Robotiq 2F-140 gripper, direct-ethernet controlled
**Deployment script:** `src/ur3e_rl/ur3e_rl/real_phase2_unclamped_run.py` (`--legacy` obs format, `--start-waypoint real_home_position`)

## 1. Interface verification (before any real motion)
The real-hardware deployment's observation/action construction was checked line-by-line against
the actual Isaac Sim training code (`ur5e_grasp_env.py`), not assumed:
- 23-D observation: 6 joint pos + 6 joint vel + 7 EE pose (gripper finger-pad midpoint xyz +
  right-pad quat) + 3 target xyz + 1 gripper pos — confirmed matching.
- 7-D action: 6 joint-position deltas (±0.08 rad) + 1 absolute gripper command — confirmed
  matching, including the exact ±0.08 rad bound and joint ordering (`shoulder_pan → wrist_3`).

## 2. Runs, in order

| # | File | Sampling | Step dur. | Steps reached | Min dist (m) | Stop cause |
|---|---|---|---|---|---|---|
| 1 | run_01 | deterministic | 0.5s (dry, no motion) | 15 (repeat) | 1.225 (static) | n/a — observation-only |
| 2 | run_02 | deterministic | 3.0s | 50 | 0.902 | `ROBOT_EMERGENCY_STOP` (user e-stop) |
| 3 | run_03 | deterministic | 2.0s | 102 | **0.412** | link drop |
| 4 | run_04 | stochastic | 2.0s | 35 | 0.922 | `ROBOT_EMERGENCY_STOP` (user e-stop) |
| 5 | run_05 | stochastic | 2.0s | 33 | 0.939 | `PROTECTIVE_STOP` C4A3 — Safety Control Board comms fault (confirmed via pendant code) |
| 6 | run_06 | stochastic | 2.0s | 34 | 0.889 | path-tolerance violation; gripper false-positive `object_grasped=True` at dist=0.94 (fingers stalled on themselves, not the target) |
| 7 | run_07 | stochastic | 2.0s | 115 | **0.407 (session best)** | link drop |
| 8 | run_08 | stochastic | 2.0s | 111 | 0.547 | link drop (user then e-stopped separately mid-diagnosis) |
| 9 | run_09 | stochastic | 1.0s | 115 | 0.662 | link drop |
| 10 | run_10 | stochastic | 0.5s | 121 | 0.615 | link drop (follow-up check found `safety_mode` had gone to `PROTECTIVE_STOP`) |
| 11 | run_11 | stochastic | 0.1s (full trained speed) | 6 | 1.225 | `ROBOT_EMERGENCY_STOP` (user e-stop, reacted to full-speed motion) |
| 12 | run_12 | stochastic | 0.3s | 118 | 0.683 | link drop (follow-up check found `PROTECTIVE_STOP`) |
| — | *(pipeline fix)* | | | | | Gripper added to URDF; joint-state merger bug found & fixed (was silently zeroing all 6 arm joints in the merged TF input); `REAL_PICK_XYZ` re-measured in the corrected finger-pad-midpoint frame |
| 13 | run_13 | stochastic | 0.3s | 10 | 0.0004 (invalid) | link drop — target constant still corrupted by the (at-that-point-still-broken) merger; data discarded |
| 14 | run_14 | stochastic | 0.3s | 29 | 0.731 | `PROTECTIVE_STOP` (mode 3) — first run on the **fully corrected** pipeline |
| 15 | run_15 | stochastic | 0.3s | 29 | 0.732 | link drop — **same step (29) as run 14**, reproduced |
| 16 | run_16 | stochastic | 0.5s | 35 | 0.787 | path-tolerance violation; **user visually confirmed a real self-collision at this stop** |

## 3. Headline results
- **`object_grasped` never once fired at a real target-proximal state, across all 16 runs.** The
  single `True` reading (run 6) occurred at dist=0.94m — a false positive from the gripper
  stalling on itself, not a grasp.
- **Best real approach: 0.407–0.412m** (runs 3, 7) against a 0.05m grasp threshold — closes
  roughly 66% of the ~1.2m starting gap, never the rest.
- **Typical/modal approach: 0.6–0.9m** — most runs plateau in this band regardless of sampling
  mode (deterministic vs. stochastic) or step speed (0.1s–3.0s tested).
- Runs 14/15, both on the corrected pipeline, stopped at **exactly the same step (29)** —
  a reproducible cluster, though its cause (step-count-linked vs. coincidence) is not fully
  isolated (see below).

## 4. Real deployment bugs found and fixed this session
1. **Out-of-distribution start** — initial deployment started the policy already at the pick
   target (dist≈0); training never saw this (fixed home pose, ~0.84m-from-base target every
   episode). Fixed by starting from `real_home_position`.
2. **Step-budget mismatch** — `MAX_STEPS` was 30 (later 80), calibrated to an older Gazebo
   checkpoint's 50-step episodes; this checkpoint trained with `max_episode_steps=500`. Raised
   to match.
3. **Missing gripper in the real URDF** — the driver's stock launch has no gripper model at all;
   real deployment approximated EE pose at `tool0` (~0.17m fixed-offset guess) instead of the
   trained reference point (gripper finger-pad midpoint). Fixed: new combined URDF
   (`real_hardware_description/ur5e_with_gripper.urdf.xacro`) attaching the Robotiq 140 at
   `tool0`, deployment script updated to read the real `left/right_inner_finger_pad` TF frames.
4. **Joint-state merger bug** — the first fix for gripper TF (via `joint_state_publisher`'s
   `source_list`) silently zeroed all 6 arm joint positions in its merged output, corrupting
   every TF-derived distance/EE-pose computation downstream (confirmed live via
   `ros2 topic echo /combined_joint_states`). Replaced with a minimal custom merger
   (`real_hardware_description/joint_state_merger.py`), unit-verified with injected known values
   before redeploying.
5. **`REAL_PICK_XYZ` measured with bug #4 active** — first re-measurement (run 13's target) was
   itself computed via the broken merger and was garbage; re-measured again after the fix,
   cross-validated against the original 2026-09-01 tool0-frame constant (X/Y match closely, Z
   differs by ~0.178m — consistent with the expected tool0-to-gripper-tip offset).

## 5. Root cause candidate for the persistent reach gap
Confirmed directly in the training environment source (`ur5e_grasp_env.py`, lines ~22-29):
**self-collision detection is hardcoded OFF** (`self_collision=False` at URDF-import time). Only
a simplified table-collision proxy exists. The policy was never able to learn 3D self-awareness
during ~630k training steps, because the simulator never modeled the possibility of the arm/
gripper contacting itself. Run 16's real, visually-confirmed self-collision — occurring in the
same 0.7-0.9m distance band where most runs plateau — is direct physical corroboration of this
gap, not just a sim-side code observation.

## 6. Open / unresolved
- Step-29 clustering (runs 14, 15) not fully isolated as step-count-based vs. coincidental —
  only 2 data points on the corrected pipeline.
- Recurring `robot_program_running` link drops throughout the session (majority of stop causes
  above) never fully root-caused; one instance confirmed as a Safety Control Board communication
  fault (C4A3), suspected to be the same underlying cause for most others given drops occurred
  regardless of what motion was being commanded (including a plain home-return move).
- No live perception in the loop — target was a fixed measured constant throughout, not
  camera-detected, a separate sim-to-real gap not addressed this session.
