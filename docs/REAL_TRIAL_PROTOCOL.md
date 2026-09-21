# Real-robot trial protocol

Follows the staged rollout used for the thesis' 16-trial deployment, so a retrained checkpoint can be
compared run for run. A human stays at the robot with the e-stop within reach for every run. The
software checks below add to that; they do not replace it.

## Before the first run

- Use the same setup as the 16 runs: real UR5e, Robotiq 2F-140 with its model in the URDF, the taught
  `real_home_position` as the start, the same target position, direct ethernet control.
- Check that the observation the script builds matches `isaac_sim/source/ur5e_grasp/spec.py`: 23
  values (6 joint positions, 6 velocities, finger-pad midpoint xyz plus quaternion, target relative to
  the base, gripper position). In the first deployment the missing gripper model, a joint-state merge
  that zeroed the arm joints, and a start pose outside the training distribution each broke the
  observation.
- Deploy at the trained control rate or train with action delay. Training steps are 0.1 s; the real
  runs used 0.3 to 3 s per step.

## Stages (do not skip)

1. **Observation only, zero motion.** Build the live observation, run the policy, log the action, send
   nothing. The action should be bounded and point roughly towards the target. If it is saturated or
   erratic, stop.
2. **Heavily clamped closed loop.** A small fixed fraction of the requested step (the thesis used 1/8),
   slow fixed step duration, a hard cap on steps and on cumulative motion per joint, gripper output
   logged but not sent.
3. **Full authority within the trained bound** (+-0.08 rad per step), slow enough for the operator to
   react to each step, gripper actuated.

Every step: check `safety_mode` (the script aborts on anything other than NORMAL), keep the log.

## The 16-run comparison

- 16 runs, same start, alternating nothing else: record for each run the step count, minimum distance
  to the target, the stop cause (e-stop, protective stop, path tolerance, link drop), and whether
  `object_grasped` fired with the tool within 0.05 m of the target. A grasp counts only if both hold.
- Keep the per-step duration fixed within the comparison (the thesis' runs varied it from 0.1 to 3 s,
  which confounds the comparison).
- Save one file per run, named like the existing `data/real_deployment_16_trials/run_NN_*.log`.

## Optional collision-check filter

`--safety-filter` (with `--table-z-base`) predicts each step with the calibrated model and scales down
or refuses steps that would cause self-collision or table contact. Measure the table height in the base
frame first: with the fingertip resting on the table (freedrive), read the tool point height in the
base frame; it is 0.0 if the arm is mounted on the table top as in simulation. The filter has not been
tested on hardware and uses approximate capsules: keep it as an extra layer, not the only one.

## What to report

Number of runs, the distribution of minimum distance, stop causes, confirmed grasps out of 16, and the
comparison with the earlier checkpoint under identical settings.
