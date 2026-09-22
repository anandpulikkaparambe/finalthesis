# Validation checklist (run on the Vast.ai GPU instance)

The new environment has only been tested against a fake physics backend. Do these in order and
stop at the first failure: a long training run on an unvalidated environment repeats the thesis'
main mistake (training in a world that does not match what you think it is).

## 0. Setup

`bash isaac_sim/vastai/setup_vastai.sh` pins Isaac Sim to 5.1.0, converts the asset (self-collision on),
runs the unit tests and then the physics validation. Do not bump the Isaac Sim version without
re-running everything below.

## 1. Unit tests (no GPU needed)

`python -m pytest tests -q` from `isaac_sim/`. Expect 49 passed. These test the maths, the reward and
termination wiring, contact holding, randomization, the curriculum, the demonstration pipeline and the
environment logic against a fake backend. They say nothing about PhysX.

## 2. Smoke test

`python scripts/smoke_test.py --episodes 3 --steps-per-episode 100 --config configs/default.json`

Expect `ENV_CREATED`, `contact source: sensor`, and episodes ending for sensible reasons. If the
environment stops with "Finger-pad contact sensor unavailable", the PhysX contact API in this Isaac
Sim version is not what `contact.py` and `_make_pad_prim` (in `ur5e_grasp_env.py`) expect. Print what
`RigidPrim` offers (`dir(RigidPrim)`, look for `contact`) and adjust those two places.

## 3. Physics validation

`python scripts/validate_collisions.py --config configs/default.json`

| Check | What a pass means | If it fails |
|---|---|---|
| contact source | the finger-pad sensor is active | see step 2 |
| hold-pose stability | arm holding still stays below 2 rad/s | compare with `"self_collision": false` in the config; if that fixes it, the wrist or gripper collision pairs need filtering |
| self-collision | folding the elbow ends in "Self Collision" or the joint is blocked | INFO only means neither direction reached it; try the wrist joints |
| table | driving the arm down ends in "Table Collision" or is blocked; deepest penetration under 6 cm | the collider wiring did not take effect (look for `DEBUG wired_count` in the log) |
| demo grasp | the scripted controller gets confirmed grasps | a wrong kinematic calibration or a contact-sensor problem; check peak forces in the output |

## 4. Demonstrations

`python scripts/collect_demos.py --episodes 50 --out demos.npz`. The success rate is a health check of
the whole chain. If it is low, do not train yet.

## 5. Attribute changes, do not stack them blindly

Run the same seed for a short training (for example 200k steps) with each of:

1. `configs/headline_like_baseline.json` (gap proxy, self-collision off, c_v = 0.05, no randomization)
2. `configs/no_randomization.json` (everything new except randomization)
3. `configs/default.json`

Compare with `scripts/evaluate.py` (deterministic, collisions on, 50 episodes) and look at curriculum
progress in the training log. The thesis' headline run needed about 300k steps before any success; if
the new configuration is much slower, look at the reward terms first (the per-step penalties are
untuned).

## 6. Seeds

`SEEDS="0 1 2" ./scripts/run_seeds.sh`. One seed is an anecdote.

## 7. Before any real-robot time

`docs/REAL_TRIAL_PROTOCOL.md`.

## What to report honestly

Success rates from `evaluate.py` with contact confirmation, with and without randomization, per seed;
the termination-reason histogram; and how often the collision proxies fire. Do not compare the new
success rate directly with the thesis' 0.97 peak: that number is defined by the finger-gap proxy in a
world without collision geometry.


## Local Isaac Sim 5.1 headless results (2026-09-22, laptop, not Vast.ai)

Run with the real physics backend, `no_randomization.json`. This is the first time the code ran against real PhysX rather than the fake backend, and it is **not a pass**.

| Check | Result | Note |
|---|---|---|
| Unit tests (fake backend) | 49/49 pass | says nothing about PhysX behaviour |
| Contact sensor API (`get_contact_force_matrix`) | works | sensor is created and read |
| Hold-pose stability | FAIL | one 20-step window peaked at 2.17 rad/s (threshold 2.0); likely settling/self-collision jitter, not yet investigated |
| Self-collision check | not conclusive | the script's probe never reached a self-collision; needs rewriting |
| Table check | vacuous | arm never got near the table; needs rewriting |
| Scripted demo grasp | FAIL, 0/5 | see below |

**Stale-asset pitfall.** A cached USD from an earlier conversion (base at z 0.80 instead of 0.78) makes every episode end at step 0 with "Target Lost". Always regenerate the USD (`assets/convert_urdf_to_usd.py`) on a new machine or point `UR5E_USD_PATH` at a fresh one. `setup_vastai.sh` regenerates it.

**Demo grasp findings.**
1. The pad collision boxes are 6 cm tall and centred on the tool point, so with active colliders the tool point cannot go below about table + 3 cm (ee_z about 0.81). The old `close_dist_m = 0.02` was unreachable and the controller stalled in `descend`. Fixed: `close_dist_m = 0.04` (the environment's own success distance is 0.05).
2. After that fix the gripper does reach the close phase, and the finger joint closes to 0.70 rad. But the pad frames stay about 5 cm apart along the closing axis even fully closed, and the pads are vertically offset from each other by about 3 cm. A 2 cm cube cannot be squeezed by that geometry. In one run the cube was shoved about 4 cm sideways while both pad sensors read 0 N.
3. Conclusion: the gripper collision geometry in the converted URDF (simple boxes at the link origins, not the real finger surfaces) does not support a physically meaningful grasp of the 2 cm cube. Until the pad collision geometry is rebuilt (or the Robotiq 2F-140 asset from Isaac Sim is used), contact-confirmed success cannot be reached, so **no training run is meaningful yet**, and no claim of "contact-verified grasping" should be made from this repo.

### Update: real 2F-140 collision meshes (2026-09-22)

The repo already contains the real Robotiq 2F-140 collision STLs (`assets/meshes/.../collision/`). The Gazebo description swaps them for 1 cm boxes; `patch_urdf.py` now puts the meshes back (8 links; `--gripper-box-collision` restores the old behaviour). Regenerate the USD after patching. Result on local Isaac Sim 5.1: 3/5 checks pass. Hold-pose now passes (1.13 rad/s) and the table check genuinely triggers a table-collision termination. The demo grasp still fails (0/5): with the meshes the finger joint stalls at about 0.15-0.25 rad instead of closing to 0.7, so the pads never reach the cube. Follower-joint gains make no difference; a lower finger-joint damping only speeds the approach to the same plateau. This looks like a mimic-joint / closed-linkage problem in the URDF-imported gripper rather than a controller problem. The next candidate is NVIDIA's own tuned Robotiq 2F-140 USD, which is not on this machine (it is fetched from NVIDIA's asset server).

### Attempted: NVIDIA Robotiq 2F-140 asset (2026-09-22) -- unresolved, abandoned for now

`assets/fetch_robotiq_2f140.py` downloads two NVIDIA-authored Robotiq 2F-140 USDs. Both were tried
as a drop-in replacement for the URDF-converted gripper's geometry, mounted onto the arm via a
synthetic fixed joint (`assets/build_nvidia_gripper_usd.py`, still in the repo since the diagnostic
work is worth keeping, but **not currently wired into the default asset path**):

1. `Robotiq_2F_140_physics_edit.usd` -- a real closed four-bar linkage with PhysX loop-closure
   joints (`excludeFromArticulation=True` on two of them), built as its own articulation. Grafting
   that loop-closure structure onto the arm's articulation left the linkage unstable: the fingers
   settled into a bent, left/right-asymmetric shape, reproducible even with self-collision and the
   contact sensor both off.
2. `2f140_instanceable.usd` -- a plain open tree, 6 independently-driven revolute joints, no loop
   closure, same link/joint names as the URDF gripper. A PhysX mimic constraint on its follower
   joints diverged even worse (finger_joint reaching hundreds of rad within 25 steps) once part of
   the arm's articulation, so the followers were switched to direct per-step position-target
   driving in `ur5e_grasp_env.py` (proven correct and stable in an unmounted, standalone test:
   gearing +1/+1/-1 relative to finger_joint converges the pads to 1.1cm apart and holds). Mounted
   onto the arm, **the whole arm diverges during reset's own settle steps**, before any policy
   action -- e.g. shoulder_pan_joint reaching -8 rad. This is not a gripper-linkage problem: it
   reproduces with the gripper's own internal joints entirely deactivated, mounting only its single
   base-link rigid body.

   Ruled out, each confirmed not to change the divergence (same trajectory to the decimal across
   most of these, which is itself a clue no one of them was the real cause): self-collision on and
   off; the asset's `UsdPhysics:MassAPI:principalAxes` shipping as an invalid all-zero quaternion
   on every body (fixed by overriding to identity -- confirmed the override lands in the file, no
   change in behaviour); the asset's `visuals`/`collisions` child scopes shipping
   `instanceable=True` the same way this repo's own arm-collider-wiring code once hit for the arm
   (fixed with the same `SetInstanceable(False)` pattern -- no change); the composed stage's own
   `metersPerUnit`/`upAxis` defaulting away from the source file's 1.0/Z-up when `Usd.Stage.CreateNew()`
   doesn't inherit sublayer metadata (fixed -- no change, because `World(stage_units_in_meters=1.0)`
   already forces the live stage's units before this file is only *referenced* in, not opened
   directly); a duplicated `ArticulationRootAPI` inside the referenced subtree (checked directly in
   the built file -- was already correctly removed); the mount joint's own frames/values (checked
   byte-for-byte identical to the URDF's own working `ur_to_gripper` joint); the placement matrix
   carrying spurious scale/shear from the transform-inversion chain (checked -- proper orthogonal
   rotation, determinant 1).

   Not yet tried: the reference chain here is unusually deep (arm's own multi-file reference chain,
   plus a new reference to `2f140_instanceable.usd`, which itself references
   `Collected_2f140_instanceable/Props/instanceable_meshes.usd`) -- worth checking whether
   `Usd.Stage.Flatten()`-ing the gripper source (or the whole composed file) before referencing it
   in changes anything, which would point at a PhysX/USD composition depth issue rather than
   anything about this specific asset's authoring.

**Decision:** given the time already spent without a root cause, this path is parked. The mesh-
collision-swapped URDF gripper below (`patch_urdf.py`'s real Robotiq STL collision meshes, no
NVIDIA asset needed) is stable and already passes 3/5 checks; effort went there instead.

### Follow-up: follower joint limits were wrong, fixed; a real geometry gap remains (2026-09-22)

Root-caused the demo grasp's earlier "0/5, 0 N" failure (both with box and real-mesh collision --
this turned out to have nothing to do with which one is used, see below): the five gripper
follower joints' *imported* limits do not match the URDF's own authored values. The URDF gives
`left_inner_knuckle_joint` a symmetric `+-0.8757` rad limit; Isaac Sim's URDF importer reads it
back as `[-0.84, 0.14]` rad -- an asymmetric, far narrower window. Once `finger_joint` closes past
the point where a follower needs to exceed `+0.14`, that follower hard-stops at its own (wrong)
limit while the mimic constraint keeps trying to enforce `follower = gearing * finger_joint`,
which it now can't -- the two fight, and the whole mechanism stalls around `finger_joint~0.47`
instead of reaching its own real limit of `0.70`, with zero contact force (nothing to do with a
real collision; reproduces identically with `self_collision=False`). Fixed in
`ur5e_grasp_env.py`'s existing mimic-gearing-fix loop: the same block now also widens these five
joints' `physics:lowerLimit`/`physics:upperLimit` to `+-60` degrees (USD revolute joint limits are
authored in degrees, not radians -- confirmed by reading the wrong values back and converting).
`finger_joint` now reaches its own full `0.70` rad and holds there cleanly.

That fix is real and worth keeping (hold-pose stability improved to 0.02 rad/s peak, from 1.13),
but it did **not** close the actual gap: even at `finger_joint = 0.70` (the joint's own authored
maximum, matching the URDF exactly), the two finger pads are **8.5cm apart, center to center** --
confirmed identical (to four decimal places) whether the box-collision or the real-mesh-collision
asset is used, which rules out collision geometry as the cause: pad separation is pure kinematics
(rigid-body poses driven by joint angles), not something a collision *shape* choice can move.
Scanning the commanded angle from 0.1 to 0.7 rad shows separation decreasing monotonically the
whole way (12.7cm to 8.5cm) with no sign of the linear `gearing=-1/+1` mimic approximation
"passing through" a true minimum and reopening -- it simply never gets close enough. This is very
likely a genuine calibration gap in the *source* Gazebo URDF's mimic joints themselves (a linear
approximation of what a real Robotiq four-bar closure needs is not exact, and apparently not close
enough here), not something introduced this session -- it was never visible before because the
thesis' headline run measured "success" from the commanded-vs-achieved finger-gap proxy, not real
pad geometry.

**Not yet tried:** hand-tuning the mimic gearing to something other than exactly +-1 (the real
four-bar relationship between finger_joint and each follower is not linear, so some other constant
-- or a full nonlinear correction -- may track the true kinematics far better very close to full
closure, which is exactly the regime that matters for grasping); checking whether the URDF's own
`left_inner_finger_pad_joint`/`right_inner_finger_pad_joint` fixed-offset values are themselves
correct; comparing against the original Gazebo project's own (ROS-side) kinematic behaviour if
that's still inspectable, to see whether Gazebo's mimic plugin used the same linear approximation
or something closer to the real linkage.

**Honest status:** nothing has been trained. No config in this repo has a demonstrated, real-PhysX
grasp on the 2cm target -- the closest state reached is full, stable finger closure with the pads
8.5cm apart, well short of touching a 2cm cube.

### Follow-up: mimic gearing sign was also inverted; real contact achieved, grasp not yet closed (2026-09-22, same session)

Continued from the follower-joint-limit fix above. Re-derived the correct mimic gearing signs
independently two ways and found a second, separate bug:

1. **Analytical**: wrote a standalone forward-kinematics model (`fk_gripper.py`, not checked in)
   directly from the gripper URDF's own link origins for both fingers. With the URDF's own
   multiplier signs (which is what `MIMIC_JOINT_MULTIPLIERS` already encodes) it predicts the
   pads converge to **7mm apart** at `finger_joint`'s own 0.70 rad limit -- essentially touching.
2. **Empirical**: reading back each follower joint's achieved position after commanding
   `finger_joint` showed the *opposite* sign from what `ur5e_grasp_env.py` was assigning to the
   PhysX mimic's `gearing` attribute, for all 5/5 joints, consistently.

Both point the same way: `PhysxMimicJointAPI`'s gearing convention on this Isaac Sim build is not
simply "follower = gearing * reference" the way the schema docs and the URDF's own
`<mimic multiplier=...>` tag both imply -- there's an extra sign flip. Fixed by negating what's
assigned to the attribute (`ur5e_grasp_env.py`, same block as the limit fix). Result: pads now
close to **7.4mm apart in free air**, matching the analytical prediction almost exactly.

That still wasn't enough for a real grasp -- with the (now-corrected) gripper, the scripted demo
approach was landing about 2cm off target even where the IK had visibly converged. Root cause:
`kinematics.py`'s `TOOL_OFFSET`/`TOOL_OFFSET_PER_GRIP` (the flange-to-pad-midpoint calibration)
was fitted against the *old, buggy-sign* gripper's telemetry, so it predicted the pad moving in
close to the opposite direction as the gripper closed. Re-measured directly against the corrected
gripper (two joint configs, consistent to 4 decimals; also matches the fk_gripper.py prediction to
3 decimals) and updated in `kinematics.py`.

A third, separate problem then showed up: the demo controller's per-step differential-IK descend
approach turned out to run right through a wrist singularity (`HOME_Q`'s own `wrist_2 ~ -pi/2`,
which the straight-down approach axis keeps it near) -- confirmed live it settles into a
persistent, non-decaying oscillation a few cm from the target instead of converging, reproduced
with the axis-alignment task on and off and with several step sizes, so it wasn't simply
"overshoot from too large a step". Fixed by switching descend to solve-then-track: converge the
IK once offline (`kinematics.solve_ik`, pure kinematics, no per-step re-linearization noise) into
a fixed joint-angle target when descend starts, then track that fixed target with a plain
joint-space step instead of continuously re-solving. A `yaw_gain` wrist alignment override was
also fighting that joint-space tracking's own (already axis-aligned) wrist_3 command and had to be
disabled specifically during descend.

**Where this leaves things:** the scripted demo now reaches real, repeatable PhysX contact -- the
target cube visibly gets pushed by the closing fingers, with genuine (if small, ~0.01N) two-pad
contact forces, a categorical change from every earlier state (0.00N, gripper closing on nothing).
It has not yet produced a *held* grasp: the target is light enough, and the two pads apparently
still reach it with enough of a timing/alignment mismatch, that it slides away before both sides
trap it, rather than being squeezed. `close_dist_m`, `grasp_height_m` and `close_ramp` were
retuned to the values that produce this (best found: `grasp_height_m~0.0`, `close_dist_m=0.02` --
the descend approach has a small, genuinely-converged residual of ~1.6cm, not 1.5cm, so the old
default never triggered -- `close_ramp~0.15-0.2`) and are now the controller's defaults, but this
is empirical tuning, not a first-principles fix, and further iteration (slower approach with force
feedback, or fixing the remaining small left/right pad timing asymmetry) is likely needed.

**Honest status:** still nothing has been trained. No config in this repo has confirmed a *held*
grasp (both pads gripping simultaneously with force above `min_pad_force_n` for `hold_steps`) on
the 2cm target -- the closest state reached is repeatable, real contact that currently pushes the
target away rather than catching it.
