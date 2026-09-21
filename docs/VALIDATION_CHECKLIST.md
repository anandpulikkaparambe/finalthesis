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
