# Changes for full sim-to-real success

Each item: what was wrong, what the code now does, where it lives, and how to check it. All of
it is **implemented but not yet run in Isaac Sim**. "Tested" below means the unit tests in
`isaac_sim/tests/` (fake physics backend), which cannot see PhysX behaviour.

## A. Fix the simulator first

| # | Change | Where | Check |
|---|---|---|---|
| 1 | **Self-collision on.** PhysX articulation self-collision is enabled before the first `world.reset()`; collision pairs inside the Robotiq gripper are filtered so its closed linkage does not fight itself. The asset converter now imports with self-collision on (`--no-self-collision` restores the old behaviour). | `ur5e_grasp_env.py` (`__init__`, `_filter_gripper_internal_pairs`), `assets/convert_urdf_to_usd.py`, `config.self_collision` | `validate_collisions.py`: hold-pose stability and self-collision checks |
| 2 | **Robot collision geometry.** Already present in the latest `main` of the original repo (added after the headline run); kept. | `ur5e_grasp_env.py` (collider wiring) | `validate_collisions.py`: table check |
| 3 | **Grasp success from real contact.** Both finger pads must press on the target (PhysX contact reports) with the gripper commanded closed and the tool near the target, held for `hold_steps` consecutive steps; optional lift requirement. A separate timer produces "False Grasp" if the gripper stays closed near the target without confirmed contact. The finger-gap proxy is only available for comparison runs. | `contact.py`, `ur5e_grasp_env.py`, `reward.py`, `config.contact` | Tested: hold logic, success/false-grasp wiring. **Unverified: the PhysX contact API.** `validate_collisions.py` check 1 and 5 |
| 4 | **Retrain from scratch.** A policy trained without collisions cannot be resumed under different physics. `train.py` starts fresh by default. | `scripts/train.py` | n/a |

## B. Reward

| # | Change | Where |
|---|---|---|
| 5 | Soft penalties when the arm gets close to itself, the table, or a joint limit, plus a terminating "Self Collision" outcome. Clearances come from the calibrated kinematic model (capsule proxy). | `reward.py`, `kinematics.py` |
| 6 | Velocity-penalty coefficient defaults to **0.001** (the headline run used 0.05). Untuned. | `config.reward` |
| 7 | Energy penalty `c_e * sum|tau * qdot| * dt` (falls back to `sum qdot^2` when the simulator gives no efforts). Untuned. | `reward.py`, `config.reward` |
| 8 | Optional approach-axis alignment term; **off by default** (`align_coef = 0`). | `reward.py`, `config.reward` |

## C. Robustness to the real robot

| # | Change | Where |
|---|---|---|
| 9 | Domain randomization: observation noise (values from the Gazebo pilot), a per-episode bias on the perceived target, action noise, and per-episode PD gains, joint friction and target mass. Strength ramps with the curriculum level. The physics setters are guarded and warn once if a setter is missing. | `randomization.py`, `ur5e_grasp_env.py`, `config.randomization` |
| 10 | Control timing: random action delay (0 to 2 steps) and optional action smoothing. The real runs used 0.3 to 3 s per step against 0.1 s in training, so the trained policy should be deployed at the trained rate or with this delay randomized. | `randomization.py`, `config.action_smoothing_alpha` |
| 11 | Start-pose jitter around the home configuration. | `randomization.py` |
| 12 | One shared observation/action layout. The ROS deployment script still builds its own observation; check it against `spec.py` by hand. | `spec.py` |

## D. Training setup

| # | Change | Where |
|---|---|---|
| 13 | More parallel environments: `NUM_ENVS` defaults to 8 in the Vast.ai entrypoint (headline run: 2). | `vastai/train_entrypoint.sh` |
| 14 | Replay buffer 1M (headline run: 100k). Checkpoints every `--save-freq` steps because buffers are large. | `scripts/train.py` |
| 15 | Seeds: `--seed`, and `scripts/run_seeds.sh` trains and evaluates several seeds. | `scripts/train.py`, `scripts/run_seeds.sh` |
| 16 | Curriculum with a configurable start, a lower advance threshold (0.8) and regression when a whole window falls below 0.2. | `curriculum.py`, `config.curriculum` |
| 17 | Demonstrations: a scripted IK controller on the calibrated model (arc around the base, above the target, descend, close). `collect_demos.py` records them in the real environment, `bc_pretrain.py` clones the actor, `train.py --demo-file/--bc-init` uses them. Experimental. | `demo_controller.py`, `scripts/collect_demos.py`, `scripts/bc_pretrain.py` |

## E. Evaluation and deployment

| # | Change | Where |
|---|---|---|
| 18 | Deterministic evaluation, separate from training success, with collisions on, writing JSON and CSV. | `scripts/evaluate.py` |
| 19 | Rerun the staged 16-trial real-hardware protocol on the retrained checkpoint. | `docs/REAL_TRIAL_PROTOCOL.md` |
| 20 | Collision-check safety filter for real steps (opt-in `--safety-filter`); needs the table height in the base frame. Not tested on hardware. | `safety_filter.py`, `deployment/real_phase2_unclamped_run.py` |

## Not done

21. **Visual perception**: excluded on purpose; the target pose is still ground truth.

## Known risks

- **PhysX contact API names** (`RigidPrim` contact arguments, `get_contact_force_matrix`) are the
  ones I expect in Isaac Sim 5.1 and were not verified. If they are wrong the environment stops at
  the first reset with a clear message rather than silently using the gap proxy.
- **Self-collision may jitter** at the wrist or gripper. If `validate_collisions.py` reports a high
  joint speed while holding still, look at the filtered pairs first.
- **The kinematic model is approximate.** Calibrated on the headline telemetry: median error about
  4 to 9 mm, 95th percentile 2 to 3 cm. The capsule radii are hand-set. Its per-grip offset is an
  empirical fit artifact. The proxies are guides for reward shaping and safety, not exact contact.
- **Hyperparameters are untuned starting points.** The soft collision and limit penalties are small (0.05 per metre) on purpose: the thesis' 0.05 velocity penalty showed how easily a penalty outweighs the ~0.002 m/step shaping reward. Termination depths (3 cm) were set from the headline telemetry (the deepest table overlap there was 2.6 cm); check the `Arm_Table_Clearance` and `Self_Clearance` columns of the new telemetry to make sure normal grasp poses do not trigger them.
- **The demonstration controller was only tested in a kinematic simulation.**
