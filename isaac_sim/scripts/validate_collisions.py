"""Physics validation for the thesis-driven environment. RUN THIS FIRST on a GPU machine.

The unit tests cannot check anything that depends on PhysX. This script does, and prints a
PASS / INFO / FAIL line per check so problems show up before a long training run:

  1. contact source   the finger-pad contact sensor exists (otherwise the run would fall back
                      to the gap proxy, or stop, depending on the config)
  2. hold-pose        with self-collision on and gripper pairs filtered, the arm holding still
                      does not explode (max joint speed stays small)
  3. self-collision   driving the elbow both ways either ends in a "Self Collision"
                      termination or the joint is physically blocked
  4. table            driving the arm down either ends in "Table Collision" or is blocked;
                      reports the deepest end-effector penetration below the table top
  5. demo grasp       the scripted controller reaches the target and the contact sensor
                      confirms the grasp (end-to-end check of kinematics + contact)

    ~/isaacsim/python.sh scripts/validate_collisions.py --config configs/default.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
parser.add_argument("--demo-episodes", type=int, default=5)
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.set_defaults(headless=True)
args = parser.parse_args()

from ur5e_grasp import spec
from ur5e_grasp.config import load_config
from ur5e_grasp.demo_controller import HOME_Q, ReachGraspDemoController
from ur5e_grasp import ur5e_grasp_env as E

cfg = load_config(args.config)
cfg.randomization.enabled = False  # nominal scene: any difference must come from physics, not noise
env = E.Ur5eGraspEnv(env_id=0, headless=args.headless, log_dir="./validate_telemetry", config=cfg, seed=0)

results = []


def record(name, status, detail):
    results.append((name, status))
    print(f"[{status}] {name}: {detail}", flush=True)


def run_towards(target_q, steps, seed=1):
    """Drive the arm from home towards a full 6-joint target, one clipped delta per step.

    Returns (min_ee_z, min_self, reason, final_q). Unlike run_constant (one joint, a repeated
    delta that isn't clamped to the joint's own reachable range), this can target configurations
    that need more than one joint to move together -- both the self-collision and table checks
    below were producing vacuous results (see their own comments) because a single-joint sweep
    from HOME_Q never actually got close to a real collision within the step budget.
    """
    obs, _ = env.reset(seed=seed)
    reason, min_z, min_self = None, np.inf, np.inf
    target_q = np.asarray(target_q, dtype=float)
    for _ in range(steps):
        q = obs[spec.OBS_JOINT_POS].astype(float)
        a = spec.neutral_action()
        a[:6] = np.clip(target_q - q, -spec.MAX_JOINT_DELTA_RAD, spec.MAX_JOINT_DELTA_RAD)
        obs, r, term, trunc, info = env.step(a)
        min_z = min(min_z, float(obs[14]))
        min_self = min(min_self, info["self_clearance"])
        if term or trunc:
            reason = info.get("termination_reason", "timeout")
            break
    final_q = env._robot.get_joint_positions()[0][:6]
    return min_z, min_self, reason, final_q


# 1. contact source
obs, _ = env.reset(seed=0)
if env._contact_source == "sensor":
    record("contact source", "PASS", f"finger-pad contact sensor active ({env._contact_sensor.method})")
else:
    record("contact source", "FAIL", f"using '{env._contact_source}' (the gap proxy is what the headline run used)")

# 2. hold-pose stability
env.reset(seed=0)
max_vel, reason = 0.0, None
for _ in range(120):
    obs, r, term, trunc, info = env.step(spec.neutral_action())
    max_vel = max(max_vel, float(np.max(np.abs(obs[6:12]))))
    if term or trunc:
        reason = info.get("termination_reason", "timeout")
        break
ok = max_vel < 2.0 and reason in (None, "timeout")
record("hold-pose stability", "PASS" if ok else "FAIL", f"max arm joint speed {max_vel:.2f} rad/s, ended: {reason}")

# 3. self-collision
# A single-joint sweep from HOME_Q (the old check) never actually reaches a self-collision within
# a sane step budget, so it always reported [INFO] regardless of whether real self-collision
# detection works at all -- confirmed live 2026-09-22 that the analytical proxy's own minimum
# clearance in that sweep was -0.001, nowhere near the env's -0.03 termination threshold. Drive
# instead at a configuration a small offline search (K.self_clearance over a joint-1/joint-2 grid)
# confirmed genuinely overlaps: HOME_Q with shoulder_lift=0.4, elbow=2.8 (shoulder-forearm
# clearance -0.13, comfortably past the threshold).
self_target = HOME_Q.copy()
self_target[1], self_target[2] = 0.4, 2.8
z, s, why, final_q = run_towards(self_target, steps=200)
gap = float(np.linalg.norm(self_target - final_q))
print(f"    towards self-collision config: ended={why} min_self_clearance={s:.3f} final-target gap={gap:.2f} rad", flush=True)
hit = why == "Self Collision"
overlapped = s < -0.005  # genuinely overlapped per the analytical proxy
# NOT gap>0.3 alone as "blocked" -- confirmed live 2026-09-22 that a 60-step budget produced
# gap=3.16 with min_self_clearance staying positive throughout (0.007): the arm was moving
# noticeably slower than spec.MAX_JOINT_DELTA_RAD would suggest (fighting gravity/inertia at this
# extended pose, not a collision), so a big gap on its own is not evidence of being blocked by
# anything -- it needs to be paired with actually running out of steps while still far off target.
ran_out = gap > 0.3 and why is None
record("self-collision", "PASS" if (hit or overlapped) else "FAIL",
       "termination fired" if hit else ("overlapping" if overlapped else
       f"reached the known-overlapping config with no collision response at all (final gap {gap:.2f} rad{', ran out of steps' if ran_out else ''})"))

# 4. table
# The old check's PASS condition (`depth < 0.06`) was backwards: depth is negative whenever the
# arm never gets near the table, so "the arm stayed 80cm above the table" satisfied `depth < 0.06`
# and passed vacuously -- confirmed live 2026-09-22 (0.000 m penetration, no termination, still
# [PASS]). Drive instead at a configuration confirmed offline to put the end-effector well below
# the table plane (K.table_clearance over a shoulder_lift sweep): HOME_Q with shoulder_lift=0.6.
table_target = HOME_Q.copy()
table_target[1] = 0.6
z, s, why, final_q = run_towards(table_target, steps=200)
depth = E.TABLE_TOP_Z - z
seen_table = why == "Table Collision"
print(f"    towards table config: ended={why} min_ee_z={z:.3f} (table top {E.TABLE_TOP_Z}) depth={depth:.3f}", flush=True)
record("table", "PASS" if (seen_table or depth > 0.01) else "FAIL",
       f"deepest end-effector penetration {max(depth, 0.0):.3f} m, table-collision termination seen: {seen_table}")

# 5. demonstration grasp
ctl = ReachGraspDemoController()
env.set_curriculum_level(1.0)
wins, reasons, peak_force = 0, {}, 0.0
for ep in range(args.demo_episodes):
    obs, _ = env.reset(seed=100 + ep)
    ctl.reset()
    force = None
    for _ in range(700):  # descend+close alone can take ~500 (their own max_steps defaults) + arc/above
        obs, r, term, trunc, info = env.step(ctl.act(obs, contact_force=force))
        if env._contact_source == "sensor":
            fl, fr = env._contact_sensor.read()
            force = (fl, fr)
            peak_force = max(peak_force, min(fl, fr))
        if term or trunc:
            break
    why = info.get("termination_reason", "timeout")
    reasons[why] = reasons.get(why, 0) + 1
    wins += int(bool(info.get("is_success", False)))
record("demo grasp", "PASS" if wins > 0 else "FAIL",
       f"{wins}/{args.demo_episodes} confirmed grasps, peak two-pad force {peak_force:.2f} N, endings: {reasons}")

n_pass = sum(s == "PASS" for _, s in results)
print(f"VALIDATION_SUMMARY: {n_pass}/{len(results)} PASS "
      f"({sum(s == 'FAIL' for _, s in results)} FAIL, {sum(s == 'INFO' for _, s in results)} INFO)", flush=True)
env.close()
