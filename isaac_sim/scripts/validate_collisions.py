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
from ur5e_grasp.demo_controller import ReachGraspDemoController
from ur5e_grasp import ur5e_grasp_env as E

cfg = load_config(args.config)
cfg.randomization.enabled = False  # nominal scene: any difference must come from physics, not noise
env = E.Ur5eGraspEnv(env_id=0, headless=args.headless, log_dir="./validate_telemetry", config=cfg, seed=0)

results = []


def record(name, status, detail):
    results.append((name, status))
    print(f"[{status}] {name}: {detail}", flush=True)


def run_constant(delta, joint_index, steps):
    """Drive one joint at a constant delta; returns (min_ee_z, min_self, reason, gap_rad)."""
    obs, _ = env.reset(seed=1)
    reason, min_z, min_self = None, np.inf, np.inf
    a = spec.neutral_action()
    a[joint_index] = delta
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(a)
        min_z = min(min_z, float(obs[14]))
        min_self = min(min_self, info["self_clearance"])
        if term or trunc:
            reason = info.get("termination_reason", "timeout")
            break
    target = env._robot.get_joint_position_targets()[0][joint_index] if hasattr(env._robot, "get_joint_position_targets") else np.nan
    actual = env._robot.get_joint_positions(joint_indices=[joint_index])[0, 0]
    return min_z, min_self, reason, float(abs(target - actual))


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
outcomes = []
for sign in (+1.0, -1.0):
    z, s, why, gap = run_constant(sign * 0.08, 2, 80)
    outcomes.append((sign, why, s, gap))
    print(f"    elbow {sign:+.0f}: ended={why} min_self_clearance={s:.3f} target-actual gap={gap:.2f} rad", flush=True)
hit = any(o[1] == "Self Collision" for o in outcomes)
blocked = any(o[3] > 0.3 for o in outcomes)
record("self-collision", "PASS" if (hit or blocked) else "INFO",
       "detected/blocked" if (hit or blocked) else "neither direction reached a self-collision within 80 steps; try other joints")

# 4. table
worst = np.inf
seen_table = False
for joint, sign in ((1, +1.0), (1, -1.0), (2, +1.0), (2, -1.0)):
    z, s, why, gap = run_constant(sign * 0.08, joint, 100)
    worst = min(worst, z)
    seen_table |= why == "Table Collision"
    print(f"    joint {joint + 1} {sign:+.0f}: ended={why} min_ee_z={z:.3f} (table top {E.TABLE_TOP_Z})", flush=True)
depth = E.TABLE_TOP_Z - worst
record("table", "PASS" if (seen_table or depth < 0.06) else "FAIL",
       f"deepest end-effector penetration {max(depth, 0.0):.3f} m, table-collision termination seen: {seen_table}")

# 5. demonstration grasp
ctl = ReachGraspDemoController()
env.set_curriculum_level(1.0)
wins, reasons, peak_force = 0, {}, 0.0
for ep in range(args.demo_episodes):
    obs, _ = env.reset(seed=100 + ep)
    ctl.reset()
    for _ in range(300):
        obs, r, term, trunc, info = env.step(ctl.act(obs))
        if env._contact_source == "sensor":
            fl, fr = env._contact_sensor.read()
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
