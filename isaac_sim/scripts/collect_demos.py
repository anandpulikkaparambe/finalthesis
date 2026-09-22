"""Collect demonstrations from the scripted IK controller (source/ur5e_grasp/demo_controller.py).

Runs the controller in the real environment (same observations, same physics, collisions on)
and stores transitions for behaviour cloning (scripts/bc_pretrain.py) or for pre-filling the
SAC replay buffer (scripts/train.py --demo-file). By default only successful episodes are
kept, so the demonstrations are also a check of the whole chain: kinematic calibration,
contact confirmation and collision handling. A low success rate here means the calibration
or the contact sensor needs attention before any training.

    ~/isaacsim/python.sh scripts/collect_demos.py --episodes 100 --out demos.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out", type=str, default="demos.npz")
parser.add_argument("--config", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--curriculum-level", type=float, default=1.0)
parser.add_argument("--keep-failures", action="store_true")
parser.add_argument("--max-steps", type=int, default=700)  # the descend+close phases alone can take up to ~500 (demo_controller.py's own descend_max_steps/close_max_steps defaults), plus arc/above
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.set_defaults(headless=True)
args = parser.parse_args()

from ur5e_grasp.config import load_config
from ur5e_grasp.demo_controller import ReachGraspDemoController
from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

cfg = load_config(args.config)
env = Ur5eGraspEnv(env_id=0, headless=args.headless, log_dir="./demo_telemetry", config=cfg, seed=args.seed)
env.set_curriculum_level(args.curriculum_level)
ctl = ReachGraspDemoController()

obs_l, act_l, rew_l, nxt_l, done_l = [], [], [], [], []
n_success = 0
reasons = {}
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    ctl.reset()
    ep_rows = []
    info = {}
    for _ in range(args.max_steps):
        force = env._contact_sensor.read() if env._contact_source == "sensor" else None
        action = ctl.act(obs, contact_force=force)
        nxt, reward, terminated, truncated, info = env.step(action)
        ep_rows.append((obs, action, reward, nxt, terminated))
        obs = nxt
        if terminated or truncated:
            break
    ok = bool(info.get("is_success", False))
    reasons[info.get("termination_reason", "timeout")] = reasons.get(info.get("termination_reason", "timeout"), 0) + 1
    n_success += int(ok)
    if ok or args.keep_failures:
        for o, a, r, n, d in ep_rows:
            obs_l.append(o); act_l.append(a); rew_l.append(r); nxt_l.append(n); done_l.append(d)
    print(f"episode {ep}: success={ok} steps={len(ep_rows)} reason={info.get('termination_reason', 'timeout')}", flush=True)

np.savez(args.out, obs=np.array(obs_l, dtype=np.float32), actions=np.array(act_l, dtype=np.float32),
         rewards=np.array(rew_l, dtype=np.float32), next_obs=np.array(nxt_l, dtype=np.float32),
         dones=np.array(done_l, dtype=np.float32))
print(f"DEMOS: {n_success}/{args.episodes} successful, {len(obs_l)} transitions saved to {args.out}", flush=True)
print(f"termination reasons: {reasons}", flush=True)
env.close()
