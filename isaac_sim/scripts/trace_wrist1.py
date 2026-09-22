"""Diagnostic: trace wrist1/wrist2 joint trajectories while driving toward the table-collision
target, to understand why wrist1 fails to converge under run_towards()'s clipped-delta commanding.
Not part of the validation suite -- throwaway debugging script.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from ur5e_grasp import spec
from ur5e_grasp.config import load_config
from ur5e_grasp.demo_controller import HOME_Q
from ur5e_grasp import ur5e_grasp_env as E

cfg = load_config("")
cfg.randomization.enabled = False
env = E.Ur5eGraspEnv(env_id=0, headless=True, log_dir="./trace_telemetry", config=cfg, seed=0)

target_q = HOME_Q.copy()
target_q[1], target_q[2], target_q[3], target_q[4] = -1.25, 2.3, 0.2, 1.5

obs, _ = env.reset(seed=1)
for i in range(200):
    q = obs[spec.OBS_JOINT_POS].astype(float)
    a = spec.neutral_action()
    a[:6] = np.clip(target_q - q, -spec.MAX_JOINT_DELTA_RAD, spec.MAX_JOINT_DELTA_RAD)
    obs, r, term, trunc, info = env.step(a)
    if i % 5 == 0 or i == 199:
        print(f"i={i:3d} q3(wrist1)={q[3]:+.3f} cmd3={a[3]:+.3f} q4(wrist2)={q[4]:+.3f} cmd4={a[4]:+.3f}", flush=True)
    if term or trunc:
        print(f"terminated at i={i}: {info.get('termination_reason')}", flush=True)
        break

env.close()
