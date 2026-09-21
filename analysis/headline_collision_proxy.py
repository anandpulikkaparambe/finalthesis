"""How often did the headline run's logged poses overlap under the capsule collision proxy?

The headline run was trained with no robot collision geometry and self-collision off, so
nothing stopped the arm from passing through itself. This applies the calibrated capsule
proxy (isaac_sim/source/ur5e_grasp/kinematics.py) to the logged joint angles. The proxy is
approximate (hand-set radii, centimetre-level model error): read the result as an indication,
not as an exact count.

    python analysis/headline_collision_proxy.py
"""
import collections
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "isaac_sim", "source"))
from ur5e_grasp import kinematics as K

cols = ["J1_Pos", "J2_Pos", "J3_Pos", "J4_Pos", "J5_Pos", "J6_Pos", "Gripper_Pos", "EE_Z"]
files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "headline_run", "telemetry_env*_every100th_step.csv")))
d = pd.concat([pd.read_csv(f, usecols=cols) for f in files])
n = len(d)

self_clear, pairs, table_clear, table_part = [], collections.Counter(), [], collections.Counter()
for _, r in d.iterrows():
    q, g = r[cols[:6]].values.astype(float), float(r["Gripper_Pos"])
    s, pair = K.self_clearance(q, g)
    self_clear.append(s)
    if s < 0:
        pairs[pair] += 1
    worst = (None, np.inf)
    for name, p, qq, rad in K.capsules(q, g):
        if name == "shoulder":
            continue
        low = (p[2] - K.RADIUS_TOOL_TABLE) if name == "tool" else (min(p[2], qq[2]) - rad)
        if low < worst[1]:
            worst = (name, low)
    table_clear.append(worst[1])
    if worst[1] < 0:
        table_part[worst[0]] += 1
self_clear, table_clear = np.array(self_clear), np.array(table_clear)

print(f"{n} logged poses from {len(files)} environments (every 100th step)\n")
tip = d["EE_Z"].values - 0.78
print(f"fingertip height above the table top (logged): median {np.median(tip):.3f} m, "
      f"5th-95th percentile {np.percentile(tip, 5):.3f} to {np.percentile(tip, 95):.3f} m; below the table top in {100 * np.mean(tip < 0):.1f} % of poses")
print(f"arm or gripper body below the table plane (proxy): {100 * np.mean(table_clear < 0):.1f} % of poses "
      f"({dict(table_part)})")
print(f"  deeper than 1 cm: {100 * np.mean(table_clear < -0.01):.1f} %, deeper than 2 cm: {100 * np.mean(table_clear < -0.02):.1f} %, deeper than 5 cm: {100 * np.mean(table_clear < -0.05):.1f} %  (deepest {-table_clear.min():.3f} m)")
print(f"\narm overlapping itself (proxy): {100 * np.mean(self_clear < 0):.1f} % of poses; "
      f"deeper than 2 cm: {100 * np.mean(self_clear < -0.02):.1f} %, deeper than 5 cm: {100 * np.mean(self_clear < -0.05):.1f} %")
print(f"worst pair when overlapping: {dict(pairs)}")