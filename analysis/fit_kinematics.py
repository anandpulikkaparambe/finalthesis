"""Fit the UR5e base pose and tool offset to logged telemetry.

The environment logs the joint angles and the world-frame finger-pad midpoint at every step.
With the published DH parameters the only unknowns are the base position, the (known,
pi) base yaw, and the flange-to-fingertip offset. This reproduces the constants in
isaac_sim/source/ur5e_grasp/kinematics.py from the headline run's telemetry.

    python analysis/fit_kinematics.py isaac_sim/tests/data/telemetry_sample.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

path = sys.argv[1] if len(sys.argv) > 1 else "isaac_sim/tests/data/telemetry_sample.csv"
d = pd.read_csv(path)
Q = d[["J1_Pos", "J2_Pos", "J3_Pos", "J4_Pos", "J5_Pos", "J6_Pos"]].values
P = d[["EE_X", "EE_Y", "EE_Z"]].values
G = d["Gripper_Pos"].values

A = np.array([0, -0.425, -0.3922, 0, 0, 0]); D = np.array([0.1625, 0, 0, 0.1333, 0.0997, 0.0996])
AL = np.array([np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0])


def flange(q):
    T = np.eye(4)
    for i in range(6):
        ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(AL[i]), np.sin(AL[i])
        T = T @ np.array([[ct, -st * ca, st * sa, A[i] * ct], [st, ct * ca, -ct * sa, A[i] * st], [0, sa, ca, D[i]], [0, 0, 0, 1]])
    return T


Ts = np.array([flange(q) for q in Q])
R = np.diag([-1.0, -1.0, 1.0])  # base yaw = pi


def predict(x):
    t = x[0:3][None, :] + np.outer(G, x[3:6])
    return (Ts[:, :3, 3] + np.einsum("nij,nj->ni", Ts[:, :3, :3], t)) @ R.T + x[6:9]


fit = least_squares(lambda x: (predict(x) - P).ravel(), np.array([0, 0.012, 0.173, 0, 0, 0, 0, 0.55, 0.78]), loss="soft_l1", f_scale=0.01)
err = np.linalg.norm(predict(fit.x) - P, axis=1)
print("tool offset (open)       :", np.round(fit.x[0:3], 4))
print("tool offset per grip unit:", np.round(fit.x[3:6], 4), "(empirical; do not read it as real geometry)")
print("base position (world)    :", np.round(fit.x[6:9], 4))
print(f"position error: median {1000 * np.median(err):.1f} mm, 95th percentile {1000 * np.percentile(err, 95):.1f} mm, n={len(err)}")