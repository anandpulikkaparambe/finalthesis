import os

import numpy as np
import pandas as pd

from ur5e_grasp import kinematics as K
from ur5e_grasp.demo_controller import HOME_Q

DATA = os.path.join(os.path.dirname(__file__), "data", "telemetry_sample.csv")


def test_fk_matches_headline_telemetry():
    """The calibrated model must reproduce the logged finger-pad position."""
    d = pd.read_csv(DATA)
    errs = []
    for _, r in d.iterrows():
        q = r[["J1_Pos", "J2_Pos", "J3_Pos", "J4_Pos", "J5_Pos", "J6_Pos"]].values.astype(float)
        p = K.tool_point_world(q, float(r["Gripper_Pos"]))
        errs.append(np.linalg.norm(p - r[["EE_X", "EE_Y", "EE_Z"]].values.astype(float)))
    assert np.median(errs) < 0.02  # median error under 2 cm on the sampled telemetry


def test_home_pose_is_sane():
    assert K.self_clearance(HOME_Q)[0] > 0.0
    assert K.table_clearance(HOME_Q) > 0.05
    p = K.tool_point_world(HOME_Q)
    assert 0.5 < p[2] < 1.2


def test_base_world_roundtrip():
    p = np.array([0.3, -0.2, 0.4])
    assert np.allclose(K.world_to_base(K.base_to_world(p)), p)


def test_jacobian_matches_finite_difference():
    q = HOME_Q + 0.1
    J = K.position_jacobian(q)
    dq = np.array([1e-3, -2e-3, 1e-3, 0.5e-3, 0.0, 1e-3])
    predicted = K.tool_point_base(q + dq) - K.tool_point_base(q)
    assert np.allclose(J @ dq, predicted, atol=2e-6)


def test_segment_distance_known_cases():
    a = np.array([0.0, 0.0, 0.0]); b = np.array([1.0, 0.0, 0.0])
    assert abs(K._segment_distance(a, b, np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])) - 1.0) < 1e-9
    assert abs(K._segment_distance(a, b, np.array([2.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])) - 1.0) < 1e-9
    assert K._segment_distance(a, b, np.array([0.5, -1.0, 0.0]), np.array([0.5, 1.0, 0.0])) < 1e-9


def test_table_clearance_detects_arm_below_plane():
    q = HOME_Q.copy()
    q[1] = -0.2   # shoulder lift lowers the arm towards the table
    q[2] = 0.1
    assert K.table_clearance(q) < K.table_clearance(HOME_Q)


def test_self_clearance_detects_folded_arm():
    q = np.array([0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0])
    q_fold = np.array([0.0, -np.pi / 2, -3.0, 0.0, 0.0, 0.0])  # elbow folded back on the upper arm
    assert K.self_clearance(q_fold)[0] < K.self_clearance(q)[0]


def test_dls_step_moves_towards_target_and_respects_limit():
    q = HOME_Q.copy()
    target = K.tool_point_base(q) + np.array([0.02, 0.0, -0.02])
    dq = K.dls_step(q, target, max_step=0.08)
    assert np.all(np.abs(dq) <= 0.08 + 1e-12)
    before = np.linalg.norm(target - K.tool_point_base(q))
    after = np.linalg.norm(target - K.tool_point_base(q + dq))
    assert after < before