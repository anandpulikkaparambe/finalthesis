import numpy as np

from ur5e_grasp import kinematics as K
from ur5e_grasp.demo_controller import HOME_Q
from ur5e_grasp.safety_filter import SafetyFilter


def test_normal_step_passes_unchanged():
    f = SafetyFilter()
    delta = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    out, note = f.filter(HOME_Q, delta)
    assert out is not None and np.allclose(out, delta) and note == ""


def test_step_into_the_table_is_scaled_or_refused():
    f = SafetyFilter()
    q = HOME_Q.copy()
    q[1], q[2] = -0.5, 0.3   # arm low over the table
    delta = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    out, note = f.filter(q, delta)
    if out is not None:
        s, t, tip = f.measure(q + out)
        s0, t0, tip0 = f.measure(q)
        assert t >= min(f.table_margin_m, t0) - 1e-9 and tip >= min(f.tip_margin_m, tip0) - 1e-9


def test_step_that_folds_arm_onto_itself_is_blocked():
    f = SafetyFilter()
    q = np.array([0.0, -np.pi / 2, -2.9, 0.0, 0.0, 0.0])
    delta = np.array([0.0, 0.0, -0.08, 0.0, 0.0, 0.0])
    s0, _, _ = f.measure(q)
    out, note = f.filter(q, delta)
    if s0 < f.self_margin_m:
        assert out is None or f.measure(q + out)[0] >= s0 - 1e-9


def test_step_that_improves_a_bad_pose_is_allowed():
    f = SafetyFilter()
    q = np.array([0.0, -np.pi / 2, -3.0, 0.0, 0.0, 0.0])   # already overlapping
    unfold = np.array([0.0, 0.0, 0.08, 0.0, 0.0, 0.0])
    out, _ = f.filter(q, unfold)
    assert out is not None