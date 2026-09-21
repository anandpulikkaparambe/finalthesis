import numpy as np

from ur5e_grasp import kinematics as K
from ur5e_grasp import spec
from ur5e_grasp.demo_controller import HOME_Q, ReachGraspDemoController


def _obs(q, grip, target_world):
    o = np.zeros(spec.OBS_DIM, dtype=np.float32)
    o[spec.OBS_JOINT_POS] = q
    o[spec.OBS_EE_POS] = K.tool_point_world(q, grip)
    o[15] = 1.0
    o[spec.OBS_TARGET] = target_world - np.array([0.0, 0.55, 0.78])
    o[spec.OBS_GRIPPER] = grip
    return o


def test_controller_reaches_random_targets_without_collision_proxy_violations():
    rng = np.random.default_rng(3)
    for _ in range(8):
        r = 0.05 * np.sqrt(rng.uniform())
        th = rng.uniform(0, 2 * np.pi)
        target = np.array([0.802 + r * np.cos(th), 0.29 + r * np.sin(th), 0.79])
        ctl = ReachGraspDemoController()
        q = HOME_Q.copy()
        worst_self, worst_table = np.inf, np.inf
        for _ in range(400):
            a = ctl.act(_obs(q, 0.0, target))
            assert np.all(np.abs(a[:6]) <= spec.MAX_JOINT_DELTA_RAD + 1e-6)
            q = q + a[:6]
            worst_self = min(worst_self, K.self_clearance(q)[0])
            worst_table = min(worst_table, K.table_clearance(q))
            if ctl.phase == "close":
                break
        assert ctl.phase == "close"
        assert np.linalg.norm(K.tool_point_world(q) - target) < 0.03
        assert worst_self > 0.0
        assert worst_table > -0.01   # the fingertip is allowed to touch the table next to the target


def test_close_phase_commands_gripper_closed():
    ctl = ReachGraspDemoController()
    ctl.phase = "close"
    target = np.array([0.802, 0.29, 0.79])
    q = HOME_Q.copy()
    a = ctl.act(_obs(q, 0.0, target))
    assert a[6] == spec.GRIPPER_ACTION_CLOSED and np.all(a[:6] == 0)