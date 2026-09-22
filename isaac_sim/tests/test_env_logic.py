"""Exercises Ur5eGraspEnv.reset()/step() with a fake physics backend.

Isaac Sim cannot run in CI, so the simulator objects are replaced by minimal kinematic
stand-ins (joints jump to their targets; pads sit at the calibrated finger-pad point). This
checks the environment's own logic: shapes, reward/termination wiring, contact confirmation,
randomization, logging. It says nothing about PhysX behaviour, which scripts/smoke_test.py
and scripts/validate_collisions.py check on a GPU machine.
"""
import csv

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from ur5e_grasp import kinematics as K
from ur5e_grasp import spec
from ur5e_grasp.config import EnvConfig
from ur5e_grasp.contact import HoldTracker
from ur5e_grasp.demo_controller import HOME_Q, ReachGraspDemoController
from ur5e_grasp.randomization import DomainRandomizer
from ur5e_grasp import ur5e_grasp_env as E

BASE = np.array([0.0, 0.55, 0.78])


class FakeRobot:
    num_dof = 12

    def __init__(self):
        self.q = np.zeros(12)
        self.q[:6] = HOME_Q
        self.target = self.q.copy()
        self.vel = np.zeros(12)

    def get_joint_positions(self, joint_indices=None):
        idx = np.arange(12) if joint_indices is None else np.asarray(joint_indices)
        return self.q[idx][None, :]

    def get_joint_velocities(self, joint_indices=None):
        idx = np.arange(12) if joint_indices is None else np.asarray(joint_indices)
        return self.vel[idx][None, :]

    def set_joint_position_targets(self, values, joint_indices=None):
        idx = np.asarray(joint_indices)
        self.target[idx] = np.asarray(values)[0]

    def set_joint_positions(self, values, joint_indices=None):
        idx = np.asarray(joint_indices)
        self.q[idx] = np.asarray(values)[0]
        self.target[idx] = self.q[idx]

    def set_joint_velocities(self, values):
        self.vel[:] = 0.0

    def set_gains(self, **kw):
        pass

    def set_friction_coefficients(self, *a, **kw):
        pass


class FakeWorld:
    def __init__(self, robot):
        self.robot = robot

    def step(self, render=False):
        r = self.robot
        r.vel = (r.target - r.q) / 0.1
        r.q = r.target.copy()

    def reset(self):
        pass


class FakePad:
    def __init__(self, robot):
        self.robot = robot

    def get_world_poses(self):
        p = K.tool_point_world(self.robot.q[:6], 0.0)  # pads stay put when closing (the fitted per-grip offset is an artifact)
        return p[None, :], np.array([[1.0, 0.0, 0.0, 0.0]])


class FakeBase:
    def get_world_poses(self):
        return BASE[None, :], np.array([[1.0, 0.0, 0.0, 0.0]])


class FakeLego:
    def __init__(self):
        self.pos = np.array([E.LEGO_BASE_X, E.LEGO_BASE_Y, E.LEGO_SPAWN_Z])

    def get_world_pose(self):
        return self.pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def set_world_pose(self, position, orientation):
        self.pos = np.asarray(position, dtype=float)

    def set_linear_velocity(self, v):
        pass

    def set_angular_velocity(self, v):
        pass

    def get_mass(self):
        return 0.02

    def set_mass(self, m):
        self.mass = m


class FakeSensor:
    """Reports 1 N on both pads while the tool is at the target and the gripper is closed."""

    available, method = True, "fake"

    def __init__(self, env):
        self.env = env

    def read(self):
        r, lego = self.env._robot, self.env._lego
        near = np.linalg.norm(K.tool_point_world(r.q[:6], 0.0) - lego.pos) < 0.05
        closed = r.q[6] > 0.4
        return (1.0, 1.0) if (near and closed) else (0.0, 0.0)


def make_env(tmp_path, cfg=None, seed=0):
    cfg = cfg or EnvConfig()
    env = E.Ur5eGraspEnv.__new__(E.Ur5eGraspEnv)
    env.env_id, env.cfg = 0, cfg
    env._rng = np.random.default_rng(seed)
    env._robot = FakeRobot()
    env._world = FakeWorld(env._robot)
    env._left_pad, env._right_pad, env._base_link, env._lego = FakePad(env._robot), FakePad(env._robot), FakeBase(), FakeLego()
    env._arm_dof_indices = np.arange(6)
    env._gripper_dof_index = 6
    env._nv_gripper = False  # fake backend always exercises the URDF-gripper (non-NV) code path
    env._nv_follower_dof = {}
    env.max_joint_delta_rad = spec.MAX_JOINT_DELTA_RAD
    m = 0.05
    env.joint_pos_min = np.array([-2 * np.pi] * 2 + [-np.pi] + [-2 * np.pi] * 3, dtype=np.float32) + m
    env.joint_pos_max = -env.joint_pos_min
    env.curriculum_level = 0.0
    env.MIN_SPAWN_RADIUS_M, env.MAX_SPAWN_RADIUS_M, env.MAX_STANDOFF_M = 0.0, 0.05, 0.30
    env.max_episode_steps, env.sim_step_time, env.physics_dt = cfg.max_episode_steps, cfg.sim_step_time, 1 / 60
    env._substeps_per_step = 6
    env.current_step, env.episode_shaping_sum, env._prev_dist = 0, 0.0, 0.0
    env.current_gripper_pos, env.current_spawn_radius_m = 0.0, 0.0
    env.home_joint_positions = HOME_Q.astype(np.float32)
    env.hardware_csv_path = str(tmp_path / "log.csv")
    with open(env.hardware_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["placeholder"])
    env.global_step_count = 0
    env._randomizer = DomainRandomizer(cfg.randomization, env._rng, spec.neutral_action())
    env._hold = HoldTracker(cfg.contact)
    env._contact_sensor, env._contact_source, env._contact_ready = None, "gap_proxy", False
    env._closed_near_count, env._prev_action, env._warned = 0, spec.neutral_action(), set()
    env._table_z_base = E.TABLE_TOP_Z - float(K.BASE_POS_WORLD[2])
    env._lego_mass0 = 0.02
    env.action_space = None
    return env


def quiet_cfg(**contact):
    cfg = EnvConfig()
    cfg.randomization.enabled = False
    cfg.contact.source = "sensor"
    cfg.contact.hold_steps = 3
    for k, v in contact.items():
        setattr(cfg.contact, k, v)
    return cfg


def install_sensor(env):
    env._contact_sensor, env._contact_source, env._contact_ready = FakeSensor(env), "sensor", True


def test_reset_and_step_shapes(tmp_path):
    env = make_env(tmp_path, quiet_cfg())
    install_sensor(env)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (spec.OBS_DIM,) and np.all(np.isfinite(obs))
    obs, r, term, trunc, info = env.step(spec.neutral_action())
    assert obs.shape == (spec.OBS_DIM,) and np.isfinite(r) and not trunc
    rows = list(csv.reader(open(env.hardware_csv_path)))
    assert len(rows[-1]) == 40  # 33 original columns + 7 new ones


def test_demo_controller_reaches_confirmed_success_through_env(tmp_path):
    env = make_env(tmp_path, quiet_cfg())
    install_sensor(env)
    ctl = ReachGraspDemoController()
    obs, _ = env.reset(seed=2)
    ctl.reset()
    reason = None
    for _ in range(400):
        obs, r, term, trunc, info = env.step(ctl.act(obs))
        if term or trunc:
            reason = info.get("termination_reason")
            break
    assert reason == "Full Task Success", reason
    assert info["is_success"] and info["contact_source"] == "sensor"


def test_no_contact_means_no_success_only_false_grasp(tmp_path):
    env = make_env(tmp_path, quiet_cfg())
    env._contact_sensor, env._contact_source, env._contact_ready = type("S", (), {"read": lambda s: (0.0, 0.0)})(), "sensor", True
    ctl = ReachGraspDemoController()
    obs, _ = env.reset(seed=3)
    ctl.reset()
    for _ in range(400):
        obs, r, term, trunc, info = env.step(ctl.act(obs))
        if term or trunc:
            break
    assert info.get("termination_reason") == "False Grasp -- No Contact Detected"
    assert not info["is_success"]


def test_self_collision_terminates_episode(tmp_path):
    env = make_env(tmp_path, quiet_cfg())
    install_sensor(env)
    env.reset(seed=4)
    env._robot.q[:6] = np.array([0.0, -np.pi / 2, -3.0, 0.0, 0.0, 0.0])  # elbow folded onto the upper arm
    env._robot.target[:6] = env._robot.q[:6]
    _, reward, term, _, info = env.step(spec.neutral_action())
    assert term and info["termination_reason"] == "Self Collision" and reward < -4.5


def test_seeded_resets_are_reproducible_with_randomization(tmp_path):
    def first_obs(seed):
        cfg = EnvConfig()
        cfg.contact.source = "gap_proxy"
        env = make_env(tmp_path, cfg, seed=seed)
        env._contact_ready = True
        env.curriculum_level = 1.0
        return env.reset(seed=seed)[0]

    a, b, c = first_obs(11), first_obs(11), first_obs(12)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_noisy_observation_but_true_reward(tmp_path):
    cfg = EnvConfig()
    cfg.contact.source = "gap_proxy"
    env = make_env(tmp_path, cfg, seed=5)
    env._contact_ready = True
    env.curriculum_level = 1.0
    env.reset(seed=5)
    obs, *_ = env.step(spec.neutral_action())
    true = env._get_obs()
    assert not np.allclose(obs[spec.OBS_JOINT_POS], true[spec.OBS_JOINT_POS])
    assert np.allclose(obs[spec.OBS_EE_POS], true[spec.OBS_EE_POS])  # end-effector pose is left untouched