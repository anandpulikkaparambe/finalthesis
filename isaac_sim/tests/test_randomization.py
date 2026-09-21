import numpy as np

from ur5e_grasp import spec
from ur5e_grasp.config import RandomizationConfig
from ur5e_grasp.randomization import DomainRandomizer


def _rand(**kw):
    return DomainRandomizer(RandomizationConfig(**kw), np.random.default_rng(0), spec.neutral_action())


def test_disabled_changes_nothing():
    r = _rand(enabled=False)
    p = r.reset_episode(1.0)
    obs = np.arange(23, dtype=np.float32)
    assert p.strength == 0.0
    assert np.array_equal(r.perturb_observation(obs), obs)
    a = np.ones(7, dtype=np.float32)
    assert np.array_equal(r.process_action(a), a)


def test_strength_ramps_with_curriculum():
    r = _rand()
    assert r.strength(0.0) < r.strength(1.0)
    assert np.isclose(r.strength(1.0), 1.0)


def test_observation_noise_only_touches_intended_slots():
    r = _rand()
    r.reset_episode(1.0)
    obs = np.zeros(23, dtype=np.float32)
    noisy = r.perturb_observation(obs)
    assert np.any(noisy[0:6] != 0) and np.any(noisy[6:12] != 0) and np.any(noisy[19:22] != 0)
    assert np.all(noisy[12:19] == 0) and noisy[22] == 0  # end-effector pose and gripper untouched


def test_noise_does_not_mutate_input():
    r = _rand()
    r.reset_episode(1.0)
    obs = np.zeros(23, dtype=np.float32)
    r.perturb_observation(obs)
    assert np.all(obs == 0)


def test_action_delay_returns_older_action():
    r = _rand(action_noise_rad=0.0, action_delay_steps_max=2, ramp_with_curriculum=False)
    p = r.reset_episode(1.0)
    p.delay_steps = 2
    outs = []
    for i in range(1, 6):
        a = np.zeros(7, dtype=np.float32)
        a[0] = 0.01 * i
        outs.append(float(r.process_action(a)[0]))
    assert outs[2:] == [0.01, 0.02, 0.03] or np.allclose(outs[2:], [0.01, 0.02, 0.03])
    assert outs[0] == 0.0 and outs[1] == 0.0  # neutral actions fill the delay at episode start


def test_sampled_parameters_stay_in_range():
    cfg = RandomizationConfig()
    r = _rand()
    for _ in range(200):
        p = r.reset_episode(1.0)
        assert 0 <= p.delay_steps <= cfg.action_delay_steps_max
        assert cfg.kp_scale_range[0] <= p.kp_scale <= cfg.kp_scale_range[1]
        assert cfg.kd_scale_range[0] <= p.kd_scale <= cfg.kd_scale_range[1]
        assert np.all(np.abs(p.start_jitter) <= cfg.start_joint_jitter_rad + 1e-9)
        assert cfg.joint_friction_range[0] <= p.joint_friction <= cfg.joint_friction_range[1]