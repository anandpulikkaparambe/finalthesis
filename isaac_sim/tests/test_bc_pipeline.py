"""Demonstration -> behaviour cloning -> replay-buffer prefill, end to end on the fake backend.

Skipped when stable-baselines3 / torch are not installed. Checks the code paths of
scripts/collect_demos.py, scripts/bc_pretrain.py and the prefill block of scripts/train.py
(shapes, action scaling, buffer contents). It does not say the demonstrations help learning.
"""
import subprocess
import sys
import os

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

import test_env_logic as T
from ur5e_grasp.demo_controller import ReachGraspDemoController

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "bc_pretrain.py")


def _collect(tmp_path, episodes=6):
    obs_l, act_l, rew_l, nxt_l, done_l = [], [], [], [], []
    for ep in range(episodes):
        env = T.make_env(tmp_path, T.quiet_cfg(), seed=ep)
        T.install_sensor(env)
        ctl = ReachGraspDemoController()
        obs, _ = env.reset(seed=ep)
        ctl.reset()
        for _ in range(300):
            a = ctl.act(obs)
            nxt, r, term, trunc, info = env.step(a)
            obs_l.append(obs); act_l.append(a); rew_l.append(r); nxt_l.append(nxt); done_l.append(term)
            obs = nxt
            if term or trunc:
                break
    return dict(obs=np.array(obs_l, np.float32), actions=np.array(act_l, np.float32), rewards=np.array(rew_l, np.float32),
                next_obs=np.array(nxt_l, np.float32), dones=np.array(done_l, np.float32))


def test_bc_pretrain_reduces_loss_and_imitates(tmp_path):
    demos = _collect(tmp_path)
    path = tmp_path / "demos.npz"
    np.savez(path, **demos)
    out = tmp_path / "bc.zip"
    res = subprocess.run([sys.executable, SCRIPT, "--demos", str(path), "--out", str(out), "--epochs", "60", "--device", "cpu"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    losses = [float(l.split("loss")[1]) for l in res.stdout.splitlines() if "BC loss" in l]
    assert losses[-1] < losses[0] * 0.5
    from stable_baselines3 import SAC
    model = SAC.load(str(out), device="cpu")
    pred, _ = model.predict(demos["obs"][:200], deterministic=True)
    lo, hi = np.array([-0.08] * 6 + [-np.pi]), np.array([0.08] * 6 + [np.pi])
    err = np.abs(pred - demos["actions"][:200]) / (hi - lo)
    assert np.median(err) < 0.15   # median error under 15% of the action range


def test_replay_buffer_prefill_path(tmp_path):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC

    demos = _collect(tmp_path, episodes=2)
    low, high = np.array([-0.08] * 6 + [-np.pi], np.float32), np.array([0.08] * 6 + [np.pi], np.float32)

    class Dummy(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (23,), np.float32)
        action_space = spaces.Box(low, high, dtype=np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(23, np.float32), {}

        def step(self, a):
            return np.zeros(23, np.float32), 0.0, True, False, {}

    model = SAC("MlpPolicy", Dummy(), buffer_size=5000, device="cpu")
    n = len(demos["actions"])
    for i in range(n):   # same calls as the prefill block in scripts/train.py
        scaled = model.policy.scale_action(demos["actions"][i][None, :])
        model.replay_buffer.add(demos["obs"][i][None, :], demos["next_obs"][i][None, :], scaled,
                                np.array([demos["rewards"][i]]), np.array([bool(demos["dones"][i])]), [{}])
    assert model.replay_buffer.size() == n
    assert np.all(np.abs(model.replay_buffer.actions[:n]) <= 1.0 + 1e-6)