"""Domain randomization for the Isaac Sim environment.

The thesis' Isaac Sim pipeline had no randomization at all; the Gazebo pilot only had
joint position/velocity observation noise. This module adds, per episode and per step:

  * observation noise (joint position/velocity, a per-episode bias plus jitter on the
    perceived target position; ground-truth target pose is still used, perception is out of scope)
  * action noise and a random action delay (models the slower, laggier real control loop)
  * start-pose jitter around the home configuration
  * physics parameters (PD gains, joint friction, target mass), applied by the environment

Strength scales with the curriculum level, so early training sees a mild version.
Everything sampled here is pure numpy so it can be tested without Isaac Sim.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ur5e_grasp.config import RandomizationConfig

# Observation layout (see spec.py): 0:6 joint pos, 6:12 joint vel, 12:19 EE pose, 19:22 target, 22 gripper
_JPOS, _JVEL, _TARGET = slice(0, 6), slice(6, 12), slice(19, 22)


@dataclass
class EpisodeRandomization:
    strength: float = 0.0
    start_jitter: np.ndarray = field(default_factory=lambda: np.zeros(6))
    target_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delay_steps: int = 0
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    joint_friction: float = 0.0
    lego_mass_scale: float = 1.0


class DomainRandomizer:
    def __init__(self, cfg: RandomizationConfig, rng: np.random.Generator, neutral_action: np.ndarray):
        self.cfg = cfg
        self.rng = rng
        self.neutral_action = np.asarray(neutral_action, dtype=np.float32)
        self.params = EpisodeRandomization()
        self._history = deque(maxlen=cfg.action_delay_steps_max + 1)

    def strength(self, curriculum_level: float) -> float:
        if not self.cfg.enabled:
            return 0.0
        ramp = (0.25 + 0.75 * float(np.clip(curriculum_level, 0.0, 1.0))) if self.cfg.ramp_with_curriculum else 1.0
        return self.cfg.scale * ramp

    def reset_episode(self, curriculum_level: float) -> EpisodeRandomization:
        s = self.strength(curriculum_level)
        c, rng = self.cfg, self.rng
        p = EpisodeRandomization(strength=s)
        if s > 0.0:
            p.start_jitter = rng.uniform(-1.0, 1.0, 6) * c.start_joint_jitter_rad * s
            p.target_bias = rng.normal(0.0, 1.0, 3) * c.obs_target_bias_m * s
            p.delay_steps = int(rng.integers(0, int(round(c.action_delay_steps_max * s)) + 1)) if c.action_delay_steps_max > 0 else 0
            if c.physics_randomization:
                p.kp_scale = 1.0 + s * (rng.uniform(*c.kp_scale_range) - 1.0)
                p.kd_scale = 1.0 + s * (rng.uniform(*c.kd_scale_range) - 1.0)
                p.joint_friction = s * rng.uniform(*c.joint_friction_range)
                p.lego_mass_scale = 1.0 + s * (rng.uniform(*c.lego_mass_scale_range) - 1.0)
        self.params = p
        self._history.clear()
        for _ in range(self._history.maxlen):
            self._history.append(self.neutral_action.copy())
        return p

    def perturb_observation(self, obs: np.ndarray) -> np.ndarray:
        """Noisy copy of the observation the policy sees. The true state is left untouched."""
        s, c = self.params.strength, self.cfg
        if s <= 0.0:
            return obs
        noisy = obs.copy()
        rng = self.rng
        noisy[_JPOS] += rng.normal(0.0, c.obs_joint_pos_noise_rad * s, 6)
        noisy[_JVEL] += rng.normal(0.0, c.obs_joint_vel_noise_rad_s * s, 6)
        noisy[_TARGET] += self.params.target_bias + rng.normal(0.0, c.obs_target_noise_m * s, 3)
        return noisy.astype(np.float32)

    def process_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action noise, then the random delay (an action issued now takes effect later)."""
        s, c = self.params.strength, self.cfg
        a = np.asarray(action, dtype=np.float32).copy()
        if s > 0.0:
            a[:6] += self.rng.normal(0.0, c.action_noise_rad * s, 6).astype(np.float32)
        self._history.append(a)
        d = min(self.params.delay_steps, len(self._history) - 1)
        return self._history[-1 - d].copy()
