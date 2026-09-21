"""Success-gated curriculum with regression and a configurable start.

The headline run used: +0.05 per 20-episode window at >= 90% success, level capped at 0.2
for the first 50k steps. In that run the success rate stayed at zero for roughly the first
300k steps of the resumed session and the level only started to move after ~360k, so this
version starts from a configurable level, uses a lower advance threshold, and steps back if
a whole window falls below a floor. Pure Python, unit-tested.
"""
from collections import deque

from ur5e_grasp.config import CurriculumConfig


class CurriculumManager:
    def __init__(self, cfg: CurriculumConfig):
        self.cfg = cfg
        self.level = float(cfg.initial_level)
        self.window = deque(maxlen=cfg.window_episodes)

    def record_episode(self, success: bool):
        self.window.append(1.0 if success else 0.0)

    def update(self, num_timesteps: int) -> float:
        cfg = self.cfg
        if len(self.window) >= cfg.window_episodes:
            rate = sum(self.window) / len(self.window)
            if rate >= cfg.advance_threshold:
                self.level += cfg.level_step
                self.window.clear()
            elif rate <= cfg.regress_threshold and self.level > cfg.initial_level:
                self.level -= cfg.level_step
                self.window.clear()
        cap = cfg.warmup_cap if num_timesteps < cfg.warmup_steps else 1.0
        self.level = max(cfg.initial_level, min(self.level, cap, 1.0))
        return self.level
