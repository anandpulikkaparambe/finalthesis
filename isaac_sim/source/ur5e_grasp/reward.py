"""Reward and terminal logic for the reach-and-grasp task, as pure functions.

Keeping this free of Isaac Sim lets it be unit-tested (tests/test_reward.py) and reused
by the evaluation script. The environment builds a StepQuantities from the simulator and
calls dense_reward() then classify_terminal().
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ur5e_grasp.config import RewardConfig


@dataclass
class StepQuantities:
    prev_dist: float
    dist: float
    joint_vel: np.ndarray                 # (6,) rad/s
    joint_pos: np.ndarray                 # (6,) rad
    joint_pos_min: np.ndarray             # (6,)
    joint_pos_max: np.ndarray             # (6,)
    efforts: Optional[np.ndarray] = None  # (6,) N*m if the simulator provides them
    dt: float = 0.1
    self_clearance: float = np.inf        # m, negative = overlap (proxy)
    table_clearance: float = np.inf       # m, negative = below the table plane (proxy)
    align_cos: float = 1.0                # cos of the angle between approach axis and direction to target


def dense_reward(q: StepQuantities, cfg: RewardConfig):
    """Per-step dense reward (before the episode floor) and its named terms."""
    shaping = q.prev_dist - q.dist
    velocity = cfg.velocity_penalty_coef * float(np.sum(np.abs(q.joint_vel)))
    if q.efforts is not None:
        power = float(np.sum(np.abs(q.efforts * q.joint_vel)))
    else:
        power = float(np.sum(np.square(q.joint_vel)))
    energy = cfg.energy_penalty_coef * power * q.dt

    lo = np.clip((q.joint_pos - q.joint_pos_min) / max(cfg.joint_limit_margin_rad, 1e-6), 0.0, 1.0)
    hi = np.clip((q.joint_pos_max - q.joint_pos) / max(cfg.joint_limit_margin_rad, 1e-6), 0.0, 1.0)
    limit = cfg.joint_limit_penalty_coef * float(np.sum(1.0 - np.minimum(lo, hi)))

    self_short = max(0.0, cfg.self_collision_margin_m - q.self_clearance) if np.isfinite(q.self_clearance) else 0.0
    table_short = max(0.0, cfg.table_margin_m - q.table_clearance) if np.isfinite(q.table_clearance) else 0.0
    self_pen = cfg.self_collision_soft_coef * self_short
    table_pen = cfg.table_soft_coef * table_short

    align = cfg.align_coef * (q.align_cos - 1.0)  # 0 when aligned, negative otherwise

    terms = {
        "shaping": shaping, "velocity": -velocity, "energy": -energy, "joint_limit": -limit,
        "self_soft": -self_pen, "table_soft": -table_pen, "align": align,
    }
    return float(sum(terms.values())), terms


def apply_floor(reward: float, shaping_sum: float, floor: float):
    """Clamp the per-episode dense sum at `floor`; returns (reward, new_sum)."""
    prospective = shaping_sum + reward
    if prospective < floor:
        return floor - shaping_sum, floor
    return reward, prospective


@dataclass
class TerminalInputs:
    grasp_confirmed: bool         # contact + hold (+ lift) confirmed by the contact monitor
    false_grasp: bool             # closed near the target long enough without confirmed contact
    joint_vel: np.ndarray
    table_collision: bool
    self_collision: bool
    target_lost: bool
    dist_to_target: float = 0.0


def classify_terminal(t: TerminalInputs, cfg: RewardConfig):
    """Return (terminated, reward_delta, reason). Later conditions override earlier ones,
    matching the ordering of the original environment."""
    terminated, delta, reason = False, 0.0, None

    if t.grasp_confirmed:
        terminated, delta, reason = True, cfg.success_bonus, "Full Task Success"
    elif t.false_grasp:
        terminated, delta, reason = True, cfg.false_grasp_penalty, "False Grasp -- No Contact Detected"

    if np.any(np.abs(t.joint_vel) > cfg.kill_switch_speed_rad_s):
        terminated, delta, reason = True, cfg.kill_switch_penalty, "Velocity Kill-Switch Triggered"

    if t.table_collision:
        terminated, delta, reason = (
            True, cfg.table_collision_penalty * (1.0 + t.dist_to_target / cfg.max_standoff_m), "Table Collision",
        )

    if t.self_collision and cfg.self_collision_terminates:
        terminated, delta, reason = True, cfg.self_collision_penalty, "Self Collision"

    if t.target_lost:
        terminated, delta, reason = True, cfg.target_lost_penalty, "Target Lost (Out of Bounds)"

    return terminated, float(delta), reason