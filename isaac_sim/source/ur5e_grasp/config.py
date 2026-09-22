"""Configuration for the UR5e reach-and-grasp environment.

Every number that the thesis-driven changes introduce lives here, so a run is fully
described by one JSON file (saved next to the checkpoints by scripts/train.py).

Defaults reproduce the "corrected" settings described in docs/CHANGES.md. They are
starting points, not tuned values: none of them has been trained against yet.
"""
import json
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from typing import Tuple


@dataclass
class RewardConfig:
    shaping_floor: float = -5.0            # per-episode floor on the dense reward sum
    velocity_penalty_coef: float = 0.001   # 0.05 in the thesis' headline run
    energy_penalty_coef: float = 0.0005    # x sum|tau*qdot| * dt (or sum qdot^2 if no torque signal)
    joint_limit_penalty_coef: float = 0.01
    joint_limit_margin_rad: float = 0.15
    self_collision_margin_m: float = 0.03  # soft penalty starts when clearance < margin
    self_collision_soft_coef: float = 0.05  # per metre of shortfall; keep the per-step penalty small next to the ~0.002-0.02 m/step shaping reward
    self_collision_terminates: bool = True
    self_collision_depth_m: float = 0.03   # terminate only when the proxy overlap is deeper than this (the proxy is approximate)
    self_collision_penalty: float = -5.0
    table_margin_m: float = 0.02
    table_soft_coef: float = 0.05
    table_collision_penalty: float = -5.0
    table_collision_depth_m: float = 0.03  # arm-table proxy penetration that terminates (headline poses overlapped by at most 2.6 cm)
    max_standoff_m: float = 0.30           # scales the table-collision penalty, as in the original env
    align_coef: float = 0.0               # optional approach-axis alignment term (off by default)
    success_bonus: float = 15.0
    false_grasp_penalty: float = -5.0
    kill_switch_penalty: float = -5.0
    kill_switch_speed_rad_s: float = 10.0
    target_lost_penalty: float = -5.0
    success_distance_m: float = 0.05


@dataclass
class ContactConfig:
    # "sensor": PhysX contact reports on the finger pads (recommended).
    # "gap_proxy": the commanded-vs-achieved finger position gap used in the thesis' headline run.
    source: str = "sensor"
    allow_gap_proxy_fallback: bool = False  # if the sensor cannot be created, fail loudly by default
    min_pad_force_n: float = 0.5            # both pads must exceed this against the target
    hold_steps: int = 5                     # consecutive steps the grasp must hold
    false_grasp_extra_steps: int = 10       # closed near the target this long without confirmed contact = false grasp
    lift_required_m: float = 0.0            # >0 also requires the target to be lifted by this much
    gap_threshold: float = 0.13             # only used by the gap proxy


@dataclass
class RandomizationConfig:
    enabled: bool = True
    scale: float = 1.0
    ramp_with_curriculum: bool = True       # strength = scale * (0.25 + 0.75 * curriculum_level)
    # observation noise (matches the Gazebo pilot's confirmed values)
    obs_joint_pos_noise_rad: float = 0.005
    obs_joint_vel_noise_rad_s: float = 0.01
    obs_target_bias_m: float = 0.01         # constant per-episode error in the perceived target
    obs_target_noise_m: float = 0.002       # per-step jitter of the perceived target
    # actuation
    action_noise_rad: float = 0.005
    action_delay_steps_max: int = 2
    # start pose
    start_joint_jitter_rad: float = 0.05
    # physics
    physics_randomization: bool = True
    kp_scale_range: Tuple[float, float] = (0.8, 1.2)
    kd_scale_range: Tuple[float, float] = (0.8, 1.2)
    joint_friction_range: Tuple[float, float] = (0.0, 0.05)
    lego_mass_scale_range: Tuple[float, float] = (0.7, 1.5)


@dataclass
class CurriculumConfig:
    initial_level: float = 0.0
    window_episodes: int = 20
    advance_threshold: float = 0.8
    regress_threshold: float = 0.2          # step back if a full window falls below this
    level_step: float = 0.05
    warmup_steps: int = 50000               # level is capped at warmup_cap until then
    warmup_cap: float = 0.2


@dataclass
class EnvConfig:
    reward: RewardConfig = field(default_factory=RewardConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    self_collision: bool = True             # enable PhysX articulation self-collision
    filter_gripper_internal_pairs: bool = True
    robot_gravity_compensated: bool = True  # no gravity on the robot links: a real UR holds its pose by itself, and joint targets here are current+delta, so any sag would compound
    max_episode_steps: int = 500
    sim_step_time: float = 0.1
    action_smoothing_alpha: float = 0.0     # 0 = off; else a_t = alpha*a_{t-1} + (1-alpha)*a_t


def _build(cls, data):
    if data is None:
        return cls()
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        sub = getattr(cls(), f.name)
        if is_dataclass(sub):
            kwargs[f.name] = _build(type(sub), value)
        elif isinstance(sub, tuple) and isinstance(value, (list, tuple)):
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def config_from_dict(data) -> EnvConfig:
    return _build(EnvConfig, data or {})


def load_config(path_or_none) -> EnvConfig:
    if not path_or_none:
        return EnvConfig()
    with open(path_or_none, "r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def save_config(cfg: EnvConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
