"""Observation and action layout shared by training, evaluation and the real-robot script.

The real deployment hit several bugs where its observation did not match the training
layout (missing gripper model, zeroed joint states, wrong end-effector frame). Keeping the
layout in one place, and checking it in tests, is the guard against that.
"""
import numpy as np

# 23-D observation
OBS_DIM = 23
OBS_JOINT_POS = slice(0, 6)
OBS_JOINT_VEL = slice(6, 12)
OBS_EE_POSE = slice(12, 19)     # finger-pad midpoint xyz (world) + right-pad quaternion
OBS_EE_POS = slice(12, 15)
OBS_TARGET = slice(19, 22)      # target position relative to base_link
OBS_GRIPPER = 22

# 7-D action: 6 joint-position deltas (rad per step) + 1 gripper value in [-pi, pi]
ACTION_DIM = 7
MAX_JOINT_DELTA_RAD = 0.08
GRIPPER_ACTION_OPEN = -np.pi
GRIPPER_ACTION_CLOSED = np.pi
GRIPPER_MAX_CMD = 0.8           # finger_joint target at fully closed


def gripper_closed_amount(gripper_action: float) -> float:
    """0 (open) .. 1 (closed). The environment treats > 0.5 as 'commanded closed'."""
    return float((np.clip(gripper_action, -np.pi, np.pi) + np.pi) / (2 * np.pi))


def gripper_cmd(gripper_action: float) -> float:
    return gripper_closed_amount(gripper_action) * GRIPPER_MAX_CMD


def neutral_action() -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[6] = GRIPPER_ACTION_OPEN
    return a
