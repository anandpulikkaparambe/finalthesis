"""Grasp confirmation from physical contact, replacing the finger-gap proxy.

The thesis' headline run declared success from the difference between the commanded and
the achieved finger position, in a scene where the robot had no collision geometry. Here
success needs (1) both finger pads pressing on the target, (2) the gripper commanded
closed, (3) the tool close to the target, and (4) the state held for several steps,
optionally with the target lifted. HoldTracker is pure Python and unit-tested;
GraspContactSensor wraps Isaac Sim's PhysX contact reports and can only be checked on a
machine that runs Isaac Sim (see scripts/validate_collisions.py).
"""
from typing import Optional

import numpy as np

from ur5e_grasp.config import ContactConfig


class HoldTracker:
    """Counts consecutive steps in which a grasp condition holds."""

    def __init__(self, cfg: ContactConfig):
        self.cfg = cfg
        self.count = 0
        self.rest_height = None

    def reset(self, target_rest_height: Optional[float] = None):
        self.count = 0
        self.rest_height = target_rest_height

    def update(self, contact_both_pads: bool, dist_ok: bool, closed: bool,
               target_height: Optional[float] = None) -> bool:
        lifted_ok = True
        if self.cfg.lift_required_m > 0.0:
            lifted_ok = (
                target_height is not None and self.rest_height is not None
                and (target_height - self.rest_height) >= self.cfg.lift_required_m
            )
        if contact_both_pads and dist_ok and closed and lifted_ok:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.cfg.hold_steps


def pads_in_contact(force_left_n: float, force_right_n: float, cfg: ContactConfig) -> bool:
    return force_left_n >= cfg.min_pad_force_n and force_right_n >= cfg.min_pad_force_n


def gap_proxy_contact(gripper_cmd: float, gripper_achieved: float, closed_amount: float, cfg: ContactConfig) -> bool:
    """The original proxy, kept for comparison runs only."""
    return closed_amount > 0.5 and (gripper_cmd - gripper_achieved) > cfg.gap_threshold


class GraspContactSensor:
    """Reads finger-pad contact forces against the target from PhysX contact reports.

    `left_pad` and `right_pad` must be RigidPrim views created with
    prepare_contact_sensors=True and the target's prim path in contact_filter_prim_paths_expr
    (see Ur5eGraspEnv._make_pad_prim). The API differs slightly between Isaac Sim versions,
    so the read path tries the known variants and reports `available=False` if none works.
    """

    def __init__(self, left_pad, right_pad, physics_dt: float):
        self.left_pad, self.right_pad, self.dt = left_pad, right_pad, physics_dt
        self.available = False
        self.method = None
        self._probe()

    def _read_one(self, prim):
        for name in ("get_contact_force_matrix", "get_net_contact_forces"):
            fn = getattr(prim, name, None)
            if fn is None:
                continue
            for args in ((), (self.dt,)):
                try:
                    forces = np.asarray(fn(*args), dtype=float)
                    return float(np.linalg.norm(forces.reshape(-1, 3), axis=1).sum()), name
                except Exception:
                    continue
        return None, None

    def _probe(self):
        left, name_l = self._read_one(self.left_pad)
        right, name_r = self._read_one(self.right_pad)
        if left is not None and right is not None:
            self.available, self.method = True, name_l

    def read(self):
        """Return (force_left_n, force_right_n); zeros if unavailable."""
        if not self.available:
            return 0.0, 0.0
        left, _ = self._read_one(self.left_pad)
        right, _ = self._read_one(self.right_pad)
        return (left or 0.0), (right or 0.0)
