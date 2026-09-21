"""Scripted reach-and-grasp controller used to generate demonstrations.

The thesis' Gazebo pilot got its reliability from a classical MoveIt planner. There is no
ROS/MoveIt inside Isaac Sim, so this uses damped-least-squares inverse kinematics on the
calibrated UR5e model instead. Phases:

  arc      swing around the robot base at a safe radius and height (the home pose and the
           target are on opposite sides of the base, so a straight line would cut through it)
  above    move above the target
  descend  go down onto the target with the approach axis pointing down
  close    close the gripper and hold

It works on the same 23-D observation the policy sees, so it runs in the environment with
no extra state. Experimental: the demonstrations are only as good as the calibrated
kinematics; check them with scripts/collect_demos.py (it reports success and clearances)
before using them to warm-start a policy.
"""
import numpy as np

from ur5e_grasp import kinematics as K
from ur5e_grasp import spec

HOME_Q = np.array([2.040, -1.589, 1.991, -1.973, -1.571, 0.0])


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ReachGraspDemoController:
    def __init__(self, above_m=0.08, close_dist_m=0.02, pre_dist_m=0.03, arc_radius_m=0.5,
                 arc_height_m=0.35, arc_step_rad=0.12, axis_target=(0.0, 0.0, -1.0), axis_gain=1.0):
        self.above_m = above_m
        self.close_dist_m = close_dist_m
        self.pre_dist_m = pre_dist_m
        self.arc_radius_m = arc_radius_m
        self.arc_height_m = arc_height_m
        self.arc_step_rad = arc_step_rad
        self.axis_target = None if axis_target is None else np.asarray(axis_target, dtype=float)
        self.axis_gain = axis_gain
        self.phase = "arc"

    def reset(self):
        self.phase = "arc"

    def _goal(self, ee_base, target_base):
        if self.phase == "arc":
            phi_now = np.arctan2(ee_base[1], ee_base[0])
            phi_goal = np.arctan2(target_base[1], target_base[0])
            dphi = _wrap(phi_goal - phi_now)
            radius = max(self.arc_radius_m, float(np.linalg.norm(ee_base[:2])) * 0.0 + self.arc_radius_m)
            if abs(dphi) < 0.15 and abs(ee_base[2] - self.arc_height_m) < 0.05:
                self.phase = "above"
            else:
                phi = phi_now + np.clip(dphi, -self.arc_step_rad, self.arc_step_rad)
                return np.array([radius * np.cos(phi), radius * np.sin(phi), self.arc_height_m])
        if self.phase == "above":
            goal = target_base + np.array([0.0, 0.0, self.above_m])
            if np.linalg.norm(goal - ee_base) < self.pre_dist_m:
                self.phase = "descend"
            else:
                return goal
        if self.phase == "descend":
            if np.linalg.norm(target_base - ee_base) < self.close_dist_m:
                self.phase = "close"
            else:
                return target_base.copy()
        return ee_base.copy()

    def act(self, obs: np.ndarray) -> np.ndarray:
        q = obs[spec.OBS_JOINT_POS].astype(float)
        grip = float(obs[spec.OBS_GRIPPER])
        # target relative to base_link, expressed in the base frame (the base is yawed by pi)
        target_base = K._rot_z(K.BASE_YAW).T @ obs[spec.OBS_TARGET].astype(float)
        ee_base = K.world_to_base(obs[spec.OBS_EE_POS].astype(float))

        goal = self._goal(ee_base, target_base)
        action = np.zeros(spec.ACTION_DIM, dtype=np.float32)
        if self.phase == "close":
            action[6] = spec.GRIPPER_ACTION_CLOSED
            return action
        # measured tool point drives the error; the calibrated model only supplies the Jacobian
        model_ee = K.tool_point_base(q, grip)
        goal_for_model = model_ee + (goal - ee_base)
        dq = K.dls_step(q, goal_for_model, grip=grip, q_ref=HOME_Q, axis_target=self.axis_target,
                        axis_gain=self.axis_gain, max_step=spec.MAX_JOINT_DELTA_RAD)
        action[:6] = np.clip(dq, -spec.MAX_JOINT_DELTA_RAD, spec.MAX_JOINT_DELTA_RAD)
        action[6] = spec.GRIPPER_ACTION_OPEN
        return action
