"""Scripted reach-and-grasp controller used to generate demonstrations.

The thesis' Gazebo pilot got its reliability from a classical MoveIt planner. There is no
ROS/MoveIt inside Isaac Sim, so this uses damped-least-squares inverse kinematics on the
calibrated UR5e model instead. Phases:

  arc      swing around the robot base at a safe radius and height (the home pose and the
           target are on opposite sides of the base, so a straight line would cut through it)
  above    move above the target
  descend  go down onto the target with the approach axis pointing down
  close    close the gripper and hold
  lift     verify the grasp by lifting the target clear of the table

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
    def __init__(self, above_m=0.08, close_dist_m=0.02, pre_dist_m=0.03, grasp_height_m=0.02, grasp_offset_xy=(0.0, 0.015),
                 close_ramp=0.15,
                 close_settle_steps=20, close_max_steps=250, descend_settle_steps=10, descend_max_steps=250,
                 descend_max_joint_step=0.01,
                 lift_m=0.10, yaw_gain=1.0, arc_radius_m=0.5,
                 arc_height_m=0.35, arc_step_rad=0.12, axis_target=(0.0, 0.0, -1.0), axis_gain=1.0):
        self.above_m = above_m
        self.close_dist_m = close_dist_m
        # grasp_offset_xy: (x, y) added to the grasp target in the base frame, alongside
        # grasp_height_m's z. Needed because the residual approach error isn't purely vertical --
        # a height-only sweep left the gripper closing fully past the target with zero contact at
        # every height tried (confirmed live 2026-09-22), which only a joint height+lateral search
        # can actually diagnose. Defaults (z=0.02, y=0.015) are the best point found by that
        # search: both pads make real contact there and one run trapped the target with only 0.8mm
        # of displacement. Flagged honestly, not glossed over: repeating the identical run
        # (same config, same seed) did NOT reproduce that -- confirmed live 2026-09-22 that PhysX's
        # GPU solver gives run-to-run variation at this scale, so this is the right *region*, not a
        # deterministic fix. A held grasp (contact.min_pad_force_n sustained for hold_steps) has
        # not been achieved -- see act()'s contact_force freeze logic and docs/VALIDATION_CHECKLIST.md.
        self.grasp_offset = np.array([grasp_offset_xy[0], grasp_offset_xy[1], grasp_height_m], dtype=float)
        # Continuously re-solving dls_step every env.step() (re-linearizing around the REAL,
        # physically-evolving joint state each time) does not actually converge for the last few
        # cm of the descend approach -- confirmed live 2026-09-22: it settles into a persistent
        # ~2-3cm limit cycle, reproduced with the axis-alignment task on and off and with several
        # max_step sizes, so it isn't simply "too large a step". A FIXED joint-angle target,
        # solved once offline (K.solve_ik, pure kinematics, no per-step re-linearization noise)
        # when descend starts and then tracked with a plain joint-space P step, converges cleanly
        # instead (confirmed live: the same physical arm holds a fixed joint-angle target with no
        # oscillation at all). descend_max_joint_step caps that per-step joint-space P step.
        self.descend_max_joint_step = descend_max_joint_step
        self.descend_settle_steps = descend_settle_steps
        self.descend_max_steps = descend_max_steps
        self._descend_steps = 0
        self._descend_close_steps = 0
        self._descend_q = None
        # Lift once the CLOSE COMMAND has fully ramped to spec.GRIPPER_ACTION_CLOSED (how long that
        # takes depends on close_ramp, so a step count alone was wrong -- confirmed live 2026-09-22:
        # lift_after_steps=14 fired while grip was still only 0.40, well short of closed) and has
        # then held for close_settle_steps more, so a real grasp's achieved position/contact forces
        # have time to settle. close_max_steps is a safety cap in case the command never finishes
        # ramping (e.g. close_ramp <= 0).
        self.close_settle_steps = close_settle_steps
        self.close_max_steps = close_max_steps
        self.lift_m = lift_m
        self._close_steps = 0
        self._commanded_closed_steps = 0
        self._contact_latched = False
        self._contact_latched_steps = 0
        self.yaw_gain = yaw_gain  # wrist_3 gain that lines the finger closing axis up with the cube faces (0 = off)
        self.close_ramp = close_ramp  # gripper action added per step while closing (a fast close knocks the 8 g cube away)
        self.grasp_height_m = grasp_height_m  # tool point above the target centre when grasping (fingertips sit on the table)
        self.pre_dist_m = pre_dist_m
        self.arc_radius_m = arc_radius_m
        self.arc_height_m = arc_height_m
        self.arc_step_rad = arc_step_rad
        self.axis_target = None if axis_target is None else np.asarray(axis_target, dtype=float)
        self.axis_gain = axis_gain
        self.phase = "arc"
        self._close_a = None

    def reset(self):
        self.phase = "arc"
        self._close_a = None
        self._close_steps = 0
        self._commanded_closed_steps = 0
        self._contact_latched = False
        self._contact_latched_steps = 0
        self._descend_steps = 0
        self._descend_close_steps = 0
        self._descend_q = None

    def _goal(self, ee_base, target_base):
        """Task-space goal for phases still driven by per-step dls_step. Also owns every
        phase transition, including descend's (which drives itself via _descend_q in act())."""
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
            goal = target_base + self.grasp_offset + np.array([0.0, 0.0, self.above_m])
            if np.linalg.norm(goal - ee_base) < self.pre_dist_m:
                self.phase = "descend"
            else:
                return goal
        if self.phase == "descend":
            grasp_goal = target_base + self.grasp_offset
            self._descend_steps += 1
            self._descend_close_steps = (
                self._descend_close_steps + 1 if np.linalg.norm(grasp_goal - ee_base) < self.close_dist_m else 0
            )
            at_max_steps = self.descend_max_steps > 0 and self._descend_steps > self.descend_max_steps
            settled = self._descend_close_steps > self.descend_settle_steps
            if at_max_steps or settled:
                self.phase = "close"
        if self.phase == "close":
            self._close_steps += 1
            at_max_steps = self.close_max_steps > 0 and self._close_steps > self.close_max_steps
            # Either a full, uncontested closure (_commanded_closed_steps, unchanged) or a contact
            # latch held for close_settle_steps -- lifting on the very first contact reading would
            # risk lifting on transient noise rather than a real, held touch.
            settled = (self._commanded_closed_steps > self.close_settle_steps
                       or self._contact_latched_steps > self.close_settle_steps)
            if at_max_steps or settled:
                self.phase = "lift"
        if self.phase == "lift":
            return target_base + self.grasp_offset + np.array([0.0, 0.0, self.lift_m])
        return target_base + self.grasp_offset  # close: keep holding the grasp pose

    def act(self, obs: np.ndarray, contact_force=None) -> np.ndarray:
        """contact_force: optional (left_n, right_n) pad force reading. Open-loop closing (the
        command ramps to fully closed regardless of contact) was confirmed live 2026-09-22 to be
        the actual reason a light (20g) target with good alignment still wasn't caught: real,
        repeatable contact force was showing up on the way to full closure, but the command kept
        ramping straight through it, sweeping the target aside before a real squeeze could form
        (checked across a height+lateral grid search that ruled out alignment as the cause -- the
        best point found still peaked at 0.003 N, two orders of magnitude under the 0.5 N confirm
        threshold). When given, this freezes the close ramp the first time either pad reports any
        force, instead of continuing on to spec.GRIPPER_ACTION_CLOSED."""
        q = obs[spec.OBS_JOINT_POS].astype(float)
        grip = float(obs[spec.OBS_GRIPPER])
        # target relative to base_link, expressed in the base frame (the base is yawed by pi)
        target_base = K._rot_z(K.BASE_YAW).T @ obs[spec.OBS_TARGET].astype(float)
        ee_base = K.world_to_base(obs[spec.OBS_EE_POS].astype(float))

        phase_before = self.phase
        goal = self._goal(ee_base, target_base)
        action = np.zeros(spec.ACTION_DIM, dtype=np.float32)

        if self.phase == "descend":
            if phase_before != "descend" or self._descend_q is None:
                grasp_goal = target_base + self.grasp_offset
                self._descend_q = K.solve_ik(q, grasp_goal, grip=0.0, q_ref=HOME_Q,
                                              axis_target=self.axis_target, axis_gain=self.axis_gain)
            dq = np.clip(self._descend_q - q, -self.descend_max_joint_step, self.descend_max_joint_step)
        else:
            # measured tool point drives the error; the calibrated model only supplies the Jacobian
            model_ee = K.tool_point_base(q, grip)
            goal_for_model = model_ee + (goal - ee_base)
            dq = K.dls_step(q, goal_for_model, grip=grip, q_ref=HOME_Q, axis_target=self.axis_target,
                            axis_gain=self.axis_gain, max_step=spec.MAX_JOINT_DELTA_RAD)
        if self.yaw_gain > 0.0 and self.phase != "descend":
            # fingers close along the flange y axis; a cube is only gripped cleanly with that axis parallel to a face.
            # Not applied during descend: _descend_q was solved with the same axis_target/axis_gain already folded
            # in, so this would fight that joint-space tracking's own (already-aligned) wrist_3 command instead of
            # complementing it -- confirmed live 2026-09-22 as the actual cause of the "solve-then-track" descend
            # still oscillating exactly like the old continuous dls_step approach: this override completely replaces
            # dq[5] every step with a correction based on the CURRENT orientation, not the target's.
            y_axis = K.forward_chain(q)[6][:3, 1]
            ang = np.arctan2(y_axis[1], y_axis[0])
            dq[5] = self.yaw_gain * (((ang + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
        action[:6] = np.clip(dq, -spec.MAX_JOINT_DELTA_RAD, spec.MAX_JOINT_DELTA_RAD)
        if self.phase in ("close", "lift"):
            if self._close_a is None:
                self._close_a = float(np.clip(grip / spec.GRIPPER_MAX_CMD, 0.0, 1.0)) * 2 * np.pi - np.pi
            if contact_force is not None and min(contact_force) > 0.0:
                # BOTH pads in contact -- a real squeeze, not one side nudging the target towards
                # the other pad's still-open path. Freeze fully. Tried continuing to close further
                # to build force past this point (self.squeeze_force_n) -- confirmed live
                # 2026-09-22 that this made things reliably *worse* (a bigger, not smaller, target
                # displacement) rather than building real grip force, so it's not done: on this
                # light (20g), low-friction target, continuing to close once both pads already
                # feel it just resumes pushing it, the same failure mode as not latching at all.
                self._contact_latched = True
            if self._contact_latched:
                self._contact_latched_steps += 1
            elif contact_force is not None and max(contact_force) > 0.0:
                # one pad touching, not the other yet -- confirmed live 2026-09-22 that freezing
                # here (the original "either pad" rule) traps nothing: the untouched pad's own
                # gap is still open, so the target just slides along the one contacting pad
                # instead of ever meeting the second one. Keep closing, just gently, so the second
                # pad still gets a chance to arrive before the target is pushed out of reach.
                self._close_a = min(self._close_a + self.close_ramp * 0.25, spec.GRIPPER_ACTION_CLOSED)
            else:
                self._close_a = min(self._close_a + self.close_ramp, spec.GRIPPER_ACTION_CLOSED)
            action[6] = self._close_a
            if self._close_a >= spec.GRIPPER_ACTION_CLOSED - 1e-6:
                self._commanded_closed_steps += 1
        else:
            action[6] = spec.GRIPPER_ACTION_OPEN
        return action
