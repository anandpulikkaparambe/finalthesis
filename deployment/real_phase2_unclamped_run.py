#!/usr/bin/env python3
"""
Phase 2 RL Policy UNCLAMPED real-hardware run (grasp phase only).

Unlike real_phase2_closed_loop_dry_run.py, this sends the policy's own raw
per-step joint delta directly (still bounded by the trained action space,
+/-MAX_JOINT_DELTA_RAD -- that bound is inherent to what the policy was ever
allowed to output, not an added safety clamp). The real gripper is actually
actuated here too, unlike every prior real-hardware script for this project.

Explicit user-requested trade-off (2026-09-03): no per-step magnitude clamp,
but every step is executed SLOWLY -- each single-point FollowJointTrajectory
goal gets a long STEP_DURATION_S, so real angular velocity stays far below
the trained/simulated regime (~0.08 rad in STEP_DURATION_S seconds, vs. the
sim's 0.08 rad per 0.1s step) even though the full delta is sent. This is a
deliberate middle ground: full policy authority over WHERE it moves, tight
control over HOW FAST.

Safety design still kept, none of this was asked to be removed:
- safety_mode checked after every single step, hard abort (not next-loop) on
  anything but NORMAL.
- Real ground truth for grasp success comes from the gripper driver's own
  /robotiq_2f_gripper/object_grasped topic (a real Modbus-reported signal,
  not a guess) -- not available in sim, a genuine advantage here.
- A hard MAX_STEPS bound still exists (this is an episode-length bound
  matching the trained task's own max_episode_steps, not a magnitude clamp).
- The arm is moved to the taught real_pick_position waypoint first (the same
  pre-grasp starting point the policy was always trained to act from via
  Gazebo's handoff_to_pregrasp), not left wherever it happened to be.

Checkpoint: ur3e_sac_interrupted.zip (step 323,840) -- 0/5 in the immediately
preceding Gazebo eval (see project_phase2_rl_status memory), so this run's
own expectation should be calibrated accordingly. This is a supervised,
watched test, not a claim the policy is ready.
"""

import argparse
import math
import threading
import time

import numpy as np
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from robotiq_2f_gripper_msgs.action import MoveTwoFingerGripper
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint
from ur_dashboard_msgs.msg import SafetyMode
from stable_baselines3 import SAC

UR3E_JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# Real Cartesian pick target (base_link frame), sampled live via TF 2026-09-01
# at the Freedrive-taught real_pick_position waypoint -- see
# real_phase2_dry_run.py's docstring for the same values and full rationale.
REAL_PICK_XYZ = np.array([0.806, -0.266, 0.059], dtype=np.float32)
# Corrected 2026-09-11 (second pass). The first re-measurement this same day ([0.817, 0.411,
# 0.063]) was taken while gripper_rsp.launch.py's joint-state merger had a since-fixed bug
# (joint_state_publisher's source_list was silently zeroing all 6 arm joints in its merged
# output -- confirmed live via `ros2 topic echo /combined_joint_states`), so that measurement's
# FK was computed at an all-zero arm pose, not the real one -- garbage, not a real calibration
# change. After fixing the merger (custom joint_state_merger.py, verified with injected known
# values before going live again), re-measured at real_pick_position/gripper-open once more:
# left_inner_finger_pad [0.746, -0.236, 0.057], right_inner_finger_pad [0.866, -0.296, 0.061]
# -- close together in all 3 axes, as a real parallel gripper should read, unlike the corrupted
# measurement's wildly-separated pads. Midpoint [0.806, -0.266, 0.059] matches the *original*
# 2026-09-01 tool0-frame constant ([0.802, -0.260, 0.237]) closely in X/Y, with a ~0.178m Z
# offset -- consistent with the expected tool0-to-gripper-tip distance cited elsewhere this
# session. The original constant was fine all along; the mid-session "0.49m Y discrepancy"
# finding was itself an artifact of the merger bug, not a real calibration drift.

MAX_JOINT_DELTA_RAD = 0.08  # the policy's own trained action-space bound -- not a safety clamp

# --- The actual safety knobs for this script ---
STEP_DURATION_S = 0.5       # raised 2026-09-11 from 0.3 at user request -- testing whether the
                             # new step-29 cluster (2 consecutive runs, post-pipeline-fix) is
                             # step-count-based (would still fail ~step 29) or time-based (would
                             # fail ~step 17, same ~8.7s elapsed as the 0.3s runs).
                             # Prior context: the 0.1s (full
                             # trained-speed) run was e-stopped at step 6, too early to be
                             # useful; 0.3s is a middle ground, ~3x below trained pace.
MAX_STEPS = 500               # raised 2026-09-11 from 80 at user request to match this
                              # checkpoint's actual Isaac Sim training budget exactly
                              # (ur5e_grasp_env.py: max_episode_steps=500) -- 30 and then 80 were
                              # both partial fractions of what this policy actually trained with.
                              # At STEP_DURATION_S=2.0 this is ~16.7min of continuous real motion.
MOVE_TO_START_DURATION_S = 8.0

# Real gripper actuation convention (robotiq_2f_gripper_msgs/MoveTwoFingerGripper):
# 0.0 = fully CLOSED, 0.14 = fully OPEN (opposite of sim's GripperCommand convention).
GRIPPER_OPEN_M = 0.14
GRIPPER_CLOSED_M = 0.0
GRIPPER_SPEED = 0.3   # slow close, matching the "but slowly" intent for the whole run
GRIPPER_FORCE = 1.0   # max real force -- this run exists specifically to test whether a
                       # firmer grip fixes the post-lift slip pattern found in Gazebo

# Same single-joint nudge real_pick_and_place.py already proved works from this exact
# pick pose (no calibrated real Cartesian IK on this rig -- see that file's docstring),
# same 15s duration as its one verified real lift.
LIFT_SHOULDER_DELTA = -0.15
LIFT_DURATION_S = 15.0
LIFT_HOLD_S = 5.0

from gymnasium import spaces
LEGACY_OBSERVATION_SPACE = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
LEGACY_ACTION_SPACE = spaces.Box(
    low=np.array([-MAX_JOINT_DELTA_RAD] * 6 + [-math.pi], dtype=np.float32),
    high=np.array([MAX_JOINT_DELTA_RAD] * 6 + [math.pi], dtype=np.float32),
    dtype=np.float32,
)


class Phase2UnclampedRunNode(Node):
    def __init__(self, checkpoint_path, legacy=False):
        super().__init__('phase2_unclamped_run_node')

        self.legacy = legacy
        self.current_joint_positions = np.zeros(6, dtype=np.float32)
        self.current_joint_velocities = np.zeros(6, dtype=np.float32)
        self.current_gripper_pos = 0.0
        self.object_grasped = None
        self.safety_mode = None

        self.create_subscription(JointState, '/joint_states', self._joint_state_cb, 10)
        self.create_subscription(JointState, '/robotiq_2f_gripper/joint_states', self._gripper_state_cb, 10)
        self.create_subscription(Bool, '/robotiq_2f_gripper/object_grasped', self._grasp_cb, 10)

        safety_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(SafetyMode, '/io_and_status_controller/safety_mode', self._safety_cb, safety_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        self._gripper_client = ActionClient(
            self, MoveTwoFingerGripper, '/robotiq_2f_gripper_action')

        self.waypoints = self._load_waypoints()

        self.get_logger().info(f"Loading SAC checkpoint: {checkpoint_path} (legacy={legacy})")
        load_kwargs = {}
        if legacy:
            load_kwargs['custom_objects'] = {
                'observation_space': LEGACY_OBSERVATION_SPACE,
                'action_space': LEGACY_ACTION_SPACE,
            }
        self.model = SAC.load(checkpoint_path, device='cpu', **load_kwargs)
        self.get_logger().info("Checkpoint loaded.")

    def _load_waypoints(self):
        share_dir = get_package_share_directory('ur3e_sorting')
        path = f"{share_dir}/config/real_hardware_waypoints.yaml"
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return {
            name: [pose[j] for j in UR3E_JOINT_NAMES]
            for name, pose in data.items()
            if isinstance(pose, dict)
        }

    def _joint_state_cb(self, msg):
        for i, name in enumerate(UR3E_JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.current_joint_velocities[i] = msg.velocity[idx]

    def _gripper_state_cb(self, msg):
        if 'finger_joint' in msg.name:
            self.current_gripper_pos = msg.position[msg.name.index('finger_joint')]

    def _grasp_cb(self, msg):
        self.object_grasped = msg.data

    def _safety_cb(self, msg):
        self.safety_mode = msg.mode

    def safety_ok(self):
        return self.safety_mode == 1  # 1 == NORMAL; None (no message yet) fails closed

    def build_observation(self):
        # Updated 2026-09-11: real base_link->left/right_inner_finger_pad TF now resolves
        # (see real_hardware_description/gripper_rsp.launch.py -- the gripper is now in the
        # live URDF and its joint states are merged into TF). Matches ur5e_grasp_env.py's own
        # _get_ee_pose() exactly: xyz = midpoint of both pads, quat = right pad only. This
        # replaces the previous tool0 approximation (a ~0.17m fixed-offset guess from where
        # the policy actually trained) with the real reference frame.
        try:
            t_left = self.tf_buffer.lookup_transform(
                'base_link', 'left_inner_finger_pad', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            t_right = self.tf_buffer.lookup_transform(
                'base_link', 'right_inner_finger_pad', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2))
            mid = np.array([
                (t_left.transform.translation.x + t_right.transform.translation.x) / 2.0,
                (t_left.transform.translation.y + t_right.transform.translation.y) / 2.0,
                (t_left.transform.translation.z + t_right.transform.translation.z) / 2.0,
            ], dtype=np.float32)
            ee_pose = np.concatenate((mid, np.array([
                t_right.transform.rotation.x, t_right.transform.rotation.y,
                t_right.transform.rotation.z, t_right.transform.rotation.w,
            ], dtype=np.float32)))
        except Exception as e:
            self.get_logger().warn(f"finger-pad TF lookup failed: {e}")
            return None, None

        dist = float(np.linalg.norm(ee_pose[:3] - REAL_PICK_XYZ))

        if self.legacy:
            obs = np.concatenate((
                self.current_joint_positions,
                self.current_joint_velocities,
                ee_pose,
                REAL_PICK_XYZ,
                [self.current_gripper_pos],
            ), dtype=np.float32)
        else:
            obs = np.concatenate((
                self.current_joint_positions,
                self.current_joint_velocities,
                ee_pose,
                REAL_PICK_XYZ,
                [self.current_gripper_pos],
                [0.0],  # phase_indicator, permanently 'grasp'
            ), dtype=np.float32)
        return obs, dist

    def move_to_joint_positions(self, target_positions, duration_s, name, tol=0.03, attempts=2):
        for attempt in range(1, attempts + 1):
            if not self._traj_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"{name}: trajectory action server unavailable!")
                return False
            current = self.current_joint_positions
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(UR3E_JOINT_NAMES)
            point = JointTrajectoryPoint()
            positions = []
            for i, target in enumerate(target_positions):
                cur = current[i]
                # shortest-angle wraparound shift, same logic as real_pick_and_place.py
                target = cur + (((target - cur) + math.pi) % (2 * math.pi) - math.pi)
                positions.append(target)
            point.positions = positions
            sec = int(duration_s)
            nsec = int(round((duration_s - sec) * 1e9))
            point.time_from_start = Duration(sec=sec, nanosec=nsec)
            goal.trajectory.points = [point]

            future = self._traj_client.send_goal_async(goal)
            while not future.done():
                time.sleep(0.05)
            handle = future.result()
            if not handle or not handle.accepted:
                self.get_logger().error(f"{name}: goal rejected (attempt {attempt}/{attempts}).")
                continue
            res_future = handle.get_result_async()
            while not res_future.done():
                time.sleep(0.05)
            result = res_future.result().result
            if result.error_code != 0:
                self.get_logger().error(f"{name}: FollowJointTrajectory failed: {result.error_string}")
                continue
            time.sleep(0.3)
            deltas = [abs(((self.current_joint_positions[i] - target_positions[i]) + math.pi) % (2 * math.pi) - math.pi)
                      for i in range(6)]
            if all(d <= tol for d in deltas):
                self.get_logger().info(f"{name}: reached and verified.")
                return True
            self.get_logger().warn(f"{name}: reported success but failed verification (attempt {attempt}/{attempts}).")
        return False

    def send_raw_step(self, raw_delta):
        """Sends the policy's own delta (bounded only by the trained action space) as a
        single-point FollowJointTrajectory goal with a generous STEP_DURATION_S."""
        target = self.current_joint_positions + raw_delta
        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory action server unavailable!")
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(UR3E_JOINT_NAMES)
        point = JointTrajectoryPoint()
        point.positions = target.astype(float).tolist()
        point.time_from_start = Duration(sec=int(STEP_DURATION_S),
                                          nanosec=int((STEP_DURATION_S % 1) * 1e9))
        goal.trajectory.points = [point]
        future = self._traj_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.05)
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error("Step goal rejected.")
            return False
        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.05)
        result = res_future.result().result
        if result.error_code != 0:
            self.get_logger().error(f"Step failed: {result.error_string}")
            return False
        return True

    def operate_gripper(self, position, speed=GRIPPER_SPEED, force=GRIPPER_FORCE):
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Gripper action server unavailable.")
            return False
        goal = MoveTwoFingerGripper.Goal()
        goal.target_position = float(position)
        goal.target_speed = float(speed)
        goal.target_force = float(force)
        future = self._gripper_client.send_goal_async(goal)
        while not future.done():
            time.sleep(0.05)
        handle = future.result()
        if not handle or not handle.accepted:
            self.get_logger().error("Gripper goal rejected.")
            return False
        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.05)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--start-waypoint', default='real_pick_position',
                         help='Taught waypoint name (from real_hardware_waypoints.yaml) to move to '
                              'before starting the policy loop -- e.g. real_look_position to test '
                              'reach-from-a-distance instead of starting already at the pick spot.')
    parser.add_argument('--stochastic', action='store_true',
                         help='Sample from the policy action distribution instead of always taking '
                              'the deterministic argmax action -- added 2026-09-11 at user request so '
                              'repeated runs from the same start pose do not retrace the identical '
                              'trajectory every time.')
    parser.add_argument('--safety-filter', action='store_true',
                         help='Opt-in: predict each step with the calibrated UR5e model and scale down or refuse steps that would cause self-collision or table contact (see isaac_sim/source/ur5e_grasp/safety_filter.py). Off by default.')
    parser.add_argument('--table-z-base', type=float, default=0.0,
                         help='Table plane height in the robot base frame (m); 0.0 when the arm is mounted on the table top. Measure it before using --safety-filter.')
    args, _ = parser.parse_known_args()

    safety_filter = None
    if args.safety_filter:
        import os as _os
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'isaac_sim', 'source'))
        from ur5e_grasp.safety_filter import SafetyFilter
        safety_filter = SafetyFilter(table_z_base=args.table_z_base)

    rclpy.init()
    node = Phase2UnclampedRunNode(args.checkpoint, legacy=args.legacy)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    wait_deadline = time.time() + 15.0
    while node.safety_mode is None and time.time() < wait_deadline:
        time.sleep(0.2)
    if node.safety_mode is None:
        node.get_logger().error("Never received a safety_mode reading within 15s -- aborting before any motion.")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)
        return
    time.sleep(0.5)

    try:
        if not node.safety_ok():
            node.get_logger().error(f"Safety mode not NORMAL at start (mode={node.safety_mode}) -- aborting.")
            return

        node.get_logger().info(f"Opening gripper and moving to {args.start_waypoint}...")
        node.operate_gripper(GRIPPER_OPEN_M, speed=0.3, force=0.5)
        if not node.move_to_joint_positions(
                node.waypoints[args.start_waypoint], MOVE_TO_START_DURATION_S, "move_to_start"):
            node.get_logger().error(f"Failed to reach {args.start_waypoint} -- aborting before policy loop.")
            return
        if not node.safety_ok():
            node.get_logger().error(f"Safety mode not NORMAL after move-to-start (mode={node.safety_mode}) -- aborting.")
            return

        node.get_logger().info(
            f"=== UNCLAMPED RUN: raw policy deltas (bounded +/-{MAX_JOINT_DELTA_RAD}rad), "
            f"{STEP_DURATION_S}s/step, max {MAX_STEPS} steps, gripper force={GRIPPER_FORCE} speed={GRIPPER_SPEED} ==="
        )

        grasp_succeeded = False
        for i in range(MAX_STEPS):
            obs, dist = node.build_observation()
            if obs is None:
                node.get_logger().warn("No valid observation -- stopping.")
                break

            action, _ = node.model.predict(obs, deterministic=not args.stochastic)
            raw_delta = np.clip(action[:6], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
            if safety_filter is not None:
                filtered, why = safety_filter.filter(node.current_joint_positions, raw_delta)
                if filtered is None:
                    node.get_logger().error(f"[step {i:02d}] Safety filter blocked the step ({why}) -- stopping run.")
                    break
                if why:
                    node.get_logger().warn(f"[step {i:02d}] Safety filter: {why}")
                raw_delta = filtered
            gripper_raw = float(action[6])
            closedness = float(np.clip((gripper_raw + math.pi) / (2 * math.pi), 0.0, 1.0))
            gripper_target_m = GRIPPER_OPEN_M - closedness * (GRIPPER_OPEN_M - GRIPPER_CLOSED_M)

            node.get_logger().info(
                f"[step {i:02d}] dist={dist:.4f} raw_delta(rad)={np.round(raw_delta, 4).tolist()} "
                f"gripper_raw={gripper_raw:.3f} -> gripper_target={gripper_target_m:.3f}m "
                f"object_grasped={node.object_grasped}"
            )

            if not node.send_raw_step(raw_delta):
                node.get_logger().error(f"[step {i:02d}] Step execution failed -- stopping run.")
                break
            node.operate_gripper(gripper_target_m)

            time.sleep(0.3)
            if not node.safety_ok():
                node.get_logger().error(
                    f"[step {i:02d}] Safety mode not NORMAL after step (mode={node.safety_mode}) -- stopping immediately.")
                break

            if dist < 0.05 and node.object_grasped:
                node.get_logger().info(
                    f"[step {i:02d}] SUCCESS: dist={dist:.4f} < 0.05 AND object_grasped=True (real Modbus signal).")
                grasp_succeeded = True
                break

        if grasp_succeeded:
            lift_target = list(node.current_joint_positions)
            lift_target[UR3E_JOINT_NAMES.index('shoulder_lift_joint')] += LIFT_SHOULDER_DELTA
            node.get_logger().info(
                f"Grasp confirmed -- lifting (shoulder_lift {LIFT_SHOULDER_DELTA:+.3f}rad over {LIFT_DURATION_S}s)...")
            lift_ok = node.move_to_joint_positions(lift_target, LIFT_DURATION_S, "lift", tol=0.05)
            if not node.safety_ok():
                node.get_logger().error(f"Safety mode not NORMAL after lift (mode={node.safety_mode}) -- stopping.")
            elif not lift_ok:
                node.get_logger().warn("Lift move did not verify -- checking grasp status anyway.")

            node.get_logger().info(f"Holding for {LIFT_HOLD_S}s to check if the grasp survives the lift...")
            hold_deadline = time.time() + LIFT_HOLD_S
            still_grasped = node.object_grasped
            while time.time() < hold_deadline:
                time.sleep(0.5)
                still_grasped = node.object_grasped
                if not node.safety_ok():
                    node.get_logger().error(f"Safety mode not NORMAL during hold (mode={node.safety_mode}) -- stopping.")
                    break

            if still_grasped:
                node.get_logger().info(
                    "LIFT-AND-HOLD SUCCESS: object_grasped=True after lift + "
                    f"{LIFT_HOLD_S}s hold (real Modbus signal) -- first confirmed real full grasp-and-lift.")
            else:
                node.get_logger().warn(
                    "LIFT-AND-HOLD FAILED: object_grasped=False after lift -- the block was lost during/after "
                    "the lift, matching the post-lift slip pattern already diagnosed in Gazebo.")

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("=== UNCLAMPED RUN ENDED ===")
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == '__main__':
    main()
