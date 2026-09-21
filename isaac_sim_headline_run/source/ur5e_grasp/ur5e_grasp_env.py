"""Gymnasium environment for the UR5e + Robotiq-140 reach-and-grasp task, running
natively in Isaac Sim.

This replaces remoteservertraining-UR5e/src/ur3e_rl/ur3e_rl/ur3e_env.py's ROS2 +
Gazebo + MoveIt2 stack. The observation space, action space, reward shaping,
termination/kill-switch logic and curriculum-widened spawn radius are ported
term-for-term from that file (see its own extensive inline history for *why* each
piece is shaped the way it is -- e.g. CONTACT_GAP_THRESHOLD=0.13, the potential-based
distance shaping, the SHAPING_FLOOR safety net) so training results stay comparable
across the Gazebo and Isaac Sim backends.

What's mechanically different, by design:
  - No ROS2/MoveIt2/OMPL anywhere. Isaac Sim's own Articulation/RigidObject API
    replaces JointTrajectory publishing + TF2 lookups + a Gazebo /set_pose service
    call. There is no classical-planner "handoff" phase -- this always runs the
    source project's own "pure RL, no handoff" variant (UR3E_USE_CLASSICAL_HANDOFF=
    false in the Gazebo env): home pose to grasp in one continuous episode. A
    per-reset OMPL plan has no equivalent without ROS/MoveIt, and parallel
    throughput on Isaac Sim comes from running multiple env processes / an Isaac
    Sim GridCloner setup (see scripts/train.py), not a classical planner per reset.
  - Collision detection is reduced for this first port: `table_collision` uses only
    the end-effector-height proxy (ee_z < 0.02, same threshold the Gazebo env used
    as its own belt-and-suspenders check), not a full arm-mesh-vs-table contact
    query. `self_collision` is hardcoded False -- self-collision is disabled at
    import time (self_collision=False in the URDF importer's config, matching the
    Gazebo env's own physical setup where self-collision was never simulated
    either), so there is nothing to detect. Wiring up PhysX contact-report-based
    detection for a true arm/table collision check is a known follow-up, flagged
    here rather than silently assumed done -- same spirit as the source file's own
    "known, unresolved blockers" notes.
  - target_source is always ground-truth (this env reads the lego cuboid's actual
    RigidObject pose directly) -- there is no YOLO/perception path in this port yet.

Coordinate convention: the URDF's own "world" -> "base_link" fixed joint origin
(assets/urdf/ur5e_robotiq140.urdf's base_joint) is xyz="0 0.55 0.78" -- that
offset is baked into the imported USD and requires no extra transform on the
"/World/UR5e" reference prim itself; confirmed empirically 2026-09-10 by
querying base_link's actual world pose directly (RigidPrim.get_world_poses()).

Two related mistakes were made and fixed the same day, both worth knowing if
touching this again:
  1. A translate was briefly (and wrongly) added on "/World/UR5e" on top of the
     already-correct URDF-baked offset, double-applying it and floating the
     whole robot 0.8m in the air -- caught visually, reverted.
  2. _robot_base_world_pos() (used for the observation's target_pos, and for the
     "Target Lost" >1.0m-from-base kill-switch in step()) used to read
     get_world_poses() on the Articulation object itself, which returns the pose
     of whatever prim carries ArticulationRootAPI -- a synthetic "root_joint"
     representing the abstract "world" frame, legitimately fixed at the scene
     origin, NOT base_link. That was treated as "harmless, just a constant
     offset" since the observation only needs a consistent relative vector --
     true for the observation, but NOT for the kill-switch, which compares
     against a fixed absolute threshold: once the lego's position was corrected
     to its real (0.802, 0.29) (see LEGO_BASE_X/Y below), distance-from-origin
     exceeded 1.0m even though the real distance-from-base_link is a safe
     ~0.843m, so every single episode was hard-terminating at step 0 regardless
     of policy behavior. Fixed by wrapping base_link as its own RigidPrim
     (self._base_link, discovered the same way as the finger pads) and reading
     _robot_base_world_pos() from that instead.

The table is a simple static cuboid sized to put its top surface at z=0.78 to
match (see TABLE_TOP_Z below).
"""
import os
import csv
import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# The converted USD asset is a generated build artifact (like the sibling Gazebo repo's
# gitignored build/install/log dirs), not checked into this repo -- see
# assets/convert_urdf_to_usd.py and assets/README.md. UR5E_USD_PATH overrides the
# default cache location (needed on Windows: Isaac Sim's asset resolver failed to
# resolve a WSL \\wsl.localhost UNC path during URDF import -- confirmed live
# 2026-09-09 -- so the default here is an OS-local cache dir, not this repo's own
# assets/usd/ when that repo checkout is itself accessed over a UNC path).
ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
if os.name == "nt":
    _DEFAULT_CACHE = os.path.join(os.environ.get("USERPROFILE", "."), "isaacsim_assets", "isaacsimtraining", "usd")
else:
    _DEFAULT_CACHE = os.path.join(os.environ.get("HOME", "."), ".cache", "isaacsimtraining", "usd")
DEFAULT_USD_PATH = os.environ.get(
    "UR5E_USD_PATH", os.path.join(_DEFAULT_CACHE, "ur5e_robotiq140.usd")
)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
GRIPPER_DRIVE_JOINT = "finger_joint"
LEFT_PAD_LINK = "left_inner_finger_pad"
RIGHT_PAD_LINK = "right_inner_finger_pad"
BASE_LINK = "base_link"

# The Robotiq gripper's 5 follower joints are driven purely by a PhysX mimic constraint
# relative to finger_joint (the URDF's own <mimic joint="finger_joint" multiplier="..."/>
# tags -- there's no independent drive on them, matching the original Gazebo setup where
# the equivalent kinematic-loop constraint was enforced by a Gazebo-specific mimic plugin,
# not a direct drive). Confirmed live 2026-09-10: Isaac Sim's URDF importer DOES create a
# real PhysxMimicJointAPI for each of these (contrary to an earlier suspicion that mimic
# support might be missing entirely), but it imports the "gearing" value with the WRONG
# SIGN relative to the URDF's own multiplier for every single one of these 5 joints (not
# a one-off glitch -- checked all 5, 5/5 inverted). Left uncorrected, this makes the
# follower joints fight the driven finger_joint from the very first physics tick: a fresh
# env (before any training step at all) showed joint velocities up to 1033 rad/s on these
# links. This is almost certainly a major contributor to the "gripper achieved position
# wildly exceeds its own URDF limit" symptom seen in training telemetry, and likely
# injects broader instability into the whole arm through the kinematic chain. Corrected
# explicitly after import -- see the gearing-sign fix in __init__ below.
MIMIC_JOINT_MULTIPLIERS = {
    "left_inner_knuckle_joint": -1.0,
    "left_inner_finger_joint": 1.0,
    "right_inner_knuckle_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "right_outer_knuckle_joint": -1.0,
}

# Base spawn point for the target cuboid ("lego"), in WORLD coordinates. Corrected
# 2026-09-10 to match the actual authoritative source (ur3e_rl/ur3e_env.py's own
# spawn_x/spawn_y = 0.802/0.290, real-hardware TF-calibrated 2026-09-03) instead of
# an earlier (0.3, 0.2) placeholder that was never actually re-derived from that file.
# LEGO_SIZE_M likewise corrected from 0.057 (ur3e_env.py's own comment: "Was 0.057
# (Rubik's-cube-sized guess)") to 0.020, the real hardware-measured size. LEGO_SPAWN_Z
# = TABLE_TOP_Z + LEGO_SIZE_M/2 (resting on the table top), matching ur3e_env.py's own
# live-measured base-relative Target_Z ~= 0.010 (i.e. world Z = base_z (0.78) + 0.010).
LEGO_BASE_X = 0.802
LEGO_BASE_Y = 0.290
LEGO_SIZE_M = 0.020
LEGO_SPAWN_Z = 0.78 + LEGO_SIZE_M / 2.0

# Corrected 2026-09-10 to match the Gazebo world file (ur_gazebo/worlds/pick_and_place_demo.world)
# and ur3e_env.py, both real-hardware calibrated -- table top was 0.80 (2cm too high) and
# footprint was 1.0 x 0.8 (much smaller than the real table, which spans X:[-0.95,0.95],
# Y:[-1.3,1.3] around a world-origin-centered model, not Y=0.55-centered).
TABLE_TOP_Z = 0.78
TABLE_SIZE = (1.9, 2.6, 0.04)  # matches the Gazebo world's actual table footprint
# Table is centered at world (0, 0) -- see its FixedCuboid position below -- so these are
# plain world-frame bounds, not base-relative.
TABLE_X_MIN, TABLE_X_MAX = -TABLE_SIZE[0] / 2.0, TABLE_SIZE[0] / 2.0
TABLE_Y_MIN, TABLE_Y_MAX = -TABLE_SIZE[1] / 2.0, TABLE_SIZE[1] / 2.0
# Margin below the table top before counting as a collision, not the top surface itself --
# grasping legitimately brings the EE to ~TABLE_TOP_Z + LEGO_SIZE_M/2 (just above the
# table), so the check needs a little clearance to not fire on normal approach.
TABLE_COLLISION_MARGIN_M = 0.02

# Sorting bin, added 2026-09-10 -- wasn't modeled at all before. Simplified as a single
# static box (this port already simplifies table/lego the same way) matching the real
# bin's overall footprint and position from the Gazebo world file; not yet referenced by
# any reward/termination logic since this env's task ends at grasp detection, before a
# place-in-bin phase.
BIN_POSITION = (0.415, 1.158, 0.85)
BIN_SIZE = (0.2, 0.2, 0.1)


class Ur5eGraspEnv(gym.Env):
    """UR5e + Robotiq-140 reach-and-grasp Gymnasium env, native Isaac Sim backend."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_id: int = 0,
        headless: bool = True,
        usd_path: str = DEFAULT_USD_PATH,
        log_dir: str = "./rl_logs",
    ):
        super().__init__()
        self.env_id = env_id
        self.usd_path = usd_path

        # SimulationApp must be created before any other isaacsim/omni import, and must
        # be created fresh inside whichever process actually owns this Env instance --
        # the same "initialize inside the worker, never the main script" constraint the
        # original ur3e_env.py documents for rclpy.init() (its "Poison Fork" comment)
        # applies equally to SimulationApp.
        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": headless})

        # Deferred imports: these modules touch omni/isaacsim internals that only exist
        # once SimulationApp has run.
        import omni.usd
        from pxr import Usd, UsdPhysics

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import Articulation, RigidPrim

        self._World = World
        self._Articulation = Articulation

        # Corrected 2026-09-10 from 0.05 to 0.08 to match ur3e_env.py's own value (this
        # was another constant that was never actually re-derived from that file -- same
        # pattern as LEGO_BASE_X/Y, LEGO_SIZE_M, TABLE_TOP_Z etc. earlier this session).
        # ur3e_env.py's own comment: "0.08 rad over the unchanged 0.1s sim_step_time caps
        # commanded joint speed at ~0.8 rad/s" -- at 0.05 this env was capping the arm to
        # ~0.5 rad/s, 60% of the validated speed, for no stated reason. Confirmed live this
        # matters: with the target sitting close to the UR5e's max reach and a 500-step
        # episode budget, actual measured per-step Cartesian approach distance was only
        # ~2mm median even while the policy was successfully closing distance -- at that
        # rate a 500-step episode barely has margin to cross the ~1m home-to-target gap
        # even under ideal play, let alone recover from exploration/mistakes.
        self.max_joint_delta_rad = 0.08  # bounded per-step joint delta, rad
        joint_limit_margin = 0.05
        self.joint_pos_min = np.array(
            [-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi], dtype=np.float32
        ) + joint_limit_margin
        self.joint_pos_max = np.array(
            [2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi], dtype=np.float32
        ) - joint_limit_margin

        action_low = np.array([-self.max_joint_delta_rad] * 6 + [-np.pi], dtype=np.float32)
        action_high = np.array([self.max_joint_delta_rad] * 6 + [np.pi], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        # Curriculum-widened reach (ported from ur3e_env.py's 2026-09-04 pass): standoff
        # (how far the home pose starts from the target) and spawn radius (lateral
        # randomization of the lego's spawn point) both ramp with curriculum_level,
        # 0.0 (easy) -> 1.0 (hard). Values here are first-pass, not re-tuned for this
        # scene's exact geometry -- same caveat the source file placed on its own numbers.
        self.curriculum_level = 0.0
        self.MIN_SPAWN_RADIUS_M = 0.0
        # Corrected 2026-09-10 from 0.10 to 0.05 to match ur3e_env.py's own value -- its
        # comments derive 0.05 from the "Target Lost" kill-switch's >1.0m-from-base radius
        # (~0.157m of margin around the fixed spawn point's own measured ~0.843m distance),
        # a constraint specific to this task's actual geometry, not an arbitrary number.
        self.MAX_SPAWN_RADIUS_M = 0.05
        self.MAX_STANDOFF_M = 0.30  # only used to scale the table-collision penalty, see step()

        self.max_episode_steps = 500  # pure-RL (no classical handoff) episode budget, matches Gazebo env's UR3E_USE_CLASSICAL_HANDOFF=false path
        self.sim_step_time = 0.1  # seconds of sim time advanced per env.step()
        self.physics_dt = 1.0 / 60.0
        self._substeps_per_step = max(1, round(self.sim_step_time / self.physics_dt))

        self.current_step = 0
        self.episode_shaping_sum = 0.0
        self._prev_dist = 0.0
        self.current_gripper_pos = 0.0
        self.current_spawn_radius_m = 0.0

        # Pure-RL home pose (wrapped-to-[-pi,pi] equivalent of ur3e_env.py's raw
        # unwrapped values [2.040, 4.6942, 1.991, -1.973, -1.571, 0.0] -- no ROS
        # JointTrajectoryController here, so there's no raw/wrapped-continuity concern
        # that made the original keep the unwrapped form).
        self.home_joint_positions = np.array(
            [2.040, -1.589, 1.991, -1.973, -1.571, 0.0], dtype=np.float32
        )

        # --- Scene setup ---
        self._world = World(stage_units_in_meters=1.0, physics_dt=self.physics_dt, rendering_dt=self.physics_dt)
        add_reference_to_stage(usd_path=self.usd_path, prim_path="/World/UR5e")

        stage = omni.usd.get_context().get_stage()

        # The URDF importer's output hierarchy is NOT stable across Isaac Sim versions --
        # confirmed live 2026-09-10 porting from a local 5.1.0 install to a freshly
        # pip-installed 6.0.1.0 on a rented GPU instance: 5.1.0 puts the articulation root
        # at the flat path /World/UR5e/world with every link as a direct sibling; 6.0.1.0
        # nests links under a nested .../Geometry/world/base_link/... kinematic chain (with
        # joints split out separately under a parallel .../Physics/ scope) and puts the
        # ArticulationRootAPI on .../Geometry/world/base_link, not .../world. Hardcoding
        # either path breaks on the other version. Discover both dynamically instead: walk
        # the stage for whichever prim actually has ArticulationRootAPI applied, and find
        # the finger pad links by name, wherever they ended up nested.
        ur5e_root_prim = stage.GetPrimAtPath("/World/UR5e")
        articulation_root_path = None
        left_pad_path = None
        right_pad_path = None
        base_link_path = None
        mimic_joint_prims = {}
        for prim in Usd.PrimRange(ur5e_root_prim):
            if articulation_root_path is None and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_root_path = prim.GetPath().pathString
            name = prim.GetName()
            if name == LEFT_PAD_LINK and left_pad_path is None:
                left_pad_path = prim.GetPath().pathString
            elif name == RIGHT_PAD_LINK and right_pad_path is None:
                right_pad_path = prim.GetPath().pathString
            elif name == BASE_LINK and base_link_path is None:
                base_link_path = prim.GetPath().pathString
            elif name in MIMIC_JOINT_MULTIPLIERS and name not in mimic_joint_prims:
                mimic_joint_prims[name] = prim
        if articulation_root_path is None:
            raise RuntimeError(
                "No prim under /World/UR5e has ArticulationRootAPI applied -- the imported "
                f"USD at {self.usd_path} may be malformed (check assets/convert_urdf_to_usd.py's "
                "output for import errors)."
            )
        if left_pad_path is None or right_pad_path is None:
            raise RuntimeError(
                f"Could not find both finger pad links ({LEFT_PAD_LINK}, {RIGHT_PAD_LINK}) under "
                f"/World/UR5e -- found left={left_pad_path} right={right_pad_path}."
            )
        if base_link_path is None:
            raise RuntimeError(f"Could not find {BASE_LINK} under /World/UR5e.")
        if len(mimic_joint_prims) != len(MIMIC_JOINT_MULTIPLIERS):
            missing = set(MIMIC_JOINT_MULTIPLIERS) - set(mimic_joint_prims)
            raise RuntimeError(f"Could not find gripper mimic joints under /World/UR5e: missing {missing}.")

        # Fix the importer's inverted mimic gearing sign -- see MIMIC_JOINT_MULTIPLIERS'
        # comment above. Must happen before self._world.reset() so physics never runs even
        # one step with the wrong sign.
        for jname, expected_gearing in MIMIC_JOINT_MULTIPLIERS.items():
            gearing_attr = mimic_joint_prims[jname].GetAttribute("physxMimicJoint:rotX:gearing")
            if not gearing_attr:
                raise RuntimeError(f"{jname} has no physxMimicJoint:rotX:gearing attribute to fix.")
            gearing_attr.Set(expected_gearing)

        # Table model is centered at world (0, 0) in the Gazebo world file (its own link
        # poses, e.g. the top slab's z=0.78, are relative to that) -- corrected 2026-09-10
        # from an incorrect (0, 0.55) center that didn't match the source.
        self._table = self._world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.0, 0.0, TABLE_TOP_Z - TABLE_SIZE[2] / 2.0]),
                size=1.0,
                scale=np.array(TABLE_SIZE),
                color=np.array([0.4, 0.3, 0.2]),
            )
        )
        self._world.scene.add_default_ground_plane()

        self._lego = self._world.scene.add(
            DynamicCuboid(
                prim_path="/World/Lego",
                name="lego",
                position=np.array([LEGO_BASE_X, LEGO_BASE_Y, LEGO_SPAWN_Z]),
                size=LEGO_SIZE_M,
                color=np.array([1.0, 0.0, 0.0]),
            )
        )

        # Sorting bin -- see BIN_POSITION/BIN_SIZE comment above. Static, not yet used by
        # any reward/termination logic.
        self._bin = self._world.scene.add(
            FixedCuboid(
                prim_path="/World/Bin",
                name="bin",
                position=np.array(BIN_POSITION),
                size=1.0,
                scale=np.array(BIN_SIZE),
                color=np.array([0.8, 0.2, 0.2]),
            )
        )

        self._robot = Articulation(prim_paths_expr=articulation_root_path, name=f"ur5e_{env_id}")
        self._world.scene.add(self._robot)

        # isaacsim.core.prims.Articulation has no per-link world-pose getter -- only
        # get_world_poses() for the articulation root -- so the two Robotiq finger pads
        # (wherever they ended up nested, see the discovery block above) are wrapped
        # directly as RigidPrim for their own get_world_poses(). base_link is wrapped the
        # same way: get_world_poses() on the Articulation itself returns the pose of
        # whatever prim carries ArticulationRootAPI, which is a synthetic "root_joint"/
        # "world" frame legitimately fixed at the scene origin -- NOT base_link, and NOT a
        # usable stand-in for "the robot's physical mount point" (confirmed live
        # 2026-09-10: using it for _robot_base_world_pos() put the observation's target_pos
        # and the "Target Lost" >1.0m-from-base kill-switch off by the full (0, 0.55, 0.78)
        # mount offset -- harmless for the relative observation vector, but not for a
        # kill-switch compared against a fixed absolute threshold: with the lego at its
        # correct position, distance-from-origin exceeds 1.0m even though the real
        # distance-from-base is a safe ~0.843m, so every episode was hard-terminating at
        # step 0 regardless of policy behavior).
        self._left_pad = RigidPrim(prim_paths_expr=left_pad_path, name=f"ur5e_{env_id}_left_pad")
        self._right_pad = RigidPrim(prim_paths_expr=right_pad_path, name=f"ur5e_{env_id}_right_pad")
        self._base_link = RigidPrim(prim_paths_expr=base_link_path, name=f"ur5e_{env_id}_base_link")
        self._world.scene.add(self._left_pad)
        self._world.scene.add(self._right_pad)
        self._world.scene.add(self._base_link)

        self._world.reset()

        self._arm_dof_indices = np.array(
            [self._robot.dof_names.index(n) for n in ARM_JOINT_NAMES], dtype=np.int64
        )
        self._gripper_dof_index = self._robot.dof_names.index(GRIPPER_DRIVE_JOINT)

        # Explicit PD gains for the implicit joint drives -- Isaac Sim 6.0.1.0's URDF
        # importer does NOT set these from the source URDF (confirmed live 2026-09-10:
        # import logs "Stiffness and damping not available joint <name>, actuator will be
        # created without gain parameters" for every joint), leaving stiffness/damping at
        # 0 -- with no restoring force at all, set_joint_position_targets() has no effect
        # and the arm just free-falls/flails under gravity, tripping the velocity
        # kill-switch within the first few steps. Values match this project's own earlier
        # Isaac Lab scaffold (ur3e_working_reference/isaac_lab_rl/ur3e_env_cfg.py's
        # ImplicitActuatorCfg) for continuity, not re-tuned for this exact asset.
        #
        # A small default on EVERY dof first: isolated single-joint testing showed the arm
        # (with gains) is perfectly stable on its own, but the full env (arm + gripper)
        # still tripped the velocity kill-switch almost immediately -- the 5 gripper mimic-
        # follower joints (left/right_inner_knuckle_joint, right_outer_knuckle_joint,
        # left/right_inner_finger_joint; only finger_joint itself, the mimic *reference*,
        # is ever commanded directly) don't get a real drive from this blanket call (they're
        # mimic-constrained, not independently driven -- set_gains() has nothing to act on
        # for them). Root-caused 2026-09-10: those 5 joints' PhysxMimicJointAPI imported
        # with the gearing sign INVERTED relative to the URDF's own <mimic multiplier="...">
        # for every single one of them (not a one-off glitch, 5/5), plus a near-zero
        # dampingRatio (~0.005, an importer default -- not something the URDF's <mimic> tag
        # even specifies). Together those made the closed 4-bar gripper linkage violently
        # unstable from the very first physics tick (a fresh env showed joint velocities up
        # to 1033 rad/s before any training step ran) -- this blanket kp=50/kd=5 floor alone
        # never actually fixed that, since it doesn't apply to mimic-constrained joints. See
        # the gearing-sign correction, dampingRatio fix, and iteration-count raise below,
        # which do fix it.
        self._robot.set_gains(
            kps=np.full((1, self._robot.num_dof), 50.0, dtype=np.float32),
            kds=np.full((1, self._robot.num_dof), 5.0, dtype=np.float32),
        )
        # Solver iteration counts: confirmed live 2026-09-10 that with only the importer's
        # own defaults, joint velocities grow gradually over ~10-15 steps even holding a
        # fixed target with zero policy action (a classic PD/TGS-solver-iteration-count
        # mismatch, not an instant collision -- isolated per-joint testing without a scene
        # was stable, but grew unstable once stepped for longer with a ground plane
        # present). Isaac Sim's own default is low for a 12-DOF articulation under these
        # gains; raising both counts is the standard PhysX fix for exactly this failure mode.
        #
        # set_solver_position/velocity_iteration_counts() writes a plain USD attribute that
        # must already exist -- the 6.0.1.0 importer applies UsdPhysics.ArticulationRootAPI
        # but not the PhysX-specific PhysxArticulationAPI schema that owns this attribute,
        # so the raw write fails with "Empty typeName" until the schema is applied first.
        from pxr import PhysxSchema

        art_root_prim = stage.GetPrimAtPath(articulation_root_path)
        if not art_root_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(art_root_prim)
        # Raised from 32/4 to 255/64 on 2026-09-10: 32/4 was enough to stabilize the arm's
        # own 6 joints (see note above), but nowhere near enough for the gripper's closed
        # 4-bar-linkage mimic loop -- confirmed live that even with the mimic gearing sign
        # corrected (below) and dampingRatio raised to 1.0, 32/4 still let the 5 follower
        # joints blow up to hundreds of rad/s and never converge. 255/64 (PhysX's max
        # position-iteration count) converges cleanly: max mimic-joint velocity drops from
        # the initial spike to <0.1 rad/s within ~100 physics steps and stays there. This
        # does cost real throughput (more solver work per step) -- if training FPS becomes
        # a problem, the first thing worth trying is a value between 32 and 255, re-tested
        # the same way (watch max|mimic joint velocity| over the first ~150 steps of a
        # fresh env), not assuming a lower number is safe without checking.
        self._robot.set_solver_position_iteration_counts(np.array([255]))
        self._robot.set_solver_velocity_iteration_counts(np.array([64]))
        self._robot.set_gains(
            kps=np.array([[400.0] * 6], dtype=np.float32), kds=np.array([[40.0] * 6], dtype=np.float32),
            joint_indices=self._arm_dof_indices,
        )
        self._robot.set_gains(
            kps=np.array([[800.0]], dtype=np.float32), kds=np.array([[80.0]], dtype=np.float32),
            joint_indices=[self._gripper_dof_index],
        )
        # The gripper mimic joints' dampingRatio imports at ~0.005 (near-undamped) -- not
        # something the URDF's <mimic> tag even specifies, purely an importer default, and
        # far too low for a real mechanical linkage. Confirmed live 2026-09-10 as part of
        # root-causing the same instability the gearing-sign fix targets: raising this to a
        # critically-damped 1.0 (alongside the gearing fix and the iteration-count raise
        # above -- none of the three alone was sufficient) is what actually gets the gripper
        # to converge instead of oscillating or diverging.
        for _jname in MIMIC_JOINT_MULTIPLIERS:
            damping_attr = mimic_joint_prims[_jname].GetAttribute("physxMimicJoint:rotX:dampingRatio")
            damping_attr.Set(1.0)

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hardware_csv_path = os.path.join(log_dir, f"hardware_log_env{env_id}_{timestamp}.csv")
        with open(self.hardware_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Global_Step", "Curriculum_Lvl", "Dist_to_Target", "Reward", "Terminated",
                "Table_Collision",
                "J1_Pos", "J2_Pos", "J3_Pos", "J4_Pos", "J5_Pos", "J6_Pos",
                "J1_Vel", "J2_Vel", "J3_Vel", "J4_Vel", "J5_Vel", "J6_Vel",
                "Gripper_Pos", "EE_X", "EE_Y", "EE_Z",
                "Target_X", "Target_Y", "Target_Z",
                "Act_J1", "Act_J2", "Act_J3", "Act_J4", "Act_J5", "Act_J6", "Act_Gripper",
                "Spawn_Radius_M",
            ])
        self.global_step_count = 0

    def set_curriculum_level(self, level):
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    # ------------------------------------------------------------------ helpers

    def _get_ee_pose(self):
        """Midpoint of the two Robotiq finger pads -- see the Gazebo env's own
        _update_ee_pose() docstring for why the midpoint (not one pad) is used."""
        left_pos, left_quat = self._left_pad.get_world_poses()
        right_pos, right_quat = self._right_pad.get_world_poses()
        mid = (left_pos[0] + right_pos[0]) / 2.0
        quat = right_quat[0]  # right-pad-only orientation, matches Gazebo env's own choice
        return np.concatenate([mid, quat]).astype(np.float32)

    def _get_obs(self):
        joint_pos = self._robot.get_joint_positions(joint_indices=self._arm_dof_indices)[0].astype(np.float32)
        joint_vel = self._robot.get_joint_velocities(joint_indices=self._arm_dof_indices)[0].astype(np.float32)
        ee_pose = self._get_ee_pose()
        target_pos = self._lego.get_world_pose()[0].astype(np.float32) - self._robot_base_world_pos()
        gripper_pos = np.array([self.current_gripper_pos], dtype=np.float32)
        return np.concatenate([joint_pos, joint_vel, ee_pose, target_pos, gripper_pos]).astype(np.float32)

    def _robot_base_world_pos(self):
        pos, _ = self._base_link.get_world_poses()
        return pos[0].astype(np.float32)

    def _active_goal_pose(self):
        return self._lego.get_world_pose()[0].astype(np.float32)

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.episode_shaping_sum = 0.0

        self._robot.set_joint_positions(
            np.array([self.home_joint_positions], dtype=np.float32), joint_indices=self._arm_dof_indices
        )
        self._robot.set_joint_positions(np.array([[0.0]], dtype=np.float32), joint_indices=[self._gripper_dof_index])
        self._robot.set_joint_velocities(np.zeros((1, self._robot.num_dof), dtype=np.float32))

        self.current_spawn_radius_m = (
            self.MIN_SPAWN_RADIUS_M
            + self.curriculum_level * (self.MAX_SPAWN_RADIUS_M - self.MIN_SPAWN_RADIUS_M)
        )
        spawn_r = self.current_spawn_radius_m * np.sqrt(np.random.uniform(0.0, 1.0))
        spawn_theta = np.random.uniform(0.0, 2 * np.pi)
        spawn_x = LEGO_BASE_X + spawn_r * np.cos(spawn_theta)
        spawn_y = LEGO_BASE_Y + spawn_r * np.sin(spawn_theta)
        self._lego.set_world_pose(
            position=np.array([spawn_x, spawn_y, LEGO_SPAWN_Z]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self._lego.set_linear_velocity(np.zeros(3))
        self._lego.set_angular_velocity(np.zeros(3))

        for _ in range(5):
            self._world.step(render=False)

        state = self._get_obs()
        ee_pos = state[12:15]
        self._prev_dist = float(np.linalg.norm(ee_pos - self._active_goal_pose()))
        self.current_gripper_pos = 0.0
        return state, {}

    def step(self, action):
        arm_delta = np.clip(action[:6], -self.max_joint_delta_rad, self.max_joint_delta_rad)
        current_arm_pos = self._robot.get_joint_positions(joint_indices=self._arm_dof_indices)[0]
        arm_target = np.clip(current_arm_pos + arm_delta, self.joint_pos_min, self.joint_pos_max)
        gripper_val = float(action[6])
        gripper_cmd_val = float(((np.clip(gripper_val, -np.pi, np.pi) + np.pi) / (2 * np.pi)) * 0.8)

        self._robot.set_joint_position_targets(
            np.array([arm_target], dtype=np.float32), joint_indices=self._arm_dof_indices
        )
        self._robot.set_joint_position_targets(
            np.array([[gripper_cmd_val]], dtype=np.float32), joint_indices=[self._gripper_dof_index]
        )

        for _ in range(self._substeps_per_step):
            self._world.step(render=False)

        achieved_gripper_pos = float(
            self._robot.get_joint_positions(joint_indices=[self._gripper_dof_index])[0, 0]
        )
        self.current_gripper_pos = achieved_gripper_pos

        state = self._get_obs()
        ee_pos = state[12:15]
        dist = float(np.linalg.norm(ee_pos - self._active_goal_pose()))

        reward = float(self._prev_dist - dist)
        self._prev_dist = dist

        joint_vel = state[6:12]
        velocity_penalty = 0.05 * float(np.sum(np.abs(joint_vel)))
        reward -= velocity_penalty

        SHAPING_FLOOR = -5.0
        prospective_sum = self.episode_shaping_sum + reward
        if prospective_sum < SHAPING_FLOOR:
            reward = SHAPING_FLOOR - self.episode_shaping_sum
            self.episode_shaping_sum = SHAPING_FLOOR
        else:
            self.episode_shaping_sum = prospective_sum

        gripper_closed_amount = (np.clip(gripper_val, -np.pi, np.pi) + np.pi) / (2 * np.pi)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {}

        # Grasp-success: contact signal is the achieved-vs-commanded finger_joint gap,
        # same CONTACT_GAP_THRESHOLD=0.13 calibrated live in the Gazebo project
        # (contact_threshold_diagnostic.py) -- kept as-is since it's measuring the same
        # physical quantity (Robotiq mimic-joint travel vs. commanded target) here.
        CONTACT_GAP_THRESHOLD = 0.13
        contact_gap = gripper_cmd_val - achieved_gripper_pos
        contact_detected = gripper_closed_amount > 0.5 and contact_gap > CONTACT_GAP_THRESHOLD

        if dist < 0.05 and gripper_closed_amount > 0.5:
            if contact_detected:
                reward += 15.0
                terminated = True
                info["termination_reason"] = "Full Task Success"
            else:
                reward -= 5.0
                terminated = True
                info["termination_reason"] = "False Grasp -- No Contact Detected"

        if np.any(np.abs(joint_vel) > 10.0):
            reward -= 5.0
            terminated = True
            info["termination_reason"] = "Velocity Kill-Switch Triggered"

        # Table collision: height-and-footprint proxy, not a full arm-mesh contact query
        # (see module docstring for why this port doesn't yet do that). Corrected
        # 2026-09-10: this used to be a bare "ee_z < 0.02" left over from before the table
        # height was fixed to its real 0.78 -- at that height, 0.02 only catches the EE
        # falling almost all the way to the floor, silently missing the much more common
        # case of it dipping into the table surface itself while still elevated (e.g.
        # z=0.5 is clearly inside/below the table top but was never flagged). Now checks
        # actual height-below-table-top AND horizontal position within the table's real
        # footprint, so being low but off the table's edge (which is fine) doesn't trigger.
        table_collision = (
            ee_pos[2] < TABLE_TOP_Z - TABLE_COLLISION_MARGIN_M
            and TABLE_X_MIN <= ee_pos[0] <= TABLE_X_MAX
            and TABLE_Y_MIN <= ee_pos[1] <= TABLE_Y_MAX
        )
        if table_collision:
            reward -= 5.0 * (1.0 + float(dist) / self.MAX_STANDOFF_M)
            terminated = True
            info["termination_reason"] = "Table Collision"

        # Target Lost: same base-relative-distance kill-switch as ur3e_env.py, computed
        # from the lego's actual RigidObject pose (no perception/TF involved).
        target_pos_base_relative = state[19:22]
        target_dist_from_base = float(np.linalg.norm(target_pos_base_relative))
        if target_dist_from_base > 1.0 or target_pos_base_relative[2] < 0.0:
            reward -= 5.0
            terminated = True
            info["termination_reason"] = "Target Lost (Out of Bounds)"

        info["is_success"] = terminated and info.get("termination_reason") == "Full Task Success"
        info["distance_to_target"] = dist

        self.global_step_count += 1
        try:
            with open(self.hardware_csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.global_step_count, self.curriculum_level, f"{dist:.4f}", f"{reward:.4f}", terminated,
                    table_collision,
                    *[f"{p:.4f}" for p in state[0:6]],
                    *[f"{v:.4f}" for v in state[6:12]],
                    f"{achieved_gripper_pos:.4f}",
                    f"{ee_pos[0]:.4f}", f"{ee_pos[1]:.4f}", f"{ee_pos[2]:.4f}",
                    f"{target_pos_base_relative[0]:.4f}", f"{target_pos_base_relative[1]:.4f}", f"{target_pos_base_relative[2]:.4f}",
                    *[f"{a:.4f}" for a in action],
                    f"{self.current_spawn_radius_m:.4f}",
                ])
        except Exception as e:
            print(f"[Ur5eGraspEnv {self.env_id}] hardware logger failed to write: {e}", flush=True)

        return state, reward, terminated, truncated, info

    def close(self):
        self._simulation_app.close()
