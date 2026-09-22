"""Swap the URDF-imported Robotiq gripper's geometry for NVIDIA's Robotiq 2F-140 asset.

Why: the box collision proxies in the URDF-converted gripper leave the pads ~5cm apart even
fully closed, so a 2cm cube cannot be grasped (docs/VALIDATION_CHECKLIST.md's first update).
NVIDIA ships two Robotiq 2F-140 USDs (fetch_robotiq_2f140.py): `Robotiq_2F_140_physics_edit.usd`
is a real closed four-bar linkage (PhysX loop-closure joints, `excludeFromArticulation=True` on
two of them) built as its OWN articulation; grafting that loop-closure structure onto a
different articulation root (the arm) leaves the linkage unstable (confirmed live 2026-09-22:
the fingers settle into a bent, left/right-asymmetric shape even with self-collision and the
contact sensor both off, so it isn't a real collision -- it's the loop-closure joint selection
depending on which prim PhysX treats as the articulation root). `2f140_instanceable.usd` is a
plain open tree -- 6 independently-driven revolute joints, no loop closure at all -- with the
SAME link and joint names as the Gazebo project's own URDF (robotiq_arg2f_base_link,
left_inner_finger_pad, right_outer_knuckle_joint, ...), so it composes cleanly into the arm's
single articulation.

It still can't use a PhysX mimic constraint, though: confirmed live 2026-09-22 that applying
PhysxMimicJointAPI to these joints (even with the correct gearing signs, verified empirically
below) makes the whole gripper diverge within ~25 steps once it's part of the arm's articulation
(finger_joint alone reached 232 rad). ur5e_grasp_env.py instead drives right_outer_knuckle_joint,
left_inner_finger_joint and right_inner_finger_joint directly every step, in software, using the
same per-joint signs as MIMIC_JOINT_MULTIPLIERS (confirmed live in a standalone test: gearing
+1/+1/-1 relative to finger_joint converges the pads to 1.1cm apart in 30 steps and holds there;
the opposite inner-finger sign only reaches 3.5cm and the mimic-based version never converges at
all). left_inner_knuckle_joint/right_inner_knuckle_joint are left undriven -- confirmed to have
zero effect on pad separation (dead-end decorative links with no downstream joint).

Input : the converted arm+gripper USD (convert_urdf_to_usd.py output) and the downloaded
        NVIDIA folder (fetch_robotiq_2f140.py).
Output: <same folder as the input>/ur5e_nvgripper.usd, a thin layer on top of the input that
        - deactivates the old (box-collision) gripper bodies and every joint that touches them,
        - references NVIDIA's gripper at /ur/nv_gripper (its articulation root removed, so the
          arm stays the only articulation) with the SAME link/joint names as before,
        - adds a fixed joint tool0 -> robotiq_arg2f_base_link (identity, matching the URDF's).
No mimic constraints are authored -- ur5e_grasp_env.py drives the follower joints directly (see
above). It stays next to the input because the env loads configuration/ur5e_robotiq140_physics.usd
from the same folder for the arm colliders.

    C:\\isaacsim\\python.bat build_nvidia_gripper_usd.py [--src ...] [--gripper-dir ...]
"""
import argparse
import os

parser = argparse.ArgumentParser()
_default_dir = os.path.join(os.path.expanduser("~"), "isaacsim_assets", "isaacsimtraining", "usd")
parser.add_argument("--src", default=os.environ.get("UR5E_SRC_USD_PATH", os.path.join(_default_dir, "ur5e_robotiq140.usd")))
parser.add_argument("--gripper-dir", default=os.path.join(os.path.expanduser("~"), "isaacsim_assets", "robotiq_2f140"))
parser.add_argument("--gripper-file", default="2f140_instanceable.usd")
parser.add_argument("--out", default=None, help="default: ur5e_nvgripper.usd next to --src")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

OLD_GRIPPER_BODIES = {
    "robotiq_arg2f_base_link", "left_outer_knuckle", "left_outer_finger", "left_inner_finger",
    "left_inner_finger_pad", "left_inner_knuckle", "right_inner_knuckle", "right_outer_knuckle",
    "right_outer_finger", "right_inner_finger", "right_inner_finger_pad",
}
# Same convention as ur5e_grasp_env.MIMIC_JOINT_MULTIPLIERS -- kept in sync manually since this
# script runs before the app (and that module) can be imported.
MIMIC_JOINT_MULTIPLIERS = {
    "left_inner_knuckle_joint": -1.0,
    "left_inner_finger_joint": 1.0,
    "right_inner_knuckle_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "right_outer_knuckle_joint": -1.0,
}
src = os.path.abspath(args.src)
gripper_usd = os.path.abspath(os.path.join(args.gripper_dir, args.gripper_file))
out = os.path.abspath(args.out or os.path.join(os.path.dirname(src), "ur5e_nvgripper.usd"))
for p in (src, gripper_usd):
    if not os.path.exists(p):
        raise SystemExit(f"BUILD_FAILED: missing {p}")

# Read-only look at the source, to find the joints to deactivate and the tool0 pose.
src_stage = Usd.Stage.Open(src)
old_body_paths = {f"/ur/{n}" for n in OLD_GRIPPER_BODIES}
joints_to_drop = []
for prim in src_stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(prim)
    touched = {str(t) for t in j.GetBody0Rel().GetTargets()} | {str(t) for t in j.GetBody1Rel().GetTargets()}
    if touched & old_body_paths:
        joints_to_drop.append(prim.GetPath())
tool0_world = UsdGeom.Xformable(src_stage.GetPrimAtPath("/ur/tool0")).ComputeLocalToWorldTransform(0)
ur_world = UsdGeom.Xformable(src_stage.GetPrimAtPath("/ur")).ComputeLocalToWorldTransform(0)
tool0_in_ur = tool0_world * ur_world.GetInverse()

g_stage = Usd.Stage.Open(gripper_usd)
g_default = g_stage.GetDefaultPrim().GetPath().pathString
base_in_gripper = UsdGeom.Xformable(g_stage.GetPrimAtPath(g_default + "/robotiq_arg2f_base_link")).ComputeLocalToWorldTransform(0)
g_root_world = UsdGeom.Xformable(g_stage.GetDefaultPrim()).ComputeLocalToWorldTransform(0)
base_rel_root = base_in_gripper * g_root_world.GetInverse()

stage = Usd.Stage.CreateNew(out)
stage.GetRootLayer().subLayerPaths.append(os.path.relpath(src, os.path.dirname(out)).replace("\\", "/"))
stage.SetDefaultPrim(stage.OverridePrim("/ur"))
# Usd.Stage.CreateNew() does NOT inherit metersPerUnit/upAxis from the sublayer -- confirmed live
# 2026-09-22: a fresh root layer with neither authored falls back to USD's own hardcoded default
# (0.01 m/unit, Y-up), a 100x scale + axis mismatch against src's 1.0 m/unit, Z-up. PhysX uses
# the composed stage's metersPerUnit for every distance and mass in the WHOLE scene, not just
# prims authored on this layer, so left unset this doesn't just misplace the gripper -- it was
# the actual cause of the arm itself diverging (shoulder_pan_joint reaching -8 rad during reset's
# own settle steps, before any policy action), which looked at first like a gripper-linkage bug.
UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(src_stage))
UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(src_stage))

for name in OLD_GRIPPER_BODIES:
    stage.OverridePrim(f"/ur/{name}").SetActive(False)
for path in joints_to_drop:
    stage.OverridePrim(path).SetActive(False)

grip = stage.DefinePrim("/ur/nv_gripper", "Xform")
grip.GetReferences().AddReference(os.path.relpath(gripper_usd, os.path.dirname(out)).replace("\\", "/"), g_default)
grip.RemoveAPI(UsdPhysics.ArticulationRootAPI)
grip.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

# Every body's "visuals"/"collisions" child scope in 2f140_instanceable.usd is marked
# instanceable=True (same pattern ur5e_grasp_env.py's own arm-collider-wiring block already
# hit and documents: a plain reference to already-multiply-referenced content collapses into a
# shared, non-enumerable scenegraph prototype). SetInstanceable(False) here so this repo's own
# collision/mass tooling (and any other per-body inspection) actually sees the real prims instead
# of an opaque instance boundary.
n_deinstanced = 0
for prim in Usd.PrimRange(g_stage.GetPrimAtPath(g_default)):
    if prim.IsInstanceable():
        rel_path = str(prim.GetPath())[len(g_default):]
        stage.OverridePrim("/ur/nv_gripper" + rel_path).SetInstanceable(False)
        n_deinstanced += 1
print(f"de-instanced {n_deinstanced} prims", flush=True)
# place the gripper so its base link coincides with tool0 (the joint below then starts satisfied)
place = base_rel_root.GetInverse() * tool0_in_ur
xf = UsdGeom.Xformable(grip)
xf.ClearXformOpOrder()
xf.AddTransformOp().Set(Gf.Matrix4d(place))

# Every body in 2f140_instanceable.usd ships UsdPhysics:MassAPI:principalAxes = (0,0,0,0) -- an
# all-zero quaternion, not a valid rotation. Confirmed live 2026-09-22 as the actual root cause
# of the arm itself diverging once this gripper is attached (isolated by mounting just the base
# link, no internal gripper joints at all: the arm alone was stable, arm+this one body was not).
# PhysX must be using the degenerate quaternion to orient the diagonal inertia tensor, injecting
# garbage torque from the very first physics step. Override it to identity for every body.
n_mass_fixed = 0
for prim in Usd.PrimRange(g_stage.GetPrimAtPath(g_default)):
    if prim.HasAPI(UsdPhysics.MassAPI):
        rel_path = str(prim.GetPath())[len(g_default):]
        override_prim = stage.OverridePrim("/ur/nv_gripper" + rel_path)
        UsdPhysics.MassAPI(override_prim).CreatePrincipalAxesAttr().Set(Gf.Quatf(1, 0, 0, 0))
        n_mass_fixed += 1
print(f"principalAxes fixed on {n_mass_fixed} bodies", flush=True)

mount = UsdPhysics.FixedJoint.Define(stage, "/ur/joints/ur_to_nv_gripper")
mount.CreateBody0Rel().SetTargets([Sdf.Path("/ur/tool0")])
mount.CreateBody1Rel().SetTargets([Sdf.Path("/ur/nv_gripper/robotiq_arg2f_base_link")])
mount.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
mount.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
mount.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
mount.CreateCollisionEnabledAttr().Set(False)

# 2f140_instanceable.usd ships each follower joint as an independent, undriven revolute joint --
# no mimic constraint, no loop closure. Confirmed live 2026-09-22 that a PhysX mimic constraint
# on these joints diverges once composed into the arm's articulation, so they are NOT wired here;
# ur5e_grasp_env.py drives right_outer_knuckle_joint / left_inner_finger_joint /
# right_inner_finger_joint directly in software each step instead (see MIMIC_JOINT_MULTIPLIERS
# above and NV_FOLLOWER_JOINTS in ur5e_grasp_env.py). Just confirm the expected joint names exist.
for jname in MIMIC_JOINT_MULTIPLIERS:
    if not any(p.GetName() == jname for p in Usd.PrimRange(g_stage.GetPrimAtPath(g_default))):
        raise SystemExit(f"BUILD_FAILED: {gripper_usd} has no joint named {jname}")

stage.GetRootLayer().Save()
print(f"deactivated bodies={len(OLD_GRIPPER_BODIES)} joints={len(joints_to_drop)}", flush=True)
print(f"BUILD_OK out={out}", flush=True)
app.close()
