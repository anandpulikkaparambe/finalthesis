"""Patch the xacro-expanded UR5e+Robotiq140 URDF for Isaac Sim import.

Rewrites package:// mesh references to real filesystem paths, and swaps every
visual .dae mesh for its .stl collision-mesh equivalent -- Isaac Sim's URDF
importer crashes on these particular Collada files (the same fix this project's
earlier ur3e_working_reference/isaac_lab_rl/patch_urdf.py scaffold needed for the
UR3e variant).

Plain python3, no ROS or Isaac Sim needed -- run this right after xacro-expanding
the URDF (see assets/README.md for the full regeneration steps) and before
convert_urdf_to_usd.py.
"""
import argparse
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("--urdf-path", default=os.path.join(os.path.dirname(__file__), "urdf", "ur5e_robotiq140.urdf"))
parser.add_argument("--meshes-root", default=os.path.join(os.path.dirname(__file__), "meshes"))
parser.add_argument(
    "--out-path", default=os.path.join(os.path.dirname(__file__), "urdf", "ur5e_robotiq140.isaac.urdf")
)
parser.add_argument(
    "--keep-visual-dae",
    action="store_true",
    help=(
        "Skip the visual .dae -> collision .stl swap below. That swap was a workaround for "
        "an Isaac Sim URDF-importer crash on these .dae files -- confirmed live 2026-09-10 "
        "that a real Isaac Sim 5.1.0 install does NOT actually crash on them (isolated "
        "single-mesh import test succeeded), so this flag exists to test/use the real "
        "visual meshes instead of the collision-hull proxies. If your install DOES crash, "
        "drop this flag."
    ),
)
parser.add_argument(
    "--gripper-box-collision",
    action="store_true",
    help=(
        "Keep the 1 cm box collision proxies the Gazebo description uses for the Robotiq knuckles "
        "and fingers. By default they are replaced by the real Robotiq 2F-140 collision meshes "
        "(meshes/collision/robotiq_arg2f_140_*.stl): with the boxes, the closed gripper leaves the "
        "pads ~5 cm apart and a 2 cm cube cannot be grasped (see docs/VALIDATION_CHECKLIST.md)."
    ),
)
args = parser.parse_args()

with open(args.urdf_path, "r") as f:
    urdf = f.read()

GRIPPER_MESH_LINKS = {
    "outer_knuckle": "robotiq_arg2f_140_outer_knuckle.stl",
    "outer_finger": "robotiq_arg2f_140_outer_finger.stl",
    "inner_knuckle": "robotiq_arg2f_140_inner_knuckle.stl",
    "inner_finger": "robotiq_arg2f_140_inner_finger.stl",
}
if not args.gripper_box_collision:
    n_swapped = 0
    for side in ("left", "right"):
        for part, stl in GRIPPER_MESH_LINKS.items():
            pat = re.compile(
                r'(<link name="%s_%s">.*?<collision>\s*<origin[^>]*/>\s*<geometry>)\s*<box [^>]*/>(\s*</geometry>)'
                % (side, part),
                re.S,
            )
            repl = (
                r'\1<mesh filename="package://robotiq_2f_gripper_description/meshes/collision/%s"/>\2' % stl
            )
            urdf, n = pat.subn(repl, urdf, count=1)
            n_swapped += n
    print(f"Gripper collision boxes replaced by meshes: {n_swapped}/8")

mesh_root = args.meshes_root.replace("\\", "/")
urdf = urdf.replace("package://ur_description", f"{mesh_root}/ur_description")
urdf = urdf.replace("package://robotiq_2f_gripper_description", f"{mesh_root}/robotiq_2f_gripper_description")

if not args.keep_visual_dae:
    for link in ["base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3"]:
        urdf = urdf.replace(f"visual/{link}.dae", f"collision/{link}.stl")

remaining_dae = re.findall(r"[\w./]+\.dae", urdf)
with open(args.out_path, "w") as f:
    f.write(urdf)

print(f"Patched URDF written to {args.out_path}")
print(f"Remaining .dae references (should be empty): {remaining_dae}")
