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
args = parser.parse_args()

with open(args.urdf_path, "r") as f:
    urdf = f.read()

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
