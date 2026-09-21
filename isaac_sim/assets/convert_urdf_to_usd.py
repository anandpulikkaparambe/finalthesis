"""Convert the UR5e + Robotiq-140 URDF (already xacro-expanded and patched, see
assets/README.md) into the USD asset the training env loads.

Run with Isaac Sim's own bundled Python:
    Windows:  C:\\isaacsim\\python.bat assets\\convert_urdf_to_usd.py
    Linux:    ~/isaacsim/python.sh assets/convert_urdf_to_usd.py

On Windows this MUST be run against a URDF + mesh tree that lives on a native Windows
path (not a \\\\wsl.localhost\\... UNC path) -- confirmed live 2026-09-09 that Isaac
Sim 5.1's URDF importer (isaacsim.asset.importer.urdf) fails to resolve mesh
references through that UNC form ("Failed to resolve mesh '//wsl.localhost/...'"),
even after switching from the double-forward-slash form to a proper backslash UNC
path. If your checkout lives under WSL, copy assets/urdf/ + assets/meshes/ to a local
Windows path first (see assets/README.md's "Regenerating on Windows" section).
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--urdf-path",
    default=os.path.join(os.path.dirname(__file__), "urdf", "ur5e_robotiq140.isaac.urdf"),
    help="Patched URDF (package:// refs already resolved to real paths, .dae swapped for .stl -- see patch_urdf.py)",
)
parser.add_argument(
    "--dest-path",
    default=None,
    help="Output .usd path. Defaults to the same cache location ur5e_grasp_env.py reads (UR5E_USD_PATH, or the OS-local isaacsim_assets/isaacsimtraining/usd cache).",
)
parser.add_argument(
    "--no-self-collision",
    dest="self_collision",
    action="store_false",
    help="Import with self-collision disabled (the setting of the thesis' headline run). Default: enabled.",
)
args = parser.parse_args()

if args.dest_path is None:
    if os.name == "nt":
        cache_dir = os.path.join(os.environ.get("USERPROFILE", "."), "isaacsim_assets", "isaacsimtraining", "usd")
    else:
        cache_dir = os.path.join(os.environ.get("HOME", "."), ".cache", "isaacsimtraining", "usd")
    args.dest_path = os.path.join(cache_dir, "ur5e_robotiq140.usd")

os.makedirs(os.path.dirname(args.dest_path), exist_ok=True)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")
for _ in range(10):
    simulation_app.update()

# Isaac Sim's URDF importer API changed completely between 5.1.0 and 6.0.1.0 -- the old
# omni.kit.commands ("URDFCreateImportConfig"/"URDFParseAndImportFile") pattern is gone
# in 6.0.1.0 (no commands.py in that version's extension at all; confirmed live
# 2026-09-10 that omni.kit.commands.execute("URDFCreateImportConfig") returns (None, None)
# there, "wasn't registered or ambigious", even though the extension itself loads fine),
# replaced by a plain dataclass+class API (isaacsim.asset.importer.urdf.URDFImporterConfig
# / URDFImporter). Support both so this script works against this project's local Isaac
# Sim 5.1.0 install and a freshly pip-installed 6.0.1.0 on a rented GPU instance.
try:
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    dest_dir = os.path.dirname(args.dest_path)
    print(f"Importing {args.urdf_path} -> {dest_dir} (new class-based API)", flush=True)
    config = URDFImporterConfig(
        urdf_path=args.urdf_path,
        usd_path=dest_dir,
        merge_fixed_joints=False,
        fix_base=True,
        collision_type="Convex Hull",
        allow_self_collision=args.self_collision,  # default True; the headline run used False (see docs/CHANGES.md)
    )
    importer = URDFImporter(config)
    output_path = importer.import_urdf()
    ok = bool(output_path) and os.path.exists(output_path)
    print(f"CONVERSION_{'OK' if ok else 'FAILED'} output_path={output_path}", flush=True)
    if ok and os.path.normpath(output_path) != os.path.normpath(args.dest_path):
        import shutil

        # The new importer writes the main USD alongside sibling layer files it
        # references by relative path (in a robot-name subfolder under usd_path) --
        # copying just the single main file (confirmed live 2026-09-10) silently
        # breaks those relative references: the resulting stage opens but has zero
        # children, no error. Copy the whole output directory's contents flat into
        # dest_dir instead, so every sibling file lands next to the main one.
        output_dir = os.path.dirname(output_path)
        shutil.copytree(output_dir, dest_dir, dirs_exist_ok=True)
        copied_dest_path = os.path.join(dest_dir, os.path.basename(output_path))
        if os.path.normpath(copied_dest_path) != os.path.normpath(args.dest_path):
            shutil.copyfile(copied_dest_path, args.dest_path)
        print(f"Copied output directory contents to: {dest_dir} (main file: {args.dest_path})", flush=True)
except ImportError:
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if import_config is None:
        raise RuntimeError(
            "URDFCreateImportConfig still not registered after enable_extension() + update "
            "ticks -- the isaacsim.asset.importer.urdf extension may have failed to load; "
            "check the log above for [omni.ext.plugin] errors."
        )
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = args.self_collision
    import_config.create_physics_scene = False
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    print(f"Importing {args.urdf_path} -> {args.dest_path} (legacy command API)", flush=True)
    result, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=args.urdf_path,
        import_config=import_config,
        dest_path=args.dest_path,
    )
    ok = bool(result) and os.path.exists(args.dest_path)
    print(f"CONVERSION_{'OK' if ok else 'FAILED'} prim_path={prim_path} dest={args.dest_path}", flush=True)

simulation_app.close()
