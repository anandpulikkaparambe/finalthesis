"""Download NVIDIA's Robotiq 2F-140 USD (Isaac Sim asset library) to a local folder.

The asset lives on NVIDIA's public asset server (about 9 MB in total, with its own LICENSE
file, which stays next to the files). It is NOT committed to this repo: run this once on each
machine (setup_vastai.sh does it), then run build_nvidia_gripper_usd.py.

    C:\\isaacsim\\python.bat fetch_robotiq_2f140.py        (Windows)
    python fetch_robotiq_2f140.py                          (Linux)
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dest", default=os.path.join(os.path.expanduser("~"), "isaacsim_assets", "robotiq_2f140"))
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.client  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402

root = get_assets_root_path()
if not root:
    raise SystemExit("FETCH_FAILED: could not resolve the Isaac Sim assets root (no network?)")
os.makedirs(args.dest, exist_ok=True)
dest = args.dest.replace("\\", "/")
res = omni.client.copy(root + "/Isaac/Robots/Robotiq/2F-140", dest, omni.client.CopyBehavior.OVERWRITE)
lic = omni.client.copy(root + "/Isaac/Robots/Robotiq/LICENSE", dest + "/LICENSE", omni.client.CopyBehavior.OVERWRITE)
ok = res == omni.client.Result.OK and os.path.exists(os.path.join(args.dest, "Robotiq_2F_140_physics_edit.usd"))
total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(args.dest) for f in fs)
print(f"FETCH_{'OK' if ok else 'FAILED'} dest={args.dest} bytes={total} license={lic}", flush=True)
app.close()
