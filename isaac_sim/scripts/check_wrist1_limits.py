"""Diagnostic: dump the imported USD's actual physics drive limits/stiffness for wrist_1_joint,
to check whether the importer set something narrower than the URDF's +-2pi. Throwaway script."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from ur5e_grasp.config import load_config
from ur5e_grasp import ur5e_grasp_env as E

cfg = load_config("")
cfg.randomization.enabled = False
env = E.Ur5eGraspEnv(env_id=0, headless=True, log_dir="./check_telemetry", config=cfg, seed=0)

from pxr import UsdPhysics

stage = env._robot._prim.GetStage() if hasattr(env._robot, "_prim") else None
if stage is None:
    import omni.usd
    stage = omni.usd.get_context().get_stage()

for name in ["wrist_1_joint", "wrist_2_joint", "elbow_joint", "shoulder_lift_joint"]:
    found = False
    for prim in stage.Traverse():
        if prim.GetName() == name:
            found = True
            print(f"--- {prim.GetPath()} ---", flush=True)
            for attr in prim.GetAttributes():
                n = attr.GetName()
                if "limit" in n.lower() or "drive" in n.lower():
                    try:
                        print(f"  {n} = {attr.Get()}", flush=True)
                    except Exception as e:
                        print(f"  {n} = <error {e}>", flush=True)
    if not found:
        print(f"--- {name}: NOT FOUND ---", flush=True)

env.close()
