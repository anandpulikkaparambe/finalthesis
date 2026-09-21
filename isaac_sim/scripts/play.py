"""Load a trained SAC checkpoint and roll it out for a few episodes -- for visually
checking a policy (pass --no-headless to open the Isaac Sim viewport) or for a
quick numeric sanity check (success rate over N episodes) without opening a window.

    C:\\isaacsim\\python.bat scripts\\play.py --checkpoint rl_logs\\ur5e_isaac_sac_final.zip --no-headless
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.set_defaults(headless=True)
args = parser.parse_args()

from stable_baselines3 import SAC

from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

env = Ur5eGraspEnv(env_id=0, headless=args.headless, log_dir="./rl_logs/play")
model = SAC.load(
    args.checkpoint,
    env=env,
    custom_objects={"action_space": env.action_space, "observation_space": env.observation_space},
)

successes = 0
for ep in range(args.episodes):
    obs, info = env.reset()
    terminated = truncated = False
    ep_reward = 0.0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
    success = bool(info.get("is_success", False))
    successes += int(success)
    print(
        f"episode {ep}: reward={ep_reward:.3f} success={success} "
        f"reason={info.get('termination_reason', 'timeout')}",
        flush=True,
    )

print(f"SUCCESS_RATE: {successes}/{args.episodes}", flush=True)
env.close()
