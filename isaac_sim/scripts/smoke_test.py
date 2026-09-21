"""Minimal headless smoke test: spawn the env, reset, take random actions for a few
episodes, confirm nothing crashes. Run this after (re)generating the USD asset and
before trusting a real training run.

    C:\\isaacsim\\python.bat scripts\\smoke_test.py        (Windows)
    ~/isaacsim/python.sh scripts/smoke_test.py             (Linux / vast.ai)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--steps-per-episode", type=int, default=30)
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

t0 = time.time()
env = Ur5eGraspEnv(env_id=0, headless=args.headless, log_dir="./rl_logs/smoke_test")
print(f"ENV_CREATED after {time.time() - t0:.1f}s", flush=True)
print(f"action_space={env.action_space} observation_space={env.observation_space}", flush=True)

for ep in range(args.episodes):
    obs, info = env.reset()
    assert obs.shape == (23,), f"unexpected obs shape {obs.shape}"
    ep_reward = 0.0
    for step in range(args.steps_per_episode):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            print(
                f"episode {ep} ended at step {step}: "
                f"reason={info.get('termination_reason', 'timeout')} reward_sum={ep_reward:.3f}",
                flush=True,
            )
            break
    else:
        print(f"episode {ep} ran full {args.steps_per_episode} steps, reward_sum={ep_reward:.3f}", flush=True)

print("SMOKE_TEST_OK", flush=True)
env.close()
