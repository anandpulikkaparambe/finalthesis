"""Deterministic evaluation of a checkpoint, separate from the training success rate.

Runs N episodes with the policy in deterministic mode in the full environment (self-collision
on, physical contact confirmation, collisions active) and writes a JSON summary plus a
per-episode CSV. Use --no-randomization for the nominal scene and the default for a
randomized one. Success here is the environment's own criterion (info["is_success"]), which
requires confirmed finger-pad contact held over several steps.

    ~/isaacsim/python.sh scripts/evaluate.py --checkpoint rl_logs/ur5e_isaac_sac_final.zip \
        --episodes 50 --config configs/default.json --curriculum-level 1.0
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--config", type=str, default="")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--curriculum-level", type=float, default=1.0)
parser.add_argument("--no-randomization", action="store_true")
parser.add_argument("--out-dir", type=str, default="./eval_out")
parser.add_argument("--no-headless", dest="headless", action="store_false")
parser.set_defaults(headless=True)
args = parser.parse_args()

from ur5e_grasp.config import load_config

cfg = load_config(args.config)
if args.no_randomization:
    cfg.randomization.enabled = False
os.makedirs(args.out_dir, exist_ok=True)

from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

env = Ur5eGraspEnv(env_id=0, headless=args.headless, log_dir=os.path.join(args.out_dir, "telemetry"), config=cfg, seed=args.seed)
env.set_curriculum_level(args.curriculum_level)

from stable_baselines3 import SAC

model = SAC.load(
    args.checkpoint, env=env,
    custom_objects={"action_space": env.action_space, "observation_space": env.observation_space},
)

rows, reasons = [], collections.Counter()
for ep in range(args.episodes):
    obs, _ = env.reset(seed=args.seed + ep)
    done = False
    steps, ep_reward, min_dist, min_self, min_table = 0, 0.0, float("inf"), float("inf"), float("inf")
    info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        ep_reward += reward
        min_dist = min(min_dist, info["distance_to_target"])
        min_self = min(min_self, info["self_clearance"])
        min_table = min(min_table, info["arm_table_clearance"])
    reason = info.get("termination_reason", "timeout")
    reasons[reason] += 1
    rows.append(dict(episode=ep, success=bool(info.get("is_success", False)), reason=reason, steps=steps,
                     reward=round(ep_reward, 3), min_distance=round(min_dist, 4),
                     min_self_clearance=round(min_self, 4), min_arm_table_clearance=round(min_table, 4)))
    print(f"episode {ep}: success={rows[-1]['success']} reason={reason} steps={steps} min_dist={min_dist:.3f}", flush=True)

summary = dict(
    checkpoint=args.checkpoint, episodes=args.episodes, randomization=not args.no_randomization,
    curriculum_level=args.curriculum_level, contact_source=env._contact_source,
    success_rate=sum(r["success"] for r in rows) / len(rows), termination_reasons=dict(reasons),
    mean_steps=sum(r["steps"] for r in rows) / len(rows),
    median_min_distance=sorted(r["min_distance"] for r in rows)[len(rows) // 2],
)
with open(os.path.join(args.out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(args.out_dir, "eval_episodes.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(json.dumps(summary, indent=2), flush=True)
env.close()
