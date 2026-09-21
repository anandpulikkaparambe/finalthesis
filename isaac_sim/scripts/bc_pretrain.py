"""Behaviour-clone the SAC actor from demonstrations (scripts/collect_demos.py).

Needs only numpy, torch and stable-baselines3; it does not start Isaac Sim. The result is a
SAC .zip whose actor is passed to scripts/train.py with --bc-init. Only the actor is
trained: the critics stay random, so use --demo-file in train.py as well (the replay buffer
then contains the demonstrations' rewards) and treat this as an experimental warm start.

    python scripts/bc_pretrain.py --demos demos.npz --out bc_actor.zip --epochs 200
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC

parser = argparse.ArgumentParser()
parser.add_argument("--demos", required=True)
parser.add_argument("--out", default="bc_actor.zip")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--device", default="auto")
args = parser.parse_args()

LOW = np.array([-0.08] * 6 + [-np.pi], dtype=np.float32)
HIGH = np.array([0.08] * 6 + [np.pi], dtype=np.float32)


class SpacesOnlyEnv(gym.Env):
    """Placeholder so SAC can be constructed without starting Isaac Sim."""

    observation_space = spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)
    action_space = spaces.Box(LOW, HIGH, dtype=np.float32)

    def reset(self, seed=None, options=None):
        return np.zeros(23, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(23, dtype=np.float32), 0.0, True, False, {}


data = np.load(args.demos)
obs = torch.as_tensor(data["obs"], dtype=torch.float32)
scaled_actions = torch.as_tensor(
    2.0 * (data["actions"] - LOW) / (HIGH - LOW) - 1.0, dtype=torch.float32
)  # SB3 policies work in [-1, 1]
print(f"{len(obs)} demonstration transitions", flush=True)

model = SAC("MlpPolicy", SpacesOnlyEnv(), policy_kwargs=dict(net_arch=[256, 256]), device=args.device)
actor = model.policy.actor
device = model.device
obs, scaled_actions = obs.to(device), scaled_actions.to(device)
opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    perm = torch.randperm(len(obs), device=device)
    total = 0.0
    for i in range(0, len(obs), args.batch_size):
        idx = perm[i:i + args.batch_size]
        pred = actor(obs[idx], deterministic=True)
        loss = torch.nn.functional.mse_loss(pred, scaled_actions[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss) * len(idx)
    if epoch % 20 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch}: BC loss {total / len(obs):.5f}", flush=True)

model.save(args.out)
print(f"saved {args.out}", flush=True)
