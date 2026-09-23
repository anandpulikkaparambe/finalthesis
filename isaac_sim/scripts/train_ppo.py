"""UR5e Isaac Sim RL training -- PPO, run alongside train.py's SAC as a comparison
(not a replacement -- see that file for the SAC training loop). Same env
(Ur5eGraspEnv), same CurriculumCallback logic, same checkpointing pattern. What's
different is algorithm-specific: PPO is on-policy (no replay buffer to save/restore
on resume, unlike SAC), and typically needs many more environment steps to reach
comparable performance on continuous control -- budget --total-timesteps
accordingly when comparing against a SAC run of the same nominal step count.

See train.py's module docstring for two things that apply here unchanged:
IMPORTANT #1 (no tensorboard_log -- same protobuf/Isaac Sim conflict) and
IMPORTANT #2 (stable_baselines3/torch imports must stay deferred into main(),
same SubprocVecEnv 'spawn' reasoning on Windows).

PPO benefits more than SAC from parallelism (on-policy: each update collects
n_steps * num_envs samples), so --num-envs > 1 matters more here than for SAC.

    C:\\isaacsim\\python.bat scripts\\train_ppo.py --num-envs 8              (Windows, local dev)
    ~/isaacsim/python.sh scripts/train_ppo.py --num-envs 8 --device cuda     (vast.ai)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))


def main():
    parser = argparse.ArgumentParser(description="UR5e Isaac Sim RL Training (PPO)")
    parser.add_argument("--resume", type=str, default="", help="Path to a .zip checkpoint to resume from")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel Isaac Sim instances (one process each)")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--log-dir", type=str, default="./rl_logs_ppo")
    parser.add_argument("--device", type=str, default="cpu", help="'auto', 'cpu', or 'cuda' -- see train.py's module docstring: 'cpu' is the safe default on a VRAM-constrained box where Isaac Sim itself already loads the GPU")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout length per env before each PPO update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    args = parser.parse_args()

    print(f"Initializing UR5e Isaac Sim Environment for PPO ({args.num_envs} parallel instance(s))...", flush=True)

    from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

    def _make_env(env_id):
        def _init():
            return Ur5eGraspEnv(env_id=env_id, headless=args.headless, log_dir=args.log_dir, num_envs=args.num_envs)
        return _init

    if args.num_envs > 1:
        # Deferred import (see train.py's module docstring): only reached in the main
        # process here, and inside _init() -- which the current process is about to
        # call -- for each SubprocVecEnv worker, so this ordering is safe for both.
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([_make_env(i) for i in range(args.num_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([_make_env(0)])

    # Safe now: at least one Ur5eGraspEnv (and therefore SimulationApp) already
    # exists in this process before stable_baselines3/torch gets imported.
    import collections

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class CurriculumCallback(BaseCallback):
        """Same logic as train.py's CurriculumCallback (SAC): +0.05 curriculum_level
        per 20-episode window at >=90% rolling success, clamped to <=0.2 for the first
        50k steps, <=1.0 after. No critic-loss kill switch here -- PPO has no
        off-policy critic loss to watch the same way; PPO's own clip-fraction/KL
        divergence are what SB3's own verbose=1 logging already surfaces if training
        goes unstable."""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.success_history = collections.deque(maxlen=20)
            self.curriculum_level = 0.0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones", [])
            for i, info in enumerate(self.locals.get("infos", [])):
                if i < len(dones) and dones[i] and "is_success" in info:
                    self.success_history.append(float(info["is_success"]))

            if len(self.success_history) >= 20:
                success_rate = sum(self.success_history) / len(self.success_history)
                if success_rate > 0.9:
                    self.curriculum_level += 0.05
                    self.success_history.clear()

            if self.num_timesteps < 50000:
                self.curriculum_level = min(self.curriculum_level, 0.2)
            else:
                self.curriculum_level = min(self.curriculum_level, 1.0)

            self.training_env.env_method("set_curriculum_level", self.curriculum_level)
            return True

    os.makedirs(args.log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=args.log_dir,
        name_prefix="ur5e_isaac_ppo",
    )

    if args.resume:
        print(f"Resuming PPO Agent from checkpoint: {args.resume}", flush=True)
        if not os.path.exists(args.resume):
            print(f"[ERROR]: Could not find file {args.resume}. Exiting...", flush=True)
            return
        model = PPO.load(
            args.resume,
            env=env,
            custom_objects={
                "action_space": env.action_space,
                "observation_space": env.observation_space,
            },
        )
    else:
        print("Initializing new PPO Agent...", flush=True)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            device=args.device,
        )

    print("Starting Training Loop...", flush=True)
    curriculum_callback = CurriculumCallback()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, curriculum_callback],
            reset_num_timesteps=not bool(args.resume),
        )
        model.save(os.path.join(args.log_dir, "ur5e_isaac_ppo_final"))
        print("Training Completed and Model Saved!", flush=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...", flush=True)
        model.save(os.path.join(args.log_dir, "ur5e_isaac_ppo_interrupted"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
