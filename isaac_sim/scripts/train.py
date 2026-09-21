"""UR5e Isaac Sim RL training (SAC), thesis-driven version.

Changes relative to the script that produced the thesis' headline run (see
../../isaac_sim_headline_run/scripts/train.py):
  * every tunable comes from a config file (--config) that is saved next to the checkpoints
  * seeds (--seed) so runs can be repeated; use scripts/run_seeds.sh for several seeds
  * a 1M-step replay buffer by default (the headline run used 100k for 600k steps)
  * curriculum with regression and a configurable start (curriculum.py)
  * optional demonstration pre-fill of the replay buffer (--demo-file) and behaviour-cloned
    actor initialisation (--bc-init), both produced by scripts/collect_demos.py and
    scripts/bc_pretrain.py
  * periodic checkpoints every --save-freq steps (replay buffers are large now)

IMPORTANT #1 -- no TensorBoard logging: with Isaac Sim in the same process, TensorBoard's
event writer crashes silently on the first .learn() call (suspected protobuf conflict).
Progress is visible in each env's hardware_log_env*.csv and SB3's console table.

IMPORTANT #2: stable_baselines3 (and therefore torch) must NOT be imported at module level.
Isaac Sim's SimulationApp has to be the first heavy native library initialised in the
process, so every SB3 import is deferred into main() after the first environment exists.

    ~/isaacsim/python.sh scripts/train.py --num-envs 8 --device cuda --config configs/default.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))


def main():
    parser = argparse.ArgumentParser(description="UR5e Isaac Sim RL Training (SAC)")
    parser.add_argument("--resume", type=str, default="", help="Path to a .zip checkpoint to resume from")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel Isaac Sim instances (one process each)")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--log-dir", type=str, default="./rl_logs")
    parser.add_argument("--device", type=str, default="cuda", help="'auto', 'cpu' or 'cuda'")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--config", type=str, default="", help="JSON file with an EnvConfig (see source/ur5e_grasp/config.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-entropy-scale", type=float, default=0.5, help="target entropy = -scale * action_dim")
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--demo-file", type=str, default="", help="npz from scripts/collect_demos.py to prefill the replay buffer")
    parser.add_argument("--bc-init", type=str, default="", help="SAC .zip from scripts/bc_pretrain.py; its actor initialises the policy")
    args = parser.parse_args()

    from ur5e_grasp.config import load_config, save_config

    cfg = load_config(args.config)
    os.makedirs(args.log_dir, exist_ok=True)
    save_config(cfg, os.path.join(args.log_dir, "env_config.json"))
    with open(os.path.join(args.log_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Initializing UR5e Isaac Sim Environment for SAC ({args.num_envs} parallel instance(s), seed {args.seed})...", flush=True)

    from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

    def _make_env(env_id):
        def _init():
            return Ur5eGraspEnv(env_id=env_id, headless=args.headless, log_dir=args.log_dir, config=cfg, seed=args.seed)
        return _init

    if args.num_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([_make_env(i) for i in range(args.num_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([_make_env(0)])

    import numpy as np
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    from ur5e_grasp.curriculum import CurriculumManager

    class CurriculumCallback(BaseCallback):
        """Feeds finished episodes to CurriculumManager and pushes the level to every env.
        Keeps the critic-loss kill switch of the original script."""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.manager = CurriculumManager(cfg.curriculum)
            self._last_print = 0

        def _on_step(self) -> bool:
            if self.model and getattr(self.model, "logger", None):
                critic_loss = self.model.logger.name_to_value.get("train/critic_loss")
                if critic_loss is not None and critic_loss > 1000.0:
                    print(f"\n[KILL SWITCH] Critic loss exploded to {critic_loss}! Stopping training early.", flush=True)
                    return False
            dones = self.locals.get("dones", [])
            for i, info in enumerate(self.locals.get("infos", [])):
                if i < len(dones) and dones[i] and "is_success" in info:
                    self.manager.record_episode(bool(info["is_success"]))
            level = self.manager.update(self.num_timesteps)
            self.training_env.env_method("set_curriculum_level", level)
            if self.num_timesteps - self._last_print >= 20000:
                self._last_print = self.num_timesteps
                print(f"[curriculum] step={self.num_timesteps} level={level:.2f}", flush=True)
            return True

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.save_freq // max(1, args.num_envs)),
        save_path=args.log_dir, name_prefix="ur5e_isaac_sac", save_replay_buffer=True,
    )

    if args.resume:
        print(f"Resuming SAC Agent from checkpoint: {args.resume}", flush=True)
        if not os.path.exists(args.resume):
            print(f"[ERROR]: Could not find file {args.resume}. Exiting...", flush=True)
            return
        model = SAC.load(
            args.resume, env=env,
            custom_objects={"action_space": env.action_space, "observation_space": env.observation_space},
        )
        resume_dir, resume_name = os.path.split(args.resume)
        replay_buffer_name = re.sub(r"_(\d+_steps)\.zip$", r"_replay_buffer_\1.pkl", resume_name)
        replay_buffer_path = os.path.join(resume_dir, replay_buffer_name)
        if os.path.exists(replay_buffer_path):
            print(f"Restoring replay buffer from: {replay_buffer_path}", flush=True)
            try:
                model.load_replay_buffer(replay_buffer_path)
            except Exception as e:
                print(f"[WARNING] Could not load replay buffer ({e}) -- resuming with an empty one.", flush=True)
        else:
            print(f"[WARNING] No matching replay buffer at {replay_buffer_path} -- resuming with an empty one.", flush=True)
    else:
        print("Initializing new SAC Agent...", flush=True)
        model = SAC(
            "MlpPolicy", env, verbose=1, learning_rate=args.learning_rate, buffer_size=args.buffer_size,
            batch_size=args.batch_size, gamma=0.99, tau=0.005,
            target_entropy=-args.target_entropy_scale * env.action_space.shape[0],
            policy_kwargs=dict(net_arch=[256, 256]), device=args.device, seed=args.seed,
        )
        if args.bc_init:
            print(f"Initialising the actor from behaviour-cloned weights: {args.bc_init}", flush=True)
            bc_model = SAC.load(args.bc_init, device=args.device)
            model.policy.actor.load_state_dict(bc_model.policy.actor.state_dict())
        if args.demo_file:
            data = np.load(args.demo_file)
            n = len(data["actions"])
            print(f"Prefilling the replay buffer with {n} demonstration transitions", flush=True)
            for i in range(n):
                scaled = model.policy.scale_action(data["actions"][i][None, :])
                model.replay_buffer.add(
                    data["obs"][i][None, :], data["next_obs"][i][None, :], scaled,
                    np.array([data["rewards"][i]]), np.array([bool(data["dones"][i])]), [{}],
                )

    print("Starting Training Loop...", flush=True)
    try:
        model.learn(
            total_timesteps=args.total_timesteps, callback=[checkpoint_callback, CurriculumCallback()],
            reset_num_timesteps=not bool(args.resume),
        )
        model.save(os.path.join(args.log_dir, "ur5e_isaac_sac_final"))
        print("Training Completed and Model Saved!", flush=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...", flush=True)
        model.save(os.path.join(args.log_dir, "ur5e_isaac_sac_interrupted"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
