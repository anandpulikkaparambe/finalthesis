"""UR5e Isaac Sim RL training -- SAC, ported from remoteservertraining-UR5e's
src/ur3e_rl/ur3e_rl/train_sac.py. Same CurriculumCallback logic (success-rate-gated
curriculum, Run-6-style 50k-step clamp), same SAC hyperparameters, same
resume/replay-buffer-restore behavior -- see that file's own extensive inline
history for *why* each of those numbers is what it is. What's different: no ROS2
namespace/GZ_PARTITION plumbing (there's no Gazebo instance to point at).

IMPORTANT #1 -- no TensorBoard logging here (tensorboard_log is deliberately never
passed to SAC(...)/`.learn()`, unlike train_sac.py). Confirmed live 2026-09-09: SAC
construction WITH tensorboard_log succeeds, but the first `.learn()` call crashes
immediately and silently (no Python exception, no traceback even under
PYTHONUNBUFFERED=1 -- exit code 0, just "Simulation App Shutting Down" seconds
later) the moment TensorBoard's logger actually initializes, while the identical
setup minus tensorboard_log runs cleanly. Root cause not fully root-caused, but the
leading suspect is a protobuf ABI conflict -- TensorBoard's event-file writer and
Isaac Sim/USD both bundle their own compiled protobuf, and loading both in one
process is a known-fragile combination. Ruled out first: GPU-memory pressure
(reproduced identically under --device cpu and --device cuda) and import order
(reproduced even after moving every stable_baselines3 import into main(), see
IMPORTANT #2 below -- that reordering is still kept, as good practice per Isaac
Sim's own docs, but it was NOT what fixed this). Training progress is still fully
visible via each env's own hardware_log_env*.csv (per-step telemetry) and SB3's
own verbose=1 console table (plain print-based, no protobuf involved) --
`tensorboard --logdir` visualization just isn't available for this in-process
setup. Follow-up idea, not yet tried: log metrics from a separate process instead.

IMPORTANT #2: stable_baselines3 (and therefore torch) must NOT be imported at
module level here. Isaac Sim's SimulationApp needs to be the first heavy native
library initialized in the process -- every sb3 import is deferred into main(),
after the first Ur5eGraspEnv (and hence SimulationApp) already exists. This also
matters for --num-envs > 1: on Windows, SubprocVecEnv's 'spawn' start method
re-imports this whole module in each child process before calling _init(), so a
module-level sb3 import would run in every worker too -- keeping it inside main()
means worker processes (which only ever call _init(), never main()) never trigger
it at all.

    C:\\isaacsim\\python.bat scripts\\train.py --num-envs 2                 (Windows, local dev)
    ~/isaacsim/python.sh scripts/train.py --num-envs 8 --device cuda        (vast.ai)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))


def main():
    parser = argparse.ArgumentParser(description="UR5e Isaac Sim RL Training (SAC)")
    parser.add_argument("--resume", type=str, default="", help="Path to a .zip checkpoint to resume from")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel Isaac Sim instances (one process each)")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--log-dir", type=str, default="./rl_logs")
    parser.add_argument("--device", type=str, default="cpu", help="'auto', 'cpu', or 'cuda' -- see module docstring: 'cpu' is the safe default on a VRAM-constrained box where Isaac Sim itself already loads the GPU")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Initializing UR5e Isaac Sim Environment for SAC ({args.num_envs} parallel instance(s))...", flush=True)

    from ur5e_grasp.ur5e_grasp_env import Ur5eGraspEnv

    def _make_env(env_id):
        def _init():
            return Ur5eGraspEnv(env_id=env_id, headless=args.headless, log_dir=args.log_dir)
        return _init

    if args.num_envs > 1:
        # Deferred import (see module docstring): only reached in the main process
        # here, and inside _init() -- which the current process is about to
        # call -- for each SubprocVecEnv worker, so this ordering is safe for both.
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv([_make_env(i) for i in range(args.num_envs)])
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([_make_env(0)])

    # Safe now: at least one Ur5eGraspEnv (and therefore SimulationApp) already
    # exists in this process before stable_baselines3/torch gets imported.
    import collections
    import re

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    class CurriculumCallback(BaseCallback):
        """Ported as-is from train_sac.py's CurriculumCallback: +0.05 curriculum_level
        per 20-episode window at >=90% rolling success, clamped to <=0.2 for the first
        50k steps ("Run 6 Rule"), <=1.0 after. Also carries the same critic-loss kill
        switch (stop training rather than let a >1000 critic loss poison the policy
        with NaNs)."""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.success_history = collections.deque(maxlen=20)
            self.curriculum_level = 0.0

        def _on_step(self) -> bool:
            if self.model and hasattr(self.model, "logger") and self.model.logger:
                if "train/critic_loss" in self.model.logger.name_to_value:
                    critic_loss = self.model.logger.name_to_value["train/critic_loss"]
                    if critic_loss > 1000.0:
                        print(f"\n[KILL SWITCH] Critic loss exploded to {critic_loss}! Stopping training early.", flush=True)
                        return False

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
        name_prefix="ur5e_isaac_sac",
        save_replay_buffer=True,
    )

    if args.resume:
        print(f"Resuming SAC Agent from checkpoint: {args.resume}", flush=True)
        if not os.path.exists(args.resume):
            print(f"[ERROR]: Could not find file {args.resume}. Exiting...", flush=True)
            return
        model = SAC.load(
            args.resume,
            env=env,
            custom_objects={
                "action_space": env.action_space,
                "observation_space": env.observation_space,
            },
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
        target_entropy = -0.5 * env.action_space.shape[0]
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            target_entropy=target_entropy,
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
        model.save(os.path.join(args.log_dir, "ur5e_isaac_sac_final"))
        print("Training Completed and Model Saved!", flush=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...", flush=True)
        model.save(os.path.join(args.log_dir, "ur5e_isaac_sac_interrupted"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
