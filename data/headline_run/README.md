# Headline run data

Produced by the code in `../../isaac_sim_headline_run/` (commit `cc9ae661` of the original repository).

| File | What it is |
|---|---|
| `ur5e_isaac_sac_final.zip` | The SAC checkpoint that was deployed on the real robot (629,668 cumulative steps). sha256 `68A2FB52CC5D1F47F4BC9011FA64A6A712347ED6285AC0B7E1AE5E69074E7CA7` |
| `train_console.log` | Console output of the 600,000-step resumed session, including the SB3 rollout tables |
| `telemetry_env0_every100th_step.csv`, `telemetry_env1_every100th_step.csv` | The per-step telemetry of the two parallel environments, every 100th step (the full files are about 70 MB each and are in the original repository under `rl_logs_7hr_final/`) |
| `trial_coldstart2*.log` | Logs of a 30k-step cold-start trial that preceded the resumed run; the last logged step in these files is 28,500 |

The telemetry columns are those of `Ur5eGraspEnv`'s hardware logger (step, curriculum level, distance,
reward, termination, joint positions and velocities, gripper, end-effector and target position,
actions, spawn radius). The `Table_Collision` column is always false for this run.