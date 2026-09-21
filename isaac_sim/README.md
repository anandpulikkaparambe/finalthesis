# isaac_sim

Isaac Sim environment and training code with the thesis-driven changes. Not yet run in Isaac Sim: see
`../docs/VALIDATION_CHECKLIST.md`. `README_original_repo.md` is the README of the original repository.

```
source/ur5e_grasp/   environment and its pure-Python parts
    ur5e_grasp_env.py  the Gymnasium environment (Isaac Sim)
    config.py          every tunable, saved next to each run
    reward.py          dense reward and terminal outcomes
    contact.py         finger-pad contact confirmation and hold tracking
    randomization.py   domain randomization, action delay
    curriculum.py      success-gated curriculum with regression
    kinematics.py      UR5e forward kinematics, collision capsules, IK
    demo_controller.py scripted IK demonstrations
    safety_filter.py   collision check for real-robot steps
    spec.py            observation/action layout
scripts/             smoke_test, validate_collisions, train, evaluate, collect_demos, bc_pretrain, run_seeds.sh, play
configs/             default, no_randomization, headline_like_baseline, grasp_and_lift
tests/               49 tests (fake physics backend); run: python -m pytest tests -q
assets/, vastai/     URDF, meshes and converter; Vast.ai setup and entrypoint
```

```bash
python -m pytest tests -q
python scripts/smoke_test.py --episodes 3 --steps-per-episode 100 --config configs/default.json
python scripts/validate_collisions.py --config configs/default.json
python scripts/collect_demos.py --episodes 50 --out demos.npz
python scripts/train.py --num-envs 8 --total-timesteps 1000000 --config configs/default.json --seed 0
python scripts/evaluate.py --checkpoint rl_logs/ur5e_isaac_sac_final.zip --config configs/default.json --episodes 50
```