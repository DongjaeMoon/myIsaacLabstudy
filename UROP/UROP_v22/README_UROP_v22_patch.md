# UROP_v22 patch

Copy this directory to:

```text
myIsaacLabstudy/UROP/UROP_v22/
```

Registered tasks:

```text
Isaac-Urop-v22
Isaac-Urop-v22-Play
Isaac-Urop-v22-Demo
```

Major changes from v21:

1. Actor observation remains 100-D and contains only real-robot-available signals.
2. Contact forces, object truth, episode type, and object parameters are critic-only.
3. Catch/hug/object-stabilization rewards are gated by `reaction_window` or `near_catch_window`, not by release alone.
4. Success requires a stricter chest-pocket state, slow object velocity, and two-hand bracket geometry.
5. Post-catch tremor is penalized with lower-body joint velocity/action-rate and base angular velocity terms.
6. `Isaac-Urop-v22-Demo` provides a clean, single-env front toss for video recording.

Quick checks:

```bash
cd myIsaacLabstudy
./isaaclab.sh -p -m py_compile UROP/UROP_v22/env_cfg.py UROP/UROP_v22/mdp/*.py UROP/UROP_v22/agents/rsl_rl_ppo_cfg.py
./isaaclab.sh -p -c "import UROP.UROP_v22; import gymnasium as gym; print(gym.spec('Isaac-Urop-v22')); print(gym.spec('Isaac-Urop-v22-Play')); print(gym.spec('Isaac-Urop-v22-Demo'))"
```

First GUI sanity check:

```bash
./isaaclab.sh -p UROP/train_rsl_rl.py --task Isaac-Urop-v22 --num_envs 4 --max_iterations 40 --seed 1
```

Smoke test:

```bash
./isaaclab.sh -p UROP/train_rsl_rl.py --task Isaac-Urop-v22 --num_envs 64 --max_iterations 10 --headless --seed 1
```

Probe run:

```bash
./isaaclab.sh -p UROP/train_rsl_rl.py --task Isaac-Urop-v22 --num_envs 1024 --max_iterations 1200 --headless --seed 1
```

Full run:

```bash
./isaaclab.sh -p UROP/train_rsl_rl.py --task Isaac-Urop-v22 --num_envs 4096 --max_iterations 6000 --headless --seed 1
```

Play:

```bash
./isaaclab.sh -p UROP/play_rsl_rl.py --task Isaac-Urop-v22-Play --num_envs 8 --checkpoint logs/rsl_rl/UROP_v22/<RUN>/model_3000.pt
```

Demo:

```bash
./isaaclab.sh -p UROP/play_rsl_rl.py --task Isaac-Urop-v22-Demo --num_envs 1 --checkpoint logs/rsl_rl/UROP_v22/<RUN>/model_3000.pt
```
