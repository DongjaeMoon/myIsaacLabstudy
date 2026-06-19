# UROP_v25 low-gain small-box policy-learning patch

This patch creates `UROP/UROP_v25` as a separate Isaac Lab task.

Task IDs:
- `Isaac-Urop-v25`
- `Isaac-Urop-v25-Play`
- `Isaac-Urop-v25-Demo`

Main changes from v24:

1. Smaller physical object
   - Previous nominal box: `(0.30, 0.23, 0.21)` m.
   - v25 nominal box: `(0.23, 0.17, 0.15)` m.
   - Volume is about 40.5% of the previous box.

2. Low-gain robot actuators
   - Legs/waist: P=42.0, D=2.6
   - Shoulder pitch/elbow: P=36.0, D=2.2
   - Other arm pose joints: P=30.0, D=1.8
   - Locked fingers: P=24.0, D=1.4

3. Stable standby offset
   - `READY_POSE = CATCH_READY_POSE = SAFE_STAND_POSE`.
   - The robot should not hold a pre-open hug pose while the tag is invisible or the object is not catchable.

4. Low-arc trajectory
   - `trajectory_mode="low_arc"`.
   - The toss apex is controlled directly with a small clearance above the release/target height.
   - Training clearance: `(0.055, 0.16)` m.
   - Demo clearance: `(0.045, 0.090)` m.

5. More explicit no-tag standby
   - `tag_on_step` can delay when the AprilTag becomes visible to the actor.
   - Before that, `tag_visible=0` and object pose/velocity observations are zeroed.

6. Active catch shaping
   - Added `hand_approach_reward`, which rewards the hands moving closer to the object during `reaction_window`.
   - This is intended to reduce passive "the box lands on the robot" behavior.

7. Milder disturbance randomization
   - Random pushes are still present for standing robustness, but less aggressive than v23/v24.

## Validation

Syntax was checked with local `py_compile` against the generated files. Runtime Isaac Lab import still needs to be checked on your GPU machine.

```bash
cd myIsaacLabstudy
./isaaclab.sh -p -m py_compile \
  UROP/UROP_v25/env_cfg.py \
  UROP/UROP_v25/scene_objects_cfg.py \
  UROP/UROP_v25/mdp/*.py \
  UROP/UROP_v25/agents/rsl_rl_ppo_cfg.py
```

Task registration:

```bash
cd myIsaacLabstudy
./isaaclab.sh -p -c "import UROP.UROP_v25; import gymnasium as gym; print(gym.spec('Isaac-Urop-v25')); print(gym.spec('Isaac-Urop-v25-Play')); print(gym.spec('Isaac-Urop-v25-Demo'))"
```

Quick GUI check:

```bash
cd myIsaacLabstudy
./isaaclab.sh -p UROP/train_rsl_rl.py \
  --task Isaac-Urop-v25 \
  --num_envs 4 \
  --max_iterations 30 \
  --seed 1
```

Probe training:

```bash
cd myIsaacLabstudy
./isaaclab.sh -p UROP/train_rsl_rl.py \
  --task Isaac-Urop-v25 \
  --num_envs 1024 \
  --max_iterations 1000 \
  --headless \
  --seed 1
```

Full training candidate:

```bash
cd myIsaacLabstudy
./isaaclab.sh -p UROP/train_rsl_rl.py \
  --task Isaac-Urop-v25 \
  --num_envs 4096 \
  --max_iterations 4000 \
  --headless \
  --seed 1
```

Demo playback:

```bash
cd myIsaacLabstudy
./isaaclab.sh -p UROP/play_rsl_rl.py \
  --task Isaac-Urop-v25-Demo \
  --num_envs 1 \
  --checkpoint "logs/rsl_rl/UROP_v25/<RUN_NAME>/model_<ITER>.pt"
```
