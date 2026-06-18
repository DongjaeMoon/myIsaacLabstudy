# UROP_v23 policy-learning patch

This patch creates a new task folder, `UROP/UROP_v23`, derived from the v21 design but focused on:

1. Lower, handover-like box trajectories instead of high ballistic lobs.
2. Much stronger anti-tremor behavior after catching.
3. Robust quiet standing when no tag/object is visible or when the box is not catchable.
4. Fewer active reward terms and fewer contact sensors to reduce training overhead.
5. Calibrated torso-mounted camera extrinsics for AprilTag-style object observations.

## Task IDs

- `Isaac-Urop-v23`
- `Isaac-Urop-v23-Play`
- `Isaac-Urop-v23-Demo`

## Actor observation contract

The actor policy observation remains 100D and contains only real-robot-available signals:

- projected gravity: 3
- base angular velocity: 3
- controlled joint positions relative to ready pose: 29
- controlled joint velocities scaled by 0.05: 29
- previous action: 29
- object position in camera optical frame: 3
- object linear velocity in camera optical frame: 3
- tag visible bit: 1

No contact force or simulator-only privileged labels are exposed to the actor.

## Camera extrinsic

The old heuristic camera fallback is replaced by the provided torso-mounted calibration:

```yaml
parent_frame: torso
translation_m: [0.05762, 0.01753, 0.42987]
quat_wxyz: [0.91496, 0.0, 0.40355, 0.0]
```

If `torso_link` exists in the USD, v23 uses `torso_link` pose and applies this torso-to-camera transform. If not, it falls back to pelvis/root using the provided torso offset.

## Main behavioral changes

- Lower toss arc: sender is closer, arrival time is shorter, vertical velocity is clamped lower.
- More idle/no-object training: no-toss episodes now often have `tag_visible=0`.
- Reduced tremor: lower-body/waist action scales are reduced, actuator damping is increased, and `post_catch_stillness_penalty` strongly penalizes lower-body action-rate, lower-body joint velocity, base angular velocity, and base linear velocity after the object enters the catch pocket.
- Fewer reward terms: v23 uses a compact reward stack instead of many overlapping hand/object terms.
- Fewer contact sensors: critic contact uses torso + left hand + right hand only.

## Suggested checks

```bash
cd myIsaacLabstudy

./isaaclab.sh -p -m py_compile \
  UROP/UROP_v23/env_cfg.py \
  UROP/UROP_v23/scene_objects_cfg.py \
  UROP/UROP_v23/mdp/*.py \
  UROP/UROP_v23/agents/rsl_rl_ppo_cfg.py

./isaaclab.sh -p -c "import UROP.UROP_v23; import gymnasium as gym; print(gym.spec('Isaac-Urop-v23')); print(gym.spec('Isaac-Urop-v23-Play')); print(gym.spec('Isaac-Urop-v23-Demo'))"
```

## Suggested training

Probe:

```bash
./isaaclab.sh -p UROP/train_rsl_rl.py \
  --task Isaac-Urop-v23 \
  --num_envs 1024 \
  --max_iterations 800 \
  --headless \
  --seed 1
```

Full candidate:

```bash
./isaaclab.sh -p UROP/train_rsl_rl.py \
  --task Isaac-Urop-v23 \
  --num_envs 4096 \
  --max_iterations 4000 \
  --headless \
  --seed 1
```

Demo replay:

```bash
./isaaclab.sh -p UROP/play_rsl_rl.py \
  --task Isaac-Urop-v23-Demo \
  --num_envs 1 \
  --checkpoint "logs/rsl_rl/UROP_v23/<RUN_NAME>/model_<ITER>.pt"
```
