# Unitree G1 UROP v21-v24 Real Run Checklist

This repo side is ready for check-only validation and scripted real runs. Do not
start policy control until obs-only output is stable.

## Today Sequence

1. Robot PC: start the camera/image server manually.
2. Host PC: enable multicast manually.
3. Verify AprilTag standalone or camera preview.
4. Run obs-only for the target UROP version.
5. Run gated policy for the same UROP version.
6. Use nogate only after gated tests are stable.

## Common Preparation

```bash
cd ~/mdj/myIsaacLabstudy
sudo ip link set enx00e04c0f3e58 multicast on
```

Required calibration files:

```text
sim2real/calib/head_camera_extrinsics.real.yaml
sim2real/calib/head_camera_intrinsics.real.yaml
sim2real/calib/box_tag_config.real.yaml
```

The wrappers refuse to run without these files. The values are not guessed in
this repo.

## Version Commands

v21:

```bash
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_obsonly.sh v21
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_policy_gated.sh v21 logs/rsl_rl/UROP_v21/<run>/exported/policy.pt
```

v22:

```bash
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_obsonly.sh v22
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_policy_gated.sh v22 logs/rsl_rl/UROP_v22/<run>/exported/policy.pt
```

v23:

```bash
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_obsonly.sh v23
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_policy_gated.sh v23 logs/rsl_rl/UROP_v23/<run>/exported/policy.pt
```

v24:

```bash
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_obsonly.sh v24
NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 bash sim2real/run_real_apriltag_policy_gated.sh v24 logs/rsl_rl/UROP_v24/<run>/exported/policy.pt
```

Nogate is blocked unless explicitly allowed:

```bash
ALLOW_NOGATE=1 NET_IFACE=enx00e04c0f3e58 PRINT_RATE=10 \
  bash sim2real/run_real_apriltag_policy_nogate.sh v23 logs/rsl_rl/UROP_v23/<run>/exported/policy.pt
```

## Generated YAML

Each version has three runtime configs:

```text
sim2real/configs/g1_catch_real_urop_vXX_apriltag_obsonly.yaml
sim2real/configs/g1_catch_real_urop_vXX_apriltag_policy_gated.yaml
sim2real/configs/g1_catch_real_urop_vXX_apriltag_policy_nogate.yaml
```

Contract snapshots live in:

```text
sim2real/configs/contracts/urop_vXX_contract.yaml
```

Older v23 configs are kept for archive/reference:

```text
sim2real/configs/g1_catch_real_real_apriltag_obsonly_traingain_v23.yaml
sim2real/configs/g1_catch_real_real_apriltag_policy_gated_traingain_v23.yaml
sim2real/configs/g1_catch_real_real_apriltag_policy_nogate_traingain_v23.yaml
sim2real/configs/g1_catch_real_real_tag0_policy_test_traingain_v23.yaml
sim2real/configs/g1_catch_real_real_v23_zeros_obsonly.yaml
```

## Policy Path Rules

Use only exported TorchScript policies:

```text
logs/rsl_rl/UROP_vXX/<run>/exported/policy.pt
```

Do not pass `model_*.pt` checkpoints to sim2real. The policy runner rejects
those paths.

## Check-Only Commands

```bash
python sim2real/tools/list_urop_policies.py
python sim2real/g1_catch_real.py --check-only --config sim2real/configs/g1_catch_real_urop_v23_apriltag_obsonly.yaml --no-policy
python sim2real/tools/check_sim2real_contract.py --version v23 --config sim2real/configs/g1_catch_real_urop_v23_apriltag_policy_gated.yaml --policy logs/rsl_rl/UROP_v23/<run>/exported/policy.pt
```

## Before Enabling Policy

Confirm in obs-only:

- `obs_dim=100`, `action_dim=29`.
- projected gravity is stable and points in the expected policy frame.
- base angular velocity is near zero while standing.
- `object_rel_pos.z > 0` for a visible object in front of the camera policy frame.
- approaching object has `object_rel_lin_vel.z < 0` in OpenCV camera frames.
- `tag_visible=1` when the tag is visible and object terms zero when it is not.
- start pose reaches the version-specific `catch_ready`.
- damping key works before policy autonomy.

## Frame Notes

- v23 uses the raw calibrated OpenCV camera optical observation.
- v21/v22/v24 use `training_camera_opencv`: AprilTag body-relative object state is
  reprojected through the training camera offset and `[-y, -z, x]` OpenCV
  convention used in their training observation code.
- `policy_body_frame=unitree` remains a required real-robot validation item.
