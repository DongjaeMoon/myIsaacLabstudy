from __future__ import annotations

import math
import os
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = (0.32, 0.24, 0.24)


def _get_stage(env) -> int:
    p = env.cfg.curriculum.stage_schedule.params
    forced = int(p.get("eval_stage", -1))
    if forced >= 0:
        return forced

    step = int(env.common_step_counter)
    nsteps = int(p["num_steps_per_env"])
    s0_limit = int(p["stage0_iters"]) * nsteps
    s1_limit = s0_limit + int(p["stage1_iters"]) * nsteps
    s2_limit = s1_limit + int(p["stage2_iters"]) * nsteps

    if step < s0_limit:
        return 0
    if step < s1_limit:
        return 1
    if step < s2_limit:
        return 2
    return 3


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _ensure_carry_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    if not hasattr(env, "_urop_carry_cmd"):
        env._urop_carry_cmd = torch.zeros((n, 3), device=d)
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.ones(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d)
    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((n, 2), device=d)
    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), 3.0, device=d)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.8, device=d)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.01, device=d)


def _load_state_bank_if_needed(env: "ManagerBasedRLEnv", state_bank_path: str) -> None:
    if hasattr(env, "_urop_state_bank"):
        return

    if not os.path.exists(state_bank_path):
        raise FileNotFoundError(
            f"Carry state bank not found: {state_bank_path}\n"
            "First build it from your successful catch policy rollouts."
        )

    raw = torch.load(state_bank_path, map_location="cpu")
    required = ["root_pose", "root_vel", "joint_pos", "joint_vel", "object_pose", "object_vel"]
    for key in required:
        if key not in raw:
            raise RuntimeError(f"State bank missing key: {key}")

    bank = {k: v.to(env.device) if torch.is_tensor(v) else v for k, v in raw.items()}
    env._urop_state_bank = bank

    if "box_size" in bank:
        env._urop_bank_box_size_mean = bank["box_size"].float().mean(dim=0)
    else:
        env._urop_bank_box_size_mean = torch.tensor(DEFAULT_BOX_SIZE, device=env.device)

    if "obj_rel_root" in bank:
        env._urop_bank_target_obj_rel = bank["obj_rel_root"].float().mean(dim=0)
    else:
        env._urop_bank_target_obj_rel = torch.tensor([0.42, 0.0, 0.22], device=env.device)

    env._urop_bank_root_z_mean = bank["root_pose"][:, 2].float().mean()


def _sample_command_by_stage(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, stage0: dict, stage1: dict, stage2: dict, stage3: dict) -> None:
    _ensure_carry_buffers(env)
    stage = _get_stage(env)

    if stage == 0:
        cfg = stage0
    elif stage == 1:
        cfg = stage1
    elif stage == 2:
        cfg = stage2
    else:
        cfg = stage3

    if cfg.get("zero", False):
        env._urop_carry_cmd[env_ids] = 0.0
        return

    n = int(env_ids.shape[0])
    d = env.device
    vx = torch.empty(n, device=d).uniform_(*cfg["vx"])
    vy = torch.empty(n, device=d).uniform_(*cfg["vy"])
    wz = torch.empty(n, device=d).uniform_(*cfg["wz"])
    env._urop_carry_cmd[env_ids] = torch.stack([vx, vy, wz], dim=-1)


def resample_carry_command(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
    stage3: dict,
) -> None:
    _sample_command_by_stage(env, env_ids, stage0, stage1, stage2, stage3)


def randomize_carry_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass_range=(2.5, 4.0),
    friction_range=(0.70, 0.95),
    restitution_range=(0.00, 0.03),
    size_jitter=(0.97, 1.03),
) -> None:
    _ensure_carry_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    mass = torch.empty((n, 1), device=d).uniform_(*mass_range)
    friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)
    size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    size = size * torch.empty((n, 3), device=d).uniform_(*size_jitter)

    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = size


def reset_from_state_bank(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    state_bank_path: str,
    stage0_cmd: dict,
    stage1_cmd: dict,
    stage2_cmd: dict,
    stage3_cmd: dict,
    xy_noise: float = 0.01,
    yaw_noise_deg: float = 0.0,
    joint_noise: float = 0.015,
    object_linvel_noise: float = 0.05,
    object_angvel_noise: float = 0.10,
) -> None:
    _ensure_carry_buffers(env)
    _load_state_bank_if_needed(env, state_bank_path)
    randomize_carry_object(env, env_ids)

    bank = env._urop_state_bank
    robot = env.scene["robot"]
    obj = env.scene["object"]
    n = int(env_ids.shape[0])
    device = env.device

    bank_n = bank["root_pose"].shape[0]
    sample_ids = torch.randint(0, bank_n, (n,), device=device)

    root_pose = bank["root_pose"][sample_ids].clone()
    root_vel = bank["root_vel"][sample_ids].clone()
    joint_pos = bank["joint_pos"][sample_ids].clone()
    joint_vel = bank["joint_vel"][sample_ids].clone()
    object_pose = bank["object_pose"][sample_ids].clone()
    object_vel = bank["object_vel"][sample_ids].clone()

    origins = env.scene.env_origins[env_ids]

    if xy_noise > 0.0:
        delta_xy = torch.empty((n, 2), device=device).uniform_(-xy_noise, xy_noise)
        root_pose[:, 0:2] += delta_xy
        object_pose[:, 0:2] += delta_xy

    if yaw_noise_deg > 0.0:
        yaw_noise = torch.empty(n, device=device).uniform_(math.radians(-yaw_noise_deg), math.radians(yaw_noise_deg))
        dq = _yaw_to_quat(yaw_noise)
        root_pose[:, 3:7] = quat_mul(dq, root_pose[:, 3:7])
        object_pose[:, 3:7] = quat_mul(dq, object_pose[:, 3:7])
        root_pose[:, 0:3] = quat_apply(dq, root_pose[:, 0:3])
        object_pose[:, 0:3] = quat_apply(dq, object_pose[:, 0:3])
        root_vel[:, 0:3] = quat_apply(dq, root_vel[:, 0:3])
        root_vel[:, 3:6] = quat_apply(dq, root_vel[:, 3:6])
        object_vel[:, 0:3] = quat_apply(dq, object_vel[:, 0:3])
        object_vel[:, 3:6] = quat_apply(dq, object_vel[:, 3:6])

    if joint_noise > 0.0:
        joint_pos += torch.randn_like(joint_pos) * joint_noise

    if object_linvel_noise > 0.0:
        object_vel[:, 0:3] += torch.randn_like(object_vel[:, 0:3]) * object_linvel_noise
    if object_angvel_noise > 0.0:
        object_vel[:, 3:6] += torch.randn_like(object_vel[:, 3:6]) * object_angvel_noise

    root_pose[:, 0:3] += origins
    object_pose[:, 0:3] += origins

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    obj.write_root_pose_to_sim(object_pose, env_ids=env_ids)
    obj.write_root_velocity_to_sim(object_vel, env_ids=env_ids)

    try:
        robot.reset(env_ids)
    except Exception:
        pass
    try:
        obj.reset(env_ids)
    except Exception:
        pass

    env._urop_hold_latched[env_ids] = True
    env._urop_hold_steps[env_ids] = 1
    env._urop_hold_anchor_xy[env_ids] = root_pose[:, 0:2]
    env._urop_spawn_xy[env_ids] = root_pose[:, 0:2]

    _sample_command_by_stage(env, env_ids, stage0_cmd, stage1_cmd, stage2_cmd, stage3_cmd)
