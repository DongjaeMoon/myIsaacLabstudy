from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .observations import get_lower_body_joint_indices, quat_rotate_inverse
from .rewards import _toss_active, _update_hold_latch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def robot_fallen_degree(
    env: "ManagerBasedRLEnv",
    min_root_z: float = 0.50,
    max_tilt_deg: float = 45.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(robot.data.root_quat_w, g_world)
    upright = -g_b[:, 2]
    upright_min = math.cos(math.radians(max_tilt_deg))
    return (z < min_root_z) | (upright < upright_min)


def object_dropped(env: "ManagerBasedRLEnv", min_z: float = 0.30, max_dist: float = 2.0) -> torch.Tensor:
    active = _toss_active(env) > 0.5
    if not torch.any(active):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj = env.scene["object"]
    robot = env.scene["robot"]
    z = obj.data.root_pos_w[:, 2]
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    dropped = (z < min_z) | (dist > max_dist)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    out[active] = dropped[active]
    return out


def successful_hold_complete(env: "ManagerBasedRLEnv", min_steps: int = 40) -> torch.Tensor:
    _update_hold_latch(env)
    if not hasattr(env, "_urop_hold_latched"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._urop_hold_latched & (env._urop_hold_steps >= int(min_steps))


def post_hold_runaway(env: "ManagerBasedRLEnv", max_anchor_drift: float = 0.28) -> torch.Tensor:
    _update_hold_latch(env)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene["robot"]
    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_hold_anchor_xy, dim=-1)
    return env._urop_hold_latched & (drift > max_anchor_drift)


def unsafe_lower_body_deviation(env: "ManagerBasedRLEnv", max_abs_dev: float = 0.85) -> torch.Tensor:
    idx = get_lower_body_joint_indices(env)
    robot = env.scene["robot"]
    if hasattr(env, "_urop_ready_joint_pos"):
        target = env._urop_ready_joint_pos[:, idx]
    else:
        target = robot.data.default_joint_pos[:, idx]
    diff = torch.abs(robot.data.joint_pos[:, idx] - target)
    return torch.any(diff > max_abs_dev, dim=-1)
