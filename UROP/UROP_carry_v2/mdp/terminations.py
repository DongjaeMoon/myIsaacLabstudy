from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def robot_fallen_degree(env: "ManagerBasedRLEnv", min_root_z: float = 0.45, max_tilt_deg: float = 60.0) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    upright = -g_b[:, 2]
    upright_min = math.cos(math.radians(max_tilt_deg))
    return (z < min_root_z) | (upright < upright_min)


def object_dropped(env: "ManagerBasedRLEnv", min_z: float = 0.25, max_dist: float = 1.20) -> torch.Tensor:
    obj = env.scene["object"]
    robot = env.scene["robot"]
    z = obj.data.root_pos_w[:, 2]
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return (z < min_z) | (dist > max_dist)


def object_tilt_exceeded(env: "ManagerBasedRLEnv", max_tilt_deg: float = 60.0) -> torch.Tensor:
    q = env.scene["object"].data.root_quat_w
    z_axis = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1))
    cos_tilt = z_axis[:, 2]
    cos_min = math.cos(math.radians(max_tilt_deg))
    return cos_tilt < cos_min
