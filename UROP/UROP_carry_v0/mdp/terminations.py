from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def robot_fallen_degree(
    env: "ManagerBasedRLEnv",
    min_root_z: float = 0.45,
    max_tilt_deg: float = 60.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    upright = -g_b[:, 2]
    upright_min = math.cos(math.radians(max_tilt_deg))
    return (z < min_root_z) | (upright < upright_min)


def object_dropped(
    env: "ManagerBasedRLEnv",
    min_z: float = 0.42,
    max_dist: float = 1.20,
    max_rel_x: float = 0.95,
) -> torch.Tensor:
    obj = env.scene["object"]
    robot = env.scene["robot"]

    world_dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    rel_p_b = quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)

    return (obj.data.root_pos_w[:, 2] < min_z) | (world_dist > max_dist) | (torch.abs(rel_p_b[:, 0]) > max_rel_x)


def object_tilted(env: "ManagerBasedRLEnv", min_up_z: float = 0.15) -> torch.Tensor:
    obj = env.scene["object"]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    up_world = quat_apply(obj.data.root_quat_w, up_local)
    return up_world[:, 2] < min_up_z
