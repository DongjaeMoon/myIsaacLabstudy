from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from .rewards import _get_stage
from .observations import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

def robot_fallen(env: "ManagerBasedRLEnv", min_root_z=0.55, min_upright=0.4) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    upright = (-g_b[:, 2])
    return (z < min_root_z) | (upright < min_upright)

def object_dropped_curriculum(env: "ManagerBasedRLEnv", min_z=0.50, max_dist=3.0) -> torch.Tensor:
    # stage0에서는 drop을 의미있게 볼 필요 없어서 off
    if _get_stage(env) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    z = obj.data.root_pos_w[:, 2]
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return (z < min_z) | (dist > max_dist)
