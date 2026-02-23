from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING
from .rewards import _get_stage
from .observations import quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

def robot_fallen_degree(
    env: "ManagerBasedRLEnv",
    min_root_z: float = 0.55,
    max_tilt_deg: float = 66.4,
) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w

    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    upright = (-g_b[:, 2])

    upright_min = math.cos(math.radians(max_tilt_deg))
    return (z < min_root_z) | (upright < upright_min)

def object_dropped_curriculum(env: "ManagerBasedRLEnv", min_z=0.50, max_dist=3.0) -> torch.Tensor:
    if _get_stage(env) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # throw 이후에만 drop 판정
    if hasattr(env, "_urop_toss_active"):
        active = env._urop_toss_active
    else:
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    obj = env.scene["object"]
    robot = env.scene["robot"]

    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if torch.any(active):
        z = obj.data.root_pos_w[:, 2]
        dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
        drop = (z < min_z) | (dist > max_dist)
        out[active] = drop[active]
    return out