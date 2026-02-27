from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .rewards import _update_hold_latch, _toss_active
from .observations import quat_rotate_inverse

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


def object_dropped(env: "ManagerBasedRLEnv", min_z: float = 0.20, max_dist: float = 3.0) -> torch.Tensor:
    """Terminate if the object is clearly dropped *after* the toss started."""
    active = _toss_active(env) > 0.5
    if not torch.any(active):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    obj = env.scene["object"]
    robot = env.scene["robot"]
    z = obj.data.root_pos_w[:, 2]
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    drop = (z < min_z) | (dist > max_dist)

    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    out[active] = drop[active]
    return out


def post_hold_runaway(env: "ManagerBasedRLEnv", max_anchor_drift: float = 0.40) -> torch.Tensor:
    """Receive-only policy: once the box is caught, walking away is considered failure."""
    _update_hold_latch(env)
    if not hasattr(env, "_urop_hold_latched") or not hasattr(env, "_urop_hold_anchor_xy"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene["robot"]
    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_hold_anchor_xy, dim=-1)
    return env._urop_hold_latched & (drift > max_anchor_drift)