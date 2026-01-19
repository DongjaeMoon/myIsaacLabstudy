# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _contact_touched(env: "ManagerBasedRLEnv", sensor_name: str, min_force: float) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    data = sensor.data

    if hasattr(data, "net_forces_w"):
        forces = data.net_forces_w
    elif hasattr(data, "forces_w"):
        forces = data.forces_w
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if forces.ndim == 4:
        forces = forces[:, :, -1, :]

    mag = torch.norm(forces, dim=-1)       
    touched = (mag > min_force).any(dim=1) 
    return touched

# --------------------------------------------------------------------------------------
# Termination / sanity utilities
# --------------------------------------------------------------------------------------
def ball_past_robot(env: "ManagerBasedRLEnv", asset_name: str, robot_name: str, goal_x_offset: float = -0.2) -> torch.Tensor:
    ball_x = env.scene[asset_name].data.root_pos_w[:, 0]
    robot_x = env.scene[robot_name].data.root_pos_w[:, 0]
    return ball_x < (robot_x + float(goal_x_offset))

def ball_out_of_bounds(env: "ManagerBasedRLEnv", asset_name: str, x_bounds=(-2.0, 4.0), y_abs_max=3.0, z_bounds=(0.0, 3.0)) -> torch.Tensor:
    p = env.scene[asset_name].data.root_pos_w
    x_ok = (p[:, 0] >= x_bounds[0]) & (p[:, 0] <= x_bounds[1])
    y_ok = (torch.abs(p[:, 1]) <= float(y_abs_max))
    z_ok = (p[:, 2] >= z_bounds[0]) & (p[:, 2] <= z_bounds[1])
    return ~(x_ok & y_ok & z_ok)

# --------------------------------------------------------------------------------------
# Reset event: shoot ball
# --------------------------------------------------------------------------------------
def shoot_ball_towards_body(
    env: "ManagerBasedRLEnv", env_ids: torch.Tensor, asset_name: str, robot_name: str,
    target_body_name: str, x_offset: float, y_range: tuple, z_range: tuple,
    speed_range: tuple, aim_noise_y: float, aim_noise_z: float,
) -> None:
    device = env.device
    env_ids = env_ids.to(device=device)
    n = env_ids.numel()

    robot = env.scene[robot_name]
    ball = env.scene[asset_name]

    body_ids, _ = robot.find_bodies(target_body_name)
    if len(body_ids) > 0:
        target_pos = robot.data.body_pos_w[env_ids][:, body_ids, :].mean(dim=1) 
    else:
        target_pos = robot.data.root_pos_w[env_ids]

    ball_pos = target_pos.clone()
    ball_pos[:, 0] += float(x_offset)
    ball_pos[:, 1] += torch.empty(n, device=device).uniform_(y_range[0], y_range[1])
    ball_pos[:, 2] = torch.empty(n, device=device).uniform_(z_range[0], z_range[1])

    aim = target_pos.clone()
    aim[:, 1] += torch.empty(n, device=device).uniform_(-aim_noise_y, aim_noise_y)
    aim[:, 2] += torch.empty(n, device=device).uniform_(-aim_noise_z, aim_noise_z)

    direction = aim - ball_pos
    direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
    speed = torch.empty(n, device=device).uniform_(speed_range[0], speed_range[1])
    
    pose = torch.zeros(n, 7, device=device)
    pose[:, :3] = ball_pos
    pose[:, 3] = 1.0 
    vel = torch.zeros(n, 6, device=device)
    vel[:, :3] = direction * speed.unsqueeze(-1)

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)

# --------------------------------------------------------------------------------------
# Rewards
# --------------------------------------------------------------------------------------
def track_ball_tip_kernel(env: "ManagerBasedRLEnv", asset_name: str, ee_body_name: str, tip_offset: tuple, sigma: float) -> torch.Tensor:
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(ee_body_name)
    body_pos = robot.data.body_pos_w[:, body_ids, :].mean(dim=1)
    body_quat = robot.data.body_quat_w[:, body_ids, :].mean(dim=1)
    
    offset = torch.tensor(tip_offset, device=env.device).repeat(env.num_envs, 1)
    tip_pos = body_pos + quat_apply(body_quat, offset)
    
    ball_pos = env.scene[asset_name].data.root_pos_w
    dist_sq = torch.sum((ball_pos - tip_pos) ** 2, dim=-1)
    return torch.exp(-dist_sq / float(sigma))

def save_ball_reward(env: "ManagerBasedRLEnv", sensor_name: str, min_force: float = 0.1, **kwargs) -> torch.Tensor:
    return _contact_touched(env, sensor_name, min_force).to(dtype=torch.float32)

def ball_saved_simple(env: "ManagerBasedRLEnv", sensor_name: str, min_force: float = 0.1, **kwargs) -> torch.Tensor:
    return _contact_touched(env, sensor_name, min_force)