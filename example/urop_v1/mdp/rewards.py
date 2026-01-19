# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_apply_inverse  # (warn: quat_rotate_inverse deprecated)
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _tip_pos_w(env: "ManagerBasedRLEnv", ee_body_name: str, tip_offset: tuple[float, float, float]) -> torch.Tensor:
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(ee_body_name)
    if len(body_ids) == 0:
        return robot.data.root_pos_w

    body_pos_w = robot.data.body_pos_w[:, body_ids, :].mean(dim=1)
    body_quat_w = robot.data.body_quat_w[:, body_ids, :].mean(dim=1)

    offset = torch.tensor(tip_offset, device=env.device, dtype=body_pos_w.dtype).unsqueeze(0).repeat(env.num_envs, 1)
    return body_pos_w + quat_apply(body_quat_w, offset)


def _contact_touched(env: "ManagerBasedRLEnv", sensor_name: str, min_force: float) -> torch.Tensor:
    """Return (num_envs,) bool: whether contact force magnitude exceeds min_force."""
    sensor = env.scene[sensor_name]
    data = sensor.data

    # robust field access
    if hasattr(data, "net_forces_w"):
        forces = data.net_forces_w
    elif hasattr(data, "forces_w"):
        forces = data.forces_w
    else:
        raise AttributeError(f"Contact sensor data has no forces field. Available: {dir(data)}")

    # shapes: (N, B, 3) or (N, B, H, 3)
    if forces.ndim == 4:
        forces = forces[:, :, -1, :]

    mag = torch.norm(forces, dim=-1)       # (N, B)
    touched = (mag > min_force).any(dim=1) # (N,)
    return touched


# --------------------------------------------------------------------------------------
# Termination / sanity utilities
# --------------------------------------------------------------------------------------
def root_height_below(env: "ManagerBasedRLEnv", minimum_height: float) -> torch.Tensor:
    root_pos = env.scene["robot"].data.root_pos_w
    return root_pos[:, 2] < float(minimum_height)


def ball_out_of_bounds(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    x_bounds: tuple[float, float] = (-2.0, 4.0),
    y_abs_max: float = 3.0,
    z_bounds: tuple[float, float] = (0.0, 3.0),
) -> torch.Tensor:
    p = env.scene[asset_name].data.root_pos_w
    x_ok = (p[:, 0] >= x_bounds[0]) & (p[:, 0] <= x_bounds[1])
    y_ok = (torch.abs(p[:, 1]) <= float(y_abs_max))
    z_ok = (p[:, 2] >= z_bounds[0]) & (p[:, 2] <= z_bounds[1])
    return ~(x_ok & y_ok & z_ok)


def ball_past_robot(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    robot_name: str,
    goal_x_offset: float = -0.2,
) -> torch.Tensor:
    ball_x = env.scene[asset_name].data.root_pos_w[:, 0]
    robot_x = env.scene[robot_name].data.root_pos_w[:, 0]
    return ball_x < (robot_x + float(goal_x_offset))


# --------------------------------------------------------------------------------------
# Reset event: shoot ball toward a target body
# --------------------------------------------------------------------------------------
def shoot_ball_towards_body(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_name: str = "target_ball",
    robot_name: str = "robot",
    target_body_name: str = "shoulder_link",
    x_offset: float = 2.2,
    y_range: tuple[float, float] = (-0.25, 0.25),
    z_range: tuple[float, float] = (0.55, 1.05),
    speed_range: tuple[float, float] = (2.0, 4.0),
    aim_noise_y: float = 0.15,
    aim_noise_z: float = 0.15,
) -> None:
    device = env.device
    env_ids = env_ids.to(device=device)
    n = env_ids.numel()

    robot = env.scene[robot_name]
    ball = env.scene[asset_name]

    # target point: target body pos (fallback root)
    body_ids, _ = robot.find_bodies(target_body_name)
    if len(body_ids) > 0:
        target_pos = robot.data.body_pos_w[env_ids][:, body_ids, :].mean(dim=1)  # (n,3)
    else:
        target_pos = robot.data.root_pos_w[env_ids]  # (n,3)

    # spawn point in world (front of target along +x)
    ball_pos = target_pos.clone()
    ball_pos[:, 0] += float(x_offset)
    ball_pos[:, 1] += torch.empty(n, device=device).uniform_(y_range[0], y_range[1])
    ball_pos[:, 2] = torch.empty(n, device=device).uniform_(z_range[0], z_range[1])

    # aim point = target_pos + noise
    aim = target_pos.clone()
    aim[:, 1] += torch.empty(n, device=device).uniform_(-aim_noise_y, aim_noise_y)
    aim[:, 2] += torch.empty(n, device=device).uniform_(-aim_noise_z, aim_noise_z)

    direction = aim - ball_pos
    direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

    speed = torch.empty(n, device=device).uniform_(speed_range[0], speed_range[1])
    lin_vel = direction * speed.unsqueeze(-1)

    # pose: (x,y,z,w,x,y,z) with identity quat
    pose = torch.zeros(n, 7, device=device)
    pose[:, :3] = ball_pos
    pose[:, 3] = 1.0  # w

    vel = torch.zeros(n, 6, device=device)
    vel[:, :3] = lin_vel

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


# --------------------------------------------------------------------------------------
# Rewards (kernels)
# --------------------------------------------------------------------------------------
def track_ball_tip_kernel(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    ee_body_name: str,
    tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sigma: float = 0.20,
) -> torch.Tensor:
    tip = _tip_pos_w(env, ee_body_name, tip_offset)
    ball_pos = env.scene[asset_name].data.root_pos_w
    dist_sq = torch.sum((ball_pos - tip) ** 2, dim=-1)
    return torch.exp(-dist_sq / float(sigma))


def track_future_ball_tip_kernel(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    ee_body_name: str,
    tip_offset: tuple[float, float, float],
    time_horizon_s: float = 0.25,
    sigma: float = 0.25,
) -> torch.Tensor:
    tip = _tip_pos_w(env, ee_body_name, tip_offset)
    ball_pos = env.scene[asset_name].data.root_pos_w
    ball_vel = env.scene[asset_name].data.root_lin_vel_w
    future = ball_pos + ball_vel * float(time_horizon_s)
    dist_sq = torch.sum((future - tip) ** 2, dim=-1)
    return torch.exp(-dist_sq / float(sigma))


# --------------------------------------------------------------------------------------
# Success definition: "tip save"
# --------------------------------------------------------------------------------------
def ball_saved_tip(
    env: "ManagerBasedRLEnv",
    sensor_name: str,
    asset_name: str,
    robot_name: str,
    ee_body_name: str,
    tip_offset: tuple[float, float, float],
    tip_radius: float = 0.08,
    min_force: float = 0.1,
    goal_x_offset: float = -0.2,
) -> torch.Tensor:
    # contact
    touched = _contact_touched(env, sensor_name, min_force=min_force)

    # geometric gate: ball near tip (prevents accidental/other-link touches)
    tip = _tip_pos_w(env, ee_body_name, tip_offset)
    ball_pos = env.scene[asset_name].data.root_pos_w
    near_tip = torch.norm(ball_pos - tip, dim=-1) < float(tip_radius)

    # don't count after it's already conceded
    not_conceded = ~ball_past_robot(env, asset_name=asset_name, robot_name=robot_name, goal_x_offset=goal_x_offset)

    return touched & near_tip & not_conceded


def save_ball_reward(
    env: "ManagerBasedRLEnv",
    sensor_name: str,
    asset_name: str,
    robot_name: str,
    ee_body_name: str,
    tip_offset: tuple[float, float, float],
    tip_radius: float = 0.08,
    min_force: float = 0.1,
    goal_x_offset: float = -0.2,
) -> torch.Tensor:
    saved = ball_saved_tip(
        env,
        sensor_name=sensor_name,
        asset_name=asset_name,
        robot_name=robot_name,
        ee_body_name=ee_body_name,
        tip_offset=tip_offset,
        tip_radius=tip_radius,
        min_force=min_force,
        goal_x_offset=goal_x_offset,
    )
    return saved.to(dtype=torch.float32)
