# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _yaw_only_quat_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_wxyz.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    qw = torch.cos(half)
    qz = torch.sin(half)
    out = torch.zeros_like(q_wxyz)
    out[:, 0] = qw
    out[:, 3] = qz
    return out

def _robot_frame_quat(env: "ManagerBasedRLEnv", yaw_only: bool) -> torch.Tensor:
    q = env.scene["robot"].data.root_quat_w  
    return _yaw_only_quat_wxyz(q) if yaw_only else q

def _tip_pos_w(env: "ManagerBasedRLEnv", ee_body_name: str, tip_offset: tuple) -> torch.Tensor:
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(ee_body_name)
    if len(body_ids) == 0:
        return robot.data.root_pos_w

    body_pos_w = robot.data.body_pos_w[:, body_ids, :].mean(dim=1)    
    body_quat_w = robot.data.body_quat_w[:, body_ids, :].mean(dim=1)  

    offset = torch.tensor(tip_offset, device=env.device, dtype=body_pos_w.dtype).unsqueeze(0).repeat(env.num_envs, 1)
    tip_pos_w = body_pos_w + quat_apply(body_quat_w, offset)
    return tip_pos_w

def object_pos_rel(env: "ManagerBasedRLEnv", asset_name: str, yaw_only: bool = False) -> torch.Tensor:
    robot_pos = env.scene["robot"].data.root_pos_w
    obj_pos = env.scene[asset_name].data.root_pos_w
    diff_w = obj_pos - robot_pos
    q = _robot_frame_quat(env, yaw_only=yaw_only)
    return quat_apply_inverse(q, diff_w)

def object_lin_vel_rel(env: "ManagerBasedRLEnv", asset_name: str, yaw_only: bool = False) -> torch.Tensor:
    robot_vel = env.scene["robot"].data.root_lin_vel_w
    obj_vel = env.scene[asset_name].data.root_lin_vel_w
    diff_w = obj_vel - robot_vel
    q = _robot_frame_quat(env, yaw_only=yaw_only)
    return quat_apply_inverse(q, diff_w)

def ee_tip_pos_rel(env: "ManagerBasedRLEnv", asset_name: str, ee_body_name: str, tip_offset: tuple, yaw_only: bool = False) -> torch.Tensor:
    tip_pos_w = _tip_pos_w(env, ee_body_name, tip_offset)
    ball_pos_w = env.scene[asset_name].data.root_pos_w
    rel_w = ball_pos_w - tip_pos_w
    q = _robot_frame_quat(env, yaw_only=yaw_only)
    return quat_apply_inverse(q, rel_w)