# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformerData
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --------------------------------------------------------------------------------------
# 기존에 쓰던 FrameTransformer helper (남겨둬도 무방)
# --------------------------------------------------------------------------------------
def rel_pos(env: "ManagerBasedRLEnv", ft: str) -> torch.Tensor:
    target_tf_data: FrameTransformerData = env.scene[ft].data
    return target_tf_data.target_pos_source[..., 0, :]


def rel_quat(env: "ManagerBasedRLEnv", ft: str) -> torch.Tensor:
    target_tf_data: FrameTransformerData = env.scene[ft].data
    return target_tf_data.target_quat_source[..., 0, :]


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _yaw_only_quat_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
    """(w,x,y,z) quat에서 yaw만 남긴 quaternion을 반환."""
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
    q = env.scene["robot"].data.root_quat_w  # (N,4) wxyz
    return _yaw_only_quat_wxyz(q) if yaw_only else q


def _tip_pos_w(
    env: "ManagerBasedRLEnv",
    ee_body_name: str,
    tip_offset: tuple[float, float, float],
) -> torch.Tensor:
    """arm_link2 body pose + rotated offset -> tip position in world."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(ee_body_name)
    if len(body_ids) == 0:
        # fallback: root pos (won't be good but avoids crash)
        return robot.data.root_pos_w

    body_pos_w = robot.data.body_pos_w[:, body_ids, :].mean(dim=1)    # (N,3)
    body_quat_w = robot.data.body_quat_w[:, body_ids, :].mean(dim=1)  # (N,4)

    offset = torch.tensor(tip_offset, device=env.device, dtype=body_pos_w.dtype).unsqueeze(0).repeat(env.num_envs, 1)
    tip_pos_w = body_pos_w + quat_apply(body_quat_w, offset)
    return tip_pos_w


# --------------------------------------------------------------------------------------
# Observation terms used in env_cfg.py
# --------------------------------------------------------------------------------------
def object_pos_rel(env: "ManagerBasedRLEnv", asset_name: str, yaw_only: bool = False) -> torch.Tensor:
    """Robot base-frame relative position of an object."""
    robot_pos = env.scene["robot"].data.root_pos_w
    obj_pos = env.scene[asset_name].data.root_pos_w
    diff_w = obj_pos - robot_pos

    q = _robot_frame_quat(env, yaw_only=yaw_only)
    diff_b = quat_apply_inverse(q, diff_w)
    return diff_b


def object_lin_vel_rel(env: "ManagerBasedRLEnv", asset_name: str, yaw_only: bool = False) -> torch.Tensor:
    """Robot base-frame relative linear velocity of an object."""
    robot_vel = env.scene["robot"].data.root_lin_vel_w
    obj_vel = env.scene[asset_name].data.root_lin_vel_w
    diff_w = obj_vel - robot_vel

    q = _robot_frame_quat(env, yaw_only=yaw_only)
    diff_b = quat_apply_inverse(q, diff_w)
    return diff_b


def ee_tip_pos_rel(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    ee_body_name: str,
    tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_only: bool = False,
) -> torch.Tensor:
    """Robot base-frame relative vector (ball - tip)."""
    tip_pos_w = _tip_pos_w(env, ee_body_name, tip_offset)
    ball_pos_w = env.scene[asset_name].data.root_pos_w
    rel_w = ball_pos_w - tip_pos_w

    q = _robot_frame_quat(env, yaw_only=yaw_only)
    rel_b = quat_apply_inverse(q, rel_w)
    return rel_b


def ball_future_pos_rel_to_tip(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    ee_body_name: str,
    tip_offset: tuple[float, float, float],
    time_horizon_s: float = 0.25,
    yaw_only: bool = False,
) -> torch.Tensor:
    """Robot base-frame relative vector (predicted_future_ball_pos - current_tip_pos)."""
    tip_pos_w = _tip_pos_w(env, ee_body_name, tip_offset)

    ball_pos_w = env.scene[asset_name].data.root_pos_w
    ball_vel_w = env.scene[asset_name].data.root_lin_vel_w
    future_ball_pos_w = ball_pos_w + ball_vel_w * float(time_horizon_s)

    rel_w = future_ball_pos_w - tip_pos_w
    q = _robot_frame_quat(env, yaw_only=yaw_only)
    rel_b = quat_apply_inverse(q, rel_w)
    return rel_b
