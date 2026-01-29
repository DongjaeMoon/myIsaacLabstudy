# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, matrix_from_quat
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hold_object_close(env: "ManagerBasedRLEnv", sigma: float = 0.6) -> torch.Tensor:
    """Object stays near robot root."""
    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = torch.linalg.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    return torch.exp(-(d / sigma) ** 2)


def object_not_dropped_bonus(env: "ManagerBasedRLEnv", min_z: float = 0.25) -> torch.Tensor:
    obj = env.scene["object"]
    return (obj.data.root_pos_w[:, 2] > min_z).float()


def impact_peak_penalty(env: "ManagerBasedRLEnv", sensor_names: list[str], force_thr: float = 250.0) -> torch.Tensor:
    peaks = []
    for name in sensor_names:
        sensor = env.scene[name]
        f = sensor.data.net_forces_w  # (N, 1, 3)
        mag = torch.linalg.norm(f, dim=-1)        # (N, 1)
        peak = mag.max(dim=-1).values             # (N,)
        peaks.append(peak)
    peak_all = torch.stack(peaks, dim=-1).max(dim=-1).values  # 센서들 중 최대
    over = torch.relu(peak_all - force_thr)
    return over * over


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """L2 of action rate (needs env.action_manager)."""
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)
