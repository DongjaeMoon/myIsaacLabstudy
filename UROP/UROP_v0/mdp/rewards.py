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


def impact_peak_penalty(env: "ManagerBasedRLEnv", force_thr: float = 250.0) -> torch.Tensor:
    """Penalize peak contact force above threshold."""
    sensor = env.scene["contact_sensor"]
    f = sensor.data.net_forces_w  # (N, B, 3)
    mag = torch.linalg.norm(f, dim=-1)      # (N, B)
    peak = mag.max(dim=-1).values           # (N,)
    over = torch.relu(peak - force_thr)
    return over * over


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """L2 of action rate (needs env.action_manager)."""
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)
