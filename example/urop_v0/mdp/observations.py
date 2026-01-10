# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def rel_pos(env: ManagerBasedRLEnv, ft: str) -> torch.Tensor:
    target_tf_data:FrameTransformerData = env.scene[ft].data
    rel_pos = target_tf_data.target_pos_source[..., 0, :]

    return rel_pos

def rel_quat(env: ManagerBasedRLEnv, ft: str) -> torch.Tensor:
    target_tf_data:FrameTransformerData = env.scene[ft].data
    rel_rot = target_tf_data.target_quat_source[..., 0, :]

    return rel_rot