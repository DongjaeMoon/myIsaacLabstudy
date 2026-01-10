# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
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



#### env.scene["tool_frame"].data의 index: [env num, target_frames, data(xyz pos or quat rot)]
#### target_frames index: 0: finger1, 1: finger2, 2: tool
#### FrameTransformerData.target_pos_source: source 좌표(relative)에서의 target의 pos
#### FrameTransformerData.target_quat_source: source 좌표(relative)에서의 target의 rot (quat)
#### FrameTransformerData.source_pos_w: world 좌표(absolute)에서의 target의 pos


def reward_example(env: ManagerBasedRLEnv, ft_name: str) -> torch.Tensor:
    """calculate the relative position error from the right gripper to the tool"""

    z = env.scene[ft_name].data.target_pos_w[:,0,2]

    return z
