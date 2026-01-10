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

# 파일 맨 아래에 추가

def object_pos_rel(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """로봇(Base) 기준으로 타겟 물체(예: 공)의 상대 위치를 반환"""
    # 1. 로봇의 Base 위치
    robot_pos = env.scene["robot"].data.root_pos_w  # (num_envs, 3)
    
    # 2. 타겟 물체의 위치
    target_pos = env.scene[asset_name].data.root_pos_w # (num_envs, 3)
    
    # 3. 상대 위치 계산 (단순 차이)
    # 필요하다면 로봇의 현재 회전(Yaw)을 고려해서 로봇 좌표계로 변환해줄 수도 있지만, 
    # 일단 월드 좌표계 차이만 줘도 학습은 됩니다.
    return target_pos - robot_pos