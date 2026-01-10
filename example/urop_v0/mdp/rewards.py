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


# 파일 맨 아래에 추가

# 1. 공에게 다가가기 (거리 보상)
def distance_to_target(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    robot_root_pos = env.scene["robot"].data.root_pos_w
    target_pos = env.scene[asset_name].data.root_pos_w
    
    # 거리 계산
    dist = torch.norm(target_pos - robot_root_pos, dim=-1)
    return dist # 이 값을 최소화하도록 음수 가중치를 주거나, exp(-dist)로 변환해서 사용

# 2. 로봇 팔 끝(End-effector)이 공에 닿기 (안전한 버전)
def ee_distance_to_target(env: ManagerBasedRLEnv, asset_name: str, ee_body_name: str) -> torch.Tensor:
    # 1. 로봇의 팔 끝(End-effector) 링크 찾기
    ee_indices = env.scene["robot"].find_bodies(ee_body_name)[0]
    
    # 2. 위치 가져오기: 결과 모양은 (num_envs, 찾은_개수, 3) 임. 예: (64, 1, 3)
    ee_pos_all = env.scene["robot"].data.body_pos_w[:, ee_indices, :]
    
    # 3. [핵심 수정] 차원 축소
    # 찾은 개수 차원(dim=1)을 평균내서 없앰. 
    # 하나만 찾았으면 그대로 그 위치고, 실수로 여러 개 찾았어도 중심점으로 계산되므로 에러 안 남.
    ee_pos = ee_pos_all.mean(dim=1)  # 결과 모양: (64, 3)

    # 4. 타겟 위치 (64, 3)
    target_pos = env.scene[asset_name].data.root_pos_w
    
    # 5. 거리 계산 (64,) -> 이제 Shape Mismatch가 절대 발생하지 않음
    dist = torch.norm(target_pos - ee_pos, dim=-1)
    
    return dist