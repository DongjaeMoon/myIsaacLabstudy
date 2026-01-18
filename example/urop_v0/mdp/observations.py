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

# [observations.py 수정]
from isaaclab.utils.math import quat_rotate_inverse, subtract_frame_transforms

def object_pos_rel(env: ManagerBasedRLEnv, asset_name: str) -> torch.Tensor:
    """로봇 Base Frame 기준 타겟 물체의 상대 위치 (Rotation 고려)"""
    # 1. 로봇과 타겟의 월드 좌표
    robot_pos = env.scene["robot"].data.root_pos_w
    robot_quat = env.scene["robot"].data.root_quat_w
    target_pos = env.scene[asset_name].data.root_pos_w
    
    # 2. 월드 좌표 차이 계산
    diff_w = target_pos - robot_pos
    
    # 3. 로봇의 회전(Orientation)을 고려하여 Body Frame으로 변환
    # (로봇이 고개를 돌려도 공의 위치를 '내 기준'으로 알 수 있음)
    diff_b = quat_rotate_inverse(robot_quat, diff_w)
    
    return diff_b

# [observations.py 에 추가]

def ee_pos_rel(env: ManagerBasedRLEnv, asset_name: str, ee_body_name: str) -> torch.Tensor:
    """로봇의 손 끝(End-effector) 기준 타겟 물체의 상대 위치를 반환"""
    # 1. End-effector의 월드 위치 찾기
    body_ids, _ = env.scene["robot"].find_bodies(ee_body_name)
    # (num_envs, 3) - 여러 개 찾아도 평균값 사용
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, body_ids, :].mean(dim=1)
    
    # 2. 타겟(공)의 월드 위치
    target_pos_w = env.scene[asset_name].data.root_pos_w

    # 3. 상대 위치 (공 위치 - 손 위치)
    # 이 벡터가 (0,0,0)이 되는 것이 목표!
    return target_pos_w - ee_pos_w

from isaaclab.utils.math import quat_apply

def ee_tip_pos_rel(
    env: ManagerBasedRLEnv, 
    asset_name: str, 
    ee_body_name: str, 
    tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.3) 
) -> torch.Tensor:
    """
    로봇 팔의 '끝점(Tip)' 기준 타겟(공)의 상대 위치 벡터를 반환합니다.
    계산식: Tip_Pos = Body_Pos + Rotation * Offset
    """
    # 1. Body(arm_link2)의 위치와 회전(Quaternion) 가져오기
    body_ids, _ = env.scene["robot"].find_bodies(ee_body_name)
    
    # (num_envs, 3) & (num_envs, 4)
    body_pos_w = env.scene["robot"].data.body_pos_w[:, body_ids, :].mean(dim=1)
    body_quat_w = env.scene["robot"].data.body_quat_w[:, body_ids, :].mean(dim=1)
    
    # 2. 오프셋 적용 (로컬 좌표 -> 월드 좌표 변환)
    # tip_offset만큼 떨어진 곳을 실제 월드 회전에 맞춰 돌립니다.
    offset_tensor = torch.tensor(tip_offset, device=env.device).repeat(env.num_envs, 1)
    offset_w = quat_apply(body_quat_w, offset_tensor)
    
    # 최종 팁 위치 = 몸통 위치 + 회전된 오프셋
    tip_pos_w = body_pos_w + offset_w
    
    # 3. 타겟(공) 위치
    target_pos_w = env.scene[asset_name].data.root_pos_w

    # 4. 상대 위치 (공 - 팁)
    # 목표: 이 벡터가 (0,0,0)이 되도록 학습
    return target_pos_w - tip_pos_w