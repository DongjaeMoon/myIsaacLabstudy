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
    #ee_indices = env.scene["robot"].find_bodies(ee_body_name)[0]
    body_ids, _ = env.scene["robot"].find_bodies(ee_body_name)
    if len(body_ids) == 0:
    # 못 찾으면 큰 거리로 페널티 주고 조용히 넘어가게
        return torch.ones(env.num_envs, device=env.device) * 1e3
    ee_indices = body_ids

    
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


# --- [urop_v0/mdp/rewards.py] 맨 아래에 추가 ---
'''
def root_height_below(env: ManagerBasedRLEnv, minimum_height: float) -> torch.Tensor:
    """로봇의 높이가 기준치보다 낮아지면 True(종료)를 반환"""
    # 1. 로봇의 Root 위치 가져오기 (num_envs, 3)
    root_pos = env.scene["robot"].data.root_pos_w
    
    # 2. Z축(높이)이 minimum_height보다 작은지 검사
    return root_pos[:, 2] < minimum_height


def ball_below_height(env: ManagerBasedRLEnv, asset_name: str, min_height: float) -> torch.Tensor:
    z = env.scene[asset_name].data.root_pos_w[:, 2]
    return z < min_height
'''
def ball_touched(env: ManagerBasedRLEnv, sensor_name: str, min_force: float = 0.1) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    data = sensor.data

    # IsaacLab 버전/센서 설정에 따라 텐서 shape가 다를 수 있어서 최대한 방어적으로 처리
    if hasattr(data, "net_forces_w"):
        forces = data.net_forces_w
    elif hasattr(data, "forces_w"):
        forces = data.forces_w
    else:
        raise AttributeError(f"Contact sensor data has no forces field. Available: {dir(data)}")

    # 가능한 shape:
    # (num_envs, num_bodies, 3) 또는 (num_envs, num_bodies, history, 3)
    if forces.ndim == 4:
        forces = forces[:, :, -1, :]  # 마지막 스텝만 사용

    mag = torch.norm(forces, dim=-1)          # (num_envs, num_bodies)
    touched = (mag > min_force).any(dim=1)    # (num_envs,)
    return touched

'''
def reset_ball_random_drop(
    env,
    env_ids,
    x_range=(0.4, 0.8),        # 로봇 기준 "앞"으로 떨어지는 거리 (m)
    y_abs_range=(0.0, 0.25),   # 로봇 기준 "좌/우" 거리의 절댓값 (m)
    z_range=(0.7, 1.1),        # 로봇 기준 "위" 높이 오프셋 (m)
    asset_name="target_ball",
) -> None:
    ball = env.scene[asset_name]
    device = env.device

    # env_ids 안전 처리
    env_ids = env_ids.to(device=device)
    n = env_ids.numel()

    # --- 1) 로봇의 월드 위치/자세 가져오기 (env_ids에 해당하는 것만) ---
    robot_pos_w  = env.scene["robot"].data.root_pos_w[env_ids]   # (n,3)
    robot_quat_w = env.scene["robot"].data.root_quat_w[env_ids]  # (n,4) (w,x,y,z)

    # --- 2) 로봇 "로컬 프레임" 기준 오프셋 샘플링 ---
    x = torch.empty(n, device=device).uniform_(x_range[0], x_range[1])
    z = torch.empty(n, device=device).uniform_(z_range[0], z_range[1])

    y_abs = torch.empty(n, device=device).uniform_(y_abs_range[0], y_abs_range[1])
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    y = sign * y_abs

    offset_body = torch.stack([x, y, z], dim=-1)  # (n,3)  로봇 기준 [앞,좌,위]

    # --- 3) 로봇 자세로 오프셋을 월드로 회전시켜서 더하기 ---
    R_wb = matrix_from_quat(robot_quat_w)                 # (n,3,3)
    offset_w = torch.bmm(R_wb, offset_body.unsqueeze(-1)).squeeze(-1)  # (n,3)

    pos_w = robot_pos_w + offset_w                        # (n,3)

    # --- 4) 공 자세/속도 설정 ---
    quat_w = torch.zeros(n, 4, device=device)
    quat_w[:, 0] = 1.0  # (w,x,y,z) identity

    lin_vel = torch.zeros(n, 3, device=device)
    ang_vel = torch.zeros(n, 3, device=device)

    pose = torch.cat([pos_w, quat_w], dim=-1)  # (n,7)
    vel  = torch.cat([lin_vel, ang_vel], dim=-1)  # (n,6)

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)

'''
import torch
from isaaclab.utils.math import quat_apply

def _yaw_only_quat(q_wxyz: torch.Tensor) -> torch.Tensor:
    """(w,x,y,z) quat에서 yaw만 남긴 quat 반환"""
    w, x, y, z = q_wxyz.unbind(-1)
    yaw = torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    half = 0.5 * yaw
    qw = torch.cos(half)
    qz = torch.sin(half)
    out = torch.zeros_like(q_wxyz)
    out[:, 0] = qw
    out[:, 3] = qz
    return out
'''
def spawn_ball_near_arm_relative(
    env,
    env_ids,
    asset_name: str = "target_ball",
    center_body_name: str = "shoulder_link",   # <- 팔 베이스 링크명(USD에서 확인)
    x_range = (-0.35, -0.35),                  # <- "뒤쪽"으로 떨어뜨리려면 음수
    y_abs_range = (0.15, 0.40),                # <- 좌/우
    z_range = (1.0, 1.6),                      # <- center 기준 위에서 떨어짐
    yaw_only: bool = True,
) -> None:
    """팔(또는 로봇) 기준으로 뒤/좌우 랜덤 위치에서 공을 스폰하고 속도를 0으로 초기화"""
    device = env.device
    env_ids = env_ids.to(device=device)

    ball = env.scene[asset_name]
    robot = env.scene["robot"]

    # center: 팔 베이스 링크(추천). 못 찾으면 root로 fallback.
    body_ids, _ = robot.find_bodies(center_body_name)
    if len(body_ids) > 0:
        center_pos = robot.data.body_pos_w[env_ids][:, body_ids, :].mean(dim=1)  # (n,3)
    else:
        center_pos = robot.data.root_pos_w[env_ids]  # (n,3)

    # 로봇 yaw (롤/피치 무시)
    root_quat = robot.data.root_quat_w[env_ids]  # (n,4) wxyz
    q = _yaw_only_quat(root_quat) if yaw_only else root_quat

    n = env_ids.numel()

    # 로봇 기준 오프셋 샘플링 (body frame)
    x = torch.empty(n, device=device).uniform_(x_range[0], x_range[1])
    y_abs = torch.empty(n, device=device).uniform_(y_abs_range[0], y_abs_range[1])
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    y = sign * y_abs
    z = torch.empty(n, device=device).uniform_(z_range[0], z_range[1])

    offset_body = torch.stack([x, y, z], dim=-1)          # (n,3)
    offset_w = quat_apply(q, offset_body)                 # (n,3) 로봇 yaw로 회전

    pos_w = center_pos + offset_w                          # (n,3)

    quat_w = torch.zeros(n, 4, device=device)
    quat_w[:, 0] = 1.0                                     # (w,x,y,z)

    lin_vel = torch.zeros(n, 3, device=device)
    ang_vel = torch.zeros(n, 3, device=device)

    pose = torch.cat([pos_w, quat_w], dim=-1)              # (n,7)
    vel  = torch.cat([lin_vel, ang_vel], dim=-1)           # (n,6)

    ball.write_root_pose_to_sim(pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(vel, env_ids=env_ids)


def teleport_ball_if_touched(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    sensor_name: str,
    asset_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    min_force: float = 0.1
):
    """
    센서에 충돌이 감지되면(터치 성공), 해당 환경의 공(asset)을 랜덤 위치로 순간이동시킵니다.
    """
    # 1. 터치 여부 확인
    is_touched = ball_touched(env, sensor_name, min_force) # (num_envs,) bool tensor
    
    # 2. 터치된 환경(env)의 인덱스만 골라냄
    # env_ids는 현재 step을 수행하는 환경들, is_touched는 전체 환경에 대한 상태일 수 있으므로 인덱싱 주의
    # 보통 pre_step에서는 전체 env_ids가 들어옵니다.
    touched_env_ids = env_ids[is_touched[env_ids]]
    
    # 3. 터치된 환경이 하나라도 있으면 공 위치 재설정
    if len(touched_env_ids) > 0:
        n = len(touched_env_ids)
        device = env.device
        
        # 로봇(Base) 위치 가져오기 (상대 좌표로 생성하기 위해)
        robot_pos = env.scene["robot"].data.root_pos_w[touched_env_ids]
        
        # 랜덤 오프셋 생성
        offset_x = torch.rand(n, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        offset_y = torch.rand(n, device=device) * (y_range[1] - y_range[0]) + y_range[0]
        offset_z = torch.rand(n, device=device) * (z_range[1] - z_range[0]) + z_range[0]
        
        # 새 위치 = 로봇 위치 + 오프셋
        new_pos = robot_pos.clone()
        new_pos[:, 0] += offset_x
        new_pos[:, 1] += offset_y
        new_pos[:, 2] += offset_z # 높이
        
        # 공의 자세(Rotation)는 그대로 (Identity)
        new_quat = torch.zeros(n, 4, device=device)
        new_quat[:, 0] = 1.0 
        
        # 속도 0으로 리셋
        zeros_vel = torch.zeros(n, 6, device=device)
        
        # 물리 상태 적용 (Teleport)
        ball = env.scene[asset_name]
        ball.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1), env_ids=touched_env_ids)
        ball.write_root_velocity_to_sim(zeros_vel, env_ids=touched_env_ids)
'''

# 1. 공 발사 함수 (Reset 시 사용)
def shoot_ball_towards_robot(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str = "target_ball",
    robot_name: str = "robot",
    x_offset: float = 2.5,    # 로봇 앞 2.5m 에서 발사
    y_range: tuple = (-0.5, 0.5), # 좌우 랜덤 범위
    z_range: tuple = (0.3, 0.8),  # 높이 랜덤 범위
    speed_range: tuple = (4.0, 6.0), # 공 속도 (m/s)
) -> None:
    device = env.device
    n = len(env_ids)
    
    # 1. 로봇 위치 가져오기
    robot_pos = env.scene[robot_name].data.root_pos_w[env_ids]
    
    # 2. 공 시작 위치 계산 (로봇 앞 x_offset 지점)
    ball_pos = robot_pos.clone()
    ball_pos[:, 0] += x_offset 
    ball_pos[:, 1] += torch.rand(n, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    ball_pos[:, 2] = torch.rand(n, device=device) * (z_range[1] - z_range[0]) + z_range[0]
    
    # 3. 속도 벡터 계산 (목표점: 로봇 몸통 쪽)
    # 목표점 = 로봇 위치 + 약간의 랜덤 노이즈 (골대 구석구석 찌르도록)
    target_pos = robot_pos.clone()
    target_pos[:, 1] += torch.rand(n, device=device) * 0.4 - 0.2 # 좌우 20cm 오차
    target_pos[:, 2] += torch.rand(n, device=device) * 0.4 + 0.2 # 높이 조절
    
    direction = target_pos - ball_pos
    direction = direction / torch.norm(direction, dim=-1, keepdim=True) # 단위 벡터
    
    speed = torch.rand(n, device=device) * (speed_range[1] - speed_range[0]) + speed_range[0]
    lin_vel = direction * speed.unsqueeze(-1)
    
    # 4. 물리 적용
    ball = env.scene[asset_name]
    # 위치 설정
    new_pose = torch.zeros(n, 7, device=device)
    new_pose[:, :3] = ball_pos
    new_pose[:, 3] = 1.0 # quat w
    
    # 속도 설정
    new_vel = torch.zeros(n, 6, device=device)
    new_vel[:, :3] = lin_vel
    
    ball.write_root_pose_to_sim(new_pose, env_ids=env_ids)
    ball.write_root_velocity_to_sim(new_vel, env_ids=env_ids)

# 2. 골 먹힘 판정 (공이 로봇 뒤로 넘어갔는가?)
def ball_past_robot(env: ManagerBasedRLEnv, asset_name: str, robot_name: str) -> torch.Tensor:
    ball_x = env.scene[asset_name].data.root_pos_w[:, 0]
    robot_x = env.scene[robot_name].data.root_pos_w[:, 0]
    
    # 공의 x좌표가 로봇보다 작아지면 (뒤로 넘어가면) 골 먹힘
    # (로봇이 0.0, 공이 2.5에서 시작해서 -방향으로 오므로)
    return ball_x < (robot_x - 0.5)


# [rewards.py 수정]

def track_ball_kernel(env: ManagerBasedRLEnv, asset_name: str, ee_body_name: str, sigma: float = 0.5) -> torch.Tensor:
    """
    로봇 팔 끝(EE)이 공에 가까울수록 1에 가까운 보상을 줍니다. (멀면 0)
    RBF Kernel: exp(-dist^2 / sigma)
    """
    # 1. EE 위치 찾기
    body_ids, _ = env.scene["robot"].find_bodies(ee_body_name)
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # (num_envs, 3)
    ee_pos = env.scene["robot"].data.body_pos_w[:, body_ids, :].mean(dim=1)
    target_pos = env.scene[asset_name].data.root_pos_w

    # 2. 거리 제곱 계산
    dist_sq = torch.sum((target_pos - ee_pos) ** 2, dim=-1)
    
    # 3. 0~1 사이 값으로 변환 (아주 가까우면 1, 멀면 0)
    # sigma 값이 작을수록 '정확히' 맞춰야 점수를 줌. 초기엔 좀 넉넉하게(0.5) 설정
    return torch.exp(-dist_sq / sigma)


def track_ball_tip_kernel(
    env: ManagerBasedRLEnv, 
    asset_name: str, 
    ee_body_name: str, 
    tip_offset: tuple[float, float, float] = (0.0, 0.0, 0.3), 
    sigma: float = 0.2
) -> torch.Tensor:
    """
    로봇 팔 '끝점(Tip)'이 공에 가까울수록 보상 (RBF Kernel)
    """
    # 1. Body 정보 가져오기
    body_ids, _ = env.scene["robot"].find_bodies(ee_body_name)
    body_pos_w = env.scene["robot"].data.body_pos_w[:, body_ids, :].mean(dim=1)
    body_quat_w = env.scene["robot"].data.body_quat_w[:, body_ids, :].mean(dim=1)

    # 2. Tip 위치 계산 (Offset + Rotation)
    offset_tensor = torch.tensor(tip_offset, device=env.device).repeat(env.num_envs, 1)
    tip_pos_w = body_pos_w + quat_apply(body_quat_w, offset_tensor)
    
    # 3. 타겟 위치
    target_pos_w = env.scene[asset_name].data.root_pos_w

    # 4. 거리 제곱 및 보상 계산 (가까우면 1점, 멀면 0점)
    dist_sq = torch.sum((target_pos_w - tip_pos_w) ** 2, dim=-1)
    return torch.exp(-dist_sq / sigma)