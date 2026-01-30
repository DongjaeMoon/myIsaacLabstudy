# UROP/UROP_v0/mdp/events.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_robot_base_velocity(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    lin_x=(-0.6, 0.6),
    lin_y=(-0.4, 0.4),
    yaw_rate=(-1.5, 1.5),
):
    robot = env.scene["robot"]
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)
    n = env_ids.shape[0]

    lin = torch.zeros((n, 3), device=device)
    ang = torch.zeros((n, 3), device=device)

    lin[:, 0] = torch.empty(n, device=device).uniform_(*lin_x)
    lin[:, 1] = torch.empty(n, device=device).uniform_(*lin_y)
    ang[:, 2] = torch.empty(n, device=device).uniform_(*yaw_rate)

    robot.write_root_velocity_to_sim(torch.cat([lin, ang], dim=-1), env_ids=env_ids)



def reset_and_toss_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,          
    asset_name: str,                
    pos_x=(0.3, 0.5),
    pos_y=(-0.15, 0.15),
    pos_z=(0.9, 1.2),
    vel_x=(-2.0, -0.8),
    vel_y=(-0.3, 0.3),
    vel_z=(-0.2, 0.2),
):
    obj = env.scene[asset_name]
    device = env.device

    # env_ids가 None이면 전체 환경 리셋
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)

    n = env_ids.shape[0]

    # 1. 로봇 기준의 랜덤 위치 생성 (Local)
    local_pos = torch.stack(
        [
            torch.empty(n, device=device).uniform_(*pos_x),
            torch.empty(n, device=device).uniform_(*pos_y),
            torch.empty(n, device=device).uniform_(*pos_z),
        ],
        dim=-1,
    )
    
    # ★ 2. [핵심 수정] 각 환경(방)의 원점 좌표를 더해줌 (Global로 변환)
    global_pos = local_pos + env.scene.env_origins[env_ids]

    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)

    # 3. 속도는 로봇 쪽으로 던지는 거니까 방향 그대로 유지
    lin_vel = torch.stack(
        [
            torch.empty(n, device=device).uniform_(*vel_x),
            torch.empty(n, device=device).uniform_(*vel_y),
            torch.empty(n, device=device).uniform_(*vel_z),
        ],
        dim=-1,
    )
    ang_vel = torch.zeros((n, 3), device=device)

    # 4. 적용
    obj.write_root_pose_to_sim(torch.cat([global_pos, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)