# UROP/UROP_v0/mdp/events.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_and_toss_object(
    env: "ManagerBasedRLEnv",
    pos_x=(0.7, 0.9),
    pos_y=(-0.15, 0.15),
    pos_z=(0.9, 1.2),
    vel_x=(-2.0, -0.8),   # 로봇을 향해 날아오게(부호는 네 월드축 기준으로 조정 가능)
    vel_y=(-0.3, 0.3),
    vel_z=(-0.2, 0.2),
):
    obj = env.scene["object"]
    N = env.num_envs
    device = env.device

    pos = torch.stack(
        [
            torch.empty(N, device=device).uniform_(*pos_x),
            torch.empty(N, device=device).uniform_(*pos_y),
            torch.empty(N, device=device).uniform_(*pos_z),
        ],
        dim=-1,
    )
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)

    lin_vel = torch.stack(
        [
            torch.empty(N, device=device).uniform_(*vel_x),
            torch.empty(N, device=device).uniform_(*vel_y),
            torch.empty(N, device=device).uniform_(*vel_z),
        ],
        dim=-1,
    )
    ang_vel = torch.zeros((N, 3), device=device)

    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
    obj.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1))
