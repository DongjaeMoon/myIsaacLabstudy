# UROP/UROP_v0/mdp/events.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_and_toss_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,          # ✅ 이벤트 매니저가 넘겨줌
    asset_name: str,                # ✅ env_cfg.py의 params에서 받음
    pos_x=(0.7, 0.9),
    pos_y=(-0.15, 0.15),
    pos_z=(0.9, 1.2),
    vel_x=(-2.0, -0.8),
    vel_y=(-0.3, 0.3),
    vel_z=(-0.2, 0.2),
):
    obj = env.scene[asset_name]
    device = env.device

    # ✅ reset 되는 env만 샘플링
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device)

    n = env_ids.shape[0]

    pos = torch.stack(
        [
            torch.empty(n, device=device).uniform_(*pos_x),
            torch.empty(n, device=device).uniform_(*pos_y),
            torch.empty(n, device=device).uniform_(*pos_z),
        ],
        dim=-1,
    )
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)

    lin_vel = torch.stack(
        [
            torch.empty(n, device=device).uniform_(*vel_x),
            torch.empty(n, device=device).uniform_(*vel_y),
            torch.empty(n, device=device).uniform_(*vel_z),
        ],
        dim=-1,
    )
    ang_vel = torch.zeros((n, 3), device=device)

    # ✅ reset env_ids에만 적용
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)
