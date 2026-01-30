# UROP_v0/mdp/observations.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_proprio(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel
    base_lin = robot.data.root_lin_vel_w
    base_ang = robot.data.root_ang_vel_w
    base_quat = robot.data.root_quat_w
    return torch.cat([jp, jv, base_lin, base_ang, base_quat], dim=-1)


def object_rel_state(
    env: "ManagerBasedRLEnv",
    drop_prob: float = 0.0,      # ✅ student에서는 1.0으로 두면 항상 가림
    noise_std: float = 0.0,      # (옵션) teacher에서도 약간 노이즈 줄 때
    pos_scale: float = 1.0,
    vel_scale: float = 1.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rel_pos = (obj.data.root_pos_w - robot.data.root_pos_w) * pos_scale
    rel_vel = (obj.data.root_lin_vel_w - robot.data.root_lin_vel_w) * vel_scale
    x = torch.cat([rel_pos, rel_vel], dim=-1)  # (N,6)

    if noise_std > 0.0:
        x = x + noise_std * torch.randn_like(x)

    if drop_prob > 0.0:
        # drop_prob 확률로 0으로 가림 (teacher: 0.0 / student: 1.0 권장)
        keep = (torch.rand(x.shape[0], device=x.device) > drop_prob).float().unsqueeze(-1)
        x = x * keep

    return x


def contact_forces(
    env: "ManagerBasedRLEnv",
    sensor_names: list[str],
    scale: float = 1.0 / 300.0,   # ✅ force scaling (학습 안정화)
) -> torch.Tensor:
    outs = []
    for name in sensor_names:
        sensor = env.scene[name]
        f = sensor.data.net_forces_w  # (N, 1, 3)
        outs.append(f.reshape(f.shape[0], -1))
    return torch.cat(outs, dim=-1) * scale
