from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply, quat_rotate_inverse, get_controlled_joint_indices

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    return (-g_b[:, 2]).clamp(0.0, 1.0)


def _body_name_to_idx(env: "ManagerBasedRLEnv") -> dict[str, int]:
    if hasattr(env, "_urop_body_name_to_id"):
        return env._urop_body_name_to_id
    robot = env.scene["robot"]
    env._urop_body_name_to_id = {n: i for i, n in enumerate(getattr(robot.data, "body_names", []))}
    return env._urop_body_name_to_id


def _body_pos(env: "ManagerBasedRLEnv", body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    mp = _body_name_to_idx(env)
    if body_name in mp:
        return robot.data.body_pos_w[:, mp[body_name], :]
    return robot.data.root_pos_w


def _body_vel(env: "ManagerBasedRLEnv", body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    mp = _body_name_to_idx(env)
    if body_name in mp:
        return robot.data.body_lin_vel_w[:, mp[body_name], :]
    return robot.data.root_lin_vel_w


def _sensor_force_mag(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    s = env.scene[sensor_name]
    f = s.data.net_forces_w.reshape(env.num_envs, -1)
    return torch.norm(f, dim=-1)


def _object_upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    q = env.scene["object"].data.root_quat_w
    z_axis = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1))
    return z_axis[:, 2].clamp(-1.0, 1.0)


def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _upright_cos(env)


def root_height_reward(env: "ManagerBasedRLEnv", target_z: float = 0.78, sigma: float = 0.10) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    if hasattr(env, "_urop_bank_root_z_mean"):
        target_z = float(env._urop_bank_root_z_mean)
    err = (z - target_z) / sigma
    return torch.exp(-err * err)


def track_lin_vel_xy_exp(env: "ManagerBasedRLEnv", sigma: float = 0.25) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    cmd = getattr(env, "_urop_carry_cmd", torch.zeros((env.num_envs, 3), device=env.device))
    err = torch.sum((v_b[:, 0:2] - cmd[:, 0:2]) ** 2, dim=-1)
    return torch.exp(-err / (sigma * sigma))


def track_ang_vel_z_exp(env: "ManagerBasedRLEnv", sigma: float = 0.35) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    cmd = getattr(env, "_urop_carry_cmd", torch.zeros((env.num_envs, 3), device=env.device))
    err = (w_b[:, 2] - cmd[:, 2]) ** 2
    return torch.exp(-err / (sigma * sigma))


def object_center_reward(env: "ManagerBasedRLEnv", sigma_xyz=(0.14, 0.12, 0.16)) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rel = quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    if hasattr(env, "_urop_bank_target_obj_rel"):
        target = env._urop_bank_target_obj_rel.view(1, 3)
    else:
        target = torch.tensor([0.42, 0.0, 0.22], device=env.device).view(1, 3)
    sigma = torch.tensor(sigma_xyz, device=env.device).view(1, 3)
    err = ((rel - target) / sigma) ** 2
    return torch.exp(-torch.sum(err, dim=-1))


def object_upright_reward(env: "ManagerBasedRLEnv", max_tilt_deg: float = 35.0) -> torch.Tensor:
    cos_tilt = _object_upright_cos(env)
    cos_min = math.cos(math.radians(max_tilt_deg))
    out = (cos_tilt - cos_min) / max(1.0 - cos_min, 1e-6)
    return out.clamp(0.0, 1.0)


def hold_object_vel_reward(env: "ManagerBasedRLEnv", torso_body_name: str = "torso_link", sigma: float = 0.35) -> torch.Tensor:
    obj = env.scene["object"]
    torso_vel = _body_vel(env, torso_body_name)
    rel_speed = torch.norm(obj.data.root_lin_vel_w - torso_vel, dim=-1)
    return torch.exp(-((rel_speed / sigma) ** 2))


def joint_vel_l2_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    jv = env.scene["robot"].data.joint_vel[:, idx]
    return torch.sum(jv * jv, dim=-1)


def torque_l2_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)
    jt = getattr(robot.data, "applied_torque", None)
    if jt is None:
        jt = getattr(robot.data, "joint_effort", None)
    if jt is None:
        return torch.zeros(env.num_envs, device=env.device)
    jt = jt[:, idx]
    return torch.sum(jt * jt, dim=-1)


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)


def hug_contact_bonus(
    env: "ManagerBasedRLEnv",
    sensor_names_left: list[str],
    sensor_names_right: list[str],
    sensor_name_torso: str,
    thr: float = 1.0,
) -> torch.Tensor:
    left_hits = torch.stack([(_sensor_force_mag(env, n) > thr).float() for n in sensor_names_left], dim=-1).sum(dim=-1)
    right_hits = torch.stack([(_sensor_force_mag(env, n) > thr).float() for n in sensor_names_right], dim=-1).sum(dim=-1)
    torso_hit = (_sensor_force_mag(env, sensor_name_torso) > thr).float() * 2.0
    bilateral_gate = (left_hits > 0.0) & (right_hits > 0.0)
    max_possible = float(len(sensor_names_left) + len(sensor_names_right)) + 2.0
    contact_score = (left_hits + right_hits + torso_hit) / max_possible
    return bilateral_gate.float() * (contact_score ** 2.0)
