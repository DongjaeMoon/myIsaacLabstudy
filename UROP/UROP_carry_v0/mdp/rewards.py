#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v0/mdp/rewards.py]
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .observations import (
    quat_apply,
    quat_rotate_inverse,
    get_controlled_joint_indices,
    get_arm_carry_joint_indices,
)

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
    sensor = env.scene[sensor_name]
    force = sensor.data.net_forces_w.reshape(env.num_envs, -1)
    return torch.norm(force, dim=-1)


def _carry_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_command"):
        return env._urop_command
    return torch.zeros((env.num_envs, 3), device=env.device)


def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _upright_cos(env)


def root_height_reward(env: "ManagerBasedRLEnv", target_z: float = 0.78, sigma: float = 0.12) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    err = (z - target_z) / sigma
    return torch.exp(-(err * err))


def command_tracking_lin_vel_reward(env: "ManagerBasedRLEnv", sigma: float = 0.28) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    cmd = _carry_command(env)[:, 0:2]
    err = torch.sum((v_b[:, 0:2] - cmd) ** 2, dim=-1)
    return torch.exp(-err / (sigma * sigma))


def command_tracking_ang_vel_reward(env: "ManagerBasedRLEnv", sigma: float = 0.35) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    cmd = _carry_command(env)[:, 2]
    err = (w_b[:, 2] - cmd) ** 2
    return torch.exp(-err / (sigma * sigma))


def base_motion_penalty(env: "ManagerBasedRLEnv", w_lin_z: float = 1.0, w_ang_xy: float = 0.4) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    return w_lin_z * (v_b[:, 2] ** 2) + w_ang_xy * torch.sum(w_b[:, 0:2] ** 2, dim=-1)


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
    return torch.sum(jt[:, idx] ** 2, dim=-1)


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1)


def arm_carry_pose_reward(env: "ManagerBasedRLEnv", sigma: float = 0.65) -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_arm_carry_joint_indices(env)
    current = robot.data.joint_pos[:, idx]
    target = robot.data.default_joint_pos[:, idx]
    diff = torch.norm(current - target, dim=-1)
    return torch.exp(-((diff / sigma) ** 2))


def object_centering_reward(env: "ManagerBasedRLEnv", sigma: float = 0.12) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    root_q = robot.data.root_quat_w
    rel_p = quat_rotate_inverse(root_q, obj.data.root_pos_w - robot.data.root_pos_w)
    target = getattr(env, "_urop_carry_rel_target", torch.tensor([0.38, 0.0, 0.34], device=env.device).unsqueeze(0).repeat(env.num_envs, 1))
    diff = torch.norm(rel_p - target, dim=-1)
    return torch.exp(-((diff / sigma) ** 2))


def object_upright_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    obj = env.scene["object"]
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    up_world = quat_apply(obj.data.root_quat_w, up_local)
    return up_world[:, 2].clamp(0.0, 1.0)


def object_relative_velocity_reward(
    env: "ManagerBasedRLEnv",
    torso_body_name: str = "torso_link",
    sigma: float = 0.45,
) -> torch.Tensor:
    obj = env.scene["object"]
    torso_vel = _body_vel(env, torso_body_name)
    rel_speed = torch.norm(obj.data.root_lin_vel_w - torso_vel, dim=-1)
    return torch.exp(-((rel_speed / sigma) ** 2))


def object_not_dropped_bonus(env: "ManagerBasedRLEnv", min_z: float = 0.45, max_dist: float = 1.10) -> torch.Tensor:
    obj = env.scene["object"]
    robot = env.scene["robot"]
    z_ok = obj.data.root_pos_w[:, 2] > min_z
    dist_ok = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1) < max_dist
    return (z_ok & dist_ok).float()


def hug_contact_bonus(
    env: "ManagerBasedRLEnv",
    sensor_names_left: list[str],
    sensor_names_right: list[str],
    sensor_name_torso: str,
    thr: float = 2.0,
) -> torch.Tensor:
    def _max_force(names: list[str]) -> torch.Tensor:
        forces = [_sensor_force_mag(env, n) for n in names]
        return torch.stack(forces, dim=-1).max(dim=-1).values

    lf = _max_force(sensor_names_left)
    rf = _max_force(sensor_names_right)
    tf = _sensor_force_mag(env, sensor_name_torso)

    both_arms = ((lf > thr) & (rf > thr)).float()
    torso_touch = (tf > thr).float()
    return 0.7 * both_arms + 0.3 * torso_touch


def impact_peak_penalty(env: "ManagerBasedRLEnv", sensor_names: list[str], force_thr: float = 280.0) -> torch.Tensor:
    peaks = [_sensor_force_mag(env, n) for n in sensor_names]
    peak = torch.stack(peaks, dim=-1).max(dim=-1).values
    return torch.relu(peak - force_thr)


def object_spin_penalty(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    obj = env.scene["object"]
    return scale * torch.sum(obj.data.root_ang_vel_w ** 2, dim=-1)
