# [/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/mdp/rewards.py]

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .. import scene_objects_cfg
from .observations import (
    quat_apply,
    quat_rotate_inverse,
    get_controlled_joint_indices,
    carry_command,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Small helpers
# =============================================================================

def _body_name_to_idx(robot) -> dict[str, int]:
    return {name: i for i, name in enumerate(robot.data.body_names)}


def _get_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        return env.action_manager.action
    return torch.zeros((env.num_envs, len(scene_objects_cfg.G1_29_JOINTS)), device=env.device)


def _get_prev_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "prev_action"):
        return env.action_manager.prev_action
    return torch.zeros((env.num_envs, len(scene_objects_cfg.G1_29_JOINTS)), device=env.device)


def _get_target_obj_rel_body(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target object relative position in ROBOT BODY frame.

    This is expected to be populated at reset by events.py.
    If unavailable, we fall back to a reasonable default.
    """
    if hasattr(env, "_carry_target_obj_rel"):
        return env._carry_target_obj_rel
    return torch.tensor([0.42, 0.0, 0.22], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)


def _sensor_force_mag(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w.reshape(env.num_envs, -1)
    return torch.norm(forces, dim=-1)


def _max_force(env: "ManagerBasedRLEnv", sensor_names: list[str]) -> torch.Tensor:
    vals = [_sensor_force_mag(env, n) for n in sensor_names]
    return torch.stack(vals, dim=-1).max(dim=-1).values


def _mean_force(env: "ManagerBasedRLEnv", sensor_names: list[str]) -> torch.Tensor:
    vals = [_sensor_force_mag(env, n) for n in sensor_names]
    return torch.stack(vals, dim=-1).mean(dim=-1)


def _foot_body_indices(env: "ManagerBasedRLEnv") -> tuple[int | None, int | None]:
    robot = env.scene["robot"]
    body_map = _body_name_to_idx(robot)

    left_candidates = [
        "left_ankle_roll_link",
        "left_ankle_pitch_link",
        "left_foot_link",
    ]
    right_candidates = [
        "right_ankle_roll_link",
        "right_ankle_pitch_link",
        "right_foot_link",
    ]

    left_idx = None
    for name in left_candidates:
        if name in body_map:
            left_idx = body_map[name]
            break

    right_idx = None
    for name in right_candidates:
        if name in body_map:
            right_idx = body_map[name]
            break

    return left_idx, right_idx


def _full_body_contact_force_tensor(env: "ManagerBasedRLEnv") -> torch.Tensor | None:
    """Return full-body contact force tensor if available.

    Expected shape is usually [N, B, 3], but we only rely on the last dim being 3.
    """
    try:
        sensor = env.scene["contact_forces"]
    except KeyError:
        return None

    if not hasattr(sensor.data, "net_forces_w"):
        return None

    f = sensor.data.net_forces_w
    if f.ndim < 3:
        return None
    return f


def _foot_contact_mask(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean contact mask for left/right feet based on full-body contact sensor."""
    f = _full_body_contact_force_tensor(env)
    if f is None:
        z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return z, z

    left_idx, right_idx = _foot_body_indices(env)
    if left_idx is None or right_idx is None:
        z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return z, z

    left_mag = torch.norm(f[:, left_idx, :], dim=-1)
    right_mag = torch.norm(f[:, right_idx, :], dim=-1)

    return left_mag > force_threshold, right_mag > force_threshold


def _projected_gravity_body(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    return quat_rotate_inverse(q, g_world)


def _object_rel_pos_body(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)


def _object_rel_vel_body(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)


def _object_rel_ang_vel_body(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_ang_vel_w - robot.data.root_ang_vel_w)


def _object_upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    obj = env.scene["object"]
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    z_world = quat_apply(obj.data.root_quat_w, local_z)
    return z_world[:, 2]


# =============================================================================
# Locomotion-style command tracking rewards
# =============================================================================

def track_lin_vel_xy_exp(
    env: "ManagerBasedRLEnv",
    std: float = 0.35,
) -> torch.Tensor:
    """Exponential tracking reward for commanded planar velocity."""
    robot = env.scene["robot"]
    cmd = carry_command(env)  # [vx, vy, wz]
    lin_body = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w)

    err = cmd[:, 0:2] - lin_body[:, 0:2]
    return torch.exp(-torch.sum(err * err, dim=-1) / (std * std))


def track_ang_vel_z_exp(
    env: "ManagerBasedRLEnv",
    std: float = 0.35,
) -> torch.Tensor:
    """Exponential tracking reward for commanded yaw rate."""
    robot = env.scene["robot"]
    cmd = carry_command(env)
    ang_body = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)

    err = cmd[:, 2] - ang_body[:, 2]
    return torch.exp(-(err * err) / (std * std))


# =============================================================================
# Basic stability / regularization terms (positive magnitudes; use negative weights)
# =============================================================================

def lin_vel_z_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_lin_vel_w[:, 2] ** 2


def ang_vel_xy_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.sum(robot.data.root_ang_vel_w[:, 0:2] ** 2, dim=-1)


def flat_orientation_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for roll/pitch tilt, using projected gravity in body frame."""
    g_b = _projected_gravity_body(env)
    return torch.sum(g_b[:, 0:2] ** 2, dim=-1)


def joint_vel_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)
    return torch.sum(robot.data.joint_vel[:, idx] ** 2, dim=-1)


def joint_acc_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)

    if hasattr(robot.data, "joint_acc"):
        acc = robot.data.joint_acc[:, idx]
        return torch.sum(acc ** 2, dim=-1)

    # fallback: if joint_acc is unavailable, return zeros rather than breaking
    return torch.zeros(env.num_envs, device=env.device)


def joint_torques_l2(
    env: "ManagerBasedRLEnv",
    torque_scale: float = 1.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)

    if hasattr(robot.data, "applied_torque"):
        tau = robot.data.applied_torque[:, idx]
    elif hasattr(robot.data, "joint_effort"):
        tau = robot.data.joint_effort[:, idx]
    else:
        tau = torch.zeros((env.num_envs, idx.numel()), device=env.device)

    tau = tau * torque_scale
    return torch.sum(tau ** 2, dim=-1)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    a = _get_action(env)
    a_prev = _get_prev_action(env)
    return torch.sum((a - a_prev) ** 2, dim=-1)


def feet_slide(
    env: "ManagerBasedRLEnv",
    contact_force_threshold: float = 5.0,
) -> torch.Tensor:
    """Penalty for foot horizontal motion while the foot is in contact."""
    robot = env.scene["robot"]
    left_idx, right_idx = _foot_body_indices(env)

    if left_idx is None or right_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    left_contact, right_contact = _foot_contact_mask(env, force_threshold=contact_force_threshold)

    lv = robot.data.body_lin_vel_w[:, left_idx, 0:2]
    rv = robot.data.body_lin_vel_w[:, right_idx, 0:2]

    left_slide = torch.norm(lv, dim=-1) * left_contact.float()
    right_slide = torch.norm(rv, dim=-1) * right_contact.float()

    return left_slide + right_slide


# =============================================================================
# Carry-specific object stabilization rewards
# =============================================================================

def object_center_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.12,
) -> torch.Tensor:
    """Reward object relative position staying near reset-time target pose.

    IMPORTANT:
      This expects env._carry_target_obj_rel to be stored in BODY FRAME.
    """
    rel_pos_body = _object_rel_pos_body(env)
    target_body = _get_target_obj_rel_body(env)
    err = rel_pos_body - target_body
    return torch.exp(-torch.sum(err * err, dim=-1) / (std * std))


def object_upright_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.35,
) -> torch.Tensor:
    """Reward the box staying upright.

    Uses the cosine of object local z-axis vs world z-axis.
    """
    z_cos = torch.clamp(_object_upright_cos(env), -1.0, 1.0)
    # ideal z_cos = 1
    return torch.exp(-((1.0 - z_cos) ** 2) / (std * std))


def hold_object_vel_reward(
    env: "ManagerBasedRLEnv",
    lin_std: float = 0.6,
    ang_std: float = 1.5,
) -> torch.Tensor:
    """Reward for keeping object motion calm relative to robot body."""
    rel_lin = _object_rel_vel_body(env)
    rel_ang = _object_rel_ang_vel_body(env)

    lin_term = torch.exp(-torch.sum(rel_lin * rel_lin, dim=-1) / (lin_std * lin_std))
    ang_term = torch.exp(-torch.sum(rel_ang * rel_ang, dim=-1) / (ang_std * ang_std))
    return 0.6 * lin_term + 0.4 * ang_term


# =============================================================================
# Whole-body support / hugging rewards
# =============================================================================

def bilateral_contact_bonus(
    env: "ManagerBasedRLEnv",
    force_scale: float = 20.0,
) -> torch.Tensor:
    """Reward balanced left/right object support.

    Uses max force over left/right carry sensor groups.
    """
    left_force = _max_force(env, scene_objects_cfg.LEFT_CARRY_CONTACT_SENSOR_NAMES)
    right_force = _max_force(env, scene_objects_cfg.RIGHT_CARRY_CONTACT_SENSOR_NAMES)

    left_act = torch.tanh(left_force / force_scale)
    right_act = torch.tanh(right_force / force_scale)

    # reward both sides being active; min encourages bilateral support
    return torch.minimum(left_act, right_act)


def hug_contact_bonus(
    env: "ManagerBasedRLEnv",
    torso_force_scale: float = 25.0,
    limb_force_scale: float = 20.0,
) -> torch.Tensor:
    """Reward torso + bilateral arm support around the object."""
    torso_force = _sensor_force_mag(env, "contact_torso")
    left_force = _max_force(env, scene_objects_cfg.LEFT_CARRY_CONTACT_SENSOR_NAMES)
    right_force = _max_force(env, scene_objects_cfg.RIGHT_CARRY_CONTACT_SENSOR_NAMES)

    torso_term = torch.tanh(torso_force / torso_force_scale)
    left_term = torch.tanh(left_force / limb_force_scale)
    right_term = torch.tanh(right_force / limb_force_scale)

    # torso support + bilateral arm support
    return 0.4 * torso_term + 0.3 * left_term + 0.3 * right_term


def torso_contact_bonus(
    env: "ManagerBasedRLEnv",
    force_scale: float = 25.0,
) -> torch.Tensor:
    torso_force = _sensor_force_mag(env, "contact_torso")
    return torch.tanh(torso_force / force_scale)


# =============================================================================
# Optional shaping / convenience terms
# =============================================================================

def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def not_drop_bonus(
    env: "ManagerBasedRLEnv",
    min_height: float = 0.18,
) -> torch.Tensor:
    obj = env.scene["object"]
    return (obj.data.root_pos_w[:, 2] > min_height).float()


def object_rel_drift_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty version of object-center deviation."""
    rel_pos_body = _object_rel_pos_body(env)
    target_body = _get_target_obj_rel_body(env)
    err = rel_pos_body - target_body
    return torch.sum(err * err, dim=-1)


def object_tilt_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    z_cos = torch.clamp(_object_upright_cos(env), -1.0, 1.0)
    return (1.0 - z_cos) ** 2


def object_rel_lin_vel_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    rel_lin = _object_rel_vel_body(env)
    return torch.sum(rel_lin * rel_lin, dim=-1)


def object_rel_ang_vel_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    rel_ang = _object_rel_ang_vel_body(env)
    return torch.sum(rel_ang * rel_ang, dim=-1)


def root_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float = 0.79,
) -> torch.Tensor:
    robot = env.scene["robot"]
    return (robot.data.root_pos_w[:, 2] - target_height) ** 2