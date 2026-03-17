# [/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/mdp/observations.py]

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .. import scene_objects_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Quaternion utils (w, x, y, z)
# =============================================================================

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for batched quaternions [N, 4]."""
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for batched quaternions [N, 4]."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors [N, 3] by batched quaternions [N, 4]."""
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:, 1:4]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vector into local/body frame."""
    return quat_apply(quat_conj(q), v)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """Quaternion -> 6D rotation representation."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    # first two columns of rotation matrix
    return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)


# =============================================================================
# Controlled joint subset (29 DOF only; fingers excluded)
# IMPORTANT: use the exact order from scene_objects_cfg.G1_29_JOINTS
# =============================================================================

CONTROLLED_JOINT_NAMES = scene_objects_cfg.G1_29_JOINTS


def get_controlled_joint_indices(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cache and return joint indices that match the 29-DOF control order."""
    if hasattr(env, "_carry_controlled_joint_indices"):
        return env._carry_controlled_joint_indices

    robot = env.scene["robot"]
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}

    indices = []
    missing = []
    for name in CONTROLLED_JOINT_NAMES:
        if name in name_to_idx:
            indices.append(name_to_idx[name])
        else:
            missing.append(name)

    if len(missing) > 0:
        raise RuntimeError(f"Missing controlled joints in articulation: {missing}")

    env._carry_controlled_joint_indices = torch.tensor(
        indices, device=env.device, dtype=torch.long
    )
    return env._carry_controlled_joint_indices


# =============================================================================
# Small helpers
# =============================================================================

def _get_prev_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "prev_action"):
        return env.action_manager.prev_action
    return torch.zeros((env.num_envs, len(CONTROLLED_JOINT_NAMES)), device=env.device)


def _get_command(
    env: "ManagerBasedRLEnv",
    command_names: tuple[str, ...] = ("base_velocity", "command"),
    default_dim: int = 3,
) -> torch.Tensor:
    """Robustly fetch carry command from IsaacLab command manager.

    We try a few likely term names so this stays robust while env_cfg evolves.
    """
    if not hasattr(env, "command_manager"):
        return torch.zeros((env.num_envs, default_dim), device=env.device)

    for name in command_names:
        try:
            cmd = env.command_manager.get_command(name)
            if cmd is None:
                continue
            if cmd.ndim == 1:
                cmd = cmd.unsqueeze(-1)
            if cmd.shape[0] != env.num_envs:
                continue
            if cmd.shape[1] >= default_dim:
                return cmd[:, :default_dim]
            # pad if shorter than expected
            out = torch.zeros((env.num_envs, default_dim), device=env.device, dtype=cmd.dtype)
            out[:, :cmd.shape[1]] = cmd
            return out
        except Exception:
            pass

    return torch.zeros((env.num_envs, default_dim), device=env.device)


def _get_applied_torque_for_controlled_joints(
    env: "ManagerBasedRLEnv",
    idx: torch.Tensor,
    torque_scale: float,
) -> torch.Tensor:
    robot = env.scene["robot"]

    if hasattr(robot.data, "applied_torque"):
        jt = robot.data.applied_torque[:, idx]
    elif hasattr(robot.data, "joint_effort"):
        jt = robot.data.joint_effort[:, idx]
    else:
        jt = torch.zeros((env.num_envs, idx.numel()), device=env.device)

    jt = torch.clamp(jt * torque_scale, -1.0, 1.0)
    return jt


# =============================================================================
# Main observation terms
# =============================================================================

def robot_proprio(
    env: "ManagerBasedRLEnv",
    include_torque: bool = True,
    torque_scale: float = 1.0 / 80.0,
) -> torch.Tensor:
    """Carry policy backbone observation.

    Default output with include_torque=True:
      projected_gravity(3)
      base_lin_vel_body(3)
      base_ang_vel_body(3)
      joint_pos_29(29)
      joint_vel_29(29)
      joint_torque_29(29)
    Total = 96
    """
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)

    root_q = robot.data.root_quat_w

    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_body = quat_rotate_inverse(root_q, g_world)

    lin_body = quat_rotate_inverse(root_q, robot.data.root_lin_vel_w)
    ang_body = quat_rotate_inverse(root_q, robot.data.root_ang_vel_w)

    joint_pos = robot.data.joint_pos[:, idx]
    joint_vel = robot.data.joint_vel[:, idx]

    parts = [g_body, lin_body, ang_body, joint_pos, joint_vel]

    if include_torque:
        joint_torque = _get_applied_torque_for_controlled_joints(env, idx, torque_scale=torque_scale)
        parts.append(joint_torque)

    return torch.cat(parts, dim=-1)


def carry_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Velocity command for carry locomotion: [vx, vy, wz]."""
    return _get_command(env, command_names=("base_velocity", "command"), default_dim=3)


def prev_actions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Previous 29-dim action."""
    return _get_prev_action(env)


def object_rel_state(
    env: "ManagerBasedRLEnv",
    pos_scale: float = 1.0,
    vel_scale: float = 1.0,
    noise_std: float = 0.0,
    clip: float | None = None,
) -> torch.Tensor:
    """Object state relative to robot root frame.

    Output:
      rel_pos_body(3)
      rel_rot6d(6)
      rel_lin_vel_body(3)
      rel_ang_vel_body(3)
    Total = 15
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    root_q = robot.data.root_quat_w
    root_p = robot.data.root_pos_w
    root_v = robot.data.root_lin_vel_w
    root_w = robot.data.root_ang_vel_w

    obj_q = obj.data.root_quat_w
    obj_p = obj.data.root_pos_w
    obj_v = obj.data.root_lin_vel_w
    obj_w = obj.data.root_ang_vel_w

    rel_pos_body = quat_rotate_inverse(root_q, obj_p - root_p) * pos_scale
    rel_lin_vel_body = quat_rotate_inverse(root_q, obj_v - root_v) * vel_scale
    rel_ang_vel_body = quat_rotate_inverse(root_q, obj_w - root_w)

    rel_q = quat_mul(quat_conj(root_q), obj_q)
    rel_rot6d = quat_to_rot6d(rel_q)

    x = torch.cat([rel_pos_body, rel_rot6d, rel_lin_vel_body, rel_ang_vel_body], dim=-1)

    if noise_std > 0.0:
        x = x + noise_std * torch.randn_like(x)

    if clip is not None:
        x = torch.clamp(x, -clip, clip)

    return x


def object_params(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Normalized object parameters for privileged critic observation.

    Output:
      size_x, size_y, size_z, mass, friction, restitution
    Total = 6
    """
    d = env.device
    n = env.num_envs

    size = getattr(
        env,
        "_urop_box_size",
        torch.tensor(scene_objects_cfg.DEFAULT_BOX_SIZE if hasattr(scene_objects_cfg, "DEFAULT_BOX_SIZE") else [0.32, 0.24, 0.24],
                     device=d).unsqueeze(0).repeat(n, 1),
    )
    mass = getattr(env, "_urop_box_mass", torch.full((n, 1), 3.0, device=d))
    fric = getattr(env, "_urop_box_friction", torch.full((n, 1), 0.7, device=d))
    rest = getattr(env, "_urop_box_restitution", torch.full((n, 1), 0.02, device=d))

    # normalize around the ranges used in events.py
    size_n = torch.stack(
        [
            (size[:, 0] - 0.32) / 0.06,
            (size[:, 1] - 0.24) / 0.05,
            (size[:, 2] - 0.24) / 0.05,
        ],
        dim=-1,
    )
    mass_n = (mass - 3.25) / 1.75
    fric_n = (fric - 0.70) / 0.20
    rest_n = (rest - 0.03) / 0.03

    return torch.cat([size_n, mass_n, fric_n, rest_n], dim=-1)


def contact_features(
    env: "ManagerBasedRLEnv",
    sensor_names: list[str] | None = None,
    scale: float = 1.0 / 300.0,
    clip: float | None = 5.0,
) -> torch.Tensor:
    """Per-sensor contact magnitude features for critic.

    Default sensor order follows scene_objects_cfg.ALL_CARRY_CONTACT_SENSOR_NAMES.
    """
    if sensor_names is None:
        sensor_names = scene_objects_cfg.ALL_CARRY_CONTACT_SENSOR_NAMES

    mags = []
    for name in sensor_names:
        sensor = env.scene[name]
        # flatten all reported force vectors for this sensor, then take magnitude
        forces = sensor.data.net_forces_w.reshape(env.num_envs, -1)
        mag = torch.norm(forces, dim=-1, keepdim=True) * scale
        mags.append(mag)

    out = torch.cat(mags, dim=-1)

    if clip is not None:
        out = torch.clamp(out, -clip, clip)

    return out


def reset_grace_feature(
    env: "ManagerBasedRLEnv",
    normalize_by: float = 10.0,
) -> torch.Tensor:
    """Privileged critic feature for short post-reset grace window.

    This is useful because carry resets start from catch-bank states and may need
    a few simulation steps to settle before strict drop/tilt penalties matter.
    """
    if hasattr(env, "_carry_reset_grace_steps"):
        x = env._carry_reset_grace_steps.float().unsqueeze(-1) / normalize_by
        return torch.clamp(x, 0.0, 1.0)
    return torch.zeros((env.num_envs, 1), device=env.device)


def carry_target_rel_pos_feature(
    env: "ManagerBasedRLEnv",
    pos_scale: float = 1.0,
) -> torch.Tensor:
    """Optional privileged feature: stored target object relative position.

    Note:
      This only makes sense if events.py has already populated
      env._carry_target_obj_rel on reset.
    """
    if hasattr(env, "_carry_target_obj_rel"):
        return env._carry_target_obj_rel * pos_scale
    return torch.zeros((env.num_envs, 3), device=env.device)


def carry_target_rel_pos_error(
    env: "ManagerBasedRLEnv",
    pos_scale: float = 1.0,
) -> torch.Tensor:
    """Optional privileged feature: current minus target object relative position.

    This is critic-friendly. For actor/policy, I would usually NOT expose it at first.
    """
    if not hasattr(env, "_carry_target_obj_rel"):
        return torch.zeros((env.num_envs, 3), device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    current_rel_world = obj.data.root_pos_w - robot.data.root_pos_w
    err = (current_rel_world - env._carry_target_obj_rel) * pos_scale
    return err


# =============================================================================
# Convenience aliases / debug-friendly pieces
# =============================================================================

def object_rel_pos_body(env: "ManagerBasedRLEnv", pos_scale: float = 1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w) * pos_scale


def object_rel_vel_body(env: "ManagerBasedRLEnv", vel_scale: float = 1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w) * vel_scale


def object_upright_feature(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cosine-like upright feature for object z-axis vs world z-axis."""
    obj = env.scene["object"]
    z_world = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    z_obj_world = quat_apply(obj.data.root_quat_w, z_world)
    return z_obj_world[:, 2:3]