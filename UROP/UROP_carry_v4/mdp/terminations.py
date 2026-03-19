# [/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/mdp/terminations.py]

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .observations import quat_apply, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Helpers
# =============================================================================

def _body_name_to_idx(robot) -> dict[str, int]:
    return {name: i for i, name in enumerate(robot.data.body_names)}


def _in_reset_grace(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env bool mask: True while post-reset grace window is active."""
    if hasattr(env, "_carry_reset_grace_steps"):
        return env._carry_reset_grace_steps > 0
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _object_upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cosine between object's local +Z and world +Z."""
    obj = env.scene["object"]
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    z_world = quat_apply(obj.data.root_quat_w, local_z)
    return z_world[:, 2]


def _robot_upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Cosine between robot body +Z and world +Z."""
    robot = env.scene["robot"]
    local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    z_world = quat_apply(robot.data.root_quat_w, local_z)
    return z_world[:, 2]


def _object_rel_dist(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    return torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)


def _full_body_contact_force_tensor(env: "ManagerBasedRLEnv") -> torch.Tensor | None:
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


# =============================================================================
# Main terminations
# =============================================================================

def robot_fallen(
    env: "ManagerBasedRLEnv",
    min_root_height: float = 0.45,
    max_tilt_deg: float = 45.0,
) -> torch.Tensor:
    """Terminate if robot is too low or too tilted.

    This is the main 'fall' termination for carry.
    """
    robot = env.scene["robot"]

    too_low = robot.data.root_pos_w[:, 2] < min_root_height

    upright_cos = torch.clamp(_robot_upright_cos(env), -1.0, 1.0)
    min_cos = torch.cos(torch.tensor(max_tilt_deg * 3.1415926535 / 180.0, device=env.device))
    too_tilted = upright_cos < min_cos

    return too_low | too_tilted


def object_dropped(
    env: "ManagerBasedRLEnv",
    min_object_height: float = 0.12,
    max_object_rel_dist: float = 0.95,
    use_grace: bool = True,
) -> torch.Tensor:
    """Terminate if box falls too low or gets too far from robot."""
    obj = env.scene["object"]

    too_low = obj.data.root_pos_w[:, 2] < min_object_height
    too_far = _object_rel_dist(env) > max_object_rel_dist

    out = too_low | too_far

    if use_grace:
        out = out & (~_in_reset_grace(env))

    return out


def object_tilt_exceeded(
    env: "ManagerBasedRLEnv",
    max_tilt_deg: float = 60.0,
    use_grace: bool = True,
) -> torch.Tensor:
    """Terminate if object is excessively tilted."""
    z_cos = torch.clamp(_object_upright_cos(env), -1.0, 1.0)
    min_cos = torch.cos(torch.tensor(max_tilt_deg * 3.1415926535 / 180.0, device=env.device))
    out = z_cos < min_cos

    if use_grace:
        out = out & (~_in_reset_grace(env))

    return out


def numerical_instability(
    env: "ManagerBasedRLEnv",
) -> torch.Tensor:
    """Terminate if NaN/Inf appears in key robot/object states."""
    robot = env.scene["robot"]
    obj = env.scene["object"]

    checks = [
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.root_lin_vel_w,
        robot.data.root_ang_vel_w,
        robot.data.joint_pos,
        robot.data.joint_vel,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
        obj.data.root_lin_vel_w,
        obj.data.root_ang_vel_w,
    ]

    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for x in checks:
        bad = bad | (~torch.isfinite(x).all(dim=-1))
    return bad


# =============================================================================
# Optional generic contact-based termination
# =============================================================================

def body_contact_exceeds(
    env: "ManagerBasedRLEnv",
    body_names: list[str],
    force_threshold: float = 150.0,
    use_grace: bool = False,
) -> torch.Tensor:
    """Generic contact-based termination for specific robot bodies.

    IMPORTANT:
      This uses the full-body contact sensor aggregate force.
      If your contact_forces sensor is NOT filtered to ground-only,
      then intended object contact can also appear here.

    So use this only for body parts that should never strongly contact anything
    during carry, or after you confirm the sensor semantics.
    """
    f = _full_body_contact_force_tensor(env)
    if f is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene["robot"]
    body_map = _body_name_to_idx(robot)

    valid_indices = [body_map[name] for name in body_names if name in body_map]
    if len(valid_indices) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    idx = torch.tensor(valid_indices, device=env.device, dtype=torch.long)
    mags = torch.norm(f[:, idx, :], dim=-1)  # [N, K]
    out = mags.max(dim=-1).values > force_threshold

    if use_grace:
        out = out & (~_in_reset_grace(env))

    return out


def knee_contact_exceeds(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 150.0,
    use_grace: bool = False,
) -> torch.Tensor:
    """Convenience wrapper for knee contacts."""
    return body_contact_exceeds(
        env,
        body_names=["left_knee_link", "right_knee_link"],
        force_threshold=force_threshold,
        use_grace=use_grace,
    )


def pelvis_contact_exceeds(
    env: "ManagerBasedRLEnv",
    force_threshold: float = 150.0,
    use_grace: bool = False,
) -> torch.Tensor:
    """Convenience wrapper for pelvis/hip-base contacts if present in USD."""
    return body_contact_exceeds(
        env,
        body_names=["pelvis", "base_link", "torso_link"],
        force_threshold=force_threshold,
        use_grace=use_grace,
    )