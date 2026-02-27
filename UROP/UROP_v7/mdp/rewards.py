from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from .observations import (
    quat_rotate_inverse,
    quat_apply,
    quat_mul,
    quat_conj,
    quat_to_rot6d,
    get_controlled_joint_indices,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _toss_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # 1 if toss has happened this episode (or is active), else 0
    if hasattr(env, "_urop_toss_active"):
        return env._urop_toss_active.float()
    return torch.zeros(env.num_envs, device=env.device)


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


# -----------------------------------------------------------------------------
# Hold latch (catch success detection)
# -----------------------------------------------------------------------------

def _ensure_hold_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d)


def _update_hold_latch(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Latch turns on once the object is genuinely 'received'.

    Criteria (robust, but not overly strict):
      - toss was active,
      - object is close to torso,
      - some meaningful contact (torso+arm OR bilateral arm contact),
      - relative speed is not huge anymore.

    Once latched:
      - store base anchor position (xy) to enforce *stay still* for receive-only policy.
    """
    _ensure_hold_buffers(env)

    active = _toss_active(env) > 0.5
    if not torch.any(active):
        env._urop_hold_steps[:] = 0
        return env._urop_hold_latched.float()

    robot = env.scene["robot"]
    obj = env.scene["object"]

    torso_pos = _body_pos(env, "torso_link")
    torso_vel = _body_vel(env, "torso_link")
    obj_pos = obj.data.root_pos_w
    obj_vel = obj.data.root_lin_vel_w

    dist = torch.norm(obj_pos - torso_pos, dim=-1)
    rel_speed = torch.norm(obj_vel - torso_vel, dim=-1)

    # simple gates
    z_ok = obj_pos[:, 2] > 0.30

    # contact gate: either bilateral arms, or torso + one arm
    lf = torch.maximum(_sensor_force_mag(env, "contact_l_elbow"), _sensor_force_mag(env, "contact_l_hand"))
    rf = torch.maximum(_sensor_force_mag(env, "contact_r_elbow"), _sensor_force_mag(env, "contact_r_hand"))
    tf = _sensor_force_mag(env, "contact_torso")

    bilateral = (lf > 5.0) & (rf > 5.0)
    torso_plus = (tf > 10.0) & ((lf > 3.0) | (rf > 3.0))
    contact_gate = bilateral | torso_plus

    stable = active & z_ok & (dist < 0.55) & (rel_speed < 1.55) & contact_gate

    new_latch = stable & (~env._urop_hold_latched)
    if torch.any(new_latch):
        env._urop_hold_latched[new_latch] = True
        env._urop_hold_anchor_xy[new_latch] = robot.data.root_pos_w[new_latch, 0:2]
        env._urop_hold_steps[new_latch] = 0

    env._urop_hold_steps[env._urop_hold_latched] += 1
    env._urop_hold_steps[~env._urop_hold_latched] = 0

    # unlatch if clearly dropped/missed after latching
    unlatch = env._urop_hold_latched & ((obj_pos[:, 2] < 0.18) | (dist > 1.35))
    if torch.any(unlatch):
        env._urop_hold_latched[unlatch] = False
        env._urop_hold_steps[unlatch] = 0

    return env._urop_hold_latched.float()


def _hold_ramp(env: "ManagerBasedRLEnv", warmup_steps: int = 12) -> torch.Tensor:
    """0->1 ramp after latch to avoid punishing the impact moment."""
    _update_hold_latch(env)
    return torch.clamp(env._urop_hold_steps.float() / float(max(warmup_steps, 1)), 0.0, 1.0)


# -----------------------------------------------------------------------------
# Base stabilization (always on)
# -----------------------------------------------------------------------------

def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _upright_cos(env)


def root_height_reward(env: "ManagerBasedRLEnv", target_z=0.78, sigma=0.12) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    err = (z - target_z) / sigma
    return torch.exp(-err * err)


def base_velocity_penalty(env: "ManagerBasedRLEnv", w_lin=1.0, w_ang=0.35) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    return w_lin * torch.sum(v_b[:, 0:2] ** 2, dim=-1) + w_ang * (w_b[:, 2] ** 2)


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


# -----------------------------------------------------------------------------
# Wait-phase structure
# -----------------------------------------------------------------------------

def wait_base_drift_penalty(env: "ManagerBasedRLEnv", sigma: float = 0.20) -> torch.Tensor:
    if not hasattr(env, "_urop_spawn_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_spawn_xy, dim=-1)
    return ((drift / sigma) ** 2) * (1.0 - _toss_active(env))


def ready_pose_when_waiting(env: "ManagerBasedRLEnv", sigma: float = 0.40) -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)
    current = robot.data.joint_pos[:, idx]
    if hasattr(env, "_urop_ready_joint_pos"):
        target = env._urop_ready_joint_pos[:, idx]
    else:
        target = robot.data.default_joint_pos[:, idx]
    diff = torch.norm(current - target, dim=-1)
    return torch.exp(-((diff / sigma) ** 2)) * (1.0 - _toss_active(env))


# -----------------------------------------------------------------------------
# Catch / hold shaping
# -----------------------------------------------------------------------------

def torso_reach_object_reward(env: "ManagerBasedRLEnv", sigma=0.80) -> torch.Tensor:
    obj = env.scene["object"]
    torso = _body_pos(env, "torso_link")
    d = torch.norm(obj.data.root_pos_w - torso, dim=-1)
    return torch.exp(-((d / sigma) ** 2)) * _toss_active(env)


def hands_reach_object_reward(env: "ManagerBasedRLEnv", sigma: float = 0.38) -> torch.Tensor:
    obj = env.scene["object"]
    l_pos = _body_pos(env, "left_hand_palm_link")
    r_pos = _body_pos(env, "right_hand_palm_link")
    d_l = torch.norm(obj.data.root_pos_w - l_pos, dim=-1)
    d_r = torch.norm(obj.data.root_pos_w - r_pos, dim=-1)
    return (torch.exp(-((d_l / sigma) ** 2)) * torch.exp(-((d_r / sigma) ** 2))) * _toss_active(env)


def hands_support_under_box_reward(
    env: "ManagerBasedRLEnv",
    default_box_size=(0.32, 0.24, 0.24),
    y_frac: float = 0.46,
    z_clearance: float = 0.03,
    sigma: float = 0.16,
) -> torch.Tensor:
    obj = env.scene["object"]
    l_pos = _body_pos(env, "left_hand_palm_link")
    r_pos = _body_pos(env, "right_hand_palm_link")

    if hasattr(env, "_urop_box_size"):
        size = env._urop_box_size
    else:
        size = torch.tensor(default_box_size, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

    y_half = 0.5 * size[:, 1]
    z_half = 0.5 * size[:, 2]
    oq = obj.data.root_quat_w
    op = obj.data.root_pos_w

    off_l = torch.stack([
        torch.zeros(env.num_envs, device=env.device),
        y_half * y_frac,
        -(z_half + z_clearance),
    ], dim=-1)
    off_r = torch.stack([
        torch.zeros(env.num_envs, device=env.device),
        -y_half * y_frac,
        -(z_half + z_clearance),
    ], dim=-1)

    p_l = op + quat_apply(oq, off_l)
    p_r = op + quat_apply(oq, off_r)
    d_l = torch.norm(l_pos - p_l, dim=-1)
    d_r = torch.norm(r_pos - p_r, dim=-1)

    return (torch.exp(-((d_l / sigma) ** 2)) * torch.exp(-((d_r / sigma) ** 2))) * _toss_active(env)


def hold_pose_reward(
    env: "ManagerBasedRLEnv",
    torso_body_name: str = "torso_link",
    sigma: float = 0.24,
    target_offset=(0.06, 0.0, 0.06),
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    torso_pos = _body_pos(env, torso_body_name)
    rq = robot.data.root_quat_w
    offset = torch.tensor(target_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    target = torso_pos + quat_apply(rq, offset)
    d = torch.norm(obj.data.root_pos_w - target, dim=-1)
    return torch.exp(-((d / sigma) ** 2)) * _toss_active(env)


def hold_object_vel_reward(env: "ManagerBasedRLEnv", torso_body_name: str = "torso_link", sigma: float = 0.65) -> torch.Tensor:
    obj = env.scene["object"]
    torso_vel = _body_vel(env, torso_body_name)
    speed = torch.norm(obj.data.root_lin_vel_w - torso_vel, dim=-1)
    return torch.exp(-((speed / sigma) ** 2)) * _toss_active(env)


def contact_hold_bonus_symmetric(
    env: "ManagerBasedRLEnv",
    sensor_names_left: list[str],
    sensor_names_right: list[str],
    sensor_names_torso: list[str] | None = None,
    thr: float = 1.0,
) -> torch.Tensor:
    left_hits = [(_sensor_force_mag(env, n) > thr).float() for n in sensor_names_left]
    right_hits = [(_sensor_force_mag(env, n) > thr).float() for n in sensor_names_right]
    l_hit = torch.stack(left_hits, dim=-1).max(dim=-1).values
    r_hit = torch.stack(right_hits, dim=-1).max(dim=-1).values
    score = l_hit * r_hit
    if sensor_names_torso:
        torso_hits = [(_sensor_force_mag(env, n) > thr).float() for n in sensor_names_torso]
        t_hit = torch.stack(torso_hits, dim=-1).max(dim=-1).values
        score = score * (1.0 + 0.8 * t_hit)
    return score * _toss_active(env)


def object_not_dropped_bonus(env: "ManagerBasedRLEnv", min_z=0.48, max_dist=2.3) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    z_ok = (obj.data.root_pos_w[:, 2] > min_z).float()
    dist_ok = (torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1) < max_dist).float()
    return z_ok * dist_ok * (_upright_cos(env) ** 2.0) * _toss_active(env)


def impact_peak_penalty(env: "ManagerBasedRLEnv", sensor_names: list[str], force_thr=320.0) -> torch.Tensor:
    peaks = [_sensor_force_mag(env, n) for n in sensor_names]
    peak = torch.stack(peaks, dim=-1).max(dim=-1).values
    pen = torch.relu(peak - force_thr) / force_thr
    return pen * _toss_active(env)


def hold_latched_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _update_hold_latch(env)
    return env._urop_hold_latched.float()


def hold_sustain_bonus(env: "ManagerBasedRLEnv", min_steps: int = 30) -> torch.Tensor:
    _update_hold_latch(env)
    ok = env._urop_hold_latched & (env._urop_hold_steps >= int(min_steps))
    return ok.float()


# -----------------------------------------------------------------------------
# Post-catch anti-shuffle terms (핵심)
# -----------------------------------------------------------------------------

def post_hold_still_reward(env: "ManagerBasedRLEnv", lin_sigma: float = 0.14, yaw_sigma: float = 0.45) -> torch.Tensor:
    gate = _hold_ramp(env)
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    vxy = torch.norm(v_b[:, 0:2], dim=-1)
    yaw = torch.abs(w_b[:, 2])
    rew = torch.exp(-((vxy / lin_sigma) ** 2) - ((yaw / yaw_sigma) ** 2))
    return rew * gate


def post_hold_anchor_penalty(env: "ManagerBasedRLEnv", sigma: float = 0.12) -> torch.Tensor:
    gate = _hold_ramp(env)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_hold_anchor_xy, dim=-1)
    return ((drift / sigma) ** 2) * gate


def post_hold_leg_motion_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    gate = _hold_ramp(env)
    robot = env.scene["robot"]
    names = robot.data.joint_names
    leg_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ]
    if not hasattr(env, "_urop_leg_joint_indices"):
        name_to_idx = {n: i for i, n in enumerate(names)}
        env._urop_leg_joint_indices = torch.tensor([name_to_idx[n] for n in leg_names], device=env.device, dtype=torch.long)
    idx = env._urop_leg_joint_indices
    leg_jv = robot.data.joint_vel[:, idx]
    pen = torch.sum(leg_jv * leg_jv, dim=-1)
    return pen * gate