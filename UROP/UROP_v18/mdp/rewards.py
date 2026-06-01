from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import (
    get_controlled_joint_indices,
    get_lower_body_joint_indices,
    LOWER_BODY_JOINT_NAMES,
    quat_apply,
    quat_rotate_inverse,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


LEFT_ARM_STRUCTURAL_SENSORS = [
    "contact_l_shoulder_yaw",
    "contact_l_elbow",
    "contact_l_wrist_roll",
    "contact_l_wrist_pitch",
    "contact_l_wrist_yaw",
]
RIGHT_ARM_STRUCTURAL_SENSORS = [
    "contact_r_shoulder_yaw",
    "contact_r_elbow",
    "contact_r_wrist_roll",
    "contact_r_wrist_pitch",
    "contact_r_wrist_yaw",
]
LEFT_ARM_ALL_SENSORS = LEFT_ARM_STRUCTURAL_SENSORS + ["contact_l_hand"]
RIGHT_ARM_ALL_SENSORS = RIGHT_ARM_STRUCTURAL_SENSORS + ["contact_r_hand"]


def _toss_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_toss_active"):
        return env._urop_toss_active.float()
    return torch.zeros(env.num_envs, device=env.device)


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _body_name_to_idx(env: "ManagerBasedRLEnv") -> dict[str, int]:
    if hasattr(env, "_urop_body_name_to_id"):
        return env._urop_body_name_to_id
    robot = env.scene["robot"]
    env._urop_body_name_to_id = {name: i for i, name in enumerate(getattr(robot.data, "body_names", []))}
    return env._urop_body_name_to_id


def _body_pos(env: "ManagerBasedRLEnv", body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    body_map = _body_name_to_idx(env)
    if body_name in body_map:
        return robot.data.body_pos_w[:, body_map[body_name], :]
    return robot.data.root_pos_w


def _body_vel(env: "ManagerBasedRLEnv", body_name: str) -> torch.Tensor:
    robot = env.scene["robot"]
    body_map = _body_name_to_idx(env)
    if body_name in body_map:
        return robot.data.body_lin_vel_w[:, body_map[body_name], :]
    return robot.data.root_lin_vel_w


def _resolve_body_idx(env: "ManagerBasedRLEnv", candidates: list[str]) -> int | None:
    body_map = _body_name_to_idx(env)
    for name in candidates:
        if name in body_map:
            return body_map[name]
    return None


def _sensor_force_mag(env: "ManagerBasedRLEnv", sensor_name: str) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    forces = sensor.data.net_forces_w.reshape(env.num_envs, -1)
    return torch.norm(forces, dim=-1)


def _max_force(env: "ManagerBasedRLEnv", sensor_names: list[str]) -> torch.Tensor:
    vals = [_sensor_force_mag(env, name) for name in sensor_names]
    return torch.stack(vals, dim=-1).max(dim=-1).values


def _sum_hits(env: "ManagerBasedRLEnv", sensor_names: list[str], thr: float) -> torch.Tensor:
    vals = [(_sensor_force_mag(env, name) > thr).float() for name in sensor_names]
    return torch.stack(vals, dim=-1).sum(dim=-1)


def _upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(robot.data.root_quat_w, g_world)
    return (-g_b[:, 2]).clamp(0.0, 1.0)


def _ensure_hold_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d)
    if not hasattr(env, "_urop_hold_cache_global_step"):
        env._urop_hold_cache_global_step = -1
    if not hasattr(env, "_urop_hold_cache_episode_len"):
        env._urop_hold_cache_episode_len = torch.full((n,), -1, device=d, dtype=torch.long)


def _hold_cache_is_fresh(env: "ManagerBasedRLEnv") -> bool:
    if not hasattr(env, "_urop_hold_cache_global_step"):
        return False
    if env._urop_hold_cache_global_step != int(env.common_step_counter):
        return False
    return torch.equal(env._urop_hold_cache_episode_len, env.episode_length_buf)


def _ready_joint_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    if hasattr(env, "_urop_ready_joint_pos"):
        return env._urop_ready_joint_pos
    return robot.data.default_joint_pos


def _chest_hold_target(env: "ManagerBasedRLEnv", target_offset=(0.20, 0.0, 0.06)) -> torch.Tensor:
    torso_pos = _body_pos(env, "torso_link")
    rq = env.scene["robot"].data.root_quat_w
    offset = torch.tensor(target_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    return torso_pos + quat_apply(rq, offset)


def _update_hold_latch(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _ensure_hold_buffers(env)
    if _hold_cache_is_fresh(env):
        return env._urop_hold_latched.float()

    active = _toss_active(env) > 0.5
    inactive = ~active
    env._urop_hold_latched[inactive] = False
    env._urop_hold_steps[inactive] = 0

    if torch.any(active):
        robot = env.scene["robot"]
        obj = env.scene["object"]

        torso_pos = _body_pos(env, "torso_link")
        torso_vel = _body_vel(env, "torso_link")
        chest_target = _chest_hold_target(env)

        obj_pos = obj.data.root_pos_w
        obj_vel = obj.data.root_lin_vel_w

        torso_dist = torch.norm(obj_pos - torso_pos, dim=-1)
        hold_region_err = torch.norm(obj_pos - chest_target, dim=-1)
        rel_speed = torch.norm(obj_vel - torso_vel, dim=-1)

        left_force = _max_force(env, LEFT_ARM_STRUCTURAL_SENSORS)
        right_force = _max_force(env, RIGHT_ARM_STRUCTURAL_SENSORS)
        torso_force = _sensor_force_mag(env, "contact_torso")

        # Dense but safe latch: bilateral arm contact is ideal; torso+one arm is allowed.
        thr = 1.15
        left_contact = left_force > thr
        right_contact = right_force > thr
        torso_contact = torso_force > thr
        bilateral_contact = left_contact & right_contact
        contact_gate = bilateral_contact | (torso_contact & (left_contact | right_contact))

        stable = (
            active
            & (_upright_cos(env) > 0.66)
            & (obj_pos[:, 2] > 0.34)
            & (torso_dist < 0.78)
            & (hold_region_err < 0.42)
            & (rel_speed < 1.15)
            & contact_gate
        )

        new_latch = stable & (~env._urop_hold_latched)
        if torch.any(new_latch):
            env._urop_hold_latched[new_latch] = True
            env._urop_hold_anchor_xy[new_latch] = robot.data.root_pos_w[new_latch, 0:2]

        critical_failure = env._urop_hold_latched & active & (
            (obj_pos[:, 2] < 0.20)
            | (torso_dist > 1.15)
            | (_upright_cos(env) < 0.42)
        )
        if torch.any(critical_failure):
            env._urop_hold_latched[critical_failure] = False
            env._urop_hold_steps[critical_failure] = 0

        stable_latched = env._urop_hold_latched & stable
        env._urop_hold_steps[stable_latched] += 1
        # Do not reset instantly on one bad contact frame; AprilTag/contact can flicker.
        soft_keep = env._urop_hold_latched & (~stable) & (rel_speed < 1.35) & (torso_dist < 0.95)
        env._urop_hold_steps[soft_keep] = torch.clamp(env._urop_hold_steps[soft_keep] - 1, min=0)
        env._urop_hold_steps[env._urop_hold_latched & (~stable) & (~soft_keep)] = 0
        env._urop_hold_steps[~env._urop_hold_latched] = 0

    env._urop_hold_cache_global_step = int(env.common_step_counter)
    env._urop_hold_cache_episode_len = env.episode_length_buf.clone()
    return env._urop_hold_latched.float()


def _hold_ramp(env: "ManagerBasedRLEnv", warmup_steps: int = 18) -> torch.Tensor:
    _update_hold_latch(env)
    return torch.clamp(env._urop_hold_steps.float() / float(max(warmup_steps, 1)), 0.0, 1.0)


def _object_rel_kinematics_body(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rel_p_b = quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    rel_v_b = quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    return rel_p_b, rel_v_b


def _scene_visible(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_object_scene_visible"):
        visible = env._urop_object_scene_visible.float()
    else:
        visible = _toss_active(env)
    if hasattr(env, "_urop_visible_start_s"):
        t = env.episode_length_buf.float().unsqueeze(-1) * float(env.step_dt)
        started = (t >= env._urop_visible_start_s).float().squeeze(-1)
        visible = visible * torch.clamp(started + _toss_active(env), 0.0, 1.0)
    return visible


def _catchability_score(
    env: "ManagerBasedRLEnv",
    min_incoming_speed: float = 0.04,
    ideal_ttc: float = 0.58,
    ttc_sigma: float = 0.62,
) -> torch.Tensor:
    """Trajectory-independent soft affordance for receiving the object now.

    This must not depend on the hidden delivery family.  It only uses object kinematics and
    scene visibility, so a carried box, a pushed box, and a ballistic toss with the same rel state
    produce the same reward signal.
    """
    rel_p_b, rel_v_b = _object_rel_kinematics_body(env)
    x = rel_p_b[:, 0]
    y = rel_p_b[:, 1]
    z = rel_p_b[:, 2]
    vx = rel_v_b[:, 0]

    incoming_score = torch.sigmoid((-vx - float(min_incoming_speed)) / 0.16)

    x_window = ((x > 0.02) & (x < 1.35)).float()
    y_window = (torch.abs(y) < 0.92).float()
    z_window = ((z > -0.35) & (z < 0.78)).float()
    distance_score = torch.exp(-(((x - 0.30) / 0.62) ** 2)) * x_window
    lateral_score = torch.exp(-((torch.abs(y) / 0.66) ** 2)) * y_window
    height_score = torch.exp(-(((z - 0.10) / 0.48) ** 2)) * z_window

    spatial = torch.clamp(distance_score * lateral_score * height_score, 0.0, 1.0).pow(1.0 / 3.0)

    ttc = x / torch.clamp(-vx, min=0.07)
    ttc_window = ((ttc > 0.05) & (ttc < 1.75)).float()
    ttc_score = torch.exp(-(((ttc - float(ideal_ttc)) / float(ttc_sigma)) ** 2)) * ttc_window

    visible = torch.clamp(_scene_visible(env) + _toss_active(env), 0.0, 1.0)
    score = incoming_score * spatial * ttc_score * visible
    return torch.clamp(score, 0.0, 1.0)


def _pre_receive_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Broad gate for preparing to receive, before contact is necessary."""
    rel_p_b, rel_v_b = _object_rel_kinematics_body(env)
    x = rel_p_b[:, 0]
    y = rel_p_b[:, 1]
    z = rel_p_b[:, 2]
    vx = rel_v_b[:, 0]
    visible = torch.clamp(_scene_visible(env) + _toss_active(env), 0.0, 1.0)
    incoming = torch.sigmoid((-vx - 0.015) / 0.14)
    in_corridor = ((x > 0.12) & (x < 1.65) & (torch.abs(y) < 1.05) & (z > -0.42) & (z < 0.85)).float()
    ttc = x / torch.clamp(-vx, min=0.06)
    ttc_ok = ((ttc > 0.08) & (ttc < 2.20)).float()
    # Include active/released objects even if instantaneous TTC is noisy.
    active_boost = torch.clamp(_toss_active(env), 0.0, 1.0)
    return torch.clamp(visible * in_corridor * (0.75 * incoming * ttc_ok + 0.25 * active_boost), 0.0, 1.0) * (1.0 - _hold_gate(env))


def _wait_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Wait when nothing is visible, or when a visible object is far/stationary/receding.
    # Do NOT wait-gate incoming objects merely because the precise receive score is still low.
    hold = _hold_gate(env)
    rel_p_b, rel_v_b = _object_rel_kinematics_body(env)
    x = rel_p_b[:, 0]
    vx = rel_v_b[:, 0]
    speed = torch.norm(rel_v_b, dim=-1)
    visible = _scene_visible(env)
    active = _toss_active(env)
    hidden = 1.0 - visible
    far = (x > 0.62).float()
    not_incoming = ((vx > -0.08) | (speed < 0.12)).float()
    safe_visible_wait = visible * far * not_incoming * (1.0 - active)
    return torch.clamp(hidden + safe_visible_wait, 0.0, 1.0) * (1.0 - hold)


def _hold_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _update_hold_latch(env)


def _catch_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    hold = _hold_gate(env)
    # Pre-receive gives dense gradients; exact catchability is still strongest near the window.
    return torch.clamp(0.35 * _pre_receive_gate(env) + _catchability_score(env), 0.0, 1.0) * (1.0 - hold)


def _visible_noncatchable_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    rel_p_b, rel_v_b = _object_rel_kinematics_body(env)
    x = rel_p_b[:, 0]
    vx = rel_v_b[:, 0]
    speed = torch.norm(rel_v_b, dim=-1)
    visible = _scene_visible(env)
    far = (x > 0.58).float()
    stationary_or_receding = ((vx > -0.06) | (speed < 0.11)).float()
    # only punish/praise waiting before release; never suppress pre-shape for incoming objects.
    return visible * far * stationary_or_receding * (1.0 - _toss_active(env)) * (1.0 - _pre_receive_gate(env)) * (1.0 - _hold_gate(env))


def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _upright_cos(env)


def root_height_reward(env: "ManagerBasedRLEnv", target_z: float = 0.78, sigma: float = 0.10) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    err = (z - target_z) / sigma
    return torch.exp(-(err * err))


def base_motion_penalty(env: "ManagerBasedRLEnv", w_lin: float = 1.0, w_ang: float = 0.35) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    raw = w_lin * torch.sum(v_b[:, 0:2] ** 2, dim=-1) + w_ang * torch.sum(w_b ** 2, dim=-1)
    # Do not train a frozen robot: during catchable lateral tosses, allow controlled base/torso motion.
    catch = _catchability_score(env)
    gate = 1.0 - 0.65 * catch
    return raw * gate


def joint_vel_l2_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    jv = env.scene["robot"].data.joint_vel[:, idx]
    return torch.sum(jv * jv, dim=-1)


def torque_l2_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)
    tau = getattr(robot.data, "applied_torque", None)
    if tau is None:
        tau = getattr(robot.data, "joint_effort", None)
    if tau is None:
        return torch.zeros(env.num_envs, device=env.device)
    tau = tau[:, idx]
    return torch.sum(tau * tau, dim=-1)


def action_magnitude_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.sum(env.action_manager.action ** 2, dim=-1)


def _ensure_action_history_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "action_manager") or not hasattr(env.action_manager, "action"):
        return

    action = env.action_manager.action
    n, action_dim = action.shape
    d = action.device

    if not hasattr(env, "_urop_prev_prev_action"):
        env._urop_prev_prev_action = torch.zeros((n, action_dim), device=d, dtype=action.dtype)
    if not hasattr(env, "_urop_cached_prev_action"):
        env._urop_cached_prev_action = torch.zeros((n, action_dim), device=d, dtype=action.dtype)
    if not hasattr(env, "_urop_action_history_step"):
        env._urop_action_history_step = -1
    if not hasattr(env, "_urop_action_history_episode_len"):
        env._urop_action_history_episode_len = torch.full((n,), -1, device=d, dtype=torch.long)


def _get_action_history(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(env, "action_manager") or not hasattr(env.action_manager, "action"):
        zeros = torch.zeros((env.num_envs, get_controlled_joint_indices(env).shape[0]), device=env.device)
        return zeros, zeros, zeros

    _ensure_action_history_buffers(env)

    action = env.action_manager.action
    prev_action_raw = env.action_manager.prev_action
    current_step = int(env.common_step_counter)
    if env._urop_action_history_step != current_step:
        fresh_mask = (env.episode_length_buf <= 1).unsqueeze(-1)
        env._urop_prev_prev_action = torch.where(
            fresh_mask,
            torch.zeros_like(env._urop_cached_prev_action),
            env._urop_cached_prev_action,
        )
        env._urop_cached_prev_action = torch.where(fresh_mask, torch.zeros_like(prev_action_raw), prev_action_raw)
        env._urop_action_history_step = current_step
        env._urop_action_history_episode_len = env.episode_length_buf.clone()

    fresh_mask = (env.episode_length_buf <= 1).unsqueeze(-1)
    prev_action = torch.where(fresh_mask, torch.zeros_like(prev_action_raw), prev_action_raw)
    prev_prev_action = torch.where(fresh_mask, torch.zeros_like(env._urop_prev_prev_action), env._urop_prev_prev_action)
    return action, prev_action, prev_prev_action


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, prev_action, _ = _get_action_history(env)
    return torch.sum((action - prev_action) ** 2, dim=-1)


def action_acceleration_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, prev_action, prev_prev_action = _get_action_history(env)
    action_accel = action - 2.0 * prev_action + prev_prev_action
    return torch.sum(action_accel ** 2, dim=-1)


def lower_body_action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize high-frequency lower-body action changes.

    IMPORTANT:
    `action` is indexed in policy-action order, not articulation joint order.
    In this task, lower-body controlled joints are the first 15 entries of
    CONTROLLED_JOINT_NAMES, so we slice the action vector directly.
    """
    action, prev_action, _ = _get_action_history(env)

    lower_body_action_dim = len(LOWER_BODY_JOINT_NAMES)  # 15 = legs 12 + waist 3
    return torch.sum(
        (action[:, :lower_body_action_dim] - prev_action[:, :lower_body_action_dim]) ** 2,
        dim=-1,
    )


def foot_slip_penalty(env: "ManagerBasedRLEnv", ground_height_thr: float = 0.16) -> torch.Tensor:
    robot = env.scene["robot"]
    left_idx = _resolve_body_idx(env, ["left_ankle_roll_link", "left_foot_link", "left_ankle_pitch_link"])
    right_idx = _resolve_body_idx(env, ["right_ankle_roll_link", "right_foot_link", "right_ankle_pitch_link"])
    if left_idx is None or right_idx is None:
        return torch.zeros(env.num_envs, device=env.device)

    body_pos = robot.data.body_pos_w
    body_vel = robot.data.body_lin_vel_w
    left_close = (body_pos[:, left_idx, 2] < ground_height_thr).float()
    right_close = (body_pos[:, right_idx, 2] < ground_height_thr).float()
    left_slip = torch.sum(body_vel[:, left_idx, 0:2] ** 2, dim=-1) * left_close
    right_slip = torch.sum(body_vel[:, right_idx, 0:2] ** 2, dim=-1) * right_close
    return left_slip + right_slip


def wait_base_drift_penalty(env: "ManagerBasedRLEnv", sigma: float = 0.14) -> torch.Tensor:
    if not hasattr(env, "_urop_spawn_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_spawn_xy, dim=-1)
    return ((drift / sigma) ** 2) * _wait_gate(env)


def wait_yaw_drift_penalty(env: "ManagerBasedRLEnv", sigma: float = 0.20) -> torch.Tensor:
    if not hasattr(env, "_urop_spawn_yaw"):
        return torch.zeros(env.num_envs, device=env.device)
    q = env.scene["robot"].data.root_quat_w
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    yaw_err = _wrap_to_pi(yaw - env._urop_spawn_yaw[:, 0])
    return ((yaw_err / sigma) ** 2) * _wait_gate(env)


def ready_pose_when_waiting(env: "ManagerBasedRLEnv", sigma: float = 0.22) -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    target = _ready_joint_target(env)[:, idx]
    diff = torch.norm(current - target, dim=-1)
    return torch.exp(-((diff / sigma) ** 2)) * _wait_gate(env)


def waiting_joint_stillness_reward(env: "ManagerBasedRLEnv", sigma: float = 1.4) -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    jv = env.scene["robot"].data.joint_vel[:, idx]
    speed = torch.norm(jv, dim=-1)
    return torch.exp(-((speed / sigma) ** 2)) * _wait_gate(env)


def lower_body_ready_reward(env: "ManagerBasedRLEnv", sigma_wait: float = 0.16, sigma_active: float = 0.26) -> torch.Tensor:
    idx = get_lower_body_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    target = _ready_joint_target(env)[:, idx]
    diff = torch.norm(current - target, dim=-1)
    active = _catchability_score(env)
    sigma = sigma_wait * (1.0 - active) + sigma_active * active
    return torch.exp(-((diff / sigma) ** 2))


def catch_target_region_reward(env: "ManagerBasedRLEnv", sigma: float = 0.28) -> torch.Tensor:
    obj = env.scene["object"]
    dist = torch.norm(obj.data.root_pos_w - _chest_hold_target(env), dim=-1)
    return torch.exp(-((dist / sigma) ** 2)) * _catch_gate(env)


def upper_body_receive_reward(env: "ManagerBasedRLEnv", sigma: float = 0.30) -> torch.Tensor:
    obj = env.scene["object"]
    left_elbow = _body_pos(env, "left_elbow_link")
    right_elbow = _body_pos(env, "right_elbow_link")
    left_wrist = _body_pos(env, "left_wrist_roll_link")
    right_wrist = _body_pos(env, "right_wrist_roll_link")
    torso = _body_pos(env, "torso_link")

    d_left = torch.minimum(
        torch.norm(obj.data.root_pos_w - left_elbow, dim=-1),
        torch.norm(obj.data.root_pos_w - left_wrist, dim=-1),
    )
    d_right = torch.minimum(
        torch.norm(obj.data.root_pos_w - right_elbow, dim=-1),
        torch.norm(obj.data.root_pos_w - right_wrist, dim=-1),
    )
    d_torso = torch.norm(obj.data.root_pos_w - torso, dim=-1)

    left_score = torch.exp(-((d_left / sigma) ** 2))
    right_score = torch.exp(-((d_right / sigma) ** 2))
    torso_score = torch.exp(-((d_torso / (sigma * 1.9)) ** 2))
    bilateral = torch.sqrt(torch.clamp(left_score * right_score, 0.0, 1.0))
    score = 0.30 * (left_score + right_score) * 0.5 + 0.45 * bilateral + 0.25 * torso_score
    return torch.clamp(score, 0.0, 1.0) * _catch_gate(env)


def catch_velocity_match_reward(env: "ManagerBasedRLEnv", torso_body_name: str = "torso_link", sigma: float = 0.75) -> torch.Tensor:
    obj = env.scene["object"]
    torso_vel = _body_vel(env, torso_body_name)
    rel_speed = torch.norm(obj.data.root_lin_vel_w - torso_vel, dim=-1)
    return torch.exp(-((rel_speed / sigma) ** 2)) * _catch_gate(env)


def hug_contact_bonus(
    env: "ManagerBasedRLEnv",
    sensor_names_left: list[str],
    sensor_names_right: list[str],
    sensor_name_torso: str,
    thr: float = 1.5,
) -> torch.Tensor:
    left_hits = _sum_hits(env, sensor_names_left, thr)
    right_hits = _sum_hits(env, sensor_names_right, thr)
    torso_hit = (_sensor_force_mag(env, sensor_name_torso) > thr).float()
    left_any = (left_hits > 0.0).float()
    right_any = (right_hits > 0.0).float()
    bilateral = left_any * right_any
    any_contact = torch.clamp(left_any + right_any + torso_hit, 0.0, 1.0)
    max_possible = float(len(sensor_names_left) + len(sensor_names_right))
    arm_contact_score = (left_hits + right_hits) / max_possible
    contact_score = 0.20 * any_contact + 0.42 * bilateral + 0.24 * torso_hit + 0.14 * arm_contact_score
    return torch.clamp(contact_score, 0.0, 1.0) * torch.clamp(_catch_gate(env) + _hold_gate(env), 0.0, 1.0)


def visual_wait_patience_reward(env: "ManagerBasedRLEnv", action_sigma: float = 0.45) -> torch.Tensor:
    """Reward seeing a non-catchable object without premature hugging."""
    gate = _visible_noncatchable_gate(env)
    if not hasattr(env, "action_manager") or not hasattr(env.action_manager, "action"):
        return gate
    action = env.action_manager.action
    upper = action[:, len(LOWER_BODY_JOINT_NAMES):]
    energy = torch.norm(upper, dim=-1)
    return torch.exp(-((energy / float(action_sigma)) ** 2)) * gate


def premature_hug_penalty(env: "ManagerBasedRLEnv", action_w: float = 1.0, pose_w: float = 0.4) -> torch.Tensor:
    """Penalize the real failure mode: tag_visible but object is far/stationary, robot hugs air."""
    gate = _visible_noncatchable_gate(env)
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    target = _ready_joint_target(env)[:, idx]
    upper_start = len(LOWER_BODY_JOINT_NAMES)
    arm_pose_dev = torch.sum((current[:, upper_start:] - target[:, upper_start:]) ** 2, dim=-1)
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        upper_action = env.action_manager.action[:, upper_start:]
        action_energy = torch.sum(upper_action * upper_action, dim=-1)
    else:
        action_energy = torch.zeros(env.num_envs, device=env.device)
    return gate * (float(action_w) * action_energy + float(pose_w) * arm_pose_dev)


def lateral_intercept_reward(env: "ManagerBasedRLEnv", deadband: float = 0.10, speed_sigma: float = 0.35) -> torch.Tensor:
    """Allow/encourage controlled lateral motion when the incoming box is off-center."""
    catch = _catchability_score(env)
    rel_p_b, _ = _object_rel_kinematics_body(env)
    y = rel_p_b[:, 1]
    lateral_need = torch.clamp((torch.abs(y) - float(deadband)) / 0.45, 0.0, 1.0)
    robot = env.scene["robot"]
    v_b = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w)
    toward_speed = torch.relu(v_b[:, 1] * torch.sign(y))
    move_score = 1.0 - torch.exp(-((toward_speed / float(speed_sigma)) ** 2))
    return catch * lateral_need * move_score


def incoming_receive_pose_reward(env: "ManagerBasedRLEnv", sigma: float = 0.48, hold_blend: float = 0.72) -> torch.Tensor:
    """Dense bootstrap: incoming object -> move upper body toward a receive/hold posture.

    This is intentionally trajectory-independent and uses only current kinematics through
    _pre_receive_gate/_catchability_score.  It should make the policy prepare, not hug distant boxes.
    """
    gate = torch.clamp(_pre_receive_gate(env) + 0.75 * _catchability_score(env), 0.0, 1.0)
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    ready = _ready_joint_target(env)[:, idx]
    hold_vec = torch.tensor(
        [scene_objects_cfg.HOLD_POSE[name] for name in scene_objects_cfg.CONTROLLED_JOINT_NAMES],
        device=env.device,
        dtype=current.dtype,
    ).unsqueeze(0).repeat(env.num_envs, 1)
    target = ready.clone()
    upper_start = len(LOWER_BODY_JOINT_NAMES)
    blend = float(hold_blend)
    target[:, upper_start:] = (1.0 - blend) * ready[:, upper_start:] + blend * hold_vec[:, upper_start:]
    err = torch.norm(current[:, upper_start:] - target[:, upper_start:], dim=-1)
    return torch.exp(-((err / float(sigma)) ** 2)) * gate


def head_region_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Discourage solving the task by letting the box hit the high head/camera region."""
    rel_p_b, _ = _object_rel_kinematics_body(env)
    x = rel_p_b[:, 0]
    z = rel_p_b[:, 2]
    y_abs = torch.abs(rel_p_b[:, 1])
    near_head = ((x > -0.05) & (x < 0.55) & (y_abs < 0.35) & (z > 0.48)).float()
    return near_head * torch.clamp(_catch_gate(env) + _hold_gate(env) + _toss_active(env), 0.0, 1.0)


def hold_object_vel_reward(env: "ManagerBasedRLEnv", torso_body_name: str = "torso_link", sigma: float = 0.45) -> torch.Tensor:
    obj = env.scene["object"]
    torso_vel = _body_vel(env, torso_body_name)
    rel_speed = torch.norm(obj.data.root_lin_vel_w - torso_vel, dim=-1)
    return torch.exp(-((rel_speed / sigma) ** 2)) * _hold_gate(env)


def hold_pose_reward(env: "ManagerBasedRLEnv", sigma: float = 0.18) -> torch.Tensor:
    obj = env.scene["object"]
    dist = torch.norm(obj.data.root_pos_w - _chest_hold_target(env), dim=-1)
    return torch.exp(-((dist / sigma) ** 2)) * _hold_gate(env)


def hold_latched_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _hold_gate(env)


def hold_sustain_bonus(env: "ManagerBasedRLEnv", min_steps: int = 20) -> torch.Tensor:
    _update_hold_latch(env)
    sustain = torch.clamp((env._urop_hold_steps.float() - float(min_steps)) / float(max(min_steps, 1)), 0.0, 1.0)
    return sustain * env._urop_hold_latched.float()


def object_not_dropped_bonus(env: "ManagerBasedRLEnv", min_z: float = 0.42, max_dist: float = 1.8) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    z_ok = (obj.data.root_pos_w[:, 2] > min_z).float()
    dist_ok = (torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1) < max_dist).float()
    return z_ok * dist_ok * _hold_gate(env)


def impact_peak_penalty(env: "ManagerBasedRLEnv", sensor_names: list[str], force_thr: float = 220.0) -> torch.Tensor:
    peaks = [_sensor_force_mag(env, name) for name in sensor_names]
    peak = torch.stack(peaks, dim=-1).max(dim=-1).values
    gate = torch.clamp(_toss_active(env) + _catchability_score(env) + _hold_gate(env), 0.0, 1.0)
    return torch.relu(peak - force_thr) / force_thr * gate


def post_hold_still_reward(env: "ManagerBasedRLEnv", lin_sigma: float = 0.10, yaw_sigma: float = 0.30) -> torch.Tensor:
    gate = _hold_ramp(env)
    robot = env.scene["robot"]
    v_b = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)
    vxy = torch.norm(v_b[:, 0:2], dim=-1)
    yaw = torch.abs(w_b[:, 2])
    return torch.exp(-((vxy / lin_sigma) ** 2) - ((yaw / yaw_sigma) ** 2)) * gate


def post_hold_anchor_penalty(env: "ManagerBasedRLEnv", sigma: float = 0.10) -> torch.Tensor:
    gate = _hold_ramp(env)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    drift = torch.norm(robot.data.root_pos_w[:, 0:2] - env._urop_hold_anchor_xy, dim=-1)
    return ((drift / sigma) ** 2) * gate
