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


def _chest_hold_target(env: "ManagerBasedRLEnv", target_offset=(0.18, 0.0, 0.12)) -> torch.Tensor:
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

        structural_contact = (left_force > 1.5) & (right_force > 1.5)
        torso_contact = torso_force > 1.5
        contact_gate = structural_contact & torso_contact

        stable = (
            active
            & (_upright_cos(env) > 0.72)
            & (obj_pos[:, 2] > 0.45)
            & (torso_dist < 0.55)
            & (hold_region_err < 0.26)
            & (rel_speed < 0.70)
            & contact_gate
        )

        new_latch = stable & (~env._urop_hold_latched)
        if torch.any(new_latch):
            env._urop_hold_latched[new_latch] = True
            env._urop_hold_anchor_xy[new_latch] = robot.data.root_pos_w[new_latch, 0:2]

        critical_failure = env._urop_hold_latched & active & (
            (obj_pos[:, 2] < 0.24)
            | (torso_dist > 1.00)
            | (_upright_cos(env) < 0.45)
        )
        if torch.any(critical_failure):
            env._urop_hold_latched[critical_failure] = False
            env._urop_hold_steps[critical_failure] = 0

        stable_latched = env._urop_hold_latched & stable
        env._urop_hold_steps[stable_latched] += 1
        env._urop_hold_steps[env._urop_hold_latched & (~stable)] = 0
        env._urop_hold_steps[~env._urop_hold_latched] = 0

    env._urop_hold_cache_global_step = int(env.common_step_counter)
    env._urop_hold_cache_episode_len = env.episode_length_buf.clone()
    return env._urop_hold_latched.float()


def _hold_ramp(env: "ManagerBasedRLEnv", warmup_steps: int = 18) -> torch.Tensor:
    _update_hold_latch(env)
    return torch.clamp(env._urop_hold_steps.float() / float(max(warmup_steps, 1)), 0.0, 1.0)


def _wait_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Wait only when the object is hidden or visibly far.

    v19b: the previous gate waited until the internal receive-active bit.
    That made slow handover too conservative.  Now a close visible handover
    stops receiving the wait reward even before physical contact.
    """
    if not hasattr(env, "_urop_obj_visible_truth"):
        return 1.0 - _toss_active(env)

    visible = _object_visible_truth(env)
    hidden_wait = 1.0 - visible
    far_wait = _far_visible_wait_gate(env)
    return torch.clamp(hidden_wait + far_wait, 0.0, 1.0) * (1.0 - _hold_gate(env))


def _hold_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _update_hold_latch(env)


def _catch_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    hold = _hold_gate(env)
    # Internal receive-active remains the authoritative gate, but v19b makes
    # that gate turn on earlier in the event config.
    return _toss_active(env) * (1.0 - hold)


def _object_visible_truth(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_obj_visible_truth"):
        return env._urop_obj_visible_truth.float().squeeze(-1)
    return _toss_active(env)


def _object_rel_body(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rel_p_b = quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    rel_v_b = quat_rotate_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w - robot.data.root_lin_vel_w)
    return rel_p_b, rel_v_b


def _receive_affordance(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Continuous proximity/approach score for human-style handover.

    Far visible box -> near 0.
    Box entering the arm/chest receive zone -> near 1, even if slow.
    This teaches "when" without adding a new actor observation.
    """
    rel_p, rel_v = _object_rel_body(env)
    x, y, z = rel_p[:, 0], rel_p[:, 1], rel_p[:, 2]
    vx = rel_v[:, 0]

    visible = _object_visible_truth(env)
    in_front = ((x > 0.05) & (x < 1.20)).float()
    lateral_ok = torch.exp(-((torch.abs(y) / 0.38) ** 2)) * (torch.abs(y) < 0.65).float()
    height_ok = torch.exp(-(((z - 0.16) / 0.34) ** 2)) * ((z > -0.22) & (z < 0.62)).float()

    # High once the object center is around arm/chest receive distance.
    near = torch.clamp((0.72 - x) / 0.42, 0.0, 1.0) * torch.clamp((x - 0.06) / 0.12, 0.0, 1.0)
    approach = torch.clamp((-vx + 0.05) / 0.40, 0.0, 1.0)
    score = near * (0.35 + 0.65 * approach)
    return torch.clamp(score * visible * in_front * lateral_ok * height_ok, 0.0, 1.0)


def _far_visible_wait_gate(env: "ManagerBasedRLEnv", far_x: float = 0.62, max_abs_y: float = 0.45) -> torch.Tensor:
    rel_p, rel_v = _object_rel_body(env)
    visible = _object_visible_truth(env)
    x, y, z = rel_p[:, 0], rel_p[:, 1], rel_p[:, 2]
    vx = rel_v[:, 0]
    far = torch.clamp((x - float(far_x)) / 0.35, 0.0, 1.0)
    lateral_ok = (torch.abs(y) < float(max_abs_y)).float()
    height_ok = ((z > -0.30) & (z < 0.70)).float()
    not_urgent = (vx > -0.45).float()
    return visible * far * lateral_ok * height_ok * not_urgent * (1.0 - _hold_gate(env))


def _visible_far_gate(env: "ManagerBasedRLEnv", min_x: float = 0.62, max_abs_y: float = 0.45) -> torch.Tensor:
    return _far_visible_wait_gate(env, far_x=min_x, max_abs_y=max_abs_y)


def _nonreceive_visible_gate(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Only far visible objects should strongly punish receiving.  Close handover
    # objects must be allowed to transition into receive posture.
    return _far_visible_wait_gate(env)


def _mass_norm(env: "ManagerBasedRLEnv") -> torch.Tensor:
    mass = getattr(
        env,
        "_urop_box_mass",
        torch.full((env.num_envs, 1), scene_objects_cfg.OBJECT_DEFAULT_MASS, device=env.device),
    )[:, 0]
    # v19b mass range is intentionally moderate: roughly 0.8-4.2 kg.
    return torch.clamp((mass - 0.8) / 3.4, 0.0, 1.0)


def _pose_vector_from_dict(env: "ManagerBasedRLEnv", pose_dict: dict[str, float]) -> torch.Tensor:
    return torch.tensor(
        [pose_dict[name] for name in scene_objects_cfg.CONTROLLED_JOINT_NAMES],
        device=env.device,
        dtype=env.scene["robot"].data.joint_pos.dtype,
    ).unsqueeze(0).repeat(env.num_envs, 1)


def _mass_conditioned_receive_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    ready = _pose_vector_from_dict(env, scene_objects_cfg.READY_POSE)
    hold = _pose_vector_from_dict(env, scene_objects_cfg.HOLD_POSE)
    m = _mass_norm(env).unsqueeze(-1)

    target = ready.clone()
    upper = slice(15, scene_objects_cfg.EXPECTED_ACTION_DIM)
    # Heavy objects get earlier/tighter arm wrap, but this remains a soft reward, not a hard script.
    blend = 0.35 + 0.45 * m
    target[:, upper] = ready[:, upper] * (1.0 - blend) + hold[:, upper] * blend

    # Whole-body bracing: small symmetric knee/hip/ankle/waist changes. These are intentionally mild
    # because real G1 stood stably with the existing lower-body profile.
    brace = m[:, 0]
    target[:, 0] += -0.04 * brace  # left hip pitch
    target[:, 3] += 0.10 * brace   # left knee
    target[:, 4] += -0.04 * brace  # left ankle pitch
    target[:, 6] += -0.04 * brace  # right hip pitch
    target[:, 9] += 0.10 * brace   # right knee
    target[:, 10] += -0.04 * brace # right ankle pitch
    target[:, 14] += -0.07 * brace # waist pitch, small forward brace if sign matches USD
    return target


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
    return w_lin * torch.sum(v_b[:, 0:2] ** 2, dim=-1) + w_ang * torch.sum(w_b ** 2, dim=-1)


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
    active = _toss_active(env)
    sigma = sigma_wait * (1.0 - active) + sigma_active * active
    return torch.exp(-((diff / sigma) ** 2))


def visible_far_ready_reward(env: "ManagerBasedRLEnv", sigma: float = 0.20, min_x: float = 0.48) -> torch.Tensor:
    """Reward explicitly ignoring a visible but far/non-committed object."""
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    target = _ready_joint_target(env)[:, idx]
    diff = torch.norm(current - target, dim=-1)
    return torch.exp(-((diff / sigma) ** 2)) * _visible_far_gate(env, min_x=min_x)


def premature_receive_penalty(env: "ManagerBasedRLEnv", min_x: float = 0.48, action_weight: float = 0.25) -> torch.Tensor:
    """Penalize arm/torso hugging when a tag is visible but the handover is not committed.

    This directly targets the real failure mode: tag_visible becomes 1 while the box is still far away,
    and the robot immediately hugs empty air.
    """
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    target = _ready_joint_target(env)[:, idx]
    upper_diff = current[:, 15:] - target[:, 15:]
    penalty = torch.sum(upper_diff * upper_diff, dim=-1)
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        penalty = penalty + float(action_weight) * torch.sum(env.action_manager.action[:, 15:] ** 2, dim=-1)
    return penalty * _visible_far_gate(env, min_x=min_x)


def progressive_receive_pose_reward(env: "ManagerBasedRLEnv", sigma: float = 0.42) -> torch.Tensor:
    """Reward a gradual transition from ready pose to receive/hug pose.

    The gate is the real receive-active gate, which v19b turns on earlier
    when a true handover object enters the near zone.  This avoids a new
    actor state flag while still teaching the policy "when to start receiving".
    """
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    ready = _pose_vector_from_dict(env, scene_objects_cfg.READY_POSE)
    hold = _pose_vector_from_dict(env, scene_objects_cfg.HOLD_POSE)
    gate = torch.clamp(_catch_gate(env), 0.0, 1.0).unsqueeze(-1)

    target = ready.clone()
    upper = slice(15, scene_objects_cfg.EXPECTED_ACTION_DIM)
    target[:, upper] = ready[:, upper] * (1.0 - gate) + hold[:, upper] * gate
    upper_err = torch.norm(current[:, upper] - target[:, upper], dim=-1)
    return torch.exp(-((upper_err / float(sigma)) ** 2)) * gate[:, 0]


def mass_conditioned_receive_pose_reward(env: "ManagerBasedRLEnv", sigma: float = 0.34) -> torch.Tensor:
    """Softly reward mass-dependent arm/torso/lower-body bracing during receive/hold.

    The target is derived from the tag mass prior distribution used at reset. Since the actor receives
    the same prior through mode_one_hot, this term encourages a visible policy difference between light
    and heavy boxes without changing action dimension or order.
    """
    idx = get_controlled_joint_indices(env)
    current = env.scene["robot"].data.joint_pos[:, idx]
    target = _mass_conditioned_receive_target(env)
    diff = torch.norm(current - target, dim=-1)
    gate = torch.clamp(_catch_gate(env) + _hold_gate(env), 0.0, 1.0)
    return torch.exp(-((diff / sigma) ** 2)) * gate


def heavy_object_stability_reward(env: "ManagerBasedRLEnv", lin_sigma: float = 0.16, ang_sigma: float = 0.38) -> torch.Tensor:
    robot = env.scene["robot"]
    v_b = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)
    motion_score = torch.exp(-((torch.norm(v_b[:, 0:2], dim=-1) / lin_sigma) ** 2) - ((torch.norm(w_b, dim=-1) / ang_sigma) ** 2))
    gate = torch.clamp(_catch_gate(env) + _hold_gate(env), 0.0, 1.0)
    return motion_score * gate * (0.35 + 0.65 * _mass_norm(env))


def catch_target_region_reward(env: "ManagerBasedRLEnv", sigma: float = 0.28) -> torch.Tensor:
    obj = env.scene["object"]
    dist = torch.norm(obj.data.root_pos_w - _chest_hold_target(env), dim=-1)
    return torch.exp(-((dist / sigma) ** 2)) * _catch_gate(env)


def upper_body_receive_reward(env: "ManagerBasedRLEnv", sigma: float = 0.26) -> torch.Tensor:
    obj = env.scene["object"]
    left_elbow = _body_pos(env, "left_elbow_link")
    right_elbow = _body_pos(env, "right_elbow_link")
    left_wrist = _body_pos(env, "left_wrist_roll_link")
    right_wrist = _body_pos(env, "right_wrist_roll_link")

    d_left = torch.minimum(
        torch.norm(obj.data.root_pos_w - left_elbow, dim=-1),
        torch.norm(obj.data.root_pos_w - left_wrist, dim=-1),
    )
    d_right = torch.minimum(
        torch.norm(obj.data.root_pos_w - right_elbow, dim=-1),
        torch.norm(obj.data.root_pos_w - right_wrist, dim=-1),
    )
    return torch.exp(-((d_left / sigma) ** 2)) * torch.exp(-((d_right / sigma) ** 2)) * _catch_gate(env)


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
    torso_hit = (_sensor_force_mag(env, sensor_name_torso) > thr).float() * 2.0
    bilateral_gate = (left_hits > 0.0) & (right_hits > 0.0)
    max_possible = float(len(sensor_names_left) + len(sensor_names_right)) + 2.0
    contact_score = (left_hits + right_hits + torso_hit) / max_possible
    return bilateral_gate.float() * (contact_score ** 2.0) * _catch_gate(env)


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
    return torch.relu(peak - force_thr) / force_thr * _toss_active(env)


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
