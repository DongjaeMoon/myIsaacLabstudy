from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import (
    EP_NO_TOSS,
    EP_PASS_BY,
    catchable_or_hold_phase,
    chest_pocket_terms,
    get_task_state,
    hand_side_errors,
    hold_condition,
    hold_quality_terms,
    hug_geometry_terms,
    near_catch_window,
    post_catch_phase,
    projected_gravity,
    reaction_window,
    ready_pose_error,
    update_action_history,
    _get_robot,
    _object_pos_quat_vel_w,
    _object_rel_in_root_frame,
    _root_lin_vel_w,
    _root_ang_vel_w,
    _root_pos_quat,
    _safe_norm,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _controlled_qd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = _get_robot(env)
    try:
        ids, _ = robot.find_joints(list(scene_objects_cfg.CONTROLLED_JOINT_NAMES), preserve_order=True)
        ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        return robot.data.joint_vel[:, ids]
    except Exception:
        return robot.data.joint_vel[:, : scene_objects_cfg.EXPECTED_ACTION_DIM]


def _phase_active_float(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return catchable_or_hold_phase(env).to(torch.float32)


def _idle_phase(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return ~catchable_or_hold_phase(env)


# -----------------------------------------------------------------------------
# Basic balance and regularization.
# -----------------------------------------------------------------------------
def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_reward(env: "ManagerBasedRLEnv", sigma: float = 0.22) -> torch.Tensor:
    g = projected_gravity(env, noise_std=0.0)
    tilt_error = torch.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + (g[:, 2] + 1.0) ** 2)
    return torch.exp(-(tilt_error * tilt_error) / (sigma * sigma))


def root_height_reward(env: "ManagerBasedRLEnv", target_z: float = 0.78, sigma: float = 0.10) -> torch.Tensor:
    robot = _get_robot(env)
    root_pos, _root_quat = _root_pos_quat(robot)
    return torch.exp(-((root_pos[:, 2] - float(target_z)) ** 2) / (float(sigma) ** 2))


def base_motion_penalty(env: "ManagerBasedRLEnv", lin_scale: float = 0.16, ang_scale: float = 0.07) -> torch.Tensor:
    robot = _get_robot(env)
    lin = _safe_norm(_root_lin_vel_w(robot)[:, :2])
    return float(lin_scale) * lin * lin + float(ang_scale) * _safe_norm(_root_ang_vel_w(robot)) ** 2


def joint_velocity_penalty(env: "ManagerBasedRLEnv", scale: float = 0.0025) -> torch.Tensor:
    qd = _controlled_qd(env)
    return float(scale) * torch.sum(qd * qd, dim=-1)


def lower_body_joint_velocity_penalty(env: "ManagerBasedRLEnv", scale: float = 0.010) -> torch.Tensor:
    qd = _controlled_qd(env)[:, :15]
    post = post_catch_phase(env).to(torch.float32)
    return float(scale) * (0.35 + 1.65 * post) * torch.sum(qd * qd, dim=-1)


def action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, prev, _prev_prev = update_action_history(env)
    diff = action - prev
    return torch.mean(diff * diff, dim=-1)


def action_acceleration_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, prev, prev_prev = update_action_history(env)
    accel = action - 2.0 * prev + prev_prev
    return torch.mean(accel * accel, dim=-1)


def action_magnitude_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, _prev, _prev_prev = update_action_history(env)
    return torch.mean(action * action, dim=-1)


def lower_body_action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, prev, _prev_prev = update_action_history(env)
    diff = action[:, :15] - prev[:, :15]
    post = post_catch_phase(env).to(torch.float32)
    return (0.5 + 2.0 * post) * torch.mean(diff * diff, dim=-1)


def post_catch_action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, prev, _prev_prev = update_action_history(env)
    diff = action - prev
    return post_catch_phase(env).to(torch.float32) * torch.mean(diff * diff, dim=-1)


def base_ang_vel_post_catch_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = _get_robot(env)
    return post_catch_phase(env).to(torch.float32) * _safe_norm(_root_ang_vel_w(robot)) ** 2


def post_catch_tremor_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    qd = _controlled_qd(env)
    action_rate = action_rate_penalty(env)
    lower_vel = torch.mean(qd[:, :15] * qd[:, :15], dim=-1)
    return post_catch_phase(env).to(torch.float32) * (lower_vel + 2.0 * action_rate)


# -----------------------------------------------------------------------------
# Idle / false-positive behavior.
# -----------------------------------------------------------------------------
def ready_posture_reward(env: "ManagerBasedRLEnv", sigma: float = 0.50) -> torch.Tensor:
    err = ready_pose_error(env, scene_objects_cfg.READY_POSE)
    return torch.exp(-torch.mean(err * err, dim=-1) / (float(sigma) ** 2))


def hold_posture_reward(env: "ManagerBasedRLEnv", sigma: float = 0.62) -> torch.Tensor:
    err = ready_pose_error(env, scene_objects_cfg.HOLD_POSE)
    arm_err = err[:, 12:]
    return torch.exp(-torch.mean(arm_err * arm_err, dim=-1) / (float(sigma) ** 2))


def idle_until_reaction_reward(env: "ManagerBasedRLEnv", sigma: float = 0.38) -> torch.Tensor:
    idle = _idle_phase(env).to(torch.float32)
    q_err = ready_pose_error(env, scene_objects_cfg.READY_POSE)
    arm_err = q_err[:, 15:]
    action, _prev, _prev_prev = update_action_history(env)
    arm_action = action[:, 15:]
    pose_score = torch.exp(-torch.mean(arm_err * arm_err, dim=-1) / (float(sigma) ** 2))
    action_score = torch.exp(-torch.mean(arm_action * arm_action, dim=-1) / 0.18)
    return idle * pose_score * action_score


def early_arm_motion_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    action, _prev, _prev_prev = update_action_history(env)
    arm_action = action[:, 15:]
    q_err = ready_pose_error(env, scene_objects_cfg.READY_POSE)[:, 15:]
    idle = _idle_phase(env).to(torch.float32)
    return idle * (torch.mean(arm_action * arm_action, dim=-1) + 0.55 * torch.mean(q_err * q_err, dim=-1))


def no_toss_contact_like_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    left_err, right_err, _lat, _depth = hand_side_errors(env)
    close_to_box = torch.exp(-((left_err + right_err) ** 2) / 0.22)
    no_toss_or_pass = (state.episode_type == EP_NO_TOSS) | (state.episode_type == EP_PASS_BY)
    return no_toss_or_pass.to(torch.float32) * close_to_box


# -----------------------------------------------------------------------------
# Timing and whole-body catching.
# -----------------------------------------------------------------------------
def reaction_timing_reward(env: "ManagerBasedRLEnv", target_ttc: float = 0.32, sigma: float = 0.23) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    dist_x = rel_pos_b[:, 0] - state.target_anchor_b[:, 0]
    closing = -rel_vel_b[:, 0]
    ttc = dist_x / torch.clamp(closing, min=1e-3)
    score = torch.exp(-((ttc - float(target_ttc)) ** 2) / (float(sigma) ** 2))
    return reaction_window(env).to(torch.float32) * score


def hand_side_proximity_reward(env: "ManagerBasedRLEnv", sigma: float = 0.20) -> torch.Tensor:
    left_err, right_err, lateral_order, depth_order = hand_side_errors(env)
    active = _phase_active_float(env)
    left_score = torch.exp(-(left_err * left_err) / (float(sigma) ** 2))
    right_score = torch.exp(-(right_err * right_err) / (float(sigma) ** 2))
    return active * left_score * right_score * (0.35 + 0.65 * lateral_order) * (0.35 + 0.65 * depth_order)


def hug_symmetry_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    bracket, symmetry, lateral_order, depth_order = hug_geometry_terms(env)
    active = _phase_active_float(env)
    return active * bracket * symmetry * (0.35 + 0.65 * lateral_order) * (0.35 + 0.65 * depth_order)


def hug_bracket_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    bracket, _sym, _lat, _depth = hug_geometry_terms(env)
    return _phase_active_float(env) * bracket


def chest_pocket_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    x_quality, y_quality, z_quality, not_far_forward = chest_pocket_terms(env)
    active = (near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    return active * x_quality * y_quality * z_quality * not_far_forward


def hug_depth_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    bracket, symmetry, _lateral_order, depth_order = hug_geometry_terms(env)
    x_quality, _y, _z, not_far_forward = chest_pocket_terms(env)
    active = (near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    return active * bracket * symmetry * depth_order * x_quality * not_far_forward


def elbow_wrap_reward(env: "ManagerBasedRLEnv", sigma: float = 0.48) -> torch.Tensor:
    active = (near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    err = ready_pose_error(env, scene_objects_cfg.HOLD_POSE)
    arm_err = err[:, 15:]
    elbow_err = torch.stack((err[:, 18], err[:, 25]), dim=-1)
    arm_score = torch.exp(-torch.mean(arm_err * arm_err, dim=-1) / (float(sigma) ** 2))
    elbow_score = torch.exp(-torch.mean(elbow_err * elbow_err, dim=-1) / (0.38 * 0.38))
    return active * (0.45 * arm_score + 0.55 * elbow_score)


def elbow_flexion_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return elbow_wrap_reward(env)


def whole_body_absorption_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    active = (near_catch_window(env) | post_catch_phase(env)).to(torch.float32)
    qd = _controlled_qd(env)
    lower_quiet = torch.exp(-torch.mean(qd[:, :15] * qd[:, :15], dim=-1) / (1.30 * 1.30))
    return active * hold_posture_reward(env, sigma=0.78) * (0.35 + 0.65 * lower_quiet)


# -----------------------------------------------------------------------------
# Object reception and stabilization.
# -----------------------------------------------------------------------------
def object_anchor_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    pos_quality, _lin_quality, _ang_quality, z_ok = hold_quality_terms(env)
    active = (near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    return active * pos_quality * (0.35 + 0.65 * z_ok)


def object_velocity_damping_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _pos_quality, lin_quality, ang_quality, _z_ok = hold_quality_terms(env)
    active = (near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    return active * lin_quality * ang_quality


def object_height_safety_reward(env: "ManagerBasedRLEnv", min_z: float = 0.58, sigma: float = 0.16) -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, _q, _v, _w = _object_pos_quat_vel_w(env)
    active = (state.has_released & (state.episode_type != EP_NO_TOSS)).to(torch.float32)
    height_err = torch.clamp(float(min_z) - obj_pos[:, 2], min=0.0)
    return active * torch.exp(-(height_err * height_err) / (float(sigma) ** 2))


def successful_hold_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return hold_condition(env).squeeze(-1).to(torch.float32)


def sustained_hold_reward(env: "ManagerBasedRLEnv", scale_steps: float = 60.0) -> torch.Tensor:
    state = get_task_state(env)
    return torch.clamp(state.hold_counter.to(torch.float32) / float(scale_steps), 0.0, 1.0)


def post_catch_stillness_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    post = post_catch_phase(env).to(torch.float32)
    qd = _controlled_qd(env)
    robot = _get_robot(env)
    action_rate = action_rate_penalty(env)
    lower_quiet = torch.exp(-torch.mean(qd[:, :15] * qd[:, :15], dim=-1) / (0.85 * 0.85))
    base_quiet = torch.exp(-_safe_norm(_root_ang_vel_w(robot)) / 0.85)
    action_quiet = torch.exp(-action_rate / 0.025)
    return post * lower_quiet * base_quiet * action_quiet


def stable_hug_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return post_catch_stillness_reward(env) * (0.5 + 0.5 * successful_hold_reward(env))


# -----------------------------------------------------------------------------
# Failure penalties.
# -----------------------------------------------------------------------------
def object_drop_penalty(env: "ManagerBasedRLEnv", drop_z: float = 0.32) -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, _obj_q, _obj_vel, _obj_w = _object_pos_quat_vel_w(env)
    dropped = (obj_pos[:, 2] < float(drop_z)) & state.has_released & (state.episode_type != EP_NO_TOSS)
    return dropped.to(torch.float32)


def object_escape_penalty(env: "ManagerBasedRLEnv", max_dist: float = 2.25) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    escaped = (_safe_norm(rel_pos_b) > float(max_dist)) & state.has_released & (state.episode_type != EP_NO_TOSS)
    behind = (rel_pos_b[:, 0] < -0.45) & state.has_released
    return (escaped | behind).to(torch.float32)


def object_far_forward_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    state = get_task_state(env)
    forward_excess = torch.clamp(rel_pos_b[:, 0] - (state.target_anchor_b[:, 0] + 0.25), min=0.0)
    active = (near_catch_window(env) | post_catch_phase(env)).to(torch.float32)
    return active * forward_excess * forward_excess


def front_shelf_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return object_far_forward_penalty(env)


__all__ = [name for name in globals() if not name.startswith("_")]
