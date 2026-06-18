from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import (
    EP_NO_TOSS,
    EP_PASS_BY,
    catchable_or_hold_phase,
    get_task_state,
    hand_side_errors,
    hold_condition,
    hold_quality_terms,
    post_catch_phase,
    projected_gravity,
    reaction_window,
    ready_pose_error,
    update_action_history,
    _get_robot,
    _object_pos_quat_vel_w,
    _object_rel_in_root_frame,
    _root_ang_vel_w,
    _root_lin_vel_w,
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


def post_catch_action_rate_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    active = post_catch_phase(env).to(torch.float32)
    action, prev, _prev_prev = update_action_history(env)
    diff = action - prev
    return active * torch.mean(diff * diff, dim=-1)


def post_catch_base_ang_vel_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    active = post_catch_phase(env).to(torch.float32)
    robot = _get_robot(env)
    return active * _safe_norm(_root_ang_vel_w(robot)) ** 2


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
    left_err, right_err, _lat, _front = hand_side_errors(env)
    close_to_box = torch.exp(-((left_err + right_err) ** 2) / 0.22)
    no_toss_or_pass = (state.episode_type == EP_NO_TOSS) | (state.episode_type == EP_PASS_BY)
    return no_toss_or_pass.to(torch.float32) * close_to_box


# -----------------------------------------------------------------------------
# Timing and whole-body catching.
# -----------------------------------------------------------------------------
def reaction_timing_reward(env: "ManagerBasedRLEnv", target_ttc: float = 0.32, sigma: float = 0.23) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    dist = _safe_norm(err)
    closing = -torch.sum(err * rel_vel_b, dim=-1) / torch.clamp(dist, min=1e-3)
    ttc = dist / torch.clamp(closing, min=1e-3)
    score = torch.exp(-((ttc - float(target_ttc)) ** 2) / (float(sigma) ** 2))
    return reaction_window(env).to(torch.float32) * score


def hand_side_proximity_reward(env: "ManagerBasedRLEnv", sigma: float = 0.20) -> torch.Tensor:
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    active = _phase_active_float(env)
    left_score = torch.exp(-(left_err * left_err) / (float(sigma) ** 2))
    right_score = torch.exp(-(right_err * right_err) / (float(sigma) ** 2))
    return active * left_score * right_score * (0.35 + 0.65 * lateral_order) * (0.35 + 0.65 * front_ok)


def hug_symmetry_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    active = _phase_active_float(env)
    symmetry = torch.exp(-((left_err - right_err) ** 2) / 0.08)
    close = torch.exp(-((left_err + right_err) ** 2) / 0.42)
    return active * symmetry * close * lateral_order * front_ok


def hug_pocket_reward(env: "ManagerBasedRLEnv", sigma_pos: float = 0.23, sigma_hands: float = 0.34) -> torch.Tensor:
    active = _phase_active_float(env)
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang_vel_b = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    pos_score = torch.exp(-(_safe_norm(err) ** 2) / (float(sigma_pos) ** 2))
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    hand_score = torch.exp(-((left_err + right_err) ** 2) / (float(sigma_hands) ** 2))
    too_far_front = rel_pos_b[:, 0] - state.target_anchor_b[:, 0]
    depth_score = torch.exp(-(torch.clamp(too_far_front, min=0.0) ** 2) / (0.16 * 0.16))
    return active * pos_score * hand_score * lateral_order * front_ok * depth_score


def whole_body_absorption_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    active = _phase_active_float(env)
    return active * hold_posture_reward(env, sigma=0.90)


# -----------------------------------------------------------------------------
# Object reception and stabilization.
# -----------------------------------------------------------------------------
def object_anchor_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    pos_quality, _lin_quality, _ang_quality, z_ok = hold_quality_terms(env)
    active = _phase_active_float(env)
    return active * pos_quality * (0.35 + 0.65 * z_ok)


def object_velocity_damping_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _pos_quality, lin_quality, ang_quality, _z_ok = hold_quality_terms(env)
    active = _phase_active_float(env)
    return active * lin_quality * ang_quality


def object_height_safety_reward(env: "ManagerBasedRLEnv", min_z: float = 0.58, sigma: float = 0.16) -> torch.Tensor:
    obj_pos, _q, _v, _w = _object_pos_quat_vel_w(env)
    active = _phase_active_float(env)
    height_err = torch.clamp(float(min_z) - obj_pos[:, 2], min=0.0)
    return active * torch.exp(-(height_err * height_err) / (float(sigma) ** 2))


def successful_hold_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return hold_condition(env).squeeze(-1).to(torch.float32)


def sustained_hold_reward(env: "ManagerBasedRLEnv", scale_steps: float = 50.0) -> torch.Tensor:
    state = get_task_state(env)
    return torch.clamp(state.hold_counter.to(torch.float32) / float(scale_steps), 0.0, 1.0)


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


__all__ = [
    "alive_bonus",
    "upright_reward",
    "root_height_reward",
    "base_motion_penalty",
    "joint_velocity_penalty",
    "action_rate_penalty",
    "action_acceleration_penalty",
    "action_magnitude_penalty",
    "ready_posture_reward",
    "hold_posture_reward",
    "idle_until_reaction_reward",
    "early_arm_motion_penalty",
    "reaction_timing_reward",
    "hand_side_proximity_reward",
    "hug_symmetry_reward",
    "hug_pocket_reward",
    "whole_body_absorption_reward",
    "object_anchor_reward",
    "object_velocity_damping_reward",
    "object_height_safety_reward",
    "successful_hold_reward",
    "sustained_hold_reward",
    "object_drop_penalty",
    "object_escape_penalty",
    "no_toss_contact_like_penalty",
    "post_catch_action_rate_penalty",
    "post_catch_base_ang_vel_penalty",
]
