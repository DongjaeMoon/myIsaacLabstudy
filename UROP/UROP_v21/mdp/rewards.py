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
    projected_gravity,
    reaction_window,
    ready_pose_error,
    update_action_history,
    _get_object,
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


# -----------------------------------------------------------------------------
# Basic balance and regularization.
# -----------------------------------------------------------------------------
def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def upright_reward(env: "ManagerBasedRLEnv", sigma: float = 0.22) -> torch.Tensor:
    # Isaac Lab projected gravity in an upright base is approximately [0, 0, -1].
    g = projected_gravity(env, noise_std=0.0)
    tilt_error = torch.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + (g[:, 2] + 1.0) ** 2)
    return torch.exp(-(tilt_error * tilt_error) / (sigma * sigma))


def root_height_reward(env: "ManagerBasedRLEnv", target_z: float = 0.78, sigma: float = 0.10) -> torch.Tensor:
    robot = _get_robot(env)
    root_pos, _root_quat = _root_pos_quat(robot)
    return torch.exp(-((root_pos[:, 2] - float(target_z)) ** 2) / (float(sigma) ** 2))


def base_motion_penalty(env: "ManagerBasedRLEnv", lin_scale: float = 0.15, ang_scale: float = 0.05) -> torch.Tensor:
    robot = _get_robot(env)
    lin = _safe_norm(_root_lin_vel_w(robot)[:, :2])
    # Keep this mild: the policy may need some torso/base motion to absorb the box.
    return float(lin_scale) * lin * lin + float(ang_scale) * _safe_norm(_root_ang_vel_w(robot)) ** 2


def joint_velocity_penalty(env: "ManagerBasedRLEnv", scale: float = 0.0025) -> torch.Tensor:
    robot = _get_robot(env)
    try:
        ids, _ = robot.find_joints(list(scene_objects_cfg.CONTROLLED_JOINT_NAMES), preserve_order=True)
        ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        qd = robot.data.joint_vel[:, ids]
    except Exception:
        qd = robot.data.joint_vel[:, : scene_objects_cfg.EXPECTED_ACTION_DIM]
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


# -----------------------------------------------------------------------------
# Idle / false-positive behavior.
# -----------------------------------------------------------------------------
def ready_posture_reward(env: "ManagerBasedRLEnv", sigma: float = 0.55) -> torch.Tensor:
    err = ready_pose_error(env, scene_objects_cfg.READY_POSE)
    return torch.exp(-torch.mean(err * err, dim=-1) / (float(sigma) ** 2))


def hold_posture_reward(env: "ManagerBasedRLEnv", sigma: float = 0.75) -> torch.Tensor:
    err = ready_pose_error(env, scene_objects_cfg.HOLD_POSE)
    return torch.exp(-torch.mean(err * err, dim=-1) / (float(sigma) ** 2))


def idle_until_reaction_reward(env: "ManagerBasedRLEnv", sigma: float = 0.42) -> torch.Tensor:
    """Reward staying quiet before the box is actually catchable.

    This directly targets the failure mode where the policy raises the arms at
    reset because it learned a prior that a box always comes.
    """
    state = get_task_state(env)
    q_err = ready_pose_error(env, scene_objects_cfg.READY_POSE)
    arm_err = q_err[:, 15:]
    action, _prev, _prev_prev = update_action_history(env)
    arm_action = action[:, 15:]
    not_reaction = ~reaction_window(env)
    false_positive = (state.episode_type == EP_NO_TOSS) | (state.episode_type == EP_PASS_BY) | (~state.has_released)
    mask = (not_reaction & false_positive).to(torch.float32)
    pose_score = torch.exp(-torch.mean(arm_err * arm_err, dim=-1) / (float(sigma) ** 2))
    action_score = torch.exp(-torch.mean(arm_action * arm_action, dim=-1) / 0.25)
    return mask * pose_score * action_score


def early_arm_motion_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    action, _prev, _prev_prev = update_action_history(env)
    arm_action = action[:, 15:]
    q_err = ready_pose_error(env, scene_objects_cfg.READY_POSE)[:, 15:]
    no_need_to_catch = ((state.episode_type == EP_NO_TOSS) | (state.episode_type == EP_PASS_BY) | (~state.has_released)) & (~reaction_window(env))
    return no_need_to_catch.to(torch.float32) * (torch.mean(arm_action * arm_action, dim=-1) + 0.35 * torch.mean(q_err * q_err, dim=-1))


# -----------------------------------------------------------------------------
# Timing and whole-body catching.
# -----------------------------------------------------------------------------
def reaction_timing_reward(env: "ManagerBasedRLEnv", target_ttc: float = 0.34, sigma: float = 0.28) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    dist_x = rel_pos_b[:, 0] - state.target_anchor_b[:, 0]
    closing = -rel_vel_b[:, 0]
    ttc = dist_x / torch.clamp(closing, min=1e-3)
    in_window = reaction_window(env)
    score = torch.exp(-((ttc - float(target_ttc)) ** 2) / (float(sigma) ** 2))
    return in_window.to(torch.float32) * score


def hand_side_proximity_reward(env: "ManagerBasedRLEnv", sigma: float = 0.22) -> torch.Tensor:
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    active = (reaction_window(env) | catchable_or_hold_phase(env)).to(torch.float32)
    left_score = torch.exp(-(left_err * left_err) / (float(sigma) ** 2))
    right_score = torch.exp(-(right_err * right_err) / (float(sigma) ** 2))
    return active * left_score * right_score * (0.5 + 0.5 * lateral_order) * (0.5 + 0.5 * front_ok)


def hug_symmetry_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    active = (reaction_window(env) | catchable_or_hold_phase(env)).to(torch.float32)
    symmetry = torch.exp(-((left_err - right_err) ** 2) / 0.08)
    close = torch.exp(-((left_err + right_err) ** 2) / 0.42)
    return active * symmetry * close * lateral_order * front_ok


def whole_body_absorption_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Small whole-body shaping toward a stable arms-hugging posture after release.

    This is intentionally weak compared to object stabilization; it prevents the
    policy from solving the task with only wrist/hand cheating while still letting
    RL discover a natural catch motion.
    """
    active = catchable_or_hold_phase(env).to(torch.float32)
    return active * hold_posture_reward(env, sigma=0.90)


# -----------------------------------------------------------------------------
# Object reception and stabilization.
# -----------------------------------------------------------------------------
def object_anchor_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    pos_quality, _lin_quality, _ang_quality, z_ok = hold_quality_terms(env)
    active = catchable_or_hold_phase(env).to(torch.float32)
    return active * pos_quality * (0.4 + 0.6 * z_ok)


def object_velocity_damping_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _pos_quality, lin_quality, ang_quality, _z_ok = hold_quality_terms(env)
    active = catchable_or_hold_phase(env).to(torch.float32)
    return active * lin_quality * ang_quality


def object_height_safety_reward(env: "ManagerBasedRLEnv", min_z: float = 0.58, sigma: float = 0.18) -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, _q, _v, _w = _object_pos_quat_vel_w(env)
    active = catchable_or_hold_phase(env).to(torch.float32)
    height_err = torch.clamp(float(min_z) - obj_pos[:, 2], min=0.0)
    return active * torch.exp(-(height_err * height_err) / (float(sigma) ** 2))


def successful_hold_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    hold = hold_condition(env).squeeze(-1)
    return hold.to(torch.float32)


def sustained_hold_reward(env: "ManagerBasedRLEnv", scale_steps: float = 50.0) -> torch.Tensor:
    # The termination term owns the hold-counter update to avoid double-counting
    # when both rewards and terminations are evaluated in the same environment step.
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


def no_toss_contact_like_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for hugging/closing in no-toss episodes without using contact force."""
    state = get_task_state(env)
    left_err, right_err, _lateral_order, _front_ok = hand_side_errors(env)
    close_to_box = torch.exp(-((left_err + right_err) ** 2) / 0.24)
    no_toss = state.episode_type == EP_NO_TOSS
    return no_toss.to(torch.float32) * close_to_box


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
    "whole_body_absorption_reward",
    "object_anchor_reward",
    "object_velocity_damping_reward",
    "object_height_safety_reward",
    "successful_hold_reward",
    "sustained_hold_reward",
    "object_drop_penalty",
    "object_escape_penalty",
    "no_toss_contact_like_penalty",
]
