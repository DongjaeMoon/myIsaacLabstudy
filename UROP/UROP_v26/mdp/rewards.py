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
    near_catch_window,
    post_catch_phase,
    projected_gravity,
    reaction_window,
    ready_pose_error,
    tag_visible,
    update_action_history,
    _hand_positions_root_frame,
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


def _controlled_qd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = _get_robot(env)
    try:
        ids, _ = robot.find_joints(list(scene_objects_cfg.CONTROLLED_JOINT_NAMES), preserve_order=True)
        ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
        return robot.data.joint_vel[:, ids]
    except Exception:
        return robot.data.joint_vel[:, : scene_objects_cfg.EXPECTED_ACTION_DIM]


def _lower_body_slice(x: torch.Tensor) -> torch.Tensor:
    return x[:, : len(scene_objects_cfg.LOWER_BODY_JOINT_NAMES)]


# -----------------------------------------------------------------------------
# Compact v26 rewards. Fewer terms, each responsible for a distinct failure mode.
# -----------------------------------------------------------------------------
def alive_bonus(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def stand_balance_reward(
    env: "ManagerBasedRLEnv",
    target_z: float = 0.78,
    sigma_tilt: float = 0.22,
    sigma_height: float = 0.10,
    sigma_motion: float = 0.55,
) -> torch.Tensor:
    """Composite standing reward: upright, correct height, low base motion.

    v26 deliberately makes standing robust as important as catching. This single
    term replaces several smaller balance rewards to reduce manager overhead.
    """
    robot = _get_robot(env)
    g = projected_gravity(env, noise_std=0.0)
    tilt_error = torch.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + (g[:, 2] + 1.0) ** 2)
    upright = torch.exp(-(tilt_error * tilt_error) / (float(sigma_tilt) ** 2))
    root_pos, _root_quat = _root_pos_quat(robot)
    height = torch.exp(-((root_pos[:, 2] - float(target_z)) ** 2) / (float(sigma_height) ** 2))
    root_motion = _safe_norm(_root_lin_vel_w(robot)[:, :2]) + 0.55 * _safe_norm(_root_ang_vel_w(robot))
    quiet = torch.exp(-(root_motion * root_motion) / (float(sigma_motion) ** 2))
    return upright * height * quiet


def idle_ready_stand_reward(env: "ManagerBasedRLEnv", sigma_pose: float = 0.42) -> torch.Tensor:
    """Reward quiet catch-ready standing whenever the object is not catchable.

    This includes tag-invisible/no-object-like time, no-toss, pass-by, pre-release,
    and delayed-toss waiting. It directly targets the real deployment situation in
    which the policy may run for a long time with tag_visible=0.
    """
    state = get_task_state(env)
    q_err = ready_pose_error(env, scene_objects_cfg.READY_POSE)
    arm_err = q_err[:, 15:]
    action, _prev, _prev_prev = update_action_history(env)
    arm_action = action[:, 15:]
    lower_action = _lower_body_slice(action)
    not_catchable = ~catchable_or_hold_phase(env)
    false_positive = (state.episode_type == EP_NO_TOSS) | (state.episode_type == EP_PASS_BY) | (~state.has_released)
    invisible = tag_visible(env).squeeze(-1) < 0.5
    mask = (not_catchable | false_positive | invisible).to(torch.float32)
    pose_score = torch.exp(-torch.mean(arm_err * arm_err, dim=-1) / (float(sigma_pose) ** 2))
    action_score = torch.exp(-torch.mean(arm_action * arm_action, dim=-1) / 0.18)
    lower_score = torch.exp(-torch.mean(lower_action * lower_action, dim=-1) / 0.12)
    robot = _get_robot(env)
    base_quiet = torch.exp(-(_safe_norm(_root_lin_vel_w(robot)[:, :2]) ** 2) / 0.025 - (_safe_norm(_root_ang_vel_w(robot)) ** 2) / 0.16)
    return mask * pose_score * action_score * lower_score * base_quiet


def pre_catch_arm_motion_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    state = get_task_state(env)
    action, _prev, _prev_prev = update_action_history(env)
    q_err = ready_pose_error(env, scene_objects_cfg.READY_POSE)
    arm_action = action[:, 15:]
    arm_pose = q_err[:, 15:]
    invisible = tag_visible(env).squeeze(-1) < 0.5
    no_need = (((state.episode_type == EP_NO_TOSS) | (state.episode_type == EP_PASS_BY) | (~state.has_released) | invisible) & (~reaction_window(env)))
    return no_need.to(torch.float32) * (torch.mean(arm_action * arm_action, dim=-1) + 0.35 * torch.mean(arm_pose * arm_pose, dim=-1))


def catch_timing_reward(env: "ManagerBasedRLEnv", target_ttc: float = 0.26, sigma_ttc: float = 0.18) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    flight_target = getattr(state, "flight_target_b", state.target_anchor_b)
    err = rel_pos_b - flight_target
    closing = -rel_vel_b[:, 0]
    ttc = err[:, 0] / torch.clamp(closing, min=1e-3)
    timing = torch.exp(-((ttc - float(target_ttc)) ** 2) / (float(sigma_ttc) ** 2))
    return reaction_window(env).to(torch.float32) * timing


def hug_catch_reward(env: "ManagerBasedRLEnv", sigma_hand: float = 0.28, sigma_anchor: float = 0.24) -> torch.Tensor:
    """Single geometric catch/hug reward.

    Combines: object near torso pocket, hands on left/right sides, and not holding
    the box too far out in front. It replaces the many overlapping v22 hug terms.
    """
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    active = (reaction_window(env) | near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    anchor_score = torch.exp(-torch.sum(err * err, dim=-1) / (float(sigma_anchor) ** 2))
    hand_score = torch.exp(-((left_err + right_err) ** 2) / (float(sigma_hand) ** 2))
    too_far_front = torch.clamp(rel_pos_b[:, 0] - state.target_anchor_b[:, 0], min=0.0)
    depth_score = torch.exp(-(too_far_front * too_far_front) / (0.12 * 0.12))
    return active * anchor_score * hand_score * (0.5 + 0.5 * lateral_order) * (0.5 + 0.5 * front_ok) * depth_score


def hand_approach_reward(env: "ManagerBasedRLEnv", sigma: float = 0.34) -> torch.Tensor:
    """Reward active bracketing motion during reaction/intercept.

    The object now flies through a front intercept target instead of directly into
    the final hold pocket.  This term pays for decreasing the left/right side-anchor
    errors so the robot visibly reaches in and hugs rather than waiting passively.
    """
    state = get_task_state(env)
    active = (reaction_window(env) | near_catch_window(env)).to(torch.float32)
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    cur = torch.stack((left_err, right_err), dim=-1)
    prev_sum = torch.sum(state.last_hand_dist, dim=-1)
    cur_sum = torch.sum(cur, dim=-1)
    valid_prev = prev_sum > 1.0e-5
    improvement = torch.clamp(prev_sum - cur_sum, min=0.0, max=0.18)
    state.last_hand_dist = cur.detach()
    side_score = torch.exp(-(cur_sum * cur_sum) / (float(sigma) ** 2))
    bracket = 0.55 + 0.45 * lateral_order * front_ok
    return active * valid_prev.to(torch.float32) * bracket * (0.65 * side_score + 5.5 * improvement)


def pull_to_hold_reward(env: "ManagerBasedRLEnv", sigma_dist: float = 0.36) -> torch.Tensor:
    """Reward moving the object from the front intercept zone toward the final hold pocket."""
    state = get_task_state(env)
    rel_pos_b, rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    err = rel_pos_b - state.target_anchor_b
    dist = _safe_norm(err)
    toward_anchor_dir = -err / torch.clamp(dist.unsqueeze(-1), min=1.0e-5)
    toward_speed = torch.sum(rel_vel_b * toward_anchor_dir, dim=-1)
    left_err, right_err, lateral_order, front_ok = hand_side_errors(env)
    hand_gate = torch.exp(-((left_err + right_err) ** 2) / (0.42 * 0.42)) * (0.55 + 0.45 * lateral_order * front_ok)
    active = (near_catch_window(env) | post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    progress = torch.clamp(toward_speed, min=0.0, max=0.80)
    dist_score = torch.exp(-(dist * dist) / (float(sigma_dist) ** 2))
    return active * hand_gate * (0.65 * dist_score + 0.55 * progress)

def object_stability_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Object reception/stabilization with one reward call."""
    pos_quality, lin_quality, ang_quality, z_ok = hold_quality_terms(env)
    active = catchable_or_hold_phase(env).to(torch.float32)
    return active * (0.52 * pos_quality + 0.30 * lin_quality * ang_quality + 0.18 * z_ok)


def successful_hold_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return hold_condition(env).squeeze(-1).to(torch.float32)


def sustained_hold_reward(env: "ManagerBasedRLEnv", scale_steps: float = 45.0) -> torch.Tensor:
    state = get_task_state(env)
    return torch.clamp(state.hold_counter.to(torch.float32) / float(scale_steps), 0.0, 1.0)


def drop_escape_penalty(env: "ManagerBasedRLEnv", drop_z: float = 0.30, max_dist: float = 2.35) -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, _obj_q, _obj_vel, _obj_w = _object_pos_quat_vel_w(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    valid = state.has_released & (state.episode_type != EP_NO_TOSS)
    dropped = (obj_pos[:, 2] < float(drop_z)) & valid
    escaped = ((_safe_norm(rel_pos_b) > float(max_dist)) | (rel_pos_b[:, 0] < -0.45)) & valid
    return dropped.to(torch.float32) + escaped.to(torch.float32)


def smooth_action_penalty(
    env: "ManagerBasedRLEnv",
    rate_scale: float = 1.0,
    accel_scale: float = 0.55,
    mag_scale: float = 0.16,
) -> torch.Tensor:
    action, prev, prev_prev = update_action_history(env)
    rate = torch.mean((action - prev) ** 2, dim=-1)
    accel = torch.mean((action - 2.0 * prev + prev_prev) ** 2, dim=-1)
    mag = torch.mean(action * action, dim=-1)
    return float(rate_scale) * rate + float(accel_scale) * accel + float(mag_scale) * mag


def post_catch_stillness_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Strong anti-tremor penalty after the box enters the pocket."""
    active = (post_catch_phase(env) | hold_condition(env).squeeze(-1).to(torch.bool)).to(torch.float32)
    robot = _get_robot(env)
    action, prev, _prev_prev = update_action_history(env)
    lower_rate = torch.mean((_lower_body_slice(action - prev)) ** 2, dim=-1)
    lower_qd = torch.mean((_lower_body_slice(_controlled_qd(env))) ** 2, dim=-1)
    base_ang = _safe_norm(_root_ang_vel_w(robot)) ** 2
    base_lin = _safe_norm(_root_lin_vel_w(robot)[:, :2]) ** 2
    return active * (1.20 * lower_rate + 0.030 * lower_qd + 0.45 * base_ang + 0.28 * base_lin)


__all__ = [
    "alive_bonus",
    "stand_balance_reward",
    "idle_ready_stand_reward",
    "pre_catch_arm_motion_penalty",
    "catch_timing_reward",
    "hand_approach_reward",
    "hug_catch_reward",
    "pull_to_hold_reward",
    "object_stability_reward",
    "successful_hold_reward",
    "sustained_hold_reward",
    "drop_escape_penalty",
    "smooth_action_penalty",
    "post_catch_stillness_penalty",
]
