from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import (
    EP_DELAYED_TOSS,
    EP_NEAR_MISS,
    EP_NO_TOSS,
    EP_PASS_BY,
    EP_TOSS,
    HOLD_ANCHOR_B,
    get_task_state,
    update_action_history,
    _as_tensor,
    _env_ids,
    _episode_step,
    _get_env_origins,
    _get_object,
    _get_robot,
    _object_pos_quat_vel_w,
    _quat_from_euler_xyz,
    _quat_rotate,
    _root_ang_vel_w,
    _root_lin_vel_w,
    _root_pos_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Low-level write helpers.
# -----------------------------------------------------------------------------
def _write_root_state(asset, pose: torch.Tensor, velocity: torch.Tensor, env_ids: torch.Tensor) -> None:
    try:
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    except TypeError:
        asset.write_root_pose_to_sim(pose, env_ids)
    try:
        asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    except TypeError:
        asset.write_root_velocity_to_sim(velocity, env_ids)


def _write_joint_state(robot, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor) -> None:
    try:
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        return
    except Exception:
        pass
    try:
        robot.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
    except Exception:
        # Leave a clear runtime error in Isaac Lab instead of silently corrupting resets.
        raise RuntimeError("Could not write G1 joint state to simulation. Check Isaac Lab Articulation API version.")


def _set_joint_targets(robot, joint_pos: torch.Tensor, env_ids: torch.Tensor) -> None:
    for method_name in ("set_joint_position_target", "set_joint_pos_target"):
        method = getattr(robot, method_name, None)
        if method is not None:
            try:
                method(joint_pos, env_ids=env_ids)
                return
            except Exception:
                pass


def _reset_asset_buffers(asset, env_ids: torch.Tensor) -> None:
    try:
        asset.reset(env_ids)
    except Exception:
        try:
            asset.reset(env_ids=env_ids)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Reset events.
# -----------------------------------------------------------------------------
def reset_scene_to_default(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    reset_joint_targets: bool = True,
    joint_pos_noise: float = 0.015,
    joint_vel_noise: float = 0.020,
) -> None:
    """Reset G1 and object to deterministic defaults plus small joint noise.

    This term is deliberately conservative: object trajectory randomization is done by
    reset_autonomous_episode() immediately afterwards.
    """
    env_ids = _env_ids(env, env_ids)
    robot = _get_robot(env)
    obj = _get_object(env)
    origins = _get_env_origins(env)

    # Robot root.
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] += origins[env_ids]
    _write_root_state(robot, root_state[:, :7], root_state[:, 7:13] * 0.0, env_ids)

    # Robot joints: default catch-ready pose with small reset noise on controlled joints only.
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone() * 0.0
    try:
        controlled_ids, _ = robot.find_joints(list(scene_objects_cfg.CONTROLLED_JOINT_NAMES), preserve_order=True)
        controlled_ids = torch.as_tensor(controlled_ids, device=joint_pos.device, dtype=torch.long)
    except Exception:
        controlled_ids = torch.arange(scene_objects_cfg.EXPECTED_ACTION_DIM, device=joint_pos.device, dtype=torch.long)
    if joint_pos_noise > 0.0:
        joint_pos[:, controlled_ids] += torch.empty_like(joint_pos[:, controlled_ids]).uniform_(-joint_pos_noise, joint_pos_noise)
    if joint_vel_noise > 0.0:
        joint_vel[:, controlled_ids] += torch.empty_like(joint_vel[:, controlled_ids]).uniform_(-joint_vel_noise, joint_vel_noise)
    _write_joint_state(robot, joint_pos, joint_vel, env_ids)
    if reset_joint_targets:
        _set_joint_targets(robot, joint_pos, env_ids)

    # Object root; overwritten by reset_autonomous_episode(), but reset buffers here.
    obj_state = obj.data.default_root_state[env_ids].clone()
    obj_state[:, :3] += origins[env_ids]
    _write_root_state(obj, obj_state[:, :7], obj_state[:, 7:13] * 0.0, env_ids)

    _reset_asset_buffers(robot, env_ids)
    _reset_asset_buffers(obj, env_ids)

    state = get_task_state(env)
    state.hold_counter[env_ids] = 0
    state.drop_counter[env_ids] = 0
    state.action_initialized[env_ids] = False
    state.last_action[env_ids] = 0.0
    state.prev_action[env_ids] = 0.0
    state.prev_prev_action[env_ids] = 0.0
    state.action_history_step = -1


def _sample_episode_types(env: "ManagerBasedRLEnv", env_ids: torch.Tensor, stage_episode_probabilities) -> torch.Tensor:
    state = get_task_state(env)
    device = env_ids.device
    probs_by_stage = torch.tensor(stage_episode_probabilities, device=device, dtype=torch.float32)
    probs_by_stage = probs_by_stage / torch.clamp(probs_by_stage.sum(dim=-1, keepdim=True), min=1e-6)
    stage = torch.clamp(state.curriculum_stage[env_ids], min=0, max=probs_by_stage.shape[0] - 1)
    probs = probs_by_stage[stage]
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def reset_autonomous_episode(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    stage_episode_probabilities: tuple[tuple[float, float, float, float, float], ...] = (
        # toss, delayed_toss, no_toss, pass_by, near_miss
        (0.78, 0.08, 0.08, 0.04, 0.02),
        (0.66, 0.14, 0.10, 0.06, 0.04),
        (0.56, 0.18, 0.12, 0.08, 0.06),
        (0.48, 0.20, 0.14, 0.10, 0.08),
        (0.42, 0.22, 0.16, 0.12, 0.08),
    ),
    release_time_range_s: tuple[float, float] = (0.02, 0.30),
    delayed_release_time_range_s: tuple[float, float] = (0.65, 2.20),
    already_flying_prob_by_stage: tuple[float, float, float, float, float] = (0.20, 0.12, 0.08, 0.05, 0.03),
    sender_x_range: tuple[float, float] = (1.05, 1.85),
    sender_y_range: tuple[float, float] = (-0.34, 0.34),
    sender_z_rel_range: tuple[float, float] = (0.16, 0.52),
    arrival_time_range_s: tuple[float, float] = (0.42, 0.92),
    target_noise_xyz: tuple[float, float, float] = (0.10, 0.14, 0.12),
    object_size_scale_range: tuple[float, float] = (0.80, 1.25),
    object_mass_range: tuple[float, float] = (1.0, 4.0),
    object_friction_range: tuple[float, float] = (0.55, 1.15),
    object_restitution_range: tuple[float, float] = (0.00, 0.12),
    obs_noise_scale_range: tuple[float, float] = (0.75, 1.45),
    tag_available_prob: float = 0.94,
) -> None:
    """Sample the autonomous handover/toss scenario for each reset env.

    The object is often visible before being released, but no-toss/pass-by/near-miss
    episodes break the bad prior that "visible box always means catch now".
    """
    env_ids = _env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    device = env_ids.device
    m = env_ids.numel()
    state = get_task_state(env)
    robot = _get_robot(env)
    obj = _get_object(env)
    root_pos, root_quat = _root_pos_quat(robot)
    dt = float(getattr(env, "step_dt", 0.02))

    ep_type = _sample_episode_types(env, env_ids, stage_episode_probabilities)
    state.episode_type[env_ids] = ep_type

    # Per-episode physical/task parameters. The built-in Isaac Lab randomization
    # terms can alter actual mass/material; these buffers expose the sampled values
    # to the privileged critic and keep reward thresholds coherent.
    base_size = _as_tensor(scene_objects_cfg.OBJECT_BASE_SIZE, env).view(1, 3)
    size_scale = torch.empty(m, 1, device=device).uniform_(*object_size_scale_range)
    aspect = torch.empty(m, 3, device=device).uniform_(0.88, 1.12)
    state.object_size[env_ids] = base_size * size_scale * aspect
    state.object_mass[env_ids] = torch.empty(m, 1, device=device).uniform_(*object_mass_range)
    state.object_friction[env_ids] = torch.empty(m, 1, device=device).uniform_(*object_friction_range)
    state.object_restitution[env_ids] = torch.empty(m, 1, device=device).uniform_(*object_restitution_range)
    state.obs_noise_scale[env_ids] = torch.empty(m, 1, device=device).uniform_(*obs_noise_scale_range)
    state.push_scale[env_ids] = torch.empty(m, 1, device=device).uniform_(0.6, 1.6)

    # Target anchor in root/body frame: around upper torso / arms-hugging region.
    anchor_b = _as_tensor(HOLD_ANCHOR_B, env).view(1, 3).repeat(m, 1)
    anchor_b[:, 0] += torch.empty(m, device=device).uniform_(-0.04, 0.06)
    anchor_b[:, 1] += torch.empty(m, device=device).uniform_(-0.035, 0.035)
    anchor_b[:, 2] += torch.empty(m, device=device).uniform_(-0.035, 0.050)
    state.target_anchor_b[env_ids] = anchor_b

    # Sender/thrower pose in root frame. The box can be visible before release.
    sender_rel_b = torch.zeros(m, 3, device=device)
    sender_rel_b[:, 0] = torch.empty(m, device=device).uniform_(*sender_x_range)
    sender_rel_b[:, 1] = torch.empty(m, device=device).uniform_(*sender_y_range)
    sender_rel_b[:, 2] = torch.empty(m, device=device).uniform_(*sender_z_rel_range)
    held_w = root_pos[env_ids] + _quat_rotate(root_quat[env_ids], sender_rel_b)
    state.held_position_w[env_ids] = held_w

    # Random initial object orientation.
    roll = torch.empty(m, device=device).uniform_(-0.35, 0.35)
    pitch = torch.empty(m, device=device).uniform_(-0.30, 0.30)
    yaw = torch.empty(m, device=device).uniform_(-0.55, 0.55)
    obj_quat_w = _quat_from_euler_xyz(roll, pitch, yaw)
    state.held_quat_w[env_ids] = obj_quat_w

    # Arrival target. Pass-by/near-miss episodes are intentionally not centered.
    noise = torch.empty(m, 3, device=device).uniform_(-1.0, 1.0)
    noise_scale = torch.tensor(target_noise_xyz, device=device).view(1, 3)
    target_b = anchor_b + noise * noise_scale
    side_sign = torch.where(torch.rand(m, device=device) > 0.5, 1.0, -1.0)
    pass_mask = ep_type == EP_PASS_BY
    near_mask = ep_type == EP_NEAR_MISS
    pass_count = int(pass_mask.sum().item())
    if pass_count > 0:
        target_b[pass_mask, 1] += side_sign[pass_mask] * torch.empty(pass_count, device=device).uniform_(0.42, 0.82)
    near_count = int(near_mask.sum().item())
    if near_count > 0:
        high = torch.rand(near_count, device=device) > 0.5
        miss = torch.where(
            high,
            torch.empty(near_count, device=device).uniform_(0.22, 0.45),
            -torch.empty(near_count, device=device).uniform_(0.18, 0.34),
        )
        target_b[near_mask, 2] += miss
    target_w = root_pos[env_ids] + _quat_rotate(root_quat[env_ids], target_b)

    arrival_t = torch.empty(m, 1, device=device).uniform_(*arrival_time_range_s)
    gravity = torch.tensor((0.0, 0.0, -9.81), device=device).view(1, 3)
    release_vel = (target_w - held_w - 0.5 * gravity * arrival_t * arrival_t) / arrival_t
    release_vel += torch.empty(m, 3, device=device).uniform_(-0.12, 0.12)
    release_vel = torch.clamp(release_vel, -5.5, 5.5)
    state.release_velocity_w[env_ids] = release_vel
    state.release_ang_velocity_w[env_ids] = torch.empty(m, 3, device=device).uniform_(-2.6, 2.6)

    # Release timing. Delayed and no-toss episodes explicitly break timing priors.
    rel_time = torch.empty(m, device=device).uniform_(*release_time_range_s)
    delayed_time = torch.empty(m, device=device).uniform_(*delayed_release_time_range_s)
    rel_time = torch.where(ep_type == EP_DELAYED_TOSS, delayed_time, rel_time)
    rel_step = torch.ceil(rel_time / dt).to(torch.long)
    rel_step = torch.where(ep_type == EP_NO_TOSS, torch.full_like(rel_step, 10_000), rel_step)

    stage = torch.clamp(state.curriculum_stage[env_ids], max=len(already_flying_prob_by_stage) - 1)
    already_prob = torch.tensor(already_flying_prob_by_stage, device=device)[stage]
    already_flying = (torch.rand(m, device=device) < already_prob) & (ep_type != EP_NO_TOSS) & (ep_type != EP_DELAYED_TOSS)
    rel_step = torch.where(already_flying, torch.zeros_like(rel_step), rel_step)
    state.release_step[env_ids] = rel_step
    state.has_released[env_ids] = already_flying
    state.ever_released[env_ids] = already_flying

    state.tag_available[env_ids] = torch.rand(m, device=device) < float(tag_available_prob)
    # Keep no-toss objects usually visible; this is what teaches the policy not to hug just because it sees a tag.
    no_toss = ep_type == EP_NO_TOSS
    no_toss_count = int(no_toss.sum().item())
    if no_toss_count > 0:
        state.tag_available[env_ids[no_toss]] = torch.rand(no_toss_count, device=device) < max(float(tag_available_prob), 0.98)

    # Write initial object state.
    pose = torch.cat((held_w, obj_quat_w), dim=-1)
    vel = torch.zeros(m, 6, device=device)
    vel[already_flying, 0:3] = release_vel[already_flying]
    vel[already_flying, 3:6] = state.release_ang_velocity_w[env_ids][already_flying]
    _write_root_state(obj, pose, vel, env_ids)
    _reset_asset_buffers(obj, env_ids)

    state.hold_counter[env_ids] = 0
    state.drop_counter[env_ids] = 0
    state._obs_cache_step = -1
    update_action_history(env)


# -----------------------------------------------------------------------------
# Interval events.
# -----------------------------------------------------------------------------
def advance_autonomous_toss(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    hold_jitter_std: float = 0.003,
) -> None:
    """Hold the object until its sampled release step, then let it fly.

    This runs as an interval event at the policy step period. Without this term,
    delayed/no-toss boxes would simply fall under gravity before the release time.
    """
    env_ids = _env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    state = get_task_state(env)
    obj = _get_object(env)
    steps = _episode_step(env)[env_ids]

    not_released = ~state.has_released[env_ids]
    no_toss = state.episode_type[env_ids] == EP_NO_TOSS
    should_release = not_released & (~no_toss) & (steps >= state.release_step[env_ids])
    should_hold = not_released & (~should_release)

    if torch.any(should_hold):
        ids = env_ids[should_hold]
        held_pos = state.held_position_w[ids].clone()
        if hold_jitter_std > 0.0:
            held_pos += torch.randn_like(held_pos) * float(hold_jitter_std)
        held_quat = state.held_quat_w[ids]
        pose = torch.cat((held_pos, held_quat), dim=-1)
        vel = torch.zeros(ids.numel(), 6, device=ids.device)
        _write_root_state(obj, pose, vel, ids)

    if torch.any(should_release):
        ids = env_ids[should_release]
        pose = torch.cat((state.held_position_w[ids], state.held_quat_w[ids]), dim=-1)
        vel = torch.cat((state.release_velocity_w[ids], state.release_ang_velocity_w[ids]), dim=-1)
        _write_root_state(obj, pose, vel, ids)
        state.has_released[ids] = True
        state.ever_released[ids] = True

    state._obs_cache_step = -1


def random_push(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    robot_push_prob: float = 0.35,
    object_push_prob: float = 0.30,
    robot_lin_vel_xy_range: tuple[float, float] = (-0.20, 0.20),
    robot_ang_vel_z_range: tuple[float, float] = (-0.35, 0.35),
    object_lin_vel_range: tuple[float, float] = (-0.30, 0.30),
    object_ang_vel_range: tuple[float, float] = (-1.10, 1.10),
) -> None:
    """Apply sparse velocity impulses to train disturbance rejection."""
    env_ids = _env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    device = env_ids.device
    state = get_task_state(env)
    robot = _get_robot(env)
    obj = _get_object(env)

    # Robot base perturbation.
    mask = torch.rand(env_ids.numel(), device=device) < float(robot_push_prob)
    if torch.any(mask):
        ids = env_ids[mask]
        root_pos, root_quat = _root_pos_quat(robot)
        vel = torch.cat((_root_lin_vel_w(robot)[ids], _root_ang_vel_w(robot)[ids]), dim=-1).clone()
        scale = state.push_scale[ids]
        vel[:, 0:2] += torch.empty(ids.numel(), 2, device=device).uniform_(*robot_lin_vel_xy_range) * scale
        vel[:, 5] += torch.empty(ids.numel(), device=device).uniform_(*robot_ang_vel_z_range) * scale.squeeze(-1)
        _write_root_state(robot, torch.cat((root_pos[ids], root_quat[ids]), dim=-1), vel, ids)

    # Object perturbation only after release; otherwise the hold event is intentionally controlling it.
    released_mask = state.has_released[env_ids] & (torch.rand(env_ids.numel(), device=device) < float(object_push_prob))
    if torch.any(released_mask):
        ids = env_ids[released_mask]
        obj_pos, obj_quat, obj_lin, obj_ang = _object_pos_quat_vel_w(env)
        vel = torch.cat((obj_lin[ids], obj_ang[ids]), dim=-1).clone()
        vel[:, 0:3] += torch.empty(ids.numel(), 3, device=device).uniform_(*object_lin_vel_range)
        vel[:, 3:6] += torch.empty(ids.numel(), 3, device=device).uniform_(*object_ang_vel_range)
        _write_root_state(obj, torch.cat((obj_pos[ids], obj_quat[ids]), dim=-1), vel, ids)


__all__ = [
    "reset_scene_to_default",
    "reset_autonomous_episode",
    "advance_autonomous_toss",
    "random_push",
]
