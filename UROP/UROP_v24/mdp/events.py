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
    joint_pos_noise: float = 0.008,
    joint_vel_noise: float = 0.010,
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
    if hasattr(state, "tag_on_step"):
        state.tag_on_step[env_ids] = 0
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
        (0.86, 0.06, 0.04, 0.03, 0.01),
        (0.74, 0.10, 0.07, 0.06, 0.03),
        (0.62, 0.14, 0.10, 0.09, 0.05),
        (0.52, 0.18, 0.13, 0.11, 0.06),
        (0.46, 0.20, 0.15, 0.12, 0.07),
    ),
    release_time_range_s: tuple[float, float] = (0.42, 1.15),
    delayed_release_time_range_s: tuple[float, float] = (1.25, 2.65),
    already_flying_prob_by_stage: tuple[float, float, float, float, float] = (0.02, 0.04, 0.06, 0.08, 0.10),
    # v24 visibility curriculum: some tosses start with zero object observation
    # until shortly before/after release, so the actor learns a true no-tag
    # catch_ready behavior instead of relying on a visible pre-release box.
    blind_until_release_prob_by_stage: tuple[float, float, float, float, float] = (0.28, 0.34, 0.40, 0.46, 0.52),
    tag_intro_time_range_s: tuple[float, float] = (0.0, 0.55),
    tag_release_margin_range_s: tuple[float, float] = (-0.10, 0.18),
    no_toss_late_tag_prob: float = 0.35,
    no_toss_late_tag_time_range_s: tuple[float, float] = (0.75, 2.60),
    sender_x_range: tuple[float, float] = (1.05, 1.65),
    sender_y_range: tuple[float, float] = (-0.28, 0.28),
    sender_z_rel_range: tuple[float, float] = (0.18, 0.46),
    # Fallback flight-time sampler used only when trajectory_mode="ballistic_time".
    # For real parcel handover training, use trajectory_mode="low_arc" so the
    # box travels on a low, human-like toss instead of a high lob.
    arrival_time_range_s: tuple[float, float] = (0.38, 0.58),
    trajectory_mode: str = "low_arc",
    toss_apex_clearance_range: tuple[float, float] = (0.14, 0.34),
    release_velocity_noise_xy: float = 0.05,
    release_velocity_noise_z: float = 0.04,
    max_release_speed: float = 4.50,
    release_ang_velocity_range: tuple[float, float] = (-1.20, 1.20),
    demo_release_ang_velocity_range: tuple[float, float] = (-0.20, 0.20),
    target_noise_xyz: tuple[float, float, float] = (0.07, 0.11, 0.09),
    object_size_scale_range: tuple[float, float] = (0.80, 1.25),
    object_mass_range: tuple[float, float] = (1.0, 4.0),
    object_friction_range: tuple[float, float] = (0.55, 1.15),
    object_restitution_range: tuple[float, float] = (0.00, 0.12),
    obs_noise_scale_range: tuple[float, float] = (0.60, 1.35),
    tag_available_prob: float = 0.96,
    demo_mode: bool = False,
    demo_release_time_s: float = 0.65,
    demo_arrival_time_s: float = 0.72,
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
    if bool(demo_mode):
        ep_type = torch.full((m,), EP_TOSS, device=device, dtype=torch.long)
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
    if bool(demo_mode):
        state.object_size[env_ids] = base_size.repeat(m, 1)
        state.object_mass[env_ids] = torch.full((m, 1), float(scene_objects_cfg.OBJECT_DEFAULT_MASS), device=device)
        state.object_friction[env_ids] = torch.full((m, 1), 0.85, device=device)
        state.object_restitution[env_ids] = torch.zeros(m, 1, device=device)
        state.obs_noise_scale[env_ids] = torch.zeros(m, 1, device=device)
        state.push_scale[env_ids] = torch.zeros(m, 1, device=device)

    # Target anchor in root/body frame: around upper torso / arms-hugging region.
    anchor_b = _as_tensor(HOLD_ANCHOR_B, env).view(1, 3).repeat(m, 1)
    if bool(demo_mode):
        anchor_b[:, 0] += 0.015
        anchor_b[:, 2] += 0.015
    else:
        anchor_b[:, 0] += torch.empty(m, device=device).uniform_(-0.04, 0.06)
        anchor_b[:, 1] += torch.empty(m, device=device).uniform_(-0.035, 0.035)
        anchor_b[:, 2] += torch.empty(m, device=device).uniform_(-0.035, 0.050)
    state.target_anchor_b[env_ids] = anchor_b

    # Sender/thrower pose in root frame. The box can be visible before release.
    sender_rel_b = torch.zeros(m, 3, device=device)
    sender_rel_b[:, 0] = torch.empty(m, device=device).uniform_(*sender_x_range)
    sender_rel_b[:, 1] = torch.empty(m, device=device).uniform_(*sender_y_range)
    sender_rel_b[:, 2] = torch.empty(m, device=device).uniform_(*sender_z_rel_range)
    if bool(demo_mode):
        sender_rel_b[:, 0] = 1.35
        sender_rel_b[:, 1] = 0.0
        sender_rel_b[:, 2] = 0.34
    held_w = root_pos[env_ids] + _quat_rotate(root_quat[env_ids], sender_rel_b)
    state.held_position_w[env_ids] = held_w

    # Random initial object orientation.
    roll = torch.empty(m, device=device).uniform_(-0.35, 0.35)
    pitch = torch.empty(m, device=device).uniform_(-0.30, 0.30)
    yaw = torch.empty(m, device=device).uniform_(-0.55, 0.55)
    if bool(demo_mode):
        roll = torch.empty(m, device=device).uniform_(-0.08, 0.08)
        pitch = torch.empty(m, device=device).uniform_(-0.06, 0.06)
        yaw = torch.empty(m, device=device).uniform_(-0.10, 0.10)
    obj_quat_w = _quat_from_euler_xyz(roll, pitch, yaw)
    state.held_quat_w[env_ids] = obj_quat_w

    # Arrival target. Pass-by/near-miss episodes are intentionally not centered.
    noise = torch.empty(m, 3, device=device).uniform_(-1.0, 1.0)
    noise_scale = torch.tensor(target_noise_xyz, device=device).view(1, 3)
    target_b = anchor_b + noise * noise_scale
    if bool(demo_mode):
        target_b = anchor_b.clone()
        target_b[:, 0] += 0.03
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

    gravity_mag = 9.81
    gravity = torch.tensor((0.0, 0.0, -gravity_mag), device=device).view(1, 3)

    # v24 gentle-toss trajectory model.  The previous time-based formula sampled
    # long arrival times, which mathematically forces a large upward velocity
    # and creates unrealistic high lobs.  Low-arc mode samples the apex height
    # directly, so randomization is expressed as "how much above the hands/chest"
    # the box rises, which matches a parcel handover toss much better.
    mode = str(trajectory_mode).lower()
    if mode in ("low_arc", "gentle", "parcel", "parcel_toss"):
        clearance = torch.empty(m, 1, device=device).uniform_(*toss_apex_clearance_range)
        if bool(demo_mode):
            lo, hi = toss_apex_clearance_range
            demo_clearance = 0.5 * (float(lo) + float(hi))
            clearance = torch.full((m, 1), demo_clearance, device=device)

        held_z = held_w[:, 2:3]
        target_z = target_w[:, 2:3]
        apex_z = torch.maximum(held_z, target_z) + clearance

        # Vertical launch speed needed to reach the sampled low apex.
        vz0 = torch.sqrt(torch.clamp(2.0 * gravity_mag * (apex_z - held_z), min=1.0e-6))
        # Positive root of: target_z = held_z + vz0*t - 0.5*g*t^2.
        disc = torch.clamp(vz0 * vz0 - 2.0 * gravity_mag * (target_z - held_z), min=1.0e-6)
        flight_t = (vz0 + torch.sqrt(disc)) / gravity_mag
        flight_t = torch.clamp(flight_t, min=0.28, max=0.78)

        release_vel = torch.zeros(m, 3, device=device)
        release_vel[:, 0:2] = (target_w[:, 0:2] - held_w[:, 0:2]) / flight_t
        release_vel[:, 2:3] = vz0
        if bool(demo_mode):
            release_vel[:, 0:2] += torch.empty(m, 2, device=device).uniform_(-0.015, 0.015)
            release_vel[:, 2:3] += torch.empty(m, 1, device=device).uniform_(-0.010, 0.010)
        else:
            release_vel[:, 0:2] += torch.empty(m, 2, device=device).uniform_(
                -float(release_velocity_noise_xy), float(release_velocity_noise_xy)
            )
            release_vel[:, 2:3] += torch.empty(m, 1, device=device).uniform_(
                -float(release_velocity_noise_z), float(release_velocity_noise_z)
            )
    else:
        # Legacy ballistic-time mode: sample arrival time and solve the velocity.
        # Keep the range short if this mode is used; long times produce high lobs.
        arrival_t = torch.empty(m, 1, device=device).uniform_(*arrival_time_range_s)
        if bool(demo_mode):
            arrival_t = torch.full((m, 1), float(demo_arrival_time_s), device=device)
        release_vel = (target_w - held_w - 0.5 * gravity * arrival_t * arrival_t) / arrival_t
        if bool(demo_mode):
            release_vel += torch.empty(m, 3, device=device).uniform_(-0.015, 0.015)
        else:
            release_vel += torch.empty(m, 3, device=device).uniform_(-0.05, 0.05)

    max_speed = max(float(max_release_speed), 0.1)
    speed = torch.linalg.norm(release_vel, dim=-1, keepdim=True)
    speed_scale = torch.clamp(max_speed / torch.clamp(speed, min=1.0e-6), max=1.0)
    release_vel = release_vel * speed_scale

    state.release_velocity_w[env_ids] = release_vel
    state.release_ang_velocity_w[env_ids] = torch.empty(m, 3, device=device).uniform_(*release_ang_velocity_range)
    if bool(demo_mode):
        state.release_ang_velocity_w[env_ids] = torch.empty(m, 3, device=device).uniform_(*demo_release_ang_velocity_range)

    # Release timing. Delayed and no-toss episodes explicitly break timing priors.
    rel_time = torch.empty(m, device=device).uniform_(*release_time_range_s)
    delayed_time = torch.empty(m, device=device).uniform_(*delayed_release_time_range_s)
    rel_time = torch.where(ep_type == EP_DELAYED_TOSS, delayed_time, rel_time)
    if bool(demo_mode):
        rel_time = torch.full((m,), float(demo_release_time_s), device=device)
    rel_step = torch.ceil(rel_time / dt).to(torch.long)
    rel_step = torch.where(ep_type == EP_NO_TOSS, torch.full_like(rel_step, 10_000), rel_step)

    stage = torch.clamp(state.curriculum_stage[env_ids], max=len(already_flying_prob_by_stage) - 1)
    already_prob = torch.tensor(already_flying_prob_by_stage, device=device)[stage]
    already_flying = (torch.rand(m, device=device) < already_prob) & (ep_type != EP_NO_TOSS) & (ep_type != EP_DELAYED_TOSS)
    if bool(demo_mode):
        already_flying = torch.zeros(m, device=device, dtype=torch.bool)
    rel_step = torch.where(already_flying, torch.zeros_like(rel_step), rel_step)
    state.release_step[env_ids] = rel_step
    state.has_released[env_ids] = already_flying
    state.ever_released[env_ids] = already_flying

    state.tag_available[env_ids] = torch.rand(m, device=device) < float(tag_available_prob)
    # Earliest observation step for the AprilTag.  The object can still exist in
    # simulation before this time, but the actor receives a zero object vector.
    early_intro_s = torch.empty(m, device=device).uniform_(*tag_intro_time_range_s)
    early_intro_step = torch.ceil(early_intro_s / dt).to(torch.long)
    stage_for_visibility = torch.clamp(state.curriculum_stage[env_ids], max=len(blind_until_release_prob_by_stage) - 1)
    blind_prob = torch.tensor(blind_until_release_prob_by_stage, device=device)[stage_for_visibility]
    valid_blind_type = (ep_type == EP_TOSS) | (ep_type == EP_DELAYED_TOSS) | (ep_type == EP_PASS_BY) | (ep_type == EP_NEAR_MISS)
    blind_until_release = (torch.rand(m, device=device) < blind_prob) & valid_blind_type
    release_margin_s = torch.empty(m, device=device).uniform_(*tag_release_margin_range_s)
    release_intro_step = torch.clamp(rel_step + torch.round(release_margin_s / dt).to(torch.long), min=0)
    tag_on_step = torch.where(blind_until_release, release_intro_step, early_intro_step)

    if bool(demo_mode):
        state.tag_available[env_ids] = True
        tag_on_step = torch.zeros(m, device=device, dtype=torch.long)

    # Keep many no-toss objects visible to teach "visible does not imply hug",
    # but delay some no-toss tags to train the real no-object/blank camera case.
    no_toss = ep_type == EP_NO_TOSS
    no_toss_count = int(no_toss.sum().item())
    if no_toss_count > 0:
        state.tag_available[env_ids[no_toss]] = torch.rand(no_toss_count, device=device) < max(float(tag_available_prob), 0.98)
        late_no_toss = torch.rand(no_toss_count, device=device) < float(no_toss_late_tag_prob)
        late_s = torch.empty(no_toss_count, device=device).uniform_(*no_toss_late_tag_time_range_s)
        late_step = torch.ceil(late_s / dt).to(torch.long)
        current_no_toss_steps = tag_on_step[no_toss]
        tag_on_step[no_toss] = torch.where(late_no_toss, late_step, current_no_toss_steps)

    if hasattr(state, "tag_on_step"):
        state.tag_on_step[env_ids] = tag_on_step

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
    robot_push_prob: float = 0.22,
    object_push_prob: float = 0.22,
    robot_lin_vel_xy_range: tuple[float, float] = (-0.14, 0.14),
    robot_ang_vel_z_range: tuple[float, float] = (-0.24, 0.24),
    object_lin_vel_range: tuple[float, float] = (-0.20, 0.20),
    object_ang_vel_range: tuple[float, float] = (-0.70, 0.70),
) -> None:
    """Apply sparse velocity impulses to train disturbance rejection."""
    env_ids = _env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    device = env_ids.device
    state = get_task_state(env)
    robot = _get_robot(env)
    obj = _get_object(env)

    # Robot base perturbation. Disturbances are curriculum-scaled: mild early, robust later.
    stage = torch.clamp(state.curriculum_stage[env_ids].to(torch.float32), 0.0, 4.0)
    stage_prob_scale = 0.25 + 0.75 * (stage / 4.0)
    mask = torch.rand(env_ids.numel(), device=device) < (float(robot_push_prob) * stage_prob_scale)
    if torch.any(mask):
        ids = env_ids[mask]
        root_pos, root_quat = _root_pos_quat(robot)
        vel = torch.cat((_root_lin_vel_w(robot)[ids], _root_ang_vel_w(robot)[ids]), dim=-1).clone()
        scale = state.push_scale[ids]
        vel[:, 0:2] += torch.empty(ids.numel(), 2, device=device).uniform_(*robot_lin_vel_xy_range) * scale
        vel[:, 5] += torch.empty(ids.numel(), device=device).uniform_(*robot_ang_vel_z_range) * scale.squeeze(-1)
        _write_root_state(robot, torch.cat((root_pos[ids], root_quat[ids]), dim=-1), vel, ids)

    # Object perturbation only after release; otherwise the hold event is intentionally controlling it.
    released_mask = state.has_released[env_ids] & (torch.rand(env_ids.numel(), device=device) < (float(object_push_prob) * stage_prob_scale))
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
