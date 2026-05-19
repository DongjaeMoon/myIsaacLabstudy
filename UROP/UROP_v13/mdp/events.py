from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg
from .observations import get_controlled_joint_indices, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = scene_objects_cfg.OBJECT_BASE_SIZE


def _get_stage(env: "ManagerBasedRLEnv") -> int:
    params = env.cfg.curriculum.stage_schedule.params
    forced = int(params.get("eval_stage", -1))
    if forced >= 0:
        return forced

    step = int(env.common_step_counter)
    nsteps = int(params["num_steps_per_env"])
    s0_limit = int(params["stage0_iters"]) * nsteps
    s1_limit = s0_limit + int(params["stage1_iters"]) * nsteps
    s2_limit = s1_limit + int(params["stage2_iters"]) * nsteps

    if step < s0_limit:
        return 0
    if step < s1_limit:
        return 1
    if step < s2_limit:
        return 2
    return 3


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dim=-1,
    )


def _ensure_urop_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype

    if not hasattr(env, "_urop_toss_done"):
        env._urop_toss_done = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_toss_active"):
        env._urop_toss_active = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_toss_wait_s"):
        env._urop_toss_wait_s = torch.full((n, 1), 999.0, device=d, dtype=dtype)
    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((n, 2), device=d, dtype=dtype)
    if not hasattr(env, "_urop_ready_joint_pos"):
        robot = env.scene["robot"]
        env._urop_ready_joint_pos = robot.data.default_joint_pos.clone()
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d, dtype=dtype)

    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d, dtype=dtype).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), scene_objects_cfg.OBJECT_DEFAULT_MASS, device=d, dtype=dtype)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.8, device=d, dtype=dtype)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.02, device=d, dtype=dtype)
    if not hasattr(env, "_urop_robot_contact_friction"):
        env._urop_robot_contact_friction = torch.full((n, 1), 0.85, device=d, dtype=dtype)
    if not hasattr(env, "_urop_floor_friction"):
        env._urop_floor_friction = torch.full((n, 1), 0.90, device=d, dtype=dtype)

    if not hasattr(env, "_urop_obj_filter_pos"):
        env._urop_obj_filter_pos = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_filter_vel"):
        env._urop_obj_filter_vel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos"):
        env._urop_obj_obs_pos = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel"):
        env._urop_obj_obs_vel = torch.zeros((n, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_visible"):
        env._urop_obj_visible = torch.zeros((n, 1), device=d, dtype=torch.bool)
    if not hasattr(env, "_urop_obj_obs_alpha"):
        env._urop_obj_obs_alpha = torch.full((n, 1), 0.65, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_drop_prob"):
        env._urop_obj_obs_drop_prob = torch.zeros((n, 1), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_pos_noise_std"):
        env._urop_obj_obs_pos_noise_std = torch.full((n, 1), 0.01, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_vel_noise_std"):
        env._urop_obj_obs_vel_noise_std = torch.full((n, 1), 0.05, device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_cache_global_step"):
        env._urop_obj_obs_cache_global_step = -1
    if not hasattr(env, "_urop_obj_obs_cache_episode_len"):
        env._urop_obj_obs_cache_episode_len = torch.full((n,), -1, device=d, dtype=torch.long)


def _apply_physx_mass_best_effort(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass: torch.Tensor,
) -> None:
    obj = env.scene["object"]
    try:
        view = obj.root_physx_view
    except Exception:
        return

    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    try:
        masses = view.get_masses().clone()
        if getattr(masses, "is_cuda", False):
            masses = masses.cpu()
        masses[env_ids_cpu, 0] = mass.detach().to(device="cpu").squeeze(-1)
        try:
            view.set_masses(masses, indices=env_ids_cpu)
        except TypeError:
            view.set_masses(masses)
    except Exception:
        pass


def _apply_physx_material_best_effort(
    view,
    env_ids: torch.Tensor,
    friction: torch.Tensor,
    restitution: torch.Tensor,
) -> None:
    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    try:
        mats = view.get_material_properties().clone()
        if getattr(mats, "is_cuda", False):
            mats = mats.cpu()
        fr = friction.detach().to(device="cpu").squeeze(-1).view(-1, 1)
        rs = restitution.detach().to(device="cpu").squeeze(-1).view(-1, 1)
        mats[env_ids_cpu, :, 0] = fr
        mats[env_ids_cpu, :, 1] = 0.92 * fr
        mats[env_ids_cpu, :, 2] = rs
        try:
            view.set_material_properties(mats, indices=env_ids_cpu)
        except TypeError:
            view.set_material_properties(mats)
    except Exception:
        pass


def randomize_receive_object(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass_range=(2.0, 4.5),
    friction_range=(0.55, 1.00),
    restitution_range=(0.00, 0.06),
    size_scale_range=(0.95, 1.05),
    apply_physx: bool = True,
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    mass = torch.empty((n, 1), device=d).uniform_(*mass_range)
    friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)
    scale = torch.empty((n, 3), device=d).uniform_(*size_scale_range)
    size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1) * scale

    env._urop_box_mass[env_ids] = mass
    env._urop_box_friction[env_ids] = friction
    env._urop_box_restitution[env_ids] = restitution
    env._urop_box_size[env_ids] = size

    if apply_physx:
        _apply_physx_mass_best_effort(env, env_ids, mass)
        try:
            _apply_physx_material_best_effort(env.scene["object"].root_physx_view, env_ids, friction, restitution)
        except Exception:
            pass


def randomize_robot_contact_material(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    friction_range=(0.70, 1.00),
    restitution_range=(0.00, 0.02),
    floor_friction_range=(0.75, 1.05),
    apply_physx: bool = True,
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    robot_friction = torch.empty((n, 1), device=d).uniform_(*friction_range)
    robot_restitution = torch.empty((n, 1), device=d).uniform_(*restitution_range)
    floor_friction = torch.empty((n, 1), device=d).uniform_(*floor_friction_range)

    env._urop_robot_contact_friction[env_ids] = robot_friction
    env._urop_floor_friction[env_ids] = floor_friction

    if apply_physx:
        try:
            _apply_physx_material_best_effort(env.scene["robot"].root_physx_view, env_ids, robot_friction, robot_restitution)
        except Exception:
            pass
    # Ground is shared across envs in the default scene, so we keep a best-effort sampled buffer here.


def randomize_object_observation(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pos_noise_range=(0.004, 0.020),
    vel_noise_range=(0.02, 0.12),
    drop_prob_range=(0.00, 0.08),
    alpha_range=(0.35, 0.85),
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    env._urop_obj_obs_pos_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*pos_noise_range)
    env._urop_obj_obs_vel_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*vel_noise_range)
    env._urop_obj_obs_drop_prob[env_ids] = torch.empty((n, 1), device=d).uniform_(*drop_prob_range)
    env._urop_obj_obs_alpha[env_ids] = torch.empty((n, 1), device=d).uniform_(*alpha_range)


def _sample_wait_times(env: "ManagerBasedRLEnv", num_envs: int, stage: int, wait_time_ranges: dict) -> torch.Tensor:
    d = env.device
    key = f"stage{stage}"
    if stage <= 0 or key not in wait_time_ranges:
        return torch.full((num_envs, 1), 999.0, device=d, dtype=env.scene["robot"].data.root_pos_w.dtype)
    low, high = wait_time_ranges[key]
    return torch.empty((num_envs, 1), device=d, dtype=env.scene["robot"].data.root_pos_w.dtype).uniform_(low, high)


def _sample_joint_noise(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    joint_noise: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_noise = joint_noise or {}
    pos_ranges = {
        "lower_body": tuple(joint_noise.get("lower_body_pos", (-0.03, 0.03))),
        "waist": tuple(joint_noise.get("waist_pos", (-0.02, 0.02))),
        "arm": tuple(joint_noise.get("arm_pos", (-0.05, 0.05))),
        "wrist": tuple(joint_noise.get("wrist_pos", (-0.03, 0.03))),
    }
    vel_range = tuple(joint_noise.get("velocity", (-0.08, 0.08)))

    n = int(env_ids.shape[0])
    d = env.device
    pos_min = torch.empty((scene_objects_cfg.EXPECTED_ACTION_DIM,), device=d)
    pos_max = torch.empty((scene_objects_cfg.EXPECTED_ACTION_DIM,), device=d)

    for index, name in enumerate(scene_objects_cfg.CONTROLLED_JOINT_NAMES):
        if "wrist" in name:
            rng = pos_ranges["wrist"]
        elif "shoulder" in name or "elbow" in name:
            rng = pos_ranges["arm"]
        elif "waist" in name:
            rng = pos_ranges["waist"]
        else:
            rng = pos_ranges["lower_body"]
        pos_min[index] = rng[0]
        pos_max[index] = rng[1]

    pos_noise = torch.rand((n, scene_objects_cfg.EXPECTED_ACTION_DIM), device=d) * (pos_max - pos_min) + pos_min
    vel_noise = torch.empty((n, scene_objects_cfg.EXPECTED_ACTION_DIM), device=d).uniform_(*vel_range)
    return pos_noise, vel_noise


def reset_autonomous_episode(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    park: dict,
    wait_time_ranges: dict,
    joint_noise: dict | None = None,
    root_xy_range=(-0.02, 0.02),
    root_yaw_range=(-0.05, 0.05),
    object_randomization: dict | None = None,
    robot_material_randomization: dict | None = None,
    floor_material_randomization: dict | None = None,
    observation_randomization: dict | None = None,
) -> None:
    _ensure_urop_buffers(env)

    object_randomization = object_randomization or {}
    robot_material_randomization = robot_material_randomization or {}
    floor_material_randomization = floor_material_randomization or {}
    observation_randomization = observation_randomization or {}

    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    n = int(env_ids.shape[0])

    stage = _get_stage(env)

    env._urop_toss_done[env_ids] = 0
    env._urop_toss_active[env_ids] = False
    env._urop_toss_wait_s[env_ids] = _sample_wait_times(env, n, stage, wait_time_ranges)
    env._urop_hold_latched[env_ids] = False
    env._urop_hold_steps[env_ids] = 0
    env._urop_hold_anchor_xy[env_ids] = 0.0
    if hasattr(env, "_urop_hold_cache_global_step"):
        env._urop_hold_cache_global_step = -1
    if hasattr(env, "_urop_hold_cache_episode_len"):
        env._urop_hold_cache_episode_len[env_ids] = -1
    env._urop_obj_filter_pos[env_ids] = 0.0
    env._urop_obj_filter_vel[env_ids] = 0.0
    env._urop_obj_obs_pos[env_ids] = 0.0
    env._urop_obj_obs_vel[env_ids] = 0.0
    env._urop_obj_visible[env_ids] = False
    env._urop_obj_obs_cache_global_step = -1
    env._urop_obj_obs_cache_episode_len[env_ids] = -1

    env._urop_ready_joint_pos[env_ids] = robot.data.default_joint_pos[env_ids].clone()

    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]

    xy_noise = torch.empty((n, 2), device=d).uniform_(*root_xy_range)
    yaw_noise = torch.empty((n,), device=d).uniform_(*root_yaw_range)
    yaw_quat = _quat_from_euler_xyz(
        torch.zeros(n, device=d),
        torch.zeros(n, device=d),
        yaw_noise,
    )

    root_pos = default_root_state[:, 0:3]
    root_pos[:, 0:2] += xy_noise
    root_quat = quat_mul(default_root_state[:, 3:7], yaw_quat)
    root_vel = torch.zeros((n, 6), device=d)

    robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    controlled_ids = get_controlled_joint_indices(env)
    pos_noise, vel_noise = _sample_joint_noise(env, env_ids, joint_noise)
    joint_pos[:, controlled_ids] += pos_noise
    joint_vel[:, controlled_ids] = vel_noise

    joint_pos_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = robot.data.soft_joint_vel_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=env_ids)
    env._urop_spawn_xy[env_ids] = root_pos[:, 0:2]

    randomize_receive_object(env, env_ids, **object_randomization)
    randomize_robot_contact_material(
        env,
        env_ids,
        friction_range=robot_material_randomization.get("friction_range", (0.70, 1.00)),
        restitution_range=robot_material_randomization.get("restitution_range", (0.00, 0.02)),
        floor_friction_range=floor_material_randomization.get("friction_range", (0.75, 1.05)),
        apply_physx=robot_material_randomization.get("apply_physx", True),
    )
    randomize_object_observation(env, env_ids, **observation_randomization)

    rel_p = torch.stack(
        [
            torch.empty(n, device=d).uniform_(*park["pos_x"]),
            torch.empty(n, device=d).uniform_(*park["pos_y"]),
            torch.empty(n, device=d).uniform_(*park["pos_z"]),
        ],
        dim=-1,
    )
    pos_w = root_pos + quat_apply(_yaw_quat(root_quat), rel_p)
    quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=d).repeat(n, 1)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=d), env_ids=env_ids)


def _select_toss_cfg(stage: int, stage1: dict, stage2: dict, stage3: dict, prob_stage1: float, prob_stage2: float, prob_stage3: float) -> tuple[dict, float]:
    if stage <= 1:
        return stage1, float(prob_stage1)
    if stage == 2:
        return stage2, float(prob_stage2)
    return stage3, float(prob_stage3)


def toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage1: dict,
    stage2: dict,
    stage3: dict,
    throw_prob_stage1: float = 1.0,
    throw_prob_stage2: float = 1.0,
    throw_prob_stage3: float = 1.0,
    max_throws_per_episode: int = 1,
) -> None:
    _ensure_urop_buffers(env)

    if max_throws_per_episode <= 0:
        return

    stage = _get_stage(env)
    if stage == 0:
        return

    if env_ids.numel() == 0:
        return

    episode_time_s = env.episode_length_buf[env_ids].float().unsqueeze(-1) * float(env.step_dt)
    due = episode_time_s >= env._urop_toss_wait_s[env_ids]
    due &= (~env._urop_toss_active[env_ids]).unsqueeze(-1)
    due &= (env._urop_toss_done[env_ids] < int(max_throws_per_episode)).unsqueeze(-1)

    if not torch.any(due):
        return

    ids_due = env_ids[due.squeeze(-1)]
    cfg, throw_prob = _select_toss_cfg(
        stage,
        stage1,
        stage2,
        stage3,
        throw_prob_stage1,
        throw_prob_stage2,
        throw_prob_stage3,
    )

    if throw_prob < 1.0:
        keep = torch.rand(ids_due.shape[0], device=env.device) < throw_prob
        ids_throw = ids_due[keep]
        ids_skip = ids_due[~keep]
        if ids_skip.numel() > 0:
            env._urop_toss_done[ids_skip] = int(max_throws_per_episode)
            env._urop_toss_active[ids_skip] = False
        if ids_throw.numel() == 0:
            return
    else:
        ids_throw = ids_due

    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = env.device
    n = int(ids_throw.shape[0])

    root_pos = robot.data.root_pos_w[ids_throw]
    root_yaw = _yaw_quat(robot.data.root_quat_w[ids_throw])

    rel_p = torch.stack(
        [
            torch.empty(n, device=d).uniform_(*cfg["pos_x"]),
            torch.empty(n, device=d).uniform_(*cfg["pos_y"]),
            torch.empty(n, device=d).uniform_(*cfg["pos_z"]),
        ],
        dim=-1,
    )
    rel_v = torch.stack(
        [
            torch.empty(n, device=d).uniform_(*cfg["vel_x"]),
            torch.empty(n, device=d).uniform_(*cfg["vel_y"]),
            torch.empty(n, device=d).uniform_(*cfg["vel_z"]),
        ],
        dim=-1,
    )

    roll = torch.empty(n, device=d).uniform_(*cfg.get("roll", (-0.03, 0.03)))
    pitch = torch.empty(n, device=d).uniform_(*cfg.get("pitch", (-0.04, 0.04)))
    yaw = torch.empty(n, device=d).uniform_(*cfg.get("yaw", (-0.06, 0.06)))
    quat_w = _quat_from_euler_xyz(roll, pitch, yaw)

    ang_vel = torch.stack(
        [
            torch.empty(n, device=d).uniform_(*cfg.get("ang_vel_x", (-0.20, 0.20))),
            torch.empty(n, device=d).uniform_(*cfg.get("ang_vel_y", (-0.20, 0.20))),
            torch.empty(n, device=d).uniform_(*cfg.get("ang_vel_z", (-0.30, 0.30))),
        ],
        dim=-1,
    )

    pos_w = root_pos + quat_apply(root_yaw, rel_p)
    vel_w = robot.data.root_lin_vel_w[ids_throw] + quat_apply(root_yaw, rel_v)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=ids_throw)
    vel6 = torch.zeros((n, 6), device=d)
    vel6[:, 0:3] = vel_w
    vel6[:, 3:6] = ang_vel
    obj.write_root_velocity_to_sim(vel6, env_ids=ids_throw)

    env._urop_toss_done[ids_throw] += 1
    env._urop_toss_active[ids_throw] = True
    env._urop_obj_obs_cache_global_step = -1
    env._urop_obj_obs_cache_episode_len[ids_throw] = -1
