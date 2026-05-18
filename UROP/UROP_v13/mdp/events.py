from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEFAULT_BOX_SIZE = (0.34, 0.26, 0.24)


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
    if not hasattr(env, "_urop_toss_done"):
        env._urop_toss_done = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_toss_active"):
        env._urop_toss_active = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((n, 2), device=d)
    if not hasattr(env, "_urop_ready_joint_pos"):
        robot = env.scene["robot"]
        env._urop_ready_joint_pos = robot.data.default_joint_pos.clone()
    if not hasattr(env, "_urop_hold_latched"):
        env._urop_hold_latched = torch.zeros(n, dtype=torch.bool, device=d)
    if not hasattr(env, "_urop_hold_steps"):
        env._urop_hold_steps = torch.zeros(n, dtype=torch.int32, device=d)
    if not hasattr(env, "_urop_hold_anchor_xy"):
        env._urop_hold_anchor_xy = torch.zeros((n, 2), device=d)

    if not hasattr(env, "_urop_box_size"):
        env._urop_box_size = torch.tensor(DEFAULT_BOX_SIZE, device=d).unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_box_mass"):
        env._urop_box_mass = torch.full((n, 1), 3.2, device=d)
    if not hasattr(env, "_urop_box_friction"):
        env._urop_box_friction = torch.full((n, 1), 0.8, device=d)
    if not hasattr(env, "_urop_box_restitution"):
        env._urop_box_restitution = torch.full((n, 1), 0.02, device=d)

    if not hasattr(env, "_urop_obj_obs_pos"):
        env._urop_obj_obs_pos = torch.zeros((n, 3), device=d)
    if not hasattr(env, "_urop_obj_obs_vel"):
        env._urop_obj_obs_vel = torch.zeros((n, 3), device=d)
    if not hasattr(env, "_urop_obj_obs_alpha"):
        env._urop_obj_obs_alpha = torch.ones((n, 1), device=d)
    if not hasattr(env, "_urop_obj_obs_drop_prob"):
        env._urop_obj_obs_drop_prob = torch.zeros((n, 1), device=d)
    if not hasattr(env, "_urop_obj_obs_pos_noise_std"):
        env._urop_obj_obs_pos_noise_std = torch.full((n, 1), 0.01, device=d)
    if not hasattr(env, "_urop_obj_obs_vel_noise_std"):
        env._urop_obj_obs_vel_noise_std = torch.full((n, 1), 0.05, device=d)


def _apply_physx_mass_material_best_effort(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass: torch.Tensor,
    friction: torch.Tensor,
    restitution: torch.Tensor,
) -> None:
    obj = env.scene["object"]
    try:
        view = obj.root_physx_view
    except Exception:
        return

    env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
    if env_ids_cpu.numel() == 0:
        return

    mass_cpu = mass.detach().to(device="cpu").squeeze(-1)
    friction_cpu = friction.detach().to(device="cpu").squeeze(-1)
    restitution_cpu = restitution.detach().to(device="cpu").squeeze(-1)

    try:
        if hasattr(view, "get_masses") and hasattr(view, "set_masses"):
            masses = view.get_masses().clone()
            if getattr(masses, "is_cuda", False):
                masses = masses.cpu()
            masses[env_ids_cpu, 0] = mass_cpu
            try:
                view.set_masses(masses, indices=env_ids_cpu)
            except TypeError:
                view.set_masses(masses)
    except Exception:
        pass

    try:
        if hasattr(view, "get_material_properties") and hasattr(view, "set_material_properties"):
            mats = view.get_material_properties().clone()
            if getattr(mats, "is_cuda", False):
                mats = mats.cpu()
            fr = friction_cpu.view(-1, 1)
            rs = restitution_cpu.view(-1, 1)
            mats[env_ids_cpu, :, 0] = fr
            mats[env_ids_cpu, :, 1] = 0.90 * fr
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
    mass_range=(2.4, 4.6),
    friction_range=(0.60, 0.95),
    restitution_range=(0.00, 0.05),
    size_scale_range=(0.94, 1.06),
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
        _apply_physx_mass_material_best_effort(env, env_ids, mass, friction, restitution)


def randomize_object_observation(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    pos_noise_range=(0.005, 0.018),
    vel_noise_range=(0.03, 0.10),
    drop_prob_range=(0.00, 0.08),
    alpha_range=(0.65, 1.00),
) -> None:
    _ensure_urop_buffers(env)
    d = env.device
    n = int(env_ids.shape[0])

    env._urop_obj_obs_pos_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*pos_noise_range)
    env._urop_obj_obs_vel_noise_std[env_ids] = torch.empty((n, 1), device=d).uniform_(*vel_noise_range)
    env._urop_obj_obs_drop_prob[env_ids] = torch.empty((n, 1), device=d).uniform_(*drop_prob_range)
    env._urop_obj_obs_alpha[env_ids] = torch.empty((n, 1), device=d).uniform_(*alpha_range)


def reset_object_parked(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    park: dict,
    object_randomization: dict | None = None,
    observation_randomization: dict | None = None,
) -> None:
    _ensure_urop_buffers(env)

    env._urop_toss_done[env_ids] = 0
    env._urop_toss_active[env_ids] = False
    env._urop_hold_latched[env_ids] = False
    env._urop_hold_steps[env_ids] = 0
    env._urop_hold_anchor_xy[env_ids] = 0.0
    env._urop_obj_obs_pos[env_ids] = 0.0
    env._urop_obj_obs_vel[env_ids] = 0.0

    robot = env.scene["robot"]
    env._urop_spawn_xy[env_ids] = robot.data.root_pos_w[env_ids, 0:2]
    env._urop_ready_joint_pos[env_ids] = robot.data.default_joint_pos[env_ids]

    object_randomization = object_randomization or {}
    observation_randomization = observation_randomization or {}
    randomize_receive_object(env, env_ids, **object_randomization)
    randomize_object_observation(env, env_ids, **observation_randomization)

    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = int(env_ids.shape[0])

    rp = robot.data.root_pos_w[env_ids]
    rq = _yaw_quat(robot.data.root_quat_w[env_ids])

    rel_p = torch.stack(
        [
            torch.empty(n, device=device).uniform_(*park["pos_x"]),
            torch.empty(n, device=device).uniform_(*park["pos_y"]),
            torch.empty(n, device=device).uniform_(*park["pos_z"]),
        ],
        dim=-1,
    )
    pos_w = rp + quat_apply(rq, rel_p)
    quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)


def toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage1: dict,
    stage2: dict,
    stage3: dict,
    throw_prob_stage1: float = 1.0,
    throw_prob_stage2: float = 0.95,
    throw_prob_stage3: float = 0.90,
    max_throws_per_episode: int = 1,
) -> None:
    _ensure_urop_buffers(env)

    if max_throws_per_episode <= 0:
        return

    can_throw = env._urop_toss_done[env_ids] < int(max_throws_per_episode)
    if not torch.any(can_throw):
        return
    env_ids = env_ids[can_throw]

    stage = _get_stage(env)
    if stage == 0:
        return

    if stage == 1:
        cfg = stage1
        prob = float(throw_prob_stage1)
    elif stage == 2:
        cfg = stage2
        prob = float(throw_prob_stage2)
    else:
        cfg = stage3
        prob = float(throw_prob_stage3)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = int(env_ids.shape[0])

    if prob < 1.0:
        keep = torch.rand(n, device=device) < prob
        ids_throw = env_ids[keep]
        ids_skip = env_ids[~keep]
        if ids_skip.numel() > 0:
            env._urop_toss_done[ids_skip] = int(max_throws_per_episode)
            env._urop_toss_active[ids_skip] = False
        if ids_throw.numel() == 0:
            return
    else:
        ids_throw = env_ids

    n2 = int(ids_throw.shape[0])
    rp = robot.data.root_pos_w[ids_throw]
    rq = _yaw_quat(robot.data.root_quat_w[ids_throw])

    rel_p = torch.stack(
        [
            torch.empty(n2, device=device).uniform_(*cfg["pos_x"]),
            torch.empty(n2, device=device).uniform_(*cfg["pos_y"]),
            torch.empty(n2, device=device).uniform_(*cfg["pos_z"]),
        ],
        dim=-1,
    )
    rel_v = torch.stack(
        [
            torch.empty(n2, device=device).uniform_(*cfg["vel_x"]),
            torch.empty(n2, device=device).uniform_(*cfg["vel_y"]),
            torch.empty(n2, device=device).uniform_(*cfg["vel_z"]),
        ],
        dim=-1,
    )

    roll = torch.empty(n2, device=device).uniform_(*cfg.get("roll", (-0.04, 0.04)))
    pitch = torch.empty(n2, device=device).uniform_(*cfg.get("pitch", (-0.06, 0.06)))
    yaw = torch.empty(n2, device=device).uniform_(*cfg.get("yaw", (-0.12, 0.12)))
    quat_w = _quat_from_euler_xyz(roll, pitch, yaw)

    ang_vel = torch.stack(
        [
            torch.empty(n2, device=device).uniform_(*cfg.get("ang_vel_x", (-0.2, 0.2))),
            torch.empty(n2, device=device).uniform_(*cfg.get("ang_vel_y", (-0.2, 0.2))),
            torch.empty(n2, device=device).uniform_(*cfg.get("ang_vel_z", (-0.3, 0.3))),
        ],
        dim=-1,
    )

    pos_w = rp + quat_apply(rq, rel_p)
    vel_w = robot.data.root_lin_vel_w[ids_throw] + quat_apply(rq, rel_v)

    obj.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1), env_ids=ids_throw)
    vel6 = torch.zeros((n2, 6), device=device)
    vel6[:, 0:3] = vel_w
    vel6[:, 3:6] = ang_vel
    obj.write_root_velocity_to_sim(vel6, env_ids=ids_throw)

    env._urop_toss_done[ids_throw] += 1
    env._urop_toss_active[ids_throw] = True
