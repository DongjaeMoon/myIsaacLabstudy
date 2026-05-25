from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import scene_objects_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CONTROLLED_JOINT_NAMES = list(scene_objects_cfg.CONTROLLED_JOINT_NAMES)
LOWER_BODY_JOINT_NAMES = list(scene_objects_cfg.LOWER_BODY_JOINT_NAMES)
ACTION_SCALE = list(scene_objects_cfg.ACTION_SCALE)
EXPECTED_POLICY_OBS_DIM = scene_objects_cfg.EXPECTED_POLICY_OBS_DIM
EXPECTED_ACTION_DIM = scene_objects_cfg.EXPECTED_ACTION_DIM
MAX_OBJECT_OBS_LATENCY_STEPS = 3


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:, 1:4]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conj(q), v)


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)


def _get_joint_indices(env: "ManagerBasedRLEnv", cache_attr: str, joint_names: list[str]) -> torch.Tensor:
    if hasattr(env, cache_attr):
        return getattr(env, cache_attr)

    robot = env.scene["robot"]
    name_to_idx = {name: i for i, name in enumerate(robot.data.joint_names)}
    missing = [name for name in joint_names if name not in name_to_idx]
    if missing:
        raise RuntimeError(f"Missing joints in articulation: {missing}")

    indices = torch.tensor([name_to_idx[name] for name in joint_names], device=env.device, dtype=torch.long)
    setattr(env, cache_attr, indices)
    return indices


def get_controlled_joint_indices(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _get_joint_indices(env, "_urop_controlled_joint_indices", CONTROLLED_JOINT_NAMES)


def get_lower_body_joint_indices(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return _get_joint_indices(env, "_urop_lower_body_joint_indices", LOWER_BODY_JOINT_NAMES)


def _ensure_ready_reference(env: "ManagerBasedRLEnv") -> None:
    if hasattr(env, "_urop_ready_ref_controlled"):
        return
    ready = torch.tensor(
        [scene_objects_cfg.READY_POSE[name] for name in CONTROLLED_JOINT_NAMES],
        device=env.device,
        dtype=env.scene["robot"].data.joint_pos.dtype,
    )
    env._urop_ready_ref_controlled = ready.unsqueeze(0).repeat(env.num_envs, 1)


def _ensure_object_obs_buffers(env: "ManagerBasedRLEnv") -> None:
    _ensure_ready_reference(env)

    n = env.num_envs
    d = env.device
    dtype = env.scene["robot"].data.root_pos_w.dtype

    if not hasattr(env, "_urop_policy_mode_one_hot"):
        base = torch.tensor([0.0, 0.0, 1.0, 0.0], device=d, dtype=dtype)
        env._urop_policy_mode_one_hot = base.unsqueeze(0).repeat(n, 1)
    if not hasattr(env, "_urop_projected_gravity_noise_std"):
        env._urop_projected_gravity_noise_std = torch.full((n, 1), 0.015, device=d, dtype=dtype)
    if not hasattr(env, "_urop_base_ang_vel_noise_std"):
        env._urop_base_ang_vel_noise_std = torch.full((n, 1), 0.04, device=d, dtype=dtype)
    if not hasattr(env, "_urop_joint_pos_noise_std"):
        env._urop_joint_pos_noise_std = torch.full((n, 1), 0.010, device=d, dtype=dtype)
    if not hasattr(env, "_urop_joint_vel_noise_std"):
        env._urop_joint_vel_noise_std = torch.full((n, 1), 0.08, device=d, dtype=dtype)

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
    if not hasattr(env, "_urop_obj_obs_latency_steps"):
        env._urop_obj_obs_latency_steps = torch.zeros((n,), device=d, dtype=torch.long)
    if not hasattr(env, "_urop_obj_obs_history_pos"):
        env._urop_obj_obs_history_pos = torch.zeros((n, MAX_OBJECT_OBS_LATENCY_STEPS + 1, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_history_vel"):
        env._urop_obj_obs_history_vel = torch.zeros((n, MAX_OBJECT_OBS_LATENCY_STEPS + 1, 3), device=d, dtype=dtype)
    if not hasattr(env, "_urop_obj_obs_history_visible"):
        env._urop_obj_obs_history_visible = torch.zeros(
            (n, MAX_OBJECT_OBS_LATENCY_STEPS + 1, 1), device=d, dtype=torch.bool
        )
    if not hasattr(env, "_urop_obj_obs_cache_global_step"):
        env._urop_obj_obs_cache_global_step = -1
    if not hasattr(env, "_urop_obj_obs_cache_episode_len"):
        env._urop_obj_obs_cache_episode_len = torch.full((n,), -1, device=d, dtype=torch.long)


def toss_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_toss_active"):
        return env._urop_toss_active.float().unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


def hold_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_hold_latched"):
        return env._urop_hold_latched.float().unsqueeze(-1)
    return torch.zeros((env.num_envs, 1), device=env.device)


def drop_state(env: "ManagerBasedRLEnv", min_z: float = 0.28, max_dist: float = 2.2) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    active = toss_state(env)
    dropped = (
        (obj.data.root_pos_w[:, 2:3] < min_z)
        | (torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1, keepdim=True) > max_dist)
    ).float()
    return dropped * active


def hold_anchor_error(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    if not hasattr(env, "_urop_hold_anchor_xy"):
        return torch.zeros((env.num_envs, 2), device=env.device)
    robot = env.scene["robot"]
    err = (robot.data.root_pos_w[:, 0:2] - env._urop_hold_anchor_xy) * scale
    if hasattr(env, "_urop_hold_latched"):
        err = err * env._urop_hold_latched.float().unsqueeze(-1)
    return err


def _object_rel_true(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rq = robot.data.root_quat_w
    rp = robot.data.root_pos_w
    rv = robot.data.root_lin_vel_w
    rw = robot.data.root_ang_vel_w
    oq = obj.data.root_quat_w
    op = obj.data.root_pos_w
    ov = obj.data.root_lin_vel_w
    ow = obj.data.root_ang_vel_w

    rel_p_b = quat_rotate_inverse(rq, op - rp)
    rel_v_b = quat_rotate_inverse(rq, ov - rv)
    rel_w_b = quat_rotate_inverse(rq, ow - rw)
    rel_q = quat_mul(quat_conj(rq), oq)
    return rel_p_b, rel_v_b, rel_w_b, rel_q


def _object_obs_cache_is_fresh(env: "ManagerBasedRLEnv") -> bool:
    if not hasattr(env, "_urop_obj_obs_cache_global_step"):
        return False
    if env._urop_obj_obs_cache_global_step != int(env.common_step_counter):
        return False
    if not hasattr(env, "_urop_obj_obs_cache_episode_len"):
        return False
    return torch.equal(env._urop_obj_obs_cache_episode_len, env.episode_length_buf)


def _refresh_object_measurement(env: "ManagerBasedRLEnv") -> None:
    _ensure_object_obs_buffers(env)
    if _object_obs_cache_is_fresh(env):
        return

    rel_p_b, rel_v_b, _, _ = _object_rel_true(env)
    active = toss_state(env) > 0.5

    meas_pos = rel_p_b + torch.randn_like(rel_p_b) * env._urop_obj_obs_pos_noise_std
    meas_vel = rel_v_b + torch.randn_like(rel_v_b) * env._urop_obj_obs_vel_noise_std

    alpha = torch.clamp(env._urop_obj_obs_alpha, 0.05, 1.0)
    filt_pos = alpha * meas_pos + (1.0 - alpha) * env._urop_obj_filter_pos
    filt_vel = alpha * meas_vel + (1.0 - alpha) * env._urop_obj_filter_vel

    env._urop_obj_filter_pos = torch.where(active, filt_pos, torch.zeros_like(filt_pos))
    env._urop_obj_filter_vel = torch.where(active, filt_vel, torch.zeros_like(filt_vel))

    dropout = torch.rand((env.num_envs, 1), device=env.device) < env._urop_obj_obs_drop_prob
    visible = active & (~dropout)
    # [v16] Keep the v15 hidden-object convention: invisible measurements become exact zeros.
    current_pos = torch.where(visible, env._urop_obj_filter_pos, torch.zeros_like(env._urop_obj_filter_pos))
    current_vel = torch.where(visible, env._urop_obj_filter_vel, torch.zeros_like(env._urop_obj_filter_vel))

    env._urop_obj_obs_history_pos = torch.roll(env._urop_obj_obs_history_pos, shifts=1, dims=1)
    env._urop_obj_obs_history_vel = torch.roll(env._urop_obj_obs_history_vel, shifts=1, dims=1)
    env._urop_obj_obs_history_visible = torch.roll(env._urop_obj_obs_history_visible, shifts=1, dims=1)
    env._urop_obj_obs_history_pos[:, 0, :] = current_pos
    env._urop_obj_obs_history_vel[:, 0, :] = current_vel
    env._urop_obj_obs_history_visible[:, 0, :] = visible

    latency = env._urop_obj_obs_latency_steps.clamp(0, MAX_OBJECT_OBS_LATENCY_STEPS)
    gather_idx_vec = latency.view(-1, 1, 1).expand(-1, 1, 3)
    gather_idx_vis = latency.view(-1, 1, 1).expand(-1, 1, 1)

    env._urop_obj_obs_pos = torch.gather(env._urop_obj_obs_history_pos, dim=1, index=gather_idx_vec).squeeze(1)
    env._urop_obj_obs_vel = torch.gather(env._urop_obj_obs_history_vel, dim=1, index=gather_idx_vec).squeeze(1)
    env._urop_obj_visible = (
        torch.gather(env._urop_obj_obs_history_visible.float(), dim=1, index=gather_idx_vis).squeeze(1) > 0.5
    )

    env._urop_obj_obs_cache_global_step = int(env.common_step_counter)
    env._urop_obj_obs_cache_episode_len = env.episode_length_buf.clone()


def _apply_gaussian_noise(value: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return value + torch.randn_like(value) * std


def _projected_gravity_clean(env: "ManagerBasedRLEnv") -> torch.Tensor:
    q = env.scene["robot"].data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    return quat_rotate_inverse(q, g_world)


def projected_gravity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Actor obs dims: 3
    _ensure_object_obs_buffers(env)
    gravity_b = _projected_gravity_clean(env)
    noisy_gravity = _apply_gaussian_noise(gravity_b, env._urop_projected_gravity_noise_std)
    return torch.clamp(noisy_gravity, -1.0, 1.0)


def _base_angular_velocity_clean(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)


def base_angular_velocity(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    # Actor obs dims: 3
    _ensure_object_obs_buffers(env)
    ang_b = _apply_gaussian_noise(_base_angular_velocity_clean(env), env._urop_base_ang_vel_noise_std)
    return ang_b * scale


def _controlled_joint_pos_rel_clean(env: "ManagerBasedRLEnv") -> torch.Tensor:
    _ensure_ready_reference(env)
    idx = get_controlled_joint_indices(env)
    return env.scene["robot"].data.joint_pos[:, idx] - env._urop_ready_ref_controlled


def controlled_joint_pos_rel(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Actor obs dims: 29
    _ensure_object_obs_buffers(env)
    joint_pos_rel = _controlled_joint_pos_rel_clean(env)
    return _apply_gaussian_noise(joint_pos_rel, env._urop_joint_pos_noise_std)


def _controlled_joint_velocities_clean(env: "ManagerBasedRLEnv") -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    return env.scene["robot"].data.joint_vel[:, idx]


def controlled_joint_velocities(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    # Actor obs dims: 29
    _ensure_object_obs_buffers(env)
    joint_vel = _apply_gaussian_noise(_controlled_joint_velocities_clean(env), env._urop_joint_vel_noise_std)
    return joint_vel * scale


def joint_torques(env: "ManagerBasedRLEnv", torque_scale: float = 1.0 / 80.0) -> torch.Tensor:
    robot = env.scene["robot"]
    idx = get_controlled_joint_indices(env)
    tau = getattr(robot.data, "applied_torque", None)
    if tau is None:
        tau = getattr(robot.data, "joint_effort", None)
    if tau is None:
        tau = torch.zeros((env.num_envs, idx.shape[0]), device=env.device)
    else:
        tau = tau[:, idx]
    return torch.clamp(tau * torque_scale, -1.0, 1.0)


def prev_actions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Actor obs dims: 29
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "prev_action"):
        return env.action_manager.prev_action
    return torch.zeros((env.num_envs, EXPECTED_ACTION_DIM), device=env.device)


def object_rel_pos(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    # Actor obs dims: 3. Before toss or when tag is hidden: exact zeros.
    _refresh_object_measurement(env)
    return env._urop_obj_obs_pos * scale


def object_rel_lin_vel(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    # Actor obs dims: 3. Before toss or when tag is hidden: exact zeros.
    _refresh_object_measurement(env)
    return env._urop_obj_obs_vel * scale


def tag_visible(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Actor obs dims: 1
    _refresh_object_measurement(env)
    return env._urop_obj_visible.float()


def mode_one_hot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    # Actor obs dims: 4. [v16] Kept unchanged to preserve the 104-D policy I/O contract.
    _ensure_object_obs_buffers(env)
    return env._urop_policy_mode_one_hot


def critic_robot_state(env: "ManagerBasedRLEnv", torque_scale: float = 1.0 / 80.0) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_b = _projected_gravity_clean(env)
    lin_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    ang_b = _base_angular_velocity_clean(env)
    jp_rel = _controlled_joint_pos_rel_clean(env)
    jv = _controlled_joint_velocities_clean(env)
    jt = joint_torques(env, torque_scale=torque_scale)
    return torch.cat([g_b, lin_b, ang_b, jp_rel, jv, jt], dim=-1)


def root_state_privileged(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    env_origins = getattr(env.scene, "env_origins", torch.zeros_like(robot.data.root_pos_w))
    root_pos_local = robot.data.root_pos_w - env_origins
    return torch.cat(
        [
            root_pos_local,
            robot.data.root_quat_w,
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
        ],
        dim=-1,
    )


def object_rel_full_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    rel_p_b, rel_v_b, rel_w_b, rel_q = _object_rel_true(env)
    rel_r6 = quat_to_rot6d(rel_q)
    state = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)
    if hasattr(env, "_urop_toss_active"):
        state = state * env._urop_toss_active.float().unsqueeze(-1)
    return state


def object_truth_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    obj = env.scene["object"]
    env_origins = getattr(env.scene, "env_origins", torch.zeros_like(obj.data.root_pos_w))
    rel_p_b, rel_v_b, rel_w_b, rel_q = _object_rel_true(env)
    obj_pos_local = obj.data.root_pos_w - env_origins
    rel_r6 = quat_to_rot6d(rel_q)
    state = torch.cat(
        [
            obj_pos_local,
            obj.data.root_quat_w,
            obj.data.root_lin_vel_w,
            obj.data.root_ang_vel_w,
            rel_p_b,
            rel_r6,
            rel_v_b,
            rel_w_b,
        ],
        dim=-1,
    )
    if hasattr(env, "_urop_toss_active"):
        state = state * env._urop_toss_active.float().unsqueeze(-1)
    return state


def object_params(env: "ManagerBasedRLEnv") -> torch.Tensor:
    dev = env.device
    n = env.num_envs
    size = getattr(
        env,
        "_urop_box_size",
        torch.tensor(scene_objects_cfg.OBJECT_BASE_SIZE, device=dev).unsqueeze(0).repeat(n, 1),
    )
    mass = getattr(env, "_urop_box_mass", torch.full((n, 1), scene_objects_cfg.OBJECT_DEFAULT_MASS, device=dev))
    fric = getattr(env, "_urop_box_friction", torch.full((n, 1), 0.8, device=dev))
    rest = getattr(env, "_urop_box_restitution", torch.full((n, 1), 0.02, device=dev))

    size_n = torch.stack(
        [
            (size[:, 0] - scene_objects_cfg.OBJECT_BASE_SIZE[0]) / 0.06,
            (size[:, 1] - scene_objects_cfg.OBJECT_BASE_SIZE[1]) / 0.05,
            (size[:, 2] - scene_objects_cfg.OBJECT_BASE_SIZE[2]) / 0.05,
        ],
        dim=-1,
    )
    mass_n = (mass - scene_objects_cfg.OBJECT_DEFAULT_MASS) / 1.6
    fric_n = (fric - 0.8) / 0.25
    rest_n = (rest - 0.02) / 0.05
    return torch.cat([size_n, mass_n, fric_n, rest_n], dim=-1)


def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str], scale: float = 1.0 / 300.0) -> torch.Tensor:
    mags = []
    for name in sensor_names:
        sensor = env.scene[name]
        forces = sensor.data.net_forces_w.reshape(env.num_envs, -1)
        mags.append(torch.norm(forces, dim=-1, keepdim=True) * scale)
    return torch.cat(mags, dim=-1)


assert EXPECTED_POLICY_OBS_DIM == 104
assert EXPECTED_ACTION_DIM == 29
