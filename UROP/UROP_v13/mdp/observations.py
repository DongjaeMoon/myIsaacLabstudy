from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


CONTROLLED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LOWER_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


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


def _ensure_object_obs_buffers(env: "ManagerBasedRLEnv") -> None:
    n = env.num_envs
    d = env.device
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


def projected_gravity(env: "ManagerBasedRLEnv") -> torch.Tensor:
    q = env.scene["robot"].data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    return quat_rotate_inverse(q, g_world)


def base_angular_velocity(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    ang_b = quat_rotate_inverse(robot.data.root_quat_w, robot.data.root_ang_vel_w)
    return ang_b * scale


def controlled_joint_positions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    return env.scene["robot"].data.joint_pos[:, idx]


def controlled_joint_velocities(env: "ManagerBasedRLEnv", scale: float = 1.0) -> torch.Tensor:
    idx = get_controlled_joint_indices(env)
    return env.scene["robot"].data.joint_vel[:, idx] * scale


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
    return env.action_manager.prev_action


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


def object_rel_pos_vel(
    env: "ManagerBasedRLEnv",
    pos_scale: float = 1.0,
    vel_scale: float = 1.0,
    apply_noise: bool = True,
) -> torch.Tensor:
    _ensure_object_obs_buffers(env)

    rel_p_b, rel_v_b, _, _ = _object_rel_true(env)
    active = toss_state(env) > 0.5

    prev_pos = env._urop_obj_obs_pos
    prev_vel = env._urop_obj_obs_vel

    meas_pos = rel_p_b * pos_scale
    meas_vel = rel_v_b * vel_scale

    if apply_noise:
        meas_pos = meas_pos + torch.randn_like(meas_pos) * env._urop_obj_obs_pos_noise_std
        meas_vel = meas_vel + torch.randn_like(meas_vel) * env._urop_obj_obs_vel_noise_std

    alpha = torch.clamp(env._urop_obj_obs_alpha, 0.05, 1.0)
    filt_pos = alpha * meas_pos + (1.0 - alpha) * prev_pos
    filt_vel = alpha * meas_vel + (1.0 - alpha) * prev_vel

    if apply_noise:
        dropout = torch.rand((env.num_envs, 1), device=env.device) < env._urop_obj_obs_drop_prob
        filt_pos = torch.where(dropout, prev_pos, filt_pos)
        filt_vel = torch.where(dropout, prev_vel, filt_vel)

    active_f = active.float()
    env._urop_obj_obs_pos = filt_pos * active_f
    env._urop_obj_obs_vel = filt_vel * active_f
    return torch.cat([env._urop_obj_obs_pos, env._urop_obj_obs_vel], dim=-1)


def critic_robot_state(env: "ManagerBasedRLEnv", torque_scale: float = 1.0 / 80.0) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    lin_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    ang_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    jp = controlled_joint_positions(env)
    jv = controlled_joint_velocities(env)
    jt = joint_torques(env, torque_scale=torque_scale)
    return torch.cat([g_b, lin_b, ang_b, jp, jv, jt], dim=-1)


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
    x = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)
    if hasattr(env, "_urop_toss_active"):
        x = x * env._urop_toss_active.float().unsqueeze(-1)
    return x


def object_truth_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    obj = env.scene["object"]
    env_origins = getattr(env.scene, "env_origins", torch.zeros_like(obj.data.root_pos_w))
    rel_p_b, rel_v_b, rel_w_b, rel_q = _object_rel_true(env)
    obj_pos_local = obj.data.root_pos_w - env_origins
    rel_r6 = quat_to_rot6d(rel_q)
    x = torch.cat(
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
        x = x * env._urop_toss_active.float().unsqueeze(-1)
    return x


def object_params(env: "ManagerBasedRLEnv") -> torch.Tensor:
    dev = env.device
    n = env.num_envs
    size = getattr(env, "_urop_box_size", torch.tensor([0.34, 0.26, 0.24], device=dev).repeat(n, 1))
    mass = getattr(env, "_urop_box_mass", torch.full((n, 1), 3.2, device=dev))
    fric = getattr(env, "_urop_box_friction", torch.full((n, 1), 0.8, device=dev))
    rest = getattr(env, "_urop_box_restitution", torch.full((n, 1), 0.02, device=dev))

    size_n = torch.stack(
        [
            (size[:, 0] - 0.34) / 0.06,
            (size[:, 1] - 0.26) / 0.05,
            (size[:, 2] - 0.24) / 0.05,
        ],
        dim=-1,
    )
    mass_n = (mass - 3.2) / 1.6
    fric_n = (fric - 0.8) / 0.2
    rest_n = (rest - 0.02) / 0.03
    return torch.cat([size_n, mass_n, fric_n, rest_n], dim=-1)


def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str], scale: float = 1.0 / 300.0) -> torch.Tensor:
    mags = []
    for name in sensor_names:
        sensor = env.scene[name]
        forces = sensor.data.net_forces_w.reshape(env.num_envs, -1)
        mags.append(torch.norm(forces, dim=-1, keepdim=True) * scale)
    return torch.cat(mags, dim=-1)
