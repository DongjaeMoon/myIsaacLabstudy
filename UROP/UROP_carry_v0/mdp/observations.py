#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v0/mdp/observations.py]
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Quaternion utilities (w, x, y, z)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Controlled joint subset (29 DOF only)
# -----------------------------------------------------------------------------

CONTROLLED_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ARM_CARRY_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _joint_index_cache(env: "ManagerBasedRLEnv") -> dict[str, int]:
    if hasattr(env, "_urop_joint_name_to_idx"):
        return env._urop_joint_name_to_idx
    robot = env.scene["robot"]
    env._urop_joint_name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    return env._urop_joint_name_to_idx


def get_controlled_joint_indices(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_controlled_joint_indices"):
        return env._urop_controlled_joint_indices

    name_to_idx = _joint_index_cache(env)
    indices = [name_to_idx[n] for n in CONTROLLED_JOINT_NAMES if n in name_to_idx]
    if len(indices) != len(CONTROLLED_JOINT_NAMES):
        missing = [n for n in CONTROLLED_JOINT_NAMES if n not in name_to_idx]
        raise RuntimeError(f"Missing controlled joints in articulation: {missing}")
    env._urop_controlled_joint_indices = torch.tensor(indices, device=env.device, dtype=torch.long)
    return env._urop_controlled_joint_indices


def get_arm_carry_joint_indices(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_arm_carry_joint_indices"):
        return env._urop_arm_carry_joint_indices

    name_to_idx = _joint_index_cache(env)
    indices = [name_to_idx[n] for n in ARM_CARRY_JOINT_NAMES if n in name_to_idx]
    if len(indices) != len(ARM_CARRY_JOINT_NAMES):
        missing = [n for n in ARM_CARRY_JOINT_NAMES if n not in name_to_idx]
        raise RuntimeError(f"Missing arm carry joints in articulation: {missing}")
    env._urop_arm_carry_joint_indices = torch.tensor(indices, device=env.device, dtype=torch.long)
    return env._urop_arm_carry_joint_indices


# -----------------------------------------------------------------------------
# Observation terms
# -----------------------------------------------------------------------------

def carry_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_command"):
        return env._urop_command
    return torch.zeros((env.num_envs, 3), device=env.device)


def carry_stage(env: "ManagerBasedRLEnv") -> torch.Tensor:
    s = float(getattr(env, "urop_stage", 0))
    return torch.full((env.num_envs, 1), s / 3.0, device=env.device)


def object_params(env: "ManagerBasedRLEnv") -> torch.Tensor:
    dev = env.device
    n = env.num_envs
    size = getattr(env, "_urop_box_size", torch.tensor([0.32, 0.24, 0.24], device=dev).unsqueeze(0).repeat(n, 1))
    mass = getattr(env, "_urop_box_mass", torch.full((n, 1), 3.0, device=dev))
    fric = getattr(env, "_urop_box_friction", torch.full((n, 1), 0.7, device=dev))
    rest = getattr(env, "_urop_box_restitution", torch.full((n, 1), 0.02, device=dev))

    size_n = torch.stack(
        [
            (size[:, 0] - 0.32) / 0.08,
            (size[:, 1] - 0.24) / 0.06,
            (size[:, 2] - 0.24) / 0.06,
        ],
        dim=-1,
    )
    mass_n = (mass - 3.75) / 1.75
    fric_n = (fric - 0.70) / 0.20
    rest_n = (rest - 0.03) / 0.03
    return torch.cat([size_n, mass_n, fric_n, rest_n], dim=-1)


def robot_proprio(env: "ManagerBasedRLEnv", torque_scale: float = 1.0 / 80.0) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w

    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    lin_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    ang_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)

    idx = get_controlled_joint_indices(env)
    jp = robot.data.joint_pos[:, idx]
    jv = robot.data.joint_vel[:, idx]

    jt = getattr(robot.data, "applied_torque", None)
    if jt is None:
        jt = getattr(robot.data, "joint_effort", None)
    if jt is None:
        jt = torch.zeros_like(robot.data.joint_pos)
    jt = torch.clamp(jt[:, idx] * torque_scale, -1.0, 1.0)

    return torch.cat([g_b, lin_b, ang_b, jp, jv, jt], dim=-1)


def prev_actions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.action_manager.prev_action


def object_rel_state(
    env: "ManagerBasedRLEnv",
    pos_scale: float = 1.0,
    vel_scale: float = 1.0,
    noise_std: float = 0.0,
    drop_prob: float = 0.0,
) -> torch.Tensor:
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

    rel_p_b = quat_rotate_inverse(rq, op - rp) * pos_scale
    rel_v_b = quat_rotate_inverse(rq, ov - rv) * vel_scale
    rel_w_b = quat_rotate_inverse(rq, ow - rw)

    rel_q = quat_mul(quat_conj(rq), oq)
    rel_r6 = quat_to_rot6d(rel_q)

    out = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * noise_std
    if drop_prob > 0.0:
        mask = (torch.rand(env.num_envs, device=env.device) < drop_prob).float().unsqueeze(-1)
        out = out * (1.0 - mask)
    return out


def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str], scale: float = 1.0 / 300.0) -> torch.Tensor:
    mags = []
    for name in sensor_names:
        sensor = env.scene[name]
        force = sensor.data.net_forces_w.reshape(env.num_envs, -1)
        mags.append(torch.norm(force, dim=-1, keepdim=True) * scale)
    return torch.cat(mags, dim=-1)
