from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -------------------------
# Quaternion utils (w, x, y, z)
# -------------------------
def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1)

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ],
        dim=-1,
    )

def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # v' = q * (0,v) * q_conj
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    vq = torch.cat([zeros, v], dim=-1)
    return quat_mul(quat_mul(q, vq), quat_conj(q))[:, 1:4]

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conj(q), v)

def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    # rotation matrix first 2 columns (6D rep)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)
    # first two columns: [r00,r10,r20,r01,r11,r21]
    return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)

# -------------------------
# Observations
# -------------------------
def robot_proprio(env: "ManagerBasedRLEnv", torque_scale: float = 1.0/80.0) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w

    # gravity in body frame (upright info)
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)

    # base velocities in body frame
    lin_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    ang_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)

    jp = robot.data.joint_pos
    jv = robot.data.joint_vel

    # torque/effort (조교님 코멘트 1)
    jt = None
    if hasattr(robot.data, "applied_torque"):
        jt = robot.data.applied_torque
    elif hasattr(robot.data, "joint_effort"):
        jt = robot.data.joint_effort
    else:
        jt = torch.zeros_like(jp)

    jt = torch.clamp(jt * torque_scale, -1.0, 1.0)

    return torch.cat([g_b, lin_b, ang_b, jp, jv, jt], dim=-1)

def prev_actions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.action_manager.prev_action

def object_rel_state(
    env: "ManagerBasedRLEnv",
    pos_scale: float = 1.0,
    vel_scale: float = 1.0,
    drop_prob: float = 0.0,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """
    조교님 코멘트 2: robot frame 기준 object state
    - rel_pos_b = R^T (p_obj - p_robot)
    - rel_vel_b = R^T (v_obj - v_robot)
    - rel_quat = q_robot^* ⊗ q_obj
    """
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

    x = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)

    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std

    # teacher/student gating용 (필요 없으면 drop_prob=0으로 고정)
    if drop_prob > 0.0:
        mask = (torch.rand(env.num_envs, device=env.device) < drop_prob).float().unsqueeze(-1)
        x = x * (1.0 - mask)

        # ✅ toss 이후에만 object 관측 활성화 (toss 전에는 0)
    if hasattr(env, "_urop_toss_count"):
        active = (env._urop_toss_count > 0).float().unsqueeze(-1)
        x = x * active

    return x

def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str], scale: float = 1.0/300.0) -> torch.Tensor:
    mags = []
    for name in sensor_names:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        mags.append(torch.norm(f, dim=-1, keepdim=True) * scale)
    out = torch.cat(mags, dim=-1)

    if hasattr(env, "_urop_toss_count"):
        active = (env._urop_toss_count > 0).float().unsqueeze(-1)
        out = out * active

    return out

