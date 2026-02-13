# UROP_v2/mdp/observations.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------
# Quaternion utilities (w,x,y,z)
# -----------------------
def quat_conj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[:, 1:] *= -1
    return out

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # (w,x,y,z)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # v' = q * (0,v) * q_conj
    qv = torch.cat([torch.zeros((v.shape[0], 1), device=v.device), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[:, 1:]

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)

def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    # 6D repr: first two columns of R
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # rotation matrix
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)
    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - w*x)
    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x*x + y*y)
    c1 = torch.stack([r00, r10, r20], dim=-1)
    c2 = torch.stack([r01, r11, r21], dim=-1)
    return torch.cat([c1, c2], dim=-1)


# -----------------------
# Observations
# -----------------------
def robot_proprio(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel

    default_jp = getattr(robot.data, "default_joint_pos", None)
    if default_jp is None:
        default_jp = torch.zeros_like(jp)
    jp_rel = jp - default_jp

    q = robot.data.root_quat_w
    lin_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    ang_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)

    g_world = torch.tensor([0.0, 0.0, -1.0], device=jp.device).unsqueeze(0).repeat(jp.shape[0], 1)
    g_b = quat_rotate_inverse(q, g_world)

    return torch.cat([jp_rel, jv, lin_b, ang_b, g_b], dim=-1)


def previous_actions(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.action_manager.prev_action


def velocity_command(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if not hasattr(env, "urop_cmd") or env.urop_cmd is None:
        device = env.scene["robot"].data.root_pos_w.device
        env.urop_cmd = torch.zeros((env.num_envs, 3), device=device)
    return env.urop_cmd


def object_rel_state(
    env: "ManagerBasedRLEnv",
    drop_prob: float = 0.0,
    noise_std: float = 0.0,
    pos_scale: float = 1.0,
    vel_scale: float = 1.0,
    ang_vel_scale: float = 1.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]

    rp = robot.data.root_pos_w
    rq = robot.data.root_quat_w
    rv = robot.data.root_lin_vel_w

    op = obj.data.root_pos_w
    oq = obj.data.root_quat_w
    ov = obj.data.root_lin_vel_w
    ow = obj.data.root_ang_vel_w

    rel_p_b = quat_rotate_inverse(rq, op - rp) * pos_scale
    rel_v_b = quat_rotate_inverse(rq, ov - rv) * vel_scale
    rel_w_b = quat_rotate_inverse(rq, ow) * ang_vel_scale

    q_rel = quat_mul(quat_conj(rq), oq)
    rel_rot6d = quat_to_rot6d(q_rel)

    feat = torch.cat([rel_p_b, rel_rot6d, rel_v_b, rel_w_b], dim=-1)

    if noise_std > 0:
        feat = feat + noise_std * torch.randn_like(feat)

    if drop_prob > 0:
        m = (torch.rand((feat.shape[0], 1), device=feat.device) > drop_prob).float()
        feat = feat * m

    return feat


def _body_index_map(env: "ManagerBasedRLEnv") -> dict:
    if hasattr(env, "_urop_body_name_to_id") and env._urop_body_name_to_id is not None:
        return env._urop_body_name_to_id
    robot = env.scene["robot"]
    names = list(getattr(robot.data, "body_names", []))
    env._urop_body_name_to_id = {n: i for i, n in enumerate(names)}
    return env._urop_body_name_to_id


def hand_object_vectors(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return (N, 9): [lhand->obj (3), rhand->obj(3), torso->obj(3)] in robot base frame."""
    robot = env.scene["robot"]
    obj = env.scene["object"]

    mp = _body_index_map(env)
    # 이 이름들은 env_cfg의 contact sensor prim_path와 맞춰둠
    l_name = "left_wrist_roll_rubber_hand"
    r_name = "right_wrist_roll_rubber_hand"
    t_name = "torso_link"

    # fallback: 이름이 다르면 여기서 바로 에러로 터져서 원인 파악이 쉬움
    li = mp[l_name]
    ri = mp[r_name]
    ti = mp[t_name]

    lpos = robot.data.body_pos_w[:, li, :]
    rpos = robot.data.body_pos_w[:, ri, :]
    tpos = robot.data.body_pos_w[:, ti, :]
    op = obj.data.root_pos_w

    rq = robot.data.root_quat_w
    lvec_b = quat_rotate_inverse(rq, op - lpos)
    rvec_b = quat_rotate_inverse(rq, op - rpos)
    tvec_b = quat_rotate_inverse(rq, op - tpos)

    return torch.cat([lvec_b, rvec_b, tvec_b], dim=-1)


def contact_forces(env: "ManagerBasedRLEnv", sensor_names: list[str], scale: float = 1.0 / 300.0) -> torch.Tensor:
    outs = []
    for name in sensor_names:
        sensor = env.scene[name]
        f = sensor.data.net_forces_w  # (N, 1, 3)
        outs.append(f.reshape(f.shape[0], -1))
    return torch.cat(outs, dim=-1) * scale
