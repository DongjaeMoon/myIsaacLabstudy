# UROP/UROP_v2/mdp/rewards.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from .observations import quat_rotate_inverse, quat_mul, quat_conj

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_stage(env) -> int:
    p = env.cfg.curriculum.stage_schedule.params
    forced = int(p.get("eval_stage", -1))
    if forced in (0, 1, 2):
        return forced

    step = int(env.common_step_counter)
    stage0 = int(p["stage0_iters"])
    stage1 = int(p["stage1_iters"])
    nsteps = int(p["num_steps_per_env"])

    s1 = stage0 * nsteps
    s2 = (stage0 + stage1) * nsteps
    if step < s1:
        return 0
    elif step < s2:
        return 1
    else:
        return 2


def _stage_w(env, w0: float, w1: float, w2: float) -> float:
    s = _get_stage(env)
    return float(w0 if s == 0 else (w1 if s == 1 else w2))


# -------------------------
# Locomotion shaping
# -------------------------
def alive_bonus_curriculum(env: "ManagerBasedRLEnv", w0=0.2, w1=0.05, w2=0.02) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device) * _stage_w(env, w0, w1, w2)


def upright_reward_curriculum(env: "ManagerBasedRLEnv", w0=1.0, w1=1.0, w2=1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=q.device).unsqueeze(0).repeat(q.shape[0], 1)
    g_b = quat_rotate_inverse(q, g_world)
    # upright => g_b ~= (0,0,-1) => -g_b.z ~= 1
    r = (-g_b[:, 2]).clamp(0.0, 1.0)
    return r * _stage_w(env, w0, w1, w2)


def root_height_reward_curriculum(
    env: "ManagerBasedRLEnv",
    target_z: float = 0.78,
    sigma: float = 0.10,
    w0: float = 1.0,
    w1: float = 0.5,
    w2: float = 0.3,
) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    err = (z - target_z) / sigma
    r = torch.exp(-err * err)
    return r * _stage_w(env, w0, w1, w2)


def track_cmd_lin_vel_xy_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.35,
    w0: float = 1.0,
    w1: float = 0.8,
    w2: float = 0.8,
) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    cmd = getattr(env, "urop_cmd", None)
    if cmd is None:
        return torch.zeros(env.num_envs, device=q.device)
    err = (v_b[:, 0:2] - cmd[:, 0:2]) / sigma
    r = torch.exp(-torch.sum(err * err, dim=-1))
    return r * _stage_w(env, w0, w1, w2)


def track_cmd_yaw_rate_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.6,
    w0: float = 0.5,
    w1: float = 0.5,
    w2: float = 0.5,
) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    cmd = getattr(env, "urop_cmd", None)
    if cmd is None:
        return torch.zeros(env.num_envs, device=q.device)
    err = (w_b[:, 2] - cmd[:, 2]) / sigma
    r = torch.exp(-err * err)
    return r * _stage_w(env, w0, w1, w2)


def joint_vel_l2_penalty_curriculum(env: "ManagerBasedRLEnv", w0=0.01, w1=0.01, w2=0.015) -> torch.Tensor:
    jv = env.scene["robot"].data.joint_vel
    p = torch.sum(jv * jv, dim=-1)
    return p * _stage_w(env, w0, w1, w2)


def action_rate_penalty_curriculum(env: "ManagerBasedRLEnv", w0=0.05, w1=0.02, w2=0.01) -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    base = torch.sum((a - a_prev) ** 2, dim=-1)
    return base * _stage_w(env, w0, w1, w2)


# -------------------------
# Catch & Carry shaping
# -------------------------
def hand_reach_object_reward_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.6,
    w0: float = 0.0,
    w1: float = 1.2,
    w2: float = 1.0,
) -> torch.Tensor:
    # use observation helper (hand_object_vectors) results if cached? recompute here:
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # body indices cached in observations module
    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    li = mp["left_wrist_roll_rubber_hand"]
    ri = mp["right_wrist_roll_rubber_hand"]

    lpos = robot.data.body_pos_w[:, li, :]
    rpos = robot.data.body_pos_w[:, ri, :]
    op = obj.data.root_pos_w

    dl = torch.norm(op - lpos, dim=-1)
    dr = torch.norm(op - rpos, dim=-1)
    d = 0.5 * (dl + dr)
    r = torch.exp(-(d / sigma) ** 2)
    return r * _stage_w(env, w0, w1, w2)

'''
def hold_pose_reward_curriculum(
    env: "ManagerBasedRLEnv",
    target_offset: tuple[float, float, float] = (0.55, 0.0, 1.05),
    sigma: float = 0.35,
    w0: float = 0.0,
    w1: float = 2.0,
    w2: float = 2.5,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rq = robot.data.root_quat_w
    rp = robot.data.root_pos_w
    op = obj.data.root_pos_w

    rel_p_b = quat_rotate_inverse(rq, op - rp)
    tgt = torch.tensor(target_offset, device=rel_p_b.device).unsqueeze(0)
    err = (rel_p_b - tgt) / sigma

    # gate: 너무 멀리 있을 땐 hold shaping을 약하게
    dist = torch.norm(rel_p_b, dim=-1)
    gate = torch.exp(-(dist / 2.0) ** 2)

    r = torch.exp(-torch.sum(err * err, dim=-1)) * gate
    return r * _stage_w(env, w0, w1, w2)'''

def hold_pose_reward_curriculum(
    env: "ManagerBasedRLEnv",
    asset_name: str = "object",
    target_offset: tuple[float, float, float] = (0.55, 0.0, 1.05),
    # 기존 단일 sigma도 지원(주면 xy/z 둘 다 sigma로)
    sigma: float | None = None,
    xy_sigma: float = 0.35,
    z_sigma: float = 0.35,
    gate_sigma: float = 2.0,
    w0: float = 0.0,
    w1: float = 2.0,
    w2: float = 2.5,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene[asset_name]

    rq = robot.data.root_quat_w
    rp = robot.data.root_pos_w
    op = obj.data.root_pos_w

    # object position in base frame
    rel_p_b = quat_rotate_inverse(rq, op - rp)

    tgt = torch.tensor(target_offset, device=rel_p_b.device).unsqueeze(0)

    # allow legacy 'sigma'
    if sigma is not None:
        xy_sigma = float(sigma)
        z_sigma = float(sigma)

    # anisotropic error (xy / z)
    exy = (rel_p_b[:, 0:2] - tgt[:, 0:2]) / xy_sigma
    ez = (rel_p_b[:, 2] - tgt[:, 2]) / z_sigma
    err = (exy * exy).sum(dim=-1) + (ez * ez)

    # gate when too far
    dist = torch.norm(rel_p_b, dim=-1)
    gate = torch.exp(-((dist / gate_sigma) ** 2))

    r = torch.exp(-err) * gate
    return r * _stage_w(env, w0, w1, w2)



def hold_object_vel_reward_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.8,
    w0: float = 0.0,
    w1: float = 0.8,
    w2: float = 1.2,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    rq = robot.data.root_quat_w
    rv = robot.data.root_lin_vel_w
    ov = obj.data.root_lin_vel_w
    rel_v_b = quat_rotate_inverse(rq, ov - rv)
    v = torch.norm(rel_v_b, dim=-1)
    r = torch.exp(-(v / sigma) ** 2)
    return r * _stage_w(env, w0, w1, w2)


def contact_hold_bonus_curriculum(
    env: "ManagerBasedRLEnv",
    sensor_names: list[str],
    thr: float = 1.0,
    w0: float = 0.0,
    w1: float = 0.4,
    w2: float = 0.6,
) -> torch.Tensor:
    # bonus if contact exists on both hands (and/or torso)
    forces = []
    for name in sensor_names:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        forces.append(torch.norm(f, dim=-1))
    mags = torch.stack(forces, dim=-1)  # (N, num_sensors)

    contact = (mags > thr).float()
    # both hands
    both = contact[:, 1] * contact[:, 2]
    torso = contact[:, 0]
    bonus = 0.6 * both + 0.4 * torso
    return bonus * _stage_w(env, w0, w1, w2)


def impact_peak_penalty_curriculum(
    env: "ManagerBasedRLEnv",
    sensor_names: list[str],
    force_thr_stage1: float = 350.0,
    force_thr_stage2: float = 300.0,
    w0: float = 0.0,
    w1: float = 0.08,
    w2: float = 0.12,
) -> torch.Tensor:
    stage = _get_stage(env)
    if stage == 0:
        return torch.zeros(env.num_envs, device=env.device)
    thr = force_thr_stage1 if stage == 1 else force_thr_stage2

    peaks = []
    for name in sensor_names:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        peaks.append(torch.norm(f, dim=-1))
    peak = torch.stack(peaks, dim=-1).max(dim=-1).values  # (N,)

    pen = torch.relu(peak - thr) / thr
    return pen * _stage_w(env, w0, w1, w2)


def object_not_dropped_bonus_curriculum(
    env: "ManagerBasedRLEnv",
    min_z: float = 0.22,
    max_dist: float = 2.5,
    w0: float = 0.0,
    w1: float = 0.3,
    w2: float = 0.4,
) -> torch.Tensor:
    stage = _get_stage(env)
    if stage == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    z_ok = (obj.data.root_pos_w[:, 2] > min_z).float()
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    d_ok = (dist < max_dist).float()
    return (z_ok * d_ok) * _stage_w(env, w0, w1, w2)
