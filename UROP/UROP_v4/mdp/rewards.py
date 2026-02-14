from __future__ import annotations
import torch
import math
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

def _upright_cos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """cos(tilt). 1=완전 직립, 0=90도 누움"""
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    return (-g_b[:, 2]).clamp(0.0, 1.0)

def _toss_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_urop_toss_count"):
        return (env._urop_toss_count > 0).float()
    return torch.ones(env.num_envs, device=env.device)



# -------------------------
# Basic stabilization
# -------------------------
def alive_bonus_curriculum(env: "ManagerBasedRLEnv", w0=0.2, w1=0.05, w2=0.02) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device) * _stage_w(env, w0, w1, w2)

def upright_reward_curriculum(env: "ManagerBasedRLEnv", w0=1.0, w1=1.0, w2=1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    g_b = quat_rotate_inverse(q, g_world)
    r = (-g_b[:, 2]).clamp(0.0, 1.0)
    return r * _stage_w(env, w0, w1, w2)

def root_height_reward_curriculum(env: "ManagerBasedRLEnv", target_z=0.78, sigma=0.10, w0=1.0, w1=0.5, w2=0.3):
    z = env.scene["robot"].data.root_pos_w[:, 2]
    err = (z - target_z) / sigma
    return torch.exp(-err * err) * _stage_w(env, w0, w1, w2)

def base_velocity_penalty_curriculum(env: "ManagerBasedRLEnv", w_lin=1.0, w_ang=0.3, w0=0.2, w1=0.08, w2=0.06):
    robot = env.scene["robot"]
    q = robot.data.root_quat_w
    v_b = quat_rotate_inverse(q, robot.data.root_lin_vel_w)
    w_b = quat_rotate_inverse(q, robot.data.root_ang_vel_w)
    pen = w_lin * torch.sum(v_b[:, 0:2] ** 2, dim=-1) + w_ang * (w_b[:, 2] ** 2)
    return pen * _stage_w(env, w0, w1, w2)

def joint_vel_l2_penalty_curriculum(env: "ManagerBasedRLEnv", w0=0.01, w1=0.01, w2=0.015) -> torch.Tensor:
    jv = env.scene["robot"].data.joint_vel
    return torch.sum(jv * jv, dim=-1) * _stage_w(env, w0, w1, w2)

def torque_l2_penalty_curriculum(env: "ManagerBasedRLEnv", w0=0.0, w1=0.002, w2=0.003) -> torch.Tensor:
    robot = env.scene["robot"]
    jt = getattr(robot.data, "applied_torque", None)
    if jt is None:
        jt = getattr(robot.data, "joint_effort", None)
    if jt is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(jt * jt, dim=-1) * _stage_w(env, w0, w1, w2)

def action_rate_penalty_curriculum(env: "ManagerBasedRLEnv", w0=0.05, w1=0.02, w2=0.01) -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return torch.sum((a - a_prev) ** 2, dim=-1) * _stage_w(env, w0, w1, w2)

# -------------------------
# Catch / Hold shaping
# -------------------------
def torso_reach_object_reward_curriculum(env: "ManagerBasedRLEnv", sigma=0.7, w0=0.0, w1=1.0, w2=0.8):
    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    r = torch.exp(-(d / sigma) ** 2) * _stage_w(env, w0, w1, w2)
    return r * _toss_active(env)


def hold_pose_reward_curriculum(
    env: "ManagerBasedRLEnv",
    torso_body_name: str = "torso_link",
    sigma: float = 0.35,
    upright_pow: float = 2.0,   # <-- 추가
    min_upright: float = 0.2,   # <-- 추가
    w0: float = 0.0,
    w1: float = 2.0,
    w2: float = 2.5,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # torso body index 찾기 (한 번만 만들고 캐시)
    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    if torso_body_name in mp and hasattr(robot.data, "body_pos_w"):
        ti = mp[torso_body_name]
        torso_pos = robot.data.body_pos_w[:, ti, :]
    else:
        # torso 이름이 안 맞으면 root로 fallback
        torso_pos = robot.data.root_pos_w

    d = torch.norm(obj.data.root_pos_w - torso_pos, dim=-1)
    r = torch.exp(-(d / sigma) ** 2)
    # ---- 추가
    u = _upright_cos(env)
    r = r * torch.clamp(u, min_upright, 1.0) ** upright_pow
    return (r * _stage_w(env, w0, w1, w2)) * _toss_active(env)


def hold_object_vel_reward_curriculum(
    env: "ManagerBasedRLEnv",
    torso_body_name: str = "torso_link",
    sigma: float = 0.6,
    upright_pow: float = 2.0,   # <-- 추가
    min_upright: float = 0.2,   # <-- 추가
    w0: float = 0.0,
    w1: float = 0.6,
    w2: float = 1.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]

    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    # torso 선속도 가져오기 (가능한 필드에 맞춰 robust하게)
    if torso_body_name in mp:
        ti = mp[torso_body_name]
        if hasattr(robot.data, "body_lin_vel_w"):
            torso_v = robot.data.body_lin_vel_w[:, ti, :]
        elif hasattr(robot.data, "body_vel_w"):
            torso_v = robot.data.body_vel_w[:, ti, 0:3]
        else:
            torso_v = robot.data.root_lin_vel_w
    else:
        torso_v = robot.data.root_lin_vel_w

    dv = torch.norm(obj.data.root_lin_vel_w - torso_v, dim=-1)
    r = torch.exp(-(dv / sigma) ** 2)

    # ---- 추가
    u = _upright_cos(env)
    r = r * torch.clamp(u, min_upright, 1.0) ** upright_pow
    return (r * _stage_w(env, w0, w1, w2)) * _toss_active(env)


def contact_hold_bonus_curriculum(env: "ManagerBasedRLEnv", sensor_names: list[str], thr=1.0, w0=0.0, w1=0.6, w2=0.9):
    # 여러 링크 접촉을 +로 준다 (조교님 코멘트 4)
    # sensor_names 순서: torso, l_upper, l_fore, l_hand, r_upper, r_fore, r_hand (권장)
    contact = []
    for name in sensor_names:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        mag = torch.norm(f, dim=-1)
        contact.append((mag > thr).float())
    c = torch.stack(contact, dim=-1)
    # torso + 양팔(최소 2~3점 이상) 접촉을 높게
    torso = c[:, 0]
    arms = c[:, 1:].sum(dim=-1) / max(c.shape[1] - 1, 1)
    score = 0.4 * torso + 0.6 * arms
    return (score * _stage_w(env, w0, w1, w2)) * _toss_active(env)

def impact_peak_penalty_curriculum(env: "ManagerBasedRLEnv", sensor_names: list[str], force_thr_stage1=400.0, force_thr_stage2=300.0, w0=0.0, w1=0.05, w2=0.10):
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, "_urop_toss_count"):
        active = (env._urop_toss_count > 0)
    else:
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    thr = force_thr_stage1 if s == 1 else force_thr_stage2
    peaks = []
    for name in sensor_names:
        try:
            ss = env.scene[name]
        except Exception:
            continue
        f = ss.data.net_forces_w.reshape(env.num_envs, -1)
        peaks.append(torch.norm(f, dim=-1))

    if len(peaks) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    peak = torch.stack(peaks, dim=-1).max(dim=-1).values
    pen = torch.relu(peak - thr) / thr
    pen = pen * _stage_w(env, w0, w1, w2)

    out = torch.zeros(env.num_envs, device=env.device)
    out[active] = pen[active]
    return out

def object_not_dropped_bonus_curriculum(env: "ManagerBasedRLEnv", min_z=0.70, max_dist=2.5, w0=0.0, w1=0.4, w2=0.5):
    # 네 말대로 "세워서 치팅" 방지하려면 min_z를 박스 크기에 맞춰 높게 잡는 게 효과적
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    z_ok = (obj.data.root_pos_w[:, 2] > min_z).float()
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    d_ok = (dist < max_dist).float()
    u = _upright_cos(env)
    upright_gate = torch.clamp(u, 0.0, 1.0) ** 2.0   # <-- 추가 (기울면 보너스 깎임)
    return ((z_ok * d_ok) * upright_gate * _stage_w(env, w0, w1, w2)) * _toss_active(env)