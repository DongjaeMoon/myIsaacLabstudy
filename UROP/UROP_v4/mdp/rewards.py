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


# [UROP_v4/mdp/rewards.py] 맨 아래에 추가

def hands_reach_object_reward_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.5,
    w0: float = 0.0,
    w1: float = 1.0,
    w2: float = 1.0,
) -> torch.Tensor:
    """
    [NEW] 양손이 모두 박스에 가까워져야만 높은 점수를 줌.
    하나라도 멀어지면 점수가 급격히 떨어지도록 설계 (Product logic).
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]
    
    # 1. Body Index 찾기 (Left Hand, Right Hand)
    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    # 손 이름은 scene_objects_cfg.py의 contact sensor 이름 참고
    # (G1 로봇의 손 링크 이름을 정확히 넣어야 함. 아래는 추정값)
    l_name = "left_wrist_roll_rubber_hand" # 또는 left_hand_link
    r_name = "right_wrist_roll_rubber_hand" # 또는 right_hand_link
    
    # 이름이 없으면 0점 처리 방지용 fallback
    if l_name in mp and r_name in mp:
        l_idx = mp[l_name]
        r_idx = mp[r_name]
        l_pos = robot.data.body_pos_w[:, l_idx, :]
        r_pos = robot.data.body_pos_w[:, r_idx, :]
    else:
        # 만약 이름을 못 찾으면 root로 대체하되 경고 느낌 (보통은 잘 찾음)
        l_pos = robot.data.root_pos_w
        r_pos = robot.data.root_pos_w

    # 2. 거리 계산
    d_l = torch.norm(obj.data.root_pos_w - l_pos, dim=-1)
    d_r = torch.norm(obj.data.root_pos_w - r_pos, dim=-1)

    # 3. 핵심 로직: 곱하기(*)를 사용해서 하나라도 멀면 0점이 되게 함!
    # (Counterweight로 한 팔 뒤로 빼면 d_r이 커져서 전체 점수가 0이 됨)
    rew_l = torch.exp(-(d_l / sigma) ** 2)
    rew_r = torch.exp(-(d_r / sigma) ** 2)
    
    return (rew_l * rew_r) * _stage_w(env, w0, w1, w2) * _toss_active(env)


def contact_hold_bonus_symmetric(
    env: "ManagerBasedRLEnv", 
    sensor_names_left: list[str], 
    sensor_names_right: list[str], 
    thr=1.0, 
    w0=0.0, 
    w1=0.6, 
    w2=0.9
) -> torch.Tensor:
    """
    [MODIFIED] 왼쪽과 오른쪽이 '동시에' 닿아야만 보상.
    한쪽만 닿으면 국물도 없음.
    """
    # 1. 왼쪽 접촉 확인
    left_contacts = []
    for name in sensor_names_left:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        left_contacts.append((torch.norm(f, dim=-1) > thr).float())
    
    # 2. 오른쪽 접촉 확인
    right_contacts = []
    for name in sensor_names_right:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        right_contacts.append((torch.norm(f, dim=-1) > thr).float())

    # 3. 각 팔이 접촉했는가? (Max로 하나라도 닿으면 True)
    l_hit = torch.stack(left_contacts, dim=-1).max(dim=-1).values
    r_hit = torch.stack(right_contacts, dim=-1).max(dim=-1).values

    # 4. 양쪽 다 닿았을 때만 보상 (AND 조건)
    score = l_hit * r_hit 

    return (score * _stage_w(env, w0, w1, w2)) * _toss_active(env)