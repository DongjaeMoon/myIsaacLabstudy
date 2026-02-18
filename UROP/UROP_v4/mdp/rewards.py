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


# [UROP_v4/mdp/rewards.py]

def contact_hold_bonus_symmetric(
    env: "ManagerBasedRLEnv", 
    sensor_names_left: list[str], 
    sensor_names_right: list[str],
    sensor_names_torso: list[str] = None,  # <-- 이 부분이 있어야 함
    thr=1.0, 
    w0=0.0, 
    w1=0.6, 
    w2=0.9
) -> torch.Tensor:
    # 1. 왼쪽
    left_contacts = []
    for name in sensor_names_left:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        left_contacts.append((torch.norm(f, dim=-1) > thr).float())
    
    # 2. 오른쪽
    right_contacts = []
    for name in sensor_names_right:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        right_contacts.append((torch.norm(f, dim=-1) > thr).float())

    # 3. 몸통 (보너스)
    if sensor_names_torso is not None:
        torso_contacts = []
        for name in sensor_names_torso:
            s = env.scene[name]
            f = s.data.net_forces_w.reshape(env.num_envs, -1)
            torso_contacts.append((torch.norm(f, dim=-1) > thr).float())
        t_hit = torch.stack(torso_contacts, dim=-1).max(dim=-1).values
    else:
        t_hit = torch.zeros(env.num_envs, device=env.device)

    l_hit = torch.stack(left_contacts, dim=-1).max(dim=-1).values
    r_hit = torch.stack(right_contacts, dim=-1).max(dim=-1).values
    
    # [로직] 양손 필수(AND) * (1 + 몸통보너스)
    hands_ok = l_hit * r_hit 
    score = hands_ok * (1.0 + 1.0 * t_hit) 

    return (score * _stage_w(env, w0, w1, w2)) * _toss_active(env)

def ready_pose_when_waiting(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.5,
    w0: float = 1.0,  # stage0 가중치
    w1: float = 1.0,  # stage1 가중치
    w2: float = 1.0,  # stage2 가중치
) -> torch.Tensor:
    """
    [핵심] Toss 신호가 0일 때(대기 중), 로봇이 초기 자세(Default Pose)와 다르면 점수를 깎음.
    이게 있어야 '팔 뒤로 뻗고 대기하는' 이상한 짓을 안 함.
    """
    # 1. 현재 던져진 상태인지 확인
    if hasattr(env, "_urop_toss_count"):
        is_tossed = (env._urop_toss_count > 0).float()
    else:
        is_tossed = torch.ones(env.num_envs, device=env.device)

    # 2. 로봇 데이터 가져오기
    robot = env.scene["robot"]
    
    # default_joint_pos는 로봇 로딩될 때의 그 '차려 자세'입니다.
    target = robot.data.default_joint_pos
    current = robot.data.joint_pos

    # 3. 자세 차이 계산
    diff = torch.norm(current - target, dim=-1)
    rew = torch.exp(-(diff / sigma) ** 2)

    # 4. [중요] 박스가 날아오면(is_tossed=1) 이 보상을 꺼버림 (0점)
    # 그래야 박스 잡으려고 자유롭게 움직임
    mask = (1.0 - is_tossed)
    
    return rew * mask * _stage_w(env, w0, w1, w2)


# UROP_v4/mdp/rewards.py

# 1. 대기 중에 움직이면 감점 (뒷걸음질 방지)
def stand_still_when_waiting_penalty(
    env: "ManagerBasedRLEnv", 
    w_lin: float = 0.1, 
    w_ang: float = 0.05
) -> torch.Tensor:
    # Toss가 0일 때(대기 중)만 작동
    if hasattr(env, "_urop_toss_count"):
        is_tossed = (env._urop_toss_count > 0).float()
    else:
        is_tossed = torch.ones(env.num_envs, device=env.device)
    
    # Toss가 1이면 mask=0 (페널티 없음), Toss가 0이면 mask=1 (페널티 적용)
    mask = (1.0 - is_tossed)
    
    robot = env.scene["robot"]
    # x,y 속도 제곱
    lin_sq = torch.sum(robot.data.root_lin_vel_w[:, 0:2] ** 2, dim=-1)
    ang_sq = torch.sum(robot.data.root_ang_vel_w[:, 2] ** 2, dim=-1)
    
    return (w_lin * lin_sq + w_ang * ang_sq) * mask

# 2. 팔 뒤로 꺾기 방지 (Soft Joint Limit)
# URDF 수정 없이 "나루토 자세"를 막는 핵심 함수
def arm_extension_penalty(env: "ManagerBasedRLEnv", limit_angle: float = -0.5) -> torch.Tensor:
    robot = env.scene["robot"]
    
    # 1. 어깨 Pitch 관절 인덱스 찾기 (한번만 찾고 캐싱)
    if not hasattr(env, "_urop_shoulder_pitch_idx"):
        indices = []
        for i, name in enumerate(robot.data.joint_names):
            if "shoulder_pitch" in name: # 좌/우 어깨 피치
                indices.append(i)
        env._urop_shoulder_pitch_idx = indices
    
    if not env._urop_shoulder_pitch_idx:
        return torch.zeros(env.num_envs, device=env.device)

    # 2. 관절 각도 가져오기
    joint_pos = robot.data.joint_pos[:, env._urop_shoulder_pitch_idx]
    
    # 3. limit_angle보다 더 뒤로(-값) 가면 페널티
    # 예: -0.5보다 작으면(더 뒤로 젖히면) 위반
    violation = torch.clamp(limit_angle - joint_pos, min=0.0)
    
    return torch.sum(violation ** 2, dim=-1)