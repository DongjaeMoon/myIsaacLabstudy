from __future__ import annotations
import torch
import math
from typing import TYPE_CHECKING

from .observations import quat_rotate_inverse, quat_mul, quat_conj, quat_apply

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
    # "실제로 던져졌는지" 기준 (no-throw episode를 확실히 분리)
    if hasattr(env, "_urop_toss_active"):
        return env._urop_toss_active.float()
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
# Catch / Hold rewards
# -------------------------
def torso_reach_object_reward_curriculum(env: "ManagerBasedRLEnv", sigma=0.8, w0=0.0, w1=1.0, w2=0.8):
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    obj = env.scene["object"]

    # torso body name은 env_cfg에서 맞춰줘야 함
    torso_name = "torso_link"
    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp
    if torso_name in mp:
        idx = mp[torso_name]
        torso = robot.data.body_pos_w[:, idx, :]
    else:
        torso = robot.data.root_pos_w

    d = torch.norm(obj.data.root_pos_w - torso, dim=-1)
    rew = torch.exp(-(d / sigma) ** 2)
    return rew * _stage_w(env, w0, w1, w2) * _toss_active(env)

def hold_pose_reward_curriculum(
    env: "ManagerBasedRLEnv",
    torso_body_name: str = "torso_link",
    sigma: float = 0.35,
    target_offset: tuple[float, float, float] = (0.2, 0.0, 0.2),
    w0: float = 0.0,
    w1: float = 2.0,
    w2: float = 2.5,
):
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    if torso_body_name in mp:
        tidx = mp[torso_body_name]
        torso_pos = robot.data.body_pos_w[:, tidx, :]
    else:
        torso_pos = robot.data.root_pos_w

    rq = robot.data.root_quat_w
    offset = torch.tensor(target_offset, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    target = torso_pos + quat_apply(rq, offset)

    d = torch.norm(obj.data.root_pos_w - target, dim=-1)
    rew = torch.exp(-(d / sigma) ** 2)
    return rew * _stage_w(env, w0, w1, w2) * _toss_active(env)

def hold_object_vel_reward_curriculum(
    env: "ManagerBasedRLEnv",
    torso_body_name: str = "torso_link",
    sigma: float = 0.8,
    w0: float = 0.0,
    w1: float = 1.0,
    w2: float = 1.2,
):
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    if torso_body_name in mp:
        tidx = mp[torso_body_name]
        torso_vel = robot.data.body_lin_vel_w[:, tidx, :]
    else:
        torso_vel = robot.data.root_lin_vel_w

    rel_v = obj.data.root_lin_vel_w - torso_vel
    speed = torch.norm(rel_v, dim=-1)
    rew = torch.exp(-(speed / sigma) ** 2)
    return rew * _stage_w(env, w0, w1, w2) * _toss_active(env)

def impact_peak_penalty_curriculum(env: "ManagerBasedRLEnv", sensor_names: list[str], force_thr_stage1=400.0, force_thr_stage2=300.0, w0=0.0, w1=0.05, w2=0.10):
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)

    if hasattr(env, "_urop_toss_active"):
        active = env._urop_toss_active
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
    s = _get_stage(env)
    if s == 0:
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    z_ok = (obj.data.root_pos_w[:, 2] > min_z).float()
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    d_ok = (dist < max_dist).float()
    u = _upright_cos(env)
    upright_gate = torch.clamp(u, 0.0, 1.0) ** 2.0
    return ((z_ok * d_ok) * upright_gate * _stage_w(env, w0, w1, w2)) * _toss_active(env)


def hands_reach_object_reward_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.5,
    w0: float = 0.0,
    w1: float = 1.0,
    w2: float = 1.0,
) -> torch.Tensor:
    """
    양손이 모두 박스에 가까워져야만 높은 점수를 줌 (곱 논리).
    """
    robot = env.scene["robot"]
    obj = env.scene["object"]

    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    l_name = "left_wrist_roll_rubber_hand"
    r_name = "right_wrist_roll_rubber_hand"

    if l_name in mp and r_name in mp:
        l_idx = mp[l_name]
        r_idx = mp[r_name]
        l_pos = robot.data.body_pos_w[:, l_idx, :]
        r_pos = robot.data.body_pos_w[:, r_idx, :]
    else:
        l_pos = robot.data.root_pos_w
        r_pos = robot.data.root_pos_w

    d_l = torch.norm(obj.data.root_pos_w - l_pos, dim=-1)
    d_r = torch.norm(obj.data.root_pos_w - r_pos, dim=-1)

    rew_l = torch.exp(-(d_l / sigma) ** 2)
    rew_r = torch.exp(-(d_r / sigma) ** 2)

    return (rew_l * rew_r) * _stage_w(env, w0, w1, w2) * _toss_active(env)


def hands_support_under_box_reward_curriculum(
    env: "ManagerBasedRLEnv",
    box_size: tuple[float, float, float] = (0.4, 0.3, 0.3),
    y_frac: float = 0.45,
    z_clearance: float = 0.03,
    sigma: float = 0.18,
    w0: float = 0.0,
    w1: float = 1.0,
    w2: float = 1.2,
) -> torch.Tensor:
    """
    [강추] '밑부분을 받쳐서 + 가슴으로 안기기' 형태를 유도하는 보상.
    - 박스 바닥면 바로 아래에 왼/오른손 목표점을 두고 각각 가까울수록 보상.
    """
    if _get_stage(env) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    active = _toss_active(env)
    if torch.all(active <= 0.0):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]

    mp = getattr(env, "_urop_body_name_to_id", None)
    if mp is None:
        names = list(getattr(robot.data, "body_names", []))
        mp = {n: i for i, n in enumerate(names)}
        env._urop_body_name_to_id = mp

    l_name = "left_wrist_roll_rubber_hand"
    r_name = "right_wrist_roll_rubber_hand"
    if l_name in mp and r_name in mp:
        l_idx = mp[l_name]
        r_idx = mp[r_name]
        l_pos = robot.data.body_pos_w[:, l_idx, :]
        r_pos = robot.data.body_pos_w[:, r_idx, :]
    else:
        l_pos = robot.data.root_pos_w
        r_pos = robot.data.root_pos_w

    y_half = 0.5 * float(box_size[1])
    z_half = 0.5 * float(box_size[2])

    oq = obj.data.root_quat_w
    op = obj.data.root_pos_w

    off_l = torch.tensor([0.0, +y_half * y_frac, -(z_half + z_clearance)], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    off_r = torch.tensor([0.0, -y_half * y_frac, -(z_half + z_clearance)], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

    p_l = op + quat_apply(oq, off_l)
    p_r = op + quat_apply(oq, off_r)

    d_l = torch.norm(l_pos - p_l, dim=-1)
    d_r = torch.norm(r_pos - p_r, dim=-1)

    rew_l = torch.exp(-(d_l / sigma) ** 2)
    rew_r = torch.exp(-(d_r / sigma) ** 2)

    return (rew_l * rew_r) * _stage_w(env, w0, w1, w2) * active


def contact_hold_bonus_symmetric(
    env: "ManagerBasedRLEnv",
    sensor_names_left: list[str],
    sensor_names_right: list[str],
    sensor_names_torso: list[str] = None,
    thr=1.0,
    w0=0.0,
    w1=0.6,
    w2=0.9
) -> torch.Tensor:
    left_contacts = []
    for name in sensor_names_left:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        left_contacts.append((torch.norm(f, dim=-1) > thr).float())

    right_contacts = []
    for name in sensor_names_right:
        s = env.scene[name]
        f = s.data.net_forces_w.reshape(env.num_envs, -1)
        right_contacts.append((torch.norm(f, dim=-1) > thr).float())

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

    hands_ok = l_hit * r_hit
    score = hands_ok * (1.0 + 1.0 * t_hit)

    return (score * _stage_w(env, w0, w1, w2)) * _toss_active(env)


def wait_base_drift_penalty(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.25,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
) -> torch.Tensor:
    """
    [핵심] Toss 전(Wait)에는 제자리에서 크게 움직이지 말게 함.
    - '박스 오기 전부터 뒷걸음 + 이상한 준비자세'를 구조적으로 막는 1순위.
    """
    if not hasattr(env, "_urop_spawn_xy"):
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    xy = robot.data.root_pos_w[:, 0:2]
    drift = torch.norm(xy - env._urop_spawn_xy, dim=-1)

    active = _toss_active(env)  # 1 if thrown, 0 if waiting
    pen = (drift / sigma) ** 2
    return pen * (1.0 - active) * _stage_w(env, w0, w1, w2)


def ready_pose_when_waiting(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.5,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
) -> torch.Tensor:
    """
    Toss 신호가 0일 때(대기 중), 로봇이 reset 당시 자세에서 크게 벗어나면 점수를 깎음.
    """
    if hasattr(env, "_urop_toss_active"):
        is_tossed = env._urop_toss_active.float()
    else:
        is_tossed = torch.ones(env.num_envs, device=env.device)

    robot = env.scene["robot"]

    # USD default_joint_pos는 T-pose/이상 포즈일 수 있음 → reset 시 저장한 ready pose를 타겟으로 사용
    target = env._urop_ready_joint_pos if hasattr(env, "_urop_ready_joint_pos") else robot.data.default_joint_pos
    current = robot.data.joint_pos

    diff = torch.norm(current - target, dim=-1)
    rew = torch.exp(-(diff / sigma) ** 2)

    mask = (1.0 - is_tossed)
    return rew * mask * _stage_w(env, w0, w1, w2)