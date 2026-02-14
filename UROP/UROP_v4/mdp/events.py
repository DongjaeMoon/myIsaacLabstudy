from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply

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

def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    # extract yaw-only quaternion from (w,x,y,z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    half = 0.5 * yaw
    return torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)

def reset_robot_base_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
):
    s = _get_stage(env)
    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)

    robot = env.scene["robot"]
    device = robot.data.root_pos_w.device
    n = env_ids.shape[0]

    vel = robot.data.root_vel_w[env_ids].clone()

    vel[:, 0] = torch.empty(n, device=device).uniform_(*cfg["lin_x"])
    vel[:, 1] = torch.empty(n, device=device).uniform_(*cfg["lin_y"])
    vel[:, 2] = 0.0
    vel[:, 3] = 0.0
    vel[:, 4] = 0.0
    vel[:, 5] = torch.empty(n, device=device).uniform_(*cfg["yaw_rate"])

    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)


def reset_and_toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
):
    s = _get_stage(env)
    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)

    robot = env.scene["robot"]
    obj   = env.scene["object"]

    device = obj.data.root_pos_w.device
    n = env_ids.shape[0]

    rp = robot.data.root_pos_w[env_ids]
    rq = _yaw_quat(robot.data.root_quat_w[env_ids])

    px = torch.empty(n, device=device).uniform_(*cfg["pos_x"])
    py = torch.empty(n, device=device).uniform_(*cfg["pos_y"])
    pz = torch.empty(n, device=device).uniform_(*cfg["pos_z"])
    rel_p = torch.stack([px, py, pz], dim=-1)

    vx = torch.empty(n, device=device).uniform_(*cfg["vel_x"])
    vy = torch.empty(n, device=device).uniform_(*cfg["vel_y"])
    vz = torch.empty(n, device=device).uniform_(*cfg["vel_z"])
    rel_v = torch.stack([vx, vy, vz], dim=-1)

    p_w = rp + quat_apply(rq, rel_p)
    v_w = robot.data.root_lin_vel_w[env_ids] + quat_apply(rq, rel_v)

    q_w = torch.tensor([1,0,0,0], device=device).repeat(n,1)

    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=env_ids)
    vel6 = torch.zeros((n,6), device=device)
    vel6[:,0:3] = v_w
    obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)


def _ensure_urop_toss_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "_urop_toss_count"):
        env._urop_toss_count = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

def reset_object_parked(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    park: dict,
) -> None:
    """
    reset 모드:
      - object를 로봇 옆(주차) 위치로 옮김
      - toss_count를 0으로 리셋 (게이팅 기준)
    park keys: pos_x,pos_y,pos_z : (min,max) tuples, robot-root 기준 상대좌표
    """
    _ensure_urop_toss_buffers(env)
    env._urop_toss_count[env_ids] = 0

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device
    n = env_ids.shape[0]

    rp = robot.data.root_pos_w[env_ids]
    rq = _yaw_quat(robot.data.root_quat_w[env_ids])

    px = torch.empty(n, device=device).uniform_(*park["pos_x"])
    py = torch.empty(n, device=device).uniform_(*park["pos_y"])
    pz = torch.empty(n, device=device).uniform_(*park["pos_z"])
    rel_p = torch.stack([px, py, pz], dim=-1)

    p_w = rp + quat_apply(rq, rel_p)
    q_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)

    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=env_ids)

    vel6 = torch.zeros((n, 6), device=device)
    obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)


def toss_object_relative_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict,
    stage1: dict,
    stage2: dict,
    throw_prob_stage1: float = 1.0,
    throw_prob_stage2: float = 0.85,
    max_throws_per_episode: int = 1,
) -> None:
    """
    interval 모드:
      - 랜덤 타이밍에 호출됨(EventManager가 interval_range_s를 관리)
      - episode당 max_throws_per_episode 번만 toss
      - stage0은 toss 자체를 하지 않음
      - stage2에서는 확률적으로 no-throw episode 섞기 가능
    """
    _ensure_urop_toss_buffers(env)

    if max_throws_per_episode <= 0:
        return

    # 이미 던진 env 제외
    can = env._urop_toss_count[env_ids] < int(max_throws_per_episode)
    if not torch.any(can):
        return
    env_ids = env_ids[can]

    s = _get_stage(env)
    if s == 0:
        return

    cfg = stage1 if s == 1 else stage2
    prob = float(throw_prob_stage1 if s == 1 else throw_prob_stage2)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    device = obj.data.root_pos_w.device

    n = env_ids.shape[0]

    # stage2에서 일부 env는 no-throw episode로 만들기
    if prob < 1.0:
        u = torch.rand(n, device=device)
        do_throw = (u < prob)
        if not torch.any(do_throw):
            env._urop_toss_count[env_ids] = int(max_throws_per_episode)
            return
        ids_throw = env_ids[do_throw]
        ids_skip = env_ids[~do_throw]
        if ids_skip.numel() > 0:
            env._urop_toss_count[ids_skip] = int(max_throws_per_episode)
    else:
        ids_throw = env_ids

    n2 = ids_throw.shape[0]

    rp = robot.data.root_pos_w[ids_throw]
    rq = _yaw_quat(robot.data.root_quat_w[ids_throw])

    px = torch.empty(n2, device=device).uniform_(*cfg["pos_x"])
    py = torch.empty(n2, device=device).uniform_(*cfg["pos_y"])
    pz = torch.empty(n2, device=device).uniform_(*cfg["pos_z"])
    rel_p = torch.stack([px, py, pz], dim=-1)

    vx = torch.empty(n2, device=device).uniform_(*cfg["vel_x"])
    vy = torch.empty(n2, device=device).uniform_(*cfg["vel_y"])
    vz = torch.empty(n2, device=device).uniform_(*cfg["vel_z"])
    rel_v = torch.stack([vx, vy, vz], dim=-1)

    p_w = rp + quat_apply(rq, rel_p)
    v_w = robot.data.root_lin_vel_w[ids_throw] + quat_apply(rq, rel_v)

    q_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n2, 1)
    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=ids_throw)

    vel6 = torch.zeros((n2, 6), device=device)
    vel6[:, 0:3] = v_w
    obj.write_root_velocity_to_sim(vel6, env_ids=ids_throw)

    env._urop_toss_count[ids_throw] += 1
