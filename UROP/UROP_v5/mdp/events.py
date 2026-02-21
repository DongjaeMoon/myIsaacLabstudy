from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from .observations import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_stage(env) -> int:
    """0/1/2 stage computed from curriculum schedule (or forced eval_stage)."""
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
    """Extract yaw-only quaternion from (w,x,y,z)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half = 0.5 * yaw
    return torch.stack(
        [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1
    )


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
    """(Optional) reset 시점에 object를 바로 던지고 싶을 때 사용."""
    s = _get_stage(env)
    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)

    robot = env.scene["robot"]
    obj = env.scene["object"]

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

    q_w = torch.tensor([1, 0, 0, 0], device=device).repeat(n, 1)

    obj.write_root_pose_to_sim(torch.cat([p_w, q_w], dim=-1), env_ids=env_ids)
    vel6 = torch.zeros((n, 6), device=device)
    vel6[:, 0:3] = v_w
    obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)


# -----------------------------------------------------------------------------
# Toss state buffers (핵심 수정)
#   - _urop_toss_done        : "이 에피소드에서 toss 이벤트 처리 여부" (interval 중복 방지)
#   - _urop_toss_active      : "실제로 던져졌는지" (관측/보상/종료 게이팅의 진짜 기준)
#   - _urop_spawn_xy         : reset 시점 base xy (Wait 제자리 유지 페널티용)
#   - _urop_ready_joint_pos  : reset 시점 joint pose (ready_pose 타겟)
# -----------------------------------------------------------------------------
def _ensure_urop_toss_buffers(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "_urop_toss_done"):
        env._urop_toss_done = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    if not hasattr(env, "_urop_toss_active"):
        env._urop_toss_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_urop_spawn_xy"):
        env._urop_spawn_xy = torch.zeros((env.num_envs, 2), device=env.device)
    if not hasattr(env, "_urop_ready_joint_pos"):
        robot = env.scene["robot"]
        env._urop_ready_joint_pos = robot.data.joint_pos.clone()


def reset_object_parked(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    park: dict,
) -> None:
    """
    reset 모드:
      - object를 로봇 옆(주차) 위치로 옮김
      - toss 상태/기준 pose 저장
    """
    _ensure_urop_toss_buffers(env)

    # toss 상태 리셋
    env._urop_toss_done[env_ids] = 0
    env._urop_toss_active[env_ids] = False

    # Wait 기준 저장(제자리/자세)
    robot = env.scene["robot"]
    env._urop_spawn_xy[env_ids] = robot.data.root_pos_w[env_ids, 0:2]
    env._urop_ready_joint_pos[env_ids] = robot.data.joint_pos[env_ids]

    # object 주차
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
      - 랜덤 타이밍 toss
      - episode당 max_throws_per_episode 번만 toss 시도
      - stage0은 toss 자체를 하지 않음
      - stage2에서는 확률적으로 no-throw episode 섞기 가능

    [중요 수정]
      - no-throw episode에서 "던졌다고 관측"되면 정책이 시작부터 catch 자세로 치팅함.
      - 그래서 toss_active(실제 throw 발생)와 toss_done(이벤트 종료)를 분리한다.
    """
    _ensure_urop_toss_buffers(env)

    if max_throws_per_episode <= 0:
        return

    # 이미 toss 시도한 env 제외
    can = env._urop_toss_done[env_ids] < int(max_throws_per_episode)
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
        do_throw = u < prob

        ids_throw = env_ids[do_throw]
        ids_skip = env_ids[~do_throw]

        # no-throw env: 이벤트는 '끝'으로 처리하되 toss_active는 False 유지
        if ids_skip.numel() > 0:
            env._urop_toss_done[ids_skip] = int(max_throws_per_episode)
            env._urop_toss_active[ids_skip] = False

        if ids_throw.numel() == 0:
            return
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

    # toss 상태 갱신
    env._urop_toss_done[ids_throw] += 1
    env._urop_toss_active[ids_throw] = True