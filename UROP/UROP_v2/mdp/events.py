# UROP/UROP_v2/mdp/events.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject

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

def _yaw_from_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """q: (N,4) in (w,x,y,z) -> yaw (N,)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def _rot_xy(yaw: torch.Tensor, v_xy: torch.Tensor) -> torch.Tensor:
    """Rotate v_xy (N,2) by yaw (N,) around +Z."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = c * v_xy[:, 0] - s * v_xy[:, 1]
    y = s * v_xy[:, 0] + c * v_xy[:, 1]
    return torch.stack([x, y], dim=-1)


def reset_velocity_command_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0: dict | None = None,
    stage1: dict | None = None,
    stage2: dict | None = None,
) -> None:
    """Reset-mode: store desired base velocity command in env.urop_cmd (body frame).
    cmd = [vx, vy, yaw_rate]
    """
    s = _get_stage(env)
    device = env.scene["robot"].data.root_pos_w.device
    n = env_ids.shape[0]

    stage0 = stage0 or {"vx": (0.0, 0.6), "vy": (-0.2, 0.2), "yaw": (-0.6, 0.6), "stand_prob": 0.15}
    stage1 = stage1 or {"vx": (0.0, 0.35), "vy": (-0.15, 0.15), "yaw": (-0.4, 0.4), "stand_prob": 0.25}
    stage2 = stage2 or {"vx": (0.2, 0.8), "vy": (-0.2, 0.2), "yaw": (-0.6, 0.6), "stand_prob": 0.05}

    cfg = stage0 if s == 0 else (stage1 if s == 1 else stage2)

    if not hasattr(env, "urop_cmd") or env.urop_cmd is None or env.urop_cmd.shape[0] != env.num_envs:
        env.urop_cmd = torch.zeros((env.num_envs, 3), device=device)

    cmd = env.urop_cmd
    vx = torch.empty(n, device=device).uniform_(cfg["vx"][0], cfg["vx"][1])
    vy = torch.empty(n, device=device).uniform_(cfg["vy"][0], cfg["vy"][1])
    wz = torch.empty(n, device=device).uniform_(cfg["yaw"][0], cfg["yaw"][1])

    # 일부는 정지 커맨드 (정지/이동 혼합이 더 안정적으로 걷기를 배움)
    stand = torch.rand(n, device=device) < float(cfg.get("stand_prob", 0.0))
    vx[stand] = 0.0
    vy[stand] = 0.0
    wz[stand] = 0.0

    cmd[env_ids, 0] = vx
    cmd[env_ids, 1] = vy
    cmd[env_ids, 2] = wz

def reset_and_throw_object_ballistic_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_name: str = "object",
    stage0: dict | None = None,
    stage1: dict | None = None,
    stage2: dict | None = None,
    intercept_with_base_vel: bool = True,
):
    """Ballistic throw that is always 'catchable':
    - spawn position is defined in robot-yaw frame (front of robot),
    - spawn_z is absolute height above ground (origin.z + spawn_z),
    - target is in robot frame near torso (root + target_offset),
    - optionally intercepts robot constant-velocity motion over flight time.
    """
    obj = env.scene[asset_name]
    robot = env.scene["robot"]
    origins = env.scene.env_origins[env_ids]

    # curriculum stage
    stage = int(getattr(env, "_urop_stage", 0))

    # defaults (override these from env_cfg params)
    if stage0 is None:
        stage0 = {"park_pos": (3.0, 0.0, 0.25)}
    if stage1 is None:
        stage1 = {
            "spawn_dist": (1.8, 2.6),
            "spawn_y": (-0.6, 0.6),
            "spawn_z": (0.95, 1.15),          # absolute height above ground
            "flight_t": (0.8, 1.2),
            "target_offset": (0.45, 0.0, 0.28),  # relative to robot root
            "spin": (-1.0, 1.0),
        }
    if stage2 is None:
        stage2 = {
            "spawn_dist": (1.5, 2.4),
            "spawn_y": (-0.7, 0.7),
            "spawn_z": (0.95, 1.25),
            "flight_t": (0.6, 1.0),
            "target_offset": (0.50, 0.0, 0.28),
            "spin": (-2.0, 2.0),
        }

    cfg = stage0 if stage == 0 else (stage1 if stage == 1 else stage2)

    n = env_ids.numel()
    device = env.device

    # robot pose/vel
    rpos = robot.data.root_pos_w[env_ids]      # (N,3)
    rquat = robot.data.root_quat_w[env_ids]    # (N,4) wxyz
    rvel = robot.data.root_lin_vel_w[env_ids]  # (N,3)
    yaw = _yaw_from_quat_wxyz(rquat)

    # stage0: park object far away (no throw)
    if stage == 0:
        park = torch.tensor(cfg["park_pos"], device=device).view(1, 3).repeat(n, 1)
        pos = origins + park
        obj.write_root_pose_to_sim(torch.cat([pos, obj.data.root_quat_w[env_ids]], dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=env_ids)
        return

    # sample spawn params
    spawn_dist = torch.empty((n,), device=device).uniform_(*cfg["spawn_dist"])
    spawn_y = torch.empty((n,), device=device).uniform_(*cfg["spawn_y"])
    spawn_z = torch.empty((n,), device=device).uniform_(*cfg["spawn_z"])
    flight_t = torch.empty((n,), device=device).uniform_(*cfg["flight_t"])

    # spawn position: in robot-yaw frame for XY, absolute for Z
    spawn_xy_body = torch.stack([spawn_dist, spawn_y], dim=-1)              # (N,2)
    spawn_xy_world = _rot_xy(yaw, spawn_xy_body)                            # (N,2)

    pos = torch.zeros((n, 3), device=device)
    pos[:, :2] = rpos[:, :2] + spawn_xy_world
    pos[:, 2] = origins[:, 2] + spawn_z                                     # <<< IMPORTANT FIX

    # target position: near torso (relative to robot root)
    toff = torch.tensor(cfg["target_offset"], device=device).view(1, 3).repeat(n, 1)
    target_xy_world = _rot_xy(yaw, toff[:, :2])
    target = torch.zeros((n, 3), device=device)
    target[:, :2] = rpos[:, :2] + target_xy_world
    target[:, 2] = rpos[:, 2] + toff[:, 2]

    # optionally compensate robot motion (constant-vel approximation)
    if intercept_with_base_vel:
        target[:, :2] = target[:, :2] + rvel[:, :2] * flight_t.unsqueeze(-1)

    # ballistic initial velocity to hit target in flight_t
    g = torch.tensor([0.0, 0.0, -9.81], device=device).view(1, 3)
    vel = (target - pos - 0.5 * g * (flight_t.unsqueeze(-1) ** 2)) / flight_t.unsqueeze(-1)

    # set angular velocity (spin)
    spin = torch.empty((n,), device=device).uniform_(*cfg["spin"])
    ang = torch.zeros((n, 3), device=device)
    ang[:, 2] = spin

    # write to sim
    obj.write_root_pose_to_sim(torch.cat([pos, obj.data.root_quat_w[env_ids]], dim=-1), env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.cat([vel, ang], dim=-1), env_ids=env_ids)


'''
def reset_and_throw_object_ballistic_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_name: str = "object",
    stage0: dict | None = None,
    stage1: dict | None = None,
    stage2: dict | None = None,
) -> None:
    """Reset-mode: Stage0 parks object away; Stage1/2 throws box toward robot chest with ballistic init vel."""
    obj: RigidObject = env.scene[asset_name]
    robot = env.scene["robot"]
    device = obj.data.root_pos_w.device
    n = env_ids.shape[0]

    # per-env origin
    origins = env.scene.env_origins[env_ids]
    rpos = robot.data.root_pos_w[env_ids]
    rquat = robot.data.root_quat_w[env_ids]  # (w,x,y,z)

    s = _get_stage(env)

    stage0 = stage0 or {
        "park_pos": (3.0, 0.0, 0.25),
    }
    stage1 = stage1 or {
        "spawn_dist": (1.8, 2.6),
        "spawn_y": (-0.6, 0.6),
        "spawn_z": (1.2, 1.6),
        "flight_t": (1.0, 1.6),
        "target_offset": (0.55, 0.0, 1.05),
        "spin": (-1.0, 1.0),
    }
    stage2 = stage2 or {
        "spawn_dist": (1.6, 2.4),
        "spawn_y": (-0.8, 0.8),
        "spawn_z": (1.2, 1.7),
        "flight_t": (0.7, 1.2),
        "target_offset": (0.60, 0.0, 1.05),
        "spin": (-2.0, 2.0),
    }

    if s == 0:
        park = torch.tensor(stage0["park_pos"], device=device).unsqueeze(0).repeat(n, 1)
        pos = park + origins
        quat = obj.data.root_quat_w[env_ids].clone()
        obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        vel6 = torch.zeros((n, 6), device=device)
        obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)
        return

    cfg = stage1 if s == 1 else stage2

    # spawn position (world)
    spawn_dist = torch.empty(n, device=device).uniform_(cfg["spawn_dist"][0], cfg["spawn_dist"][1])
    spawn_y = torch.empty(n, device=device).uniform_(cfg["spawn_y"][0], cfg["spawn_y"][1])
    spawn_z = torch.empty(n, device=device).uniform_(cfg["spawn_z"][0], cfg["spawn_z"][1])

    pos = torch.zeros((n, 3), device=device)
    pos[:, 0] = rpos[:, 0] + spawn_dist
    pos[:, 1] = rpos[:, 1] + spawn_y
    pos[:, 2] = rpos[:, 2] + spawn_z

    # target point near chest (world)
    off = torch.tensor(cfg["target_offset"], device=device).unsqueeze(0).repeat(n, 1)
    tgt = rpos + off

    # flight time
    t = torch.empty(n, device=device).uniform_(cfg["flight_t"][0], cfg["flight_t"][1]).unsqueeze(-1)

    # ballistic initial velocity: tgt = pos + v*t + 0.5*g*t^2  => v = (tgt-pos-0.5*g*t^2)/t
    g = torch.tensor([0.0, 0.0, -9.81], device=device).unsqueeze(0).repeat(n, 1)
    v = (tgt - pos - 0.5 * g * (t * t)) / t  # (n,3)

    # add noise so it isn't perfectly predictable
    v[:, 0] += torch.empty(n, device=device).uniform_(-0.2, 0.2)
    v[:, 1] += torch.empty(n, device=device).uniform_(-0.2, 0.2)
    v[:, 2] += torch.empty(n, device=device).uniform_(-0.2, 0.2)

    # write pose/vel
    quat = obj.data.root_quat_w[env_ids].clone()
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    vel6 = torch.zeros((n, 6), device=device)
    vel6[:, 0:3] = v
    vel6[:, 3] = torch.empty(n, device=device).uniform_(cfg["spin"][0], cfg["spin"][1])
    vel6[:, 4] = torch.empty(n, device=device).uniform_(cfg["spin"][0], cfg["spin"][1])
    vel6[:, 5] = torch.empty(n, device=device).uniform_(cfg["spin"][0], cfg["spin"][1])
    obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)'''


def push_robot_velocity_impulse(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_name: str = "robot",
    lin_vel_xy: tuple[float, float] = (0.15, 0.45),
    lin_vel_z: tuple[float, float] = (0.00, 0.05),
    yaw_rate: tuple[float, float] = (-0.5, 0.5),
    stage0_scale: float = 1.0,
    stage1_scale: float = 0.6,
    stage2_scale: float = 0.35,
    body_names: list[str] | None = None,
) -> None:
    """Interval-mode perturbation: add delta to root velocity."""
    stage = _get_stage(env)
    scale = stage0_scale if stage == 0 else (stage1_scale if stage == 1 else stage2_scale)

    robot = env.scene[asset_name]
    device = robot.data.root_pos_w.device
    n = env_ids.shape[0]

    # root vel (lin + ang)
    if hasattr(robot.data, "root_vel_w"):
        vel6 = robot.data.root_vel_w[env_ids].clone()
    else:
        lin = robot.data.root_lin_vel_w[env_ids].clone()
        ang = robot.data.root_ang_vel_w[env_ids].clone()
        vel6 = torch.cat([lin, ang], dim=-1)

    theta = torch.empty(n, device=device).uniform_(0.0, 2.0 * torch.pi)
    mag = torch.empty(n, device=device).uniform_(lin_vel_xy[0], lin_vel_xy[1]) * scale
    vel6[:, 0] += mag * torch.cos(theta)
    vel6[:, 1] += mag * torch.sin(theta)
    vel6[:, 2] += torch.empty(n, device=device).uniform_(lin_vel_z[0], lin_vel_z[1]) * scale
    vel6[:, 5] += torch.empty(n, device=device).uniform_(yaw_rate[0], yaw_rate[1]) * scale

    robot.write_root_velocity_to_sim(vel6, env_ids=env_ids)
