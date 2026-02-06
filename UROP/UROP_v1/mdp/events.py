# UROP/UROP_v1/mdp/events.py
from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------
# Stage utilities
# ---------------------------

'''def _get_stage(env: "ManagerBasedRLEnv") -> int:
    """Return stage from global step counter, unless forced by UROP_EVAL_STAGE."""
    forced = os.environ.get("UROP_EVAL_STAGE", None)
    if forced is not None:
        return int(forced)

    step = int(getattr(env, "common_step_counter", 0))
    s1 = int(getattr(env.cfg.curriculum, "stage1_start_steps", 0))
    s2 = int(getattr(env.cfg.curriculum, "stage2_start_steps", 0))
    if step < s1:
        return 0
    elif step < s2:
        return 1
    else:
        return 2'''
def _get_stage(env) -> int:
    # curriculum term params에서 읽어옴
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



def _pick_stage_dict(env: "ManagerBasedRLEnv", stage0: dict, stage1: dict, stage2: dict) -> dict:
    s = _get_stage(env)
    return stage0 if s == 0 else (stage1 if s == 1 else stage2)


def _root_vel6(robot, env_ids: torch.Tensor) -> torch.Tensor:
    """Get (n,6) root velocity [lin(3), ang(3)] robustly."""
    if hasattr(robot.data, "root_vel_w"):
        return robot.data.root_vel_w[env_ids].clone()
    # fallback
    lin = robot.data.root_lin_vel_w[env_ids].clone()
    ang = robot.data.root_ang_vel_w[env_ids].clone()
    return torch.cat([lin, ang], dim=-1)


# ---------------------------
# Reset base velocity (stage-aware)
# ---------------------------

def reset_robot_base_velocity_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_name: str = "robot",
    stage0: dict | None = None,
    stage1: dict | None = None,
    stage2: dict | None = None,
) -> None:
    """Reset-mode: randomize base/root velocity depending on stage."""
    robot = env.scene[asset_name]
    device = robot.data.root_pos_w.device
    n = env_ids.shape[0]

    # defaults if not provided
    stage0 = stage0 or {"lin_x": (-0.6, 0.6), "lin_y": (-0.4, 0.4), "yaw_rate": (-1.5, 1.5)}
    stage1 = stage1 or {"lin_x": (-0.3, 0.3), "lin_y": (-0.2, 0.2), "yaw_rate": (-0.8, 0.8)}
    stage2 = stage2 or {"lin_x": (-0.15, 0.15), "lin_y": (-0.1, 0.1), "yaw_rate": (-0.4, 0.4)}

    cfg = _pick_stage_dict(env, stage0, stage1, stage2)

    vel = _root_vel6(robot, env_ids)
    vel[:, 0] = torch.empty(n, device=device).uniform_(cfg["lin_x"][0], cfg["lin_x"][1])
    vel[:, 1] = torch.empty(n, device=device).uniform_(cfg["lin_y"][0], cfg["lin_y"][1])
    vel[:, 2] = 0.0
    vel[:, 3] = 0.0
    vel[:, 4] = 0.0
    vel[:, 5] = torch.empty(n, device=device).uniform_(cfg["yaw_rate"][0], cfg["yaw_rate"][1])
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)


# ---------------------------
# Reset & toss object (stage-aware)
# ---------------------------

def reset_and_toss_object_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_name: str = "object",
    stage0: dict | None = None,
    stage1: dict | None = None,
    stage2: dict | None = None,
) -> None:
    """Reset-mode: stage0 disables toss (park object away), stage1 gentle toss, stage2 harder toss."""
    obj: RigidObject = env.scene[asset_name]
    device = obj.data.root_pos_w.device
    n = env_ids.shape[0]
    origins = env.scene.env_origins[env_ids]

    stage0 = stage0 or {
        "pos_x": (2.0, 2.4), "pos_y": (-0.2, 0.2), "pos_z": (0.25, 0.35),
        "vel_x": (0.0, 0.0), "vel_y": (0.0, 0.0), "vel_z": (0.0, 0.0),
    }
    stage1 = stage1 or {
        "pos_x": (0.35, 0.55), "pos_y": (-0.15, 0.15), "pos_z": (0.9, 1.1),
        "vel_x": (-0.8, -0.3), "vel_y": (-0.2, 0.2), "vel_z": (-0.1, 0.1),
    }
    stage2 = stage2 or {
        "pos_x": (0.3, 0.5), "pos_y": (-0.15, 0.15), "pos_z": (0.9, 1.2),
        "vel_x": (-2.0, -0.8), "vel_y": (-0.3, 0.3), "vel_z": (-0.2, 0.2),
    }

    cfg = _pick_stage_dict(env, stage0, stage1, stage2)

    # position
    pos = torch.zeros((n, 3), device=device)
    pos[:, 0] = torch.empty(n, device=device).uniform_(cfg["pos_x"][0], cfg["pos_x"][1])
    pos[:, 1] = torch.empty(n, device=device).uniform_(cfg["pos_y"][0], cfg["pos_y"][1])
    pos[:, 2] = torch.empty(n, device=device).uniform_(cfg["pos_z"][0], cfg["pos_z"][1])
    pos = pos + origins

    # keep quaternion as current (or you can set fixed)
    quat = obj.data.root_quat_w[env_ids].clone()
    obj.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)

    # velocity
    vel6 = torch.zeros((n, 6), device=device)
    vel6[:, 0] = torch.empty(n, device=device).uniform_(cfg["vel_x"][0], cfg["vel_x"][1])
    vel6[:, 1] = torch.empty(n, device=device).uniform_(cfg["vel_y"][0], cfg["vel_y"][1])
    vel6[:, 2] = torch.empty(n, device=device).uniform_(cfg["vel_z"][0], cfg["vel_z"][1])
    obj.write_root_velocity_to_sim(vel6, env_ids=env_ids)


# ---------------------------
# Push recovery perturbation (interval mode)
# ---------------------------

def push_robot_velocity_impulse(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_name: str = "robot",
    # impulse magnitude ranges (m/s)
    lin_vel_xy: tuple[float, float] = (0.25, 0.75),
    lin_vel_z: tuple[float, float] = (0.00, 0.10),
    yaw_rate: tuple[float, float] = (-0.8, 0.8),
    # stage scaling
    stage0_scale: float = 1.0,
    stage1_scale: float = 0.6,
    stage2_scale: float = 0.5,
    # (A) external-force에서 torso 지정용으로 남겨둠
    body_names: list[str] | None = None,
) -> None:
    """Interval-mode perturbation.

    (B) Velocity impulse ✅ (REAL IMPLEMENTATION):
        Adds a random delta to robot root velocity.

    (A) External force ❗(COMMENTED OUT):
        Sketch only; API differs by IsaacLab version.
    """
    stage = _get_stage(env)
    scale = stage0_scale if stage == 0 else (stage1_scale if stage == 1 else stage2_scale)

    robot = env.scene[asset_name]
    device = robot.data.root_pos_w.device
    n = env_ids.shape[0]

    # ---------------------------
    # (B) Velocity impulse  ✅
    # ---------------------------
    theta = torch.empty(n, device=device).uniform_(0.0, 2.0 * 3.1415926535)
    mag = torch.empty(n, device=device).uniform_(lin_vel_xy[0], lin_vel_xy[1]) * scale
    dvx = mag * torch.cos(theta)
    dvy = mag * torch.sin(theta)
    dvz = torch.empty(n, device=device).uniform_(lin_vel_z[0], lin_vel_z[1]) * scale
    dyaw = torch.empty(n, device=device).uniform_(yaw_rate[0], yaw_rate[1]) * scale

    root_vel = _root_vel6(robot, env_ids)  # (n,6)
    root_vel[:, 0] += dvx
    root_vel[:, 1] += dvy
    root_vel[:, 2] += dvz
    root_vel[:, 5] += dyaw
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    # ---------------------------
    # (A) External force  ❗(COMMENTED)
    # ---------------------------
    # 아래는 “torso_link에 외력을 일정 시간 가하는 방식”의 스케치야.
    # IsaacLab 버전에 따라 API가 달라서, 네 환경에서 가능한 함수로 바꿔야 함.
    #
    # if body_names is None:
    #     body_names = ["torso_link"]
    #
    # # 예: robot.find_bodies(body_names) -> body_ids
    # body_ids = robot.find_bodies(body_names)
    #
    # # Force 방향/크기 (N)
    # fmag = torch.empty(n, device=device).uniform_(50.0, 200.0) * scale
    # fx = fmag * torch.cos(theta)
    # fy = fmag * torch.sin(theta)
    # fz = torch.zeros_like(fx)
    #
    # forces = torch.zeros((n, 1, 3), device=device)
    # torques = torch.zeros((n, 1, 3), device=device)
    # forces[:, 0, 0] = fx
    # forces[:, 0, 1] = fy
    # forces[:, 0, 2] = fz
    #
    # # 예시) 0.1초 동안 external wrench 적용
    # robot.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=body_ids)
