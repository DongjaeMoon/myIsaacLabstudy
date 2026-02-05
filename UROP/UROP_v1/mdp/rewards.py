# UROP/UROP_v1/mdp/rewards.py
from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------
# Stage utilities
# ---------------------------

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



def _stage_w(env: "ManagerBasedRLEnv", w0: float, w1: float, w2: float) -> float:
    s = _get_stage(env)
    return w0 if s == 0 else (w1 if s == 1 else w2)


# ---------------------------
# Stage-scaled rewards used by env_cfg.py
# ---------------------------

def alive_bonus_curriculum(env: "ManagerBasedRLEnv", w0: float = 0.2, w1: float = 0.05, w2: float = 0.02) -> torch.Tensor:
    robot = env.scene["robot"]
    n = robot.data.root_pos_w.shape[0]
    device = robot.data.root_pos_w.device
    w = _stage_w(env, w0, w1, w2)
    return torch.ones(n, device=device) * w


def root_height_reward_curriculum(
    env: "ManagerBasedRLEnv",
    target_z: float = 0.78,
    sigma: float = 0.08,
    w0: float = 1.0,
    w1: float = 0.2,
    w2: float = 0.1,
) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    base = torch.exp(-((z - target_z) / sigma) ** 2)
    return base * _stage_w(env, w0, w1, w2)


def base_velocity_penalty_curriculum(
    env: "ManagerBasedRLEnv",
    w_lin: float = 1.0,
    w_ang: float = 0.2,
    w0: float = 0.2,
    w1: float = 0.05,
    w2: float = 0.03,
) -> torch.Tensor:
    """Positive penalty (env_cfg sets weight=-1.0)."""
    robot = env.scene["robot"]
    v = robot.data.root_lin_vel_w  # (N,3)
    w = robot.data.root_ang_vel_w  # (N,3)

    lin_pen = torch.sum(v * v, dim=-1)                 # ||v||^2
    yaw_pen = w[:, 2] * w[:, 2]                        # yaw^2
    base = w_lin * lin_pen + w_ang * yaw_pen
    return base * _stage_w(env, w0, w1, w2)


def hold_object_close_curriculum(
    env: "ManagerBasedRLEnv",
    sigma: float = 0.7,
    w0: float = 0.0,
    w1: float = 2.0,
    w2: float = 2.0,
) -> torch.Tensor:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    d = torch.linalg.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    base = torch.exp(-(d / sigma) ** 2)
    return base * _stage_w(env, w0, w1, w2)


def object_not_dropped_bonus_curriculum(
    env: "ManagerBasedRLEnv",
    min_z: float = 0.25,
    w0: float = 0.0,
    w1: float = 0.5,
    w2: float = 0.5,
) -> torch.Tensor:
    obj = env.scene["object"]
    base = (obj.data.root_pos_w[:, 2] > min_z).float()
    return base * _stage_w(env, w0, w1, w2)


def impact_peak_penalty_curriculum(
    env: "ManagerBasedRLEnv",
    sensor_names: list[str],
    force_thr_stage1: float = 400.0,
    force_thr_stage2: float = 300.0,
    w0: float = 0.0,
    w1: float = 0.05,
    w2: float = 0.10,
) -> torch.Tensor:
    """Positive penalty (env_cfg sets weight=-1.0). stage0 off."""
    s = _get_stage(env)
    if s == 0:
        robot = env.scene["robot"]
        return torch.zeros(robot.data.root_pos_w.shape[0], device=robot.data.root_pos_w.device)

    force_thr = force_thr_stage1 if s == 1 else force_thr_stage2

    peaks = []
    for name in sensor_names:
        sensor = env.scene[name]
        f = sensor.data.net_forces_w      # (N,1,3) with history_length=1
        mag = torch.linalg.norm(f, dim=-1)         # (N,1)
        peak = mag.max(dim=-1).values              # (N,)
        peaks.append(peak)

    peak_all = torch.stack(peaks, dim=-1).max(dim=-1).values
    over = torch.relu(peak_all - force_thr)
    base = over * over
    return base * _stage_w(env, w0, w1, w2)


def action_rate_penalty_curriculum(env: "ManagerBasedRLEnv", w0: float = 0.05, w1: float = 0.02, w2: float = 0.01) -> torch.Tensor:
    """Positive penalty (env_cfg sets weight=-1.0)."""
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    base = torch.sum((a - a_prev) ** 2, dim=-1)
    return base * _stage_w(env, w0, w1, w2)
