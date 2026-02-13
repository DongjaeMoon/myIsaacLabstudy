# UROP/UROP_v2/mdp/terminations.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from .observations import quat_rotate_inverse

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


def robot_fallen(env: "ManagerBasedRLEnv", min_root_z: float = 0.55, min_upright: float = 0.6) -> torch.Tensor:
    robot = env.scene["robot"]
    z = robot.data.root_pos_w[:, 2]
    q = robot.data.root_quat_w
    g_world = torch.tensor([0.0, 0.0, -1.0], device=q.device).unsqueeze(0).repeat(q.shape[0], 1)
    g_b = quat_rotate_inverse(q, g_world)
    upright = (-g_b[:, 2]).clamp(0.0, 1.0)
    return (z < min_root_z) | (upright < min_upright)


def object_dropped_curriculum(
    env: "ManagerBasedRLEnv",
    min_z: float = 0.22,
    max_dist: float = 3.0,
) -> torch.Tensor:
    stage = _get_stage(env)
    if stage == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    robot = env.scene["robot"]
    obj = env.scene["object"]
    z_fail = obj.data.root_pos_w[:, 2] < min_z
    dist = torch.norm(obj.data.root_pos_w - robot.data.root_pos_w, dim=-1)
    d_fail = dist > max_dist
    return z_fail | d_fail
