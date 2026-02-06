# UROP/UROP_v1/mdp/terminations.py
from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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



def robot_fallen(env: "ManagerBasedRLEnv", min_root_z: float = 0.55) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, 2] < min_root_z


def object_dropped_curriculum(env: "ManagerBasedRLEnv", min_z: float = 0.35) -> torch.Tensor:
    """stage0: OFF, stage1/2: ON."""
    s = _get_stage(env)
    obj = env.scene["object"]
    if s == 0:
        return torch.zeros(obj.data.root_pos_w.shape[0], device=obj.data.root_pos_w.device, dtype=torch.bool)
    return obj.data.root_pos_w[:, 2] < min_z
