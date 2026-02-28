#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v7/mdp/curriculum.py]
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stage_schedule(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0_iters: int,
    stage1_iters: int,
    num_steps_per_env: int,
    eval_stage: int = -1,
) -> torch.Tensor:
    """Global curriculum stage (scalar).

    IsaacLab's CurriculumManager expects the term output to be a *scalar* tensor because it calls `.item()`.
    We still keep `env.urop_stage` for other modules (e.g., toss difficulty).
    """

    if eval_stage in (0, 1, 2):
        s = int(eval_stage)
    else:
        step = int(env.common_step_counter)
        s1 = int(stage0_iters) * int(num_steps_per_env)
        s2 = (int(stage0_iters) + int(stage1_iters)) * int(num_steps_per_env)
        if step < s1:
            s = 0
        elif step < s2:
            s = 1
        else:
            s = 2

    env.urop_stage = int(s)
    return torch.tensor(float(s), device=env.device)