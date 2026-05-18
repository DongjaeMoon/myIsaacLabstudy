from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stage_schedule(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    stage0_iters: int,
    stage1_iters: int,
    stage2_iters: int,
    num_steps_per_env: int,
    eval_stage: int = -1,
) -> torch.Tensor:
    del env_ids
    if eval_stage >= 0:
        stage = int(eval_stage)
    else:
        step = int(env.common_step_counter)
        s1 = int(stage0_iters) * int(num_steps_per_env)
        s2 = s1 + int(stage1_iters) * int(num_steps_per_env)
        s3 = s2 + int(stage2_iters) * int(num_steps_per_env)

        if step < s1:
            stage = 0
        elif step < s2:
            stage = 1
        elif step < s3:
            stage = 2
        else:
            stage = 3

    env.urop_stage = int(stage)
    return torch.tensor(float(stage), device=env.device)
