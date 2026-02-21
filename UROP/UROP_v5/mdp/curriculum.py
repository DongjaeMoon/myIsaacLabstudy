# UROP/UROP_v5/mdp/curriculum.py

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def stage_schedule(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,          # ✅ 추가 (중요)
    stage0_iters: int,
    stage1_iters: int,
    num_steps_per_env: int,
    eval_stage: int = -1,
) -> torch.Tensor:
    if eval_stage in (0, 1, 2):
        s = eval_stage
    else:
        step = int(env.common_step_counter)
        s1 = stage0_iters * num_steps_per_env
        s2 = (stage0_iters + stage1_iters) * num_steps_per_env
        if step < s1:
            s = 0
        elif step < s2:
            s = 1
        else:
            s = 2

    env.urop_stage = int(s)

    # ✅ env_ids 길이에 맞춰 리턴 (CurriculumManager가 env_ids 기반으로 term을 평가함)
    #return torch.tensor(float(s), device=env.device)
    return torch.full((env_ids.shape[0],), float(s), device=env.device)
