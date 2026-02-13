# /home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v2/mdp/curriculum.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stage_schedule(
    env: "ManagerBasedRLEnv",
    env_ids,  # <-- CurriculumManager가 두 번째 인자로 넘겨주는 env index 텐서(대부분 필요없으면 무시 가능)
    stage0_iters: int = 4000,
    stage1_iters: int = 4000,
    num_steps_per_env: int = 96,
    eval_stage: int = -1,
) -> dict:
    """Global stage schedule.
    Stage 0: locomotion only
    Stage 1: catching (no pushes)
    Stage 2: catching + carrying + pushes
    """

    # evaluation에서는 고정 stage 사용
    if eval_stage in (0, 1, 2):
        stage = int(eval_stage)
    else:
        # rsl-rl에서는 "iteration"이 아니라 "environment steps" 기준으로 스케줄링하는 게 안전
        # common_step_counter는 (num_envs 동시)에서도 증가하는 전역 스텝 카운터
        step = int(env.common_step_counter)
        s0_end = int(stage0_iters) * int(num_steps_per_env)
        s1_end = s0_end + int(stage1_iters) * int(num_steps_per_env)

        if step < s0_end:
            stage = 0
        elif step < s1_end:
            stage = 1
        else:
            stage = 2

    env.urop_stage = stage
    return {"UROP/stage": float(stage)}
