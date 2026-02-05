# /home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v1/mdp/curriculum.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _compute_stage(step: int, s1: int, s2: int) -> int:
    if step < s1:
        return 0
    elif step < s2:
        return 1
    else:
        return 2


def stage_schedule(
    env: "ManagerBasedRLEnv",
    env_ids,  # CurriculumManager 시그니처 요구(사용 안 해도 됨)
    stage0_iters: int,
    stage1_iters: int,
    num_steps_per_env: int,
    eval_stage: int = -1,
):
    """single-run stage scheduler (0->1->2)

    - stage0_iters, stage1_iters는 'RSL-RL iteration 수' 기준
    - common_step_counter는 env.step()마다 +1
    - iteration 1회에 env.step()가 num_steps_per_env번 호출되므로,
      stage boundary steps = iters * num_steps_per_env
    """
    # play/eval에서 강제로 stage 고정하고 싶을 때
    if eval_stage in (0, 1, 2):
        stage = int(eval_stage)
    else:
        step = int(env.common_step_counter)
        s1 = int(stage0_iters) * int(num_steps_per_env)
        s2 = (int(stage0_iters) + int(stage1_iters)) * int(num_steps_per_env)
        stage = _compute_stage(step, s1, s2)

    # 다른 모듈에서 빠르게 쓰고 싶으면 캐시(선택)
    env.urop_stage = stage

    # 로깅(원하면 dict로 여러 값 로깅 가능)
    return {"stage": float(stage)}
