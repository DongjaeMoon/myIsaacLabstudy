from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import get_task_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stage_schedule(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    thresholds: tuple[int, int, int, int] = (20_000, 55_000, 110_000, 180_000),
    force_stage: int | None = None,
) -> torch.Tensor:
    """Update the UROP-v27 curriculum stage.

    Stages are consumed by reset_autonomous_episode(): later stages increase delayed
    toss/no-toss/pass-by probability and keep observation/trajectory randomization active.
    """
    state = get_task_state(env)
    if force_stage is not None:
        stage_value = int(force_stage)
    else:
        step = getattr(env, "common_step_counter", 0)
        try:
            step = int(step)
        except Exception:
            step = int(step.item())
        stage_value = 0
        for threshold in thresholds:
            if step >= int(threshold):
                stage_value += 1
    stage_value = max(0, min(stage_value, 4))
    state.curriculum_stage[:] = stage_value
    return torch.tensor(float(stage_value), device=env.device)


__all__ = ["stage_schedule"]
