from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import (
    EP_NO_TOSS,
    catchable_or_hold_phase,
    get_task_state,
    hold_condition,
    projected_gravity,
    _episode_step,
    _get_object,
    _get_robot,
    _object_pos_quat_vel_w,
    _object_rel_in_root_frame,
    _root_pos_quat,
    _safe_norm,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _global_step(env: "ManagerBasedRLEnv") -> int:
    step = getattr(env, "common_step_counter", 0)
    try:
        return int(step)
    except Exception:
        return int(step.item())


def time_out(env: "ManagerBasedRLEnv") -> torch.Tensor:
    max_len = getattr(env, "max_episode_length", None)
    if max_len is None:
        max_len = getattr(env, "max_episode_length_s", 0.0) / max(float(getattr(env, "step_dt", 0.02)), 1e-6)
    return _episode_step(env) >= int(max_len)


def successful_hold_complete(env: "ManagerBasedRLEnv", min_steps: int = 40) -> torch.Tensor:
    """Terminate when the box has been stabilized in the hug region long enough."""
    state = get_task_state(env)
    step = _global_step(env)
    if state.hold_counter_step != step:
        hold = hold_condition(env).squeeze(-1).to(torch.bool)
        state.hold_counter = torch.where(hold, state.hold_counter + 1, torch.zeros_like(state.hold_counter))
        state.hold_counter_step = step
    return state.hold_counter >= int(min_steps)


def robot_fell(env: "ManagerBasedRLEnv", min_root_z: float = 0.50, max_tilt_xy: float = 0.70) -> torch.Tensor:
    robot = _get_robot(env)
    root_pos, _root_quat = _root_pos_quat(robot)
    g = projected_gravity(env, noise_std=0.0)
    tilt_xy = torch.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1])
    return (root_pos[:, 2] < float(min_root_z)) | (tilt_xy > float(max_tilt_xy))


def unsafe_idle_posture(env: "ManagerBasedRLEnv", min_root_z: float = 0.62, max_tilt_xy: float = 0.36) -> torch.Tensor:
    """Terminate unsafe pre-reaction coiling while still allowing catch absorption."""
    idle = ~catchable_or_hold_phase(env)
    robot = _get_robot(env)
    root_pos, _root_quat = _root_pos_quat(robot)
    g = projected_gravity(env, noise_std=0.0)
    tilt_xy = torch.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1])
    return idle & ((root_pos[:, 2] < float(min_root_z)) | (tilt_xy > float(max_tilt_xy)))


def object_dropped(env: "ManagerBasedRLEnv", drop_z: float = 0.22, grace_steps_after_release: int = 8) -> torch.Tensor:
    state = get_task_state(env)
    obj_pos, _obj_quat, _obj_lin, _obj_ang = _object_pos_quat_vel_w(env)
    released_long_enough = _episode_step(env) > (state.release_step + int(grace_steps_after_release))
    return (obj_pos[:, 2] < float(drop_z)) & state.has_released & released_long_enough & (state.episode_type != EP_NO_TOSS)


def object_escaped(env: "ManagerBasedRLEnv", max_dist: float = 2.75, behind_x: float = -0.65) -> torch.Tensor:
    state = get_task_state(env)
    rel_pos_b, _rel_vel_b, _obj_ang = _object_rel_in_root_frame(env)
    escaped_far = _safe_norm(rel_pos_b) > float(max_dist)
    escaped_behind = rel_pos_b[:, 0] < float(behind_x)
    return (escaped_far | escaped_behind) & state.has_released & (state.episode_type != EP_NO_TOSS)


def invalid_object_state(env: "ManagerBasedRLEnv") -> torch.Tensor:
    obj_pos, obj_quat, obj_lin, obj_ang = _object_pos_quat_vel_w(env)
    del obj_quat
    values = torch.cat((obj_pos, obj_lin, obj_ang), dim=-1)
    return ~torch.isfinite(values).all(dim=-1)


__all__ = [
    "time_out",
    "successful_hold_complete",
    "robot_fell",
    "unsafe_idle_posture",
    "object_dropped",
    "object_escaped",
    "invalid_object_state",
]
