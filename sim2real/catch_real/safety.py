from __future__ import annotations

import numpy as np

from .config_schema import CatchRealConfig


def is_lowstate_fresh(low_state, low_state_time: float, now: float, timeout_s: float) -> bool:
    return low_state is not None and (now - low_state_time) <= timeout_s


def apply_joint_target_safety(
    cfg: CatchRealConfig,
    nominal_target_q: np.ndarray,
    last_sent_q_des: np.ndarray | None,
) -> np.ndarray:
    q_des = nominal_target_q.copy()

    if cfg.safety.clamp_joint_targets_to_limits:
        lower = cfg.robot.joint_limit_lower
        upper = cfg.robot.joint_limit_upper
        if lower is not None and upper is not None:
            q_des = np.clip(q_des, lower, upper)

    if last_sent_q_des is not None and cfg.safety.max_target_delta_per_control_step is not None:
        max_delta = float(cfg.safety.max_target_delta_per_control_step)
        delta = q_des - last_sent_q_des
        q_des = last_sent_q_des + np.clip(delta, -max_delta, max_delta)

    return q_des
