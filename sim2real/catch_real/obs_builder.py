from __future__ import annotations

import numpy as np

from .config_schema import CatchRealConfig


class ObservationBuilder:
    def __init__(self, cfg: CatchRealConfig):
        self.cfg = cfg
        self.last_obs: np.ndarray | None = None

    def build(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        projected_gravity: np.ndarray,
        base_ang_vel: np.ndarray,
        reference_pose_q: np.ndarray,
        prev_action: np.ndarray,
        current_mode_value: str,
    ) -> np.ndarray:
        mode_one_hot = np.zeros(len(self.cfg.modes.names), dtype=np.float64)
        if current_mode_value in self.cfg.modes.names:
            mode_one_hot[self.cfg.modes.names.index(current_mode_value)] = 1.0

        term_values = {
            "projected_gravity": projected_gravity,
            "base_ang_vel": base_ang_vel,
            "joint_pos_rel": q - reference_pose_q,
            "joint_vel": dq,
            "previous_action": prev_action,
            "object_rel_pos": np.zeros(3, dtype=np.float64),
            "object_rel_lin_vel": np.zeros(3, dtype=np.float64),
            "tag_visible": np.zeros(1, dtype=np.float64),
            "mode_one_hot": mode_one_hot,
        }

        obs_parts: list[np.ndarray] = []
        for term in self.cfg.observation.terms:
            if term.name not in term_values:
                raise KeyError(f"Observation term '{term.name}' is not implemented in g1_catch_real.py")
            value = np.asarray(term_values[term.name], dtype=np.float64).reshape(-1)
            if value.size != term.dim:
                raise ValueError(
                    f"Observation term '{term.name}' dim mismatch: expected {term.dim}, got {value.size}"
                )
            obs_parts.append((value * term.scale).astype(np.float32))

        obs = np.concatenate(obs_parts, axis=0)
        if obs.size != self.cfg.observation.num_obs:
            raise ValueError(f"Observation size mismatch: expected {self.cfg.observation.num_obs}, got {obs.size}")

        self.last_obs = obs
        return obs
