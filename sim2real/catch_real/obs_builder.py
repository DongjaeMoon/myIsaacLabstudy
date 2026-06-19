from __future__ import annotations

import numpy as np

from .config_schema import CatchRealConfig


class ObservationBuilder:
    """Build the actor observation vector from real robot signals.

    The builder is config-driven.  For UROP-v23 the actor contract is:

        projected_gravity(3), base_ang_vel(3), joint_pos_rel(29), joint_vel(29),
        prev_actions(29), object_rel_pos(3), object_rel_lin_vel(3), tag_visible(1)

    Total: 100.  Older configs that still use previous_action or mode_one_hot are
    kept as aliases for safety, but the generated v23 YAML does not include mode.
    """

    # Observation function clamps used by UROP_v23.mdp.observations.  No artificial
    # noise is injected on hardware; these clamps simply keep runtime values in the
    # same numerical envelope as training.
    _TERM_CLIPS: dict[str, tuple[float, float]] = {
        "projected_gravity": (-1.5, 1.5),
        "base_ang_vel": (-12.0, 12.0),
        "joint_pos_rel": (-3.5, 3.5),
        "joint_vel": (-8.0, 8.0),
        "object_rel_pos": (-4.0, 4.0),
        "object_rel_lin_vel": (-8.0, 8.0),
    }

    def __init__(self, cfg: CatchRealConfig):
        self.cfg = cfg
        self.last_obs: np.ndarray | None = None
        self.last_slices: dict[str, slice] = {}

    def _observation_reference_pose(self, current_reference_pose_q: np.ndarray) -> np.ndarray:
        q_reference = str(self.cfg.observation.q_reference).lower()
        if q_reference in {
            "policy_reference_pose",
            "default_policy_reference_pose",
            "ready_pose",
            "catch_ready",
        }:
            return self.cfg.poses[self.cfg.policy_runtime.default_policy_reference_pose]
        if q_reference in self.cfg.poses:
            return self.cfg.poses[q_reference]
        # Backward-compatible behavior for older configs: use the active scripted
        # pose reference supplied by the controller.
        return current_reference_pose_q

    def _mode_one_hot(self, current_mode_value: str) -> np.ndarray:
        mode_names = list(self.cfg.modes.names)
        one_hot = np.zeros(len(mode_names), dtype=np.float64)
        try:
            one_hot[mode_names.index(current_mode_value)] = 1.0
        except ValueError:
            pass
        return one_hot

    def build(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        projected_gravity: np.ndarray,
        base_ang_vel: np.ndarray,
        reference_pose_q: np.ndarray,
        prev_action: np.ndarray,
        object_rel_pos: np.ndarray,
        object_rel_lin_vel: np.ndarray,
        tag_visible: np.ndarray,
        current_mode_value: str,
    ) -> np.ndarray:
        q_ref_obs = self._observation_reference_pose(reference_pose_q)

        term_values = {
            "projected_gravity": projected_gravity,
            "base_ang_vel": base_ang_vel,
            "joint_pos_rel": q - q_ref_obs,
            "joint_vel": dq,
            "previous_action": prev_action,  # legacy name
            "prev_actions": prev_action,     # UROP-v23 name
            "object_rel_pos": object_rel_pos,
            "object_rel_lin_vel": object_rel_lin_vel,
            "tag_visible": tag_visible,
            "mode_one_hot": self._mode_one_hot(current_mode_value),
        }

        obs_parts: list[np.ndarray] = []
        slices: dict[str, slice] = {}
        cursor = 0
        for term in self.cfg.observation.terms:
            if term.name not in term_values:
                raise KeyError(f"Observation term '{term.name}' is not implemented in ObservationBuilder")
            value = np.asarray(term_values[term.name], dtype=np.float64).reshape(-1)
            if value.size != term.dim:
                raise ValueError(
                    f"Observation term '{term.name}' dim mismatch: expected {term.dim}, got {value.size}"
                )

            scaled = value * float(term.scale)
            clip_range = self._TERM_CLIPS.get(term.name)
            if clip_range is not None:
                scaled = np.clip(scaled, clip_range[0], clip_range[1])
            part = scaled.astype(np.float32)
            obs_parts.append(part)
            slices[term.name] = slice(cursor, cursor + part.size)
            cursor += part.size

        obs = np.concatenate(obs_parts, axis=0) if obs_parts else np.zeros(0, dtype=np.float32)
        if obs.size != self.cfg.observation.num_obs:
            raise ValueError(f"Observation size mismatch: expected {self.cfg.observation.num_obs}, got {obs.size}")

        self.last_obs = obs
        self.last_slices = slices
        return obs
