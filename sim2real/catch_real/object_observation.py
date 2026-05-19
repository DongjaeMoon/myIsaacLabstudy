from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config_schema import CatchRealConfig


@dataclass(frozen=True)
class ObjectObservation:
    rel_pos: np.ndarray
    rel_lin_vel: np.ndarray
    tag_visible: np.ndarray

    @staticmethod
    def zeros() -> "ObjectObservation":
        return ObjectObservation(
            rel_pos=np.zeros(3, dtype=np.float64),
            rel_lin_vel=np.zeros(3, dtype=np.float64),
            tag_visible=np.zeros(1, dtype=np.float64),
        )

    @staticmethod
    def fake_debug() -> "ObjectObservation":
        return ObjectObservation(
            rel_pos=np.array([0.8, 0.0, 0.4], dtype=np.float64),
            rel_lin_vel=np.array([-0.5, 0.0, 0.0], dtype=np.float64),
            tag_visible=np.ones(1, dtype=np.float64),
        )


class ObjectObservationProvider:
    def __init__(self, cfg: CatchRealConfig):
        self.cfg = cfg
        self.fake_enabled = bool(cfg.policy_runtime.fake_object_debug)

    def get(self) -> ObjectObservation:
        if self.fake_enabled:
            return ObjectObservation.fake_debug()
        return ObjectObservation.zeros()

    def toggle_fake(self) -> bool:
        self.fake_enabled = not self.fake_enabled
        return self.fake_enabled

    def status_label(self) -> str:
        if self.fake_enabled:
            return "fake"
        if self.cfg.policy_runtime.object_source == "zeros":
            return "zeros"
        return f"{self.cfg.policy_runtime.object_source} later (stub->zeros)"
