from __future__ import annotations

import numpy as np

from .config_schema import CatchRealConfig
from .math_utils import clamp, interpolate

try:
    import torch
except Exception:
    torch = None


class PolicyRunner:
    LOWER_BODY_JOINTS = {
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    }

    def __init__(self, cfg: CatchRealConfig, no_policy: bool, device: str):
        self.cfg = cfg
        self.no_policy = no_policy
        self.device = device
        self.policy = None
        self.policy_device = None
        self.prev_action = np.zeros(self.cfg.policy.num_actions, dtype=np.float64)
        self.last_target_q = self.cfg.poses[self.cfg.policy_runtime.default_policy_reference_pose].copy()
        self.action_multipliers = self._build_action_multipliers()

    @property
    def is_loaded(self) -> bool:
        return self.policy is not None and self.policy_device is not None

    def load_if_enabled(self) -> None:
        if self.no_policy:
            return
        if torch is None:
            raise RuntimeError("torch is required when policy execution is enabled")
        if self.cfg.policy.path is None:
            raise FileNotFoundError("Policy execution requested, but no policy path is configured or overridden.")

        if self.device == "auto":
            self.policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.policy_device = torch.device(self.device)

        self.policy = torch.jit.load(str(self.cfg.policy.path), map_location=self.policy_device)
        self.policy.eval()

    def zero_action(self) -> None:
        self.prev_action[:] = 0.0

    def reset_target(self, target_q: np.ndarray) -> None:
        self.prev_action[:] = 0.0
        self.last_target_q = target_q.copy()

    def _build_action_multipliers(self) -> np.ndarray:
        multipliers = np.ones(self.cfg.policy.num_actions, dtype=np.float64)
        if not self.cfg.policy_runtime.gantry_upper_body_only:
            return multipliers

        lower_scale = float(self.cfg.policy_runtime.gantry_lower_body_action_scale)
        upper_scale = float(self.cfg.policy_runtime.gantry_upper_body_action_scale)
        for index, entry in enumerate(self.cfg.policy.action_order):
            if entry.name in self.LOWER_BODY_JOINTS:
                multipliers[index] = lower_scale
            else:
                multipliers[index] = upper_scale
        return multipliers

    def compute_target(self, obs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        if not self.is_loaded or torch is None:
            return q_ref.copy()

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.policy_device)
        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)
        clipped_action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64).reshape(-1)
        if clipped_action.size != self.cfg.policy.num_actions:
            raise RuntimeError(
                f"Policy action dimension mismatch: expected {self.cfg.policy.num_actions}, "
                f"got {clipped_action.size}. Check exported policy vs v23 YAML."
            )

        if self.cfg.safety.clamp_action:
            clipped_action = clamp(clipped_action, -self.cfg.policy.action_clip, self.cfg.policy.action_clip)

        clipped_action = clipped_action * self.action_multipliers

        q_target_raw = q_ref.copy()
        q_target_raw[self.cfg.policy.action_slot_indices] = (
            q_ref[self.cfg.policy.action_slot_indices] + self.cfg.policy.action_scales * clipped_action
        )

        if self.cfg.safety.max_target_delta_per_policy_step is not None:
            delta = q_target_raw - self.last_target_q
            max_delta = float(self.cfg.safety.max_target_delta_per_policy_step)
            q_target_raw = self.last_target_q + np.clip(delta, -max_delta, max_delta)

        alpha = float(np.clip(self.cfg.runtime.target_lowpass_alpha, 0.0, 1.0))
        q_target_filtered = interpolate(self.last_target_q, q_target_raw, alpha)
        self.prev_action = clipped_action.copy()
        self.last_target_q = q_target_filtered.copy()
        return q_target_filtered
