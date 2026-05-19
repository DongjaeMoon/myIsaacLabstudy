from __future__ import annotations

import numpy as np

from .config_schema import CatchRealConfig
from .math_utils import clamp, interpolate

try:
    import torch
except Exception:
    torch = None


class PolicyRunner:
    def __init__(self, cfg: CatchRealConfig, no_policy: bool, device: str):
        self.cfg = cfg
        self.no_policy = no_policy
        self.device = device
        self.policy = None
        self.policy_device = None
        self.prev_action = np.zeros(self.cfg.policy.num_actions, dtype=np.float64)
        self.last_target_q = self.cfg.poses["catch"].copy()

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

    def compute_target(self, obs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        if self.policy is None or torch is None or self.policy_device is None:
            return q_ref.copy()

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.policy_device)
        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)
        action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)

        if self.cfg.safety.clamp_action:
            action = clamp(action, -self.cfg.policy.action_clip, self.cfg.policy.action_clip)

        target = q_ref.copy()
        target[self.cfg.policy.action_slot_indices] = (
            q_ref[self.cfg.policy.action_slot_indices] + self.cfg.policy.action_scales * action
        )

        if self.cfg.safety.max_target_delta_per_policy_step is not None:
            delta = target - self.last_target_q
            max_delta = float(self.cfg.safety.max_target_delta_per_policy_step)
            target = self.last_target_q + np.clip(delta, -max_delta, max_delta)

        alpha = float(np.clip(self.cfg.runtime.target_lowpass_alpha, 0.0, 1.0))
        target = interpolate(self.last_target_q, target, alpha)
        self.prev_action = action.copy()
        self.last_target_q = target.copy()
        return target
