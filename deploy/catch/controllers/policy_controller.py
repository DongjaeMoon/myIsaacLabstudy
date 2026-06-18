# [/home/<user>/.../myIsaacLabstudy/deploy/catch/controllers/policy_controller.py]
"""Base controller used by Isaac-Sim policy deploy examples."""
from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import torch

try:
    import carb
except Exception:  # pragma: no cover - only outside Isaac Sim
    carb = None  # type: ignore

try:
    import omni.client  # type: ignore
except Exception:  # pragma: no cover - only outside Isaac Sim
    omni = None  # type: ignore

from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from omni.physx import get_physx_simulation_interface

from catch.controllers.config_loader import (
    get_articulation_props,
    get_physics_properties,
    get_robot_joint_properties,
    parse_env_config,
)


def _torch_jit_load_any(policy_file_path: str, map_location: str = "cpu"):
    expanded = os.path.expanduser(os.path.expandvars(str(policy_file_path)))
    if os.path.exists(expanded):
        return torch.jit.load(expanded, map_location=map_location)

    if omni is not None:
        result = omni.client.read_file(expanded)  # type: ignore[attr-defined]
        content = result[-1]
        file_obj = io.BytesIO(memoryview(content).tobytes())
        return torch.jit.load(file_obj, map_location=map_location)

    raise FileNotFoundError(f"Could not read policy file: {policy_file_path}")


class PolicyController(BaseController):
    def __init__(
        self,
        name: str,
        prim_path: str,
        root_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name)
        self.prim_path = prim_path
        self.root_path = root_path
        self.usd_path = usd_path
        self.articulation_path = prim_path if root_path is None else root_path

        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(os.path.expanduser(os.path.expandvars(str(usd_path))))
            elif carb is not None:
                carb.log_error("Unable to add robot USD: usd_path was not provided")

        articulation_path = self.articulation_path
        self.robot = SingleArticulation(
            prim_path=articulation_path,
            name=name,
            position=position,
            orientation=orientation,
        )

    def load_policy(self, policy_file_path: str, policy_env_path: str, map_location: str = "cpu") -> None:
        self.policy = _torch_jit_load_any(policy_file_path, map_location=map_location)
        self.policy.eval()
        self.policy_env_params = parse_env_config(policy_env_path)
        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)
        self.policy_file_path = policy_file_path
        self.policy_env_path = policy_env_path

    def initialize(
        self,
        physics_sim_view=None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        self.robot.initialize(physics_sim_view=physics_sim_view)
        controller = self.robot.get_articulation_controller()
        if controller is not None:
            controller.set_effort_modes(effort_modes)

        get_physx_simulation_interface().flush_changes()

        if controller is not None:
            controller.switch_control_mode(control_mode)

        (
            self.max_effort,
            self.max_vel,
            self.stiffness,
            self.damping,
            self.armature,
            self.default_pos,
            self.default_vel,
        ) = get_robot_joint_properties(self.policy_env_params, self.robot.dof_names)

        # Isaac Lab's implicit actuators are effectively high-authority position
        # drives when no explicit effort/velocity limits are given.  Avoid passing
        # sys.maxsize into PhysX tensors, but keep the limits high enough for sim.
        safe_effort = np.where(self.max_effort > 1.0e8, 1.0e8, self.max_effort).astype(np.float32)
        safe_vel = np.where(self.max_vel > 1.0e6, 1.0e6, self.max_vel).astype(np.float32)

        articulation_view = getattr(self.robot, "_articulation_view", None)
        if set_gains and articulation_view is not None:
            articulation_view.set_gains(self.stiffness, self.damping)

        if set_limits and articulation_view is not None:
            articulation_view.set_max_efforts(safe_effort)
            get_physx_simulation_interface().flush_changes()
            articulation_view.set_max_joint_velocities(safe_vel)

        if set_articulation_props:
            self._set_articulation_props()

        print("\n" + "=" * 72)
        print(f"[PolicyController] Loaded policy: {self.policy_file_path}")
        print(f"[PolicyController] Loaded env:    {self.policy_env_path}")
        print(f"[PolicyController] dt={self._dt:.5f}, decimation={self._decimation}, policy_dt={self._dt * self._decimation:.5f}")
        print("[PolicyController] Joint gains/defaults from env.yaml")
        for name, stiff, damp, d_pos in zip(self.robot.dof_names, self.stiffness, self.damping, self.default_pos):
            if (
                "hip" in name
                or "knee" in name
                or "ankle" in name
                or "waist" in name
                or "shoulder" in name
                or "elbow" in name
            ):
                print(f"  {name:28s} | Kp={stiff:7.2f} | Kd={damp:6.2f} | q0={d_pos:+7.3f}")
        print("=" * 72 + "\n")

    def _set_articulation_props(self) -> None:
        articulation_prop = get_articulation_props(self.policy_env_params)
        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in (None, float("inf")):
            self.robot.set_solver_position_iteration_count(int(solver_position_iteration_count))
        if solver_velocity_iteration_count not in (None, float("inf")):
            self.robot.set_solver_velocity_iteration_count(int(solver_velocity_iteration_count))
        if stabilization_threshold not in (None, float("inf")):
            self.robot.set_stabilization_threshold(float(stabilization_threshold))
        if isinstance(enabled_self_collisions, bool):
            self.robot.set_enabled_self_collisions(enabled_self_collisions)
        if sleep_threshold not in (None, float("inf")):
            self.robot.set_sleep_threshold(float(sleep_threshold))

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            out = self.policy(obs_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            action = out.detach().view(-1).cpu().numpy()
        return action.astype(np.float32)

    def post_reset(self) -> None:
        self.robot.post_reset()
