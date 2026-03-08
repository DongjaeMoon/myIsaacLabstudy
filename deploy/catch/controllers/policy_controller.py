#[/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch/controllers/policy_controller.py]
import io
from typing import Optional
import carb
import numpy as np
import omni
import torch
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from omni.physx import get_physx_simulation_interface

from loco.controllers.config_loader import get_articulation_props, get_physics_properties, get_robot_joint_properties, parse_env_config

class PolicyController(BaseController):
    def __init__(self, name: str, prim_path: str, root_path: Optional[str] = None, usd_path: Optional[str] = None, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("unable to add robot usd, usd_path not provided")

        if root_path == None:
            self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
        else:
            self.robot = SingleArticulation(prim_path=root_path, name=name, position=position, orientation=orientation)

    def load_policy(self, policy_file_path, policy_env_path) -> None:
        file_content = omni.client.read_file(policy_file_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        self.policy = torch.jit.load(file)
        self.policy_env_params = parse_env_config(policy_env_path)
        self._decimation, self._dt, self.render_interval = get_physics_properties(self.policy_env_params)

    def initialize(
        self,
        physics_sim_view: omni.physics.tensors.SimulationView = None,
        effort_modes: str = "force",
        control_mode: str = "position",
        set_gains: bool = True,
        set_limits: bool = True,
        set_articulation_props: bool = True,
    ) -> None:
        self.robot.initialize(physics_sim_view=physics_sim_view)
        self.robot.get_articulation_controller().set_effort_modes(effort_modes)

        get_physx_simulation_interface().flush_changes()

        self.robot.get_articulation_controller().switch_control_mode(control_mode)
        
        # 🚨 [수정된 부분] 7개의 값을 전부 받고, 외부에서 쓸 수 있게 self 변수로 저장합니다.
        (
            self.max_effort, 
            self.max_vel, 
            self.stiffness, 
            self.damping, 
            self.armature,       # config_loader.py에서 추가한 7번째 리턴값
            self.default_pos, 
            self.default_vel
        ) = get_robot_joint_properties(
            self.policy_env_params, self.robot.dof_names
        )

        if set_gains:
            # 지역 변수가 아닌 self 변수를 사용하도록 변경
            self.robot._articulation_view.set_gains(self.stiffness, self.damping)
            
        if set_limits:
            self.robot._articulation_view.set_max_efforts(self.max_effort)

            get_physx_simulation_interface().flush_changes()

            self.robot._articulation_view.set_max_joint_velocities(self.max_vel)
            
        if set_articulation_props:
            self._set_articulation_props()
        
        # [디버깅 추가] 파싱된 주요 관절의 Gain과 Default Position 확인
        print("\n" + "="*50)
        print("[DEBUG] 물리 속성 로드 결과 (일부 관절 확인)")
        for name, stiff, damp, d_pos in zip(self.robot.dof_names, self.stiffness, self.damping, self.default_pos):
            # 너무 기니까 다리 관절 몇 개만 필터링해서 확인
            if "left_hip" in name or "left_knee" in name or "waist" in name:
                print(f"{name:25s} | Kp(stiff): {stiff:6.1f} | Kd(damp): {damp:5.1f} | D_Pos: {d_pos:6.2f}")
        print("="*50 + "\n")

    def _set_articulation_props(self) -> None:
        articulation_prop = get_articulation_props(self.policy_env_params)
        solver_position_iteration_count = articulation_prop.get("solver_position_iteration_count")
        solver_velocity_iteration_count = articulation_prop.get("solver_velocity_iteration_count")
        stabilization_threshold = articulation_prop.get("stabilization_threshold")
        enabled_self_collisions = articulation_prop.get("enabled_self_collisions")
        sleep_threshold = articulation_prop.get("sleep_threshold")

        if solver_position_iteration_count not in [None, float("inf")]: self.robot.set_solver_position_iteration_count(solver_position_iteration_count)
        if solver_velocity_iteration_count not in [None, float("inf")]: self.robot.set_solver_velocity_iteration_count(solver_velocity_iteration_count)
        if stabilization_threshold not in [None, float("inf")]: self.robot.set_stabilization_threshold(stabilization_threshold)
        if isinstance(enabled_self_collisions, bool): self.robot.set_enabled_self_collisions(enabled_self_collisions)
        if sleep_threshold not in [None, float("inf")]: self.robot.set_sleep_threshold(sleep_threshold)

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs = torch.from_numpy(obs).view(1, -1).float()
            action = self.policy(obs).detach().view(-1).numpy()
        return action

    def post_reset(self) -> None:
        self.robot.post_reset()
