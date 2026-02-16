# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example_h1.py]

import torch
import numpy as np
import carb
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction

POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/h1_flat/2026-02-16_17-43-27/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/usd/G1_23DOF_UROP.usd"
ROBOT_PRIM_PATH = "/World/h1"

class H1Example(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_mode = True # 디버깅용 로그 출력 여부
    # ----------------------------------------------------------------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # (1) 로봇 추가
        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="h1"))
        
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 1.0]))

        
        return
    
    # ----------------------------------------------------------------------
    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("h1")
        self._box = self._world.scene.get_object("box")

        
        self._robot.initialize()
        
        # (1) Policy 로드
        print(f">>> Loading Policy from: {POLICY_PATH}")
        try:
            self.policy = torch.jit.load(POLICY_PATH).to(self.device)
            self.policy.eval()
            print(">>> Policy Loaded Successfully!")
        except Exception as e:
            carb.log_error(f"Policy Load Failed: {e}")
            return

        self.num_dof = self._robot.num_dof
        self.joint_names = self._robot.dof_names
        print(f">>> Robot DOFs: {self.num_dof}")

        # 텐서 버퍼 초기화
        self.default_pos = torch.zeros(self.num_dof, device=self.device)
        self.action_scale = torch.zeros(self.num_dof, device=self.device)

        init_pos_dict = {
            
        }

        

        # (4) 물리 콜백 등록 (매 프레임 실행될 함수)
        self._world.add_physics_callback("physics_step", callback_fn=self._robot_control)
        return

    # ----------------------------------------------------------------------
    def _get_observation(self):
        
        return

    def _robot_control(self, step_size):
        try:
            obs = self._get_observation()
        except Exception as e:
            
            return 

        
        with torch.no_grad():
            
            action = self.policy(obs.unsqueeze(0)).squeeze(0)

        
        target_pos = action * self.action_scale + self.default_pos
        target_np = target_pos.detach().cpu().numpy()
        
        action_cmd = ArticulationAction(joint_positions=target_np)
        self._robot.apply_action(action_cmd)
        
        return

    def world_cleanup(self):
        # 종료 시 정리할 게 있다면 작성
        return