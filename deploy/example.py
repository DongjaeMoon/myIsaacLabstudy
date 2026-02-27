# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example.py]

import torch
import numpy as np
import carb

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction

# --------------------------------------------------------------------------
# [설정] sim2sim_g1.py에 있던 상수들을 가져옵니다.
# --------------------------------------------------------------------------
# ★★★ 경로가 맞는지 꼭 확인하세요! ★★★
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_g1_loco_v0/2026-02-27_17-13-18/exported/policy.pt"
ROBOT_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd"
ROBOT_PRIM_PATH = "/World/G1"

class Example(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_mode = True # 디버깅용 로그 출력 여부
    # ----------------------------------------------------------------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.78]))

        
        return
    # ----------------------------------------------------------------------
    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")
        self._box = self._world.scene.get_object("box")

        
        self._robot.initialize()
        
        
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

        
        self.default_pos = torch.zeros(self.num_dof, device=self.device)
        self.action_scale = torch.zeros(self.num_dof, device=self.device)

        
        init_pos_dict = {
            "hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2,
            "shoulder_pitch": 0.2, "elbow": 0.5
        }

        
        for i, name in enumerate(self.joint_names):
            # Gains & Scale
            if ("shoulder" in name) or ("elbow" in name) or ("wrist" in name):
                self.action_scale[i] = 1.5
            else:
                self.action_scale[i] = 1.0

            # Default Pose
            for key, val in init_pos_dict.items():
                if key in name:
                    self.default_pos[i] = float(val)

        self._world.add_physics_callback("physics_step", callback_fn=self._robot_control)
        return


    # ----------------------------------------------------------------------
    def _get_observation(self):
        j_pos_np = self._robot.get_joint_positions()
        j_vel_np = self._robot.get_joint_velocities()
        
        j_pos = torch.tensor(j_pos_np, device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(j_vel_np, device=self.device, dtype=torch.float32)

        base_pos, base_quat = self._robot.get_world_pose()
        base_lin = torch.tensor(self._robot.get_linear_velocity(), device=self.device, dtype=torch.float32)
        base_ang = torch.tensor(self._robot.get_angular_velocity(), device=self.device, dtype=torch.float32)
        base_quat = torch.tensor(base_quat, device=self.device, dtype=torch.float32) # (w, x, y, z)

        obs = torch.cat([j_pos, j_vel, base_lin, base_ang, base_quat])
        return obs

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
        return