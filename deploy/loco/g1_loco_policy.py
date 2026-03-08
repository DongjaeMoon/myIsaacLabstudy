#[/home/idim5080-2/mdj/myIsaacLabstudy/deploy/loco/g1.py]
import os
from typing import Optional
import numpy as np

from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from loco.controllers.policy_controller import PolicyController

G1_29_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

class G1FlatTerrainPolicy(PolicyController):
    def __init__(
        self,
        prim_path: str,
        policy_path: str,
        usd_path: str,
        name: str = "G1",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, None, usd_path, position, orientation)

        self.load_policy(
            os.path.join(policy_path, "policy.pt"),
            os.path.join(policy_path, "env.yaml")
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(29, dtype=np.float32)   
        self.action = np.zeros(29, dtype=np.float32)             
        self._policy_counter = 0

        init_state = self.policy_env_params["scene"]["robot"]["init_state"]     
        self.default_root_pos = np.array(init_state["pos"], dtype=np.float32)   
        self.default_root_rot = np.array(init_state["rot"], dtype=np.float32)   
        self.default_root_lin_vel = np.array(init_state["lin_vel"], dtype=np.float32)  
        self.default_root_ang_vel = np.array(init_state["ang_vel"], dtype=np.float32)  

    def initialize(self):
        super().initialize(set_articulation_props=True)  

        usd_dof_names = self.robot.dof_names
        missing = [name for name in G1_29_JOINTS if name not in usd_dof_names]   
        if len(missing) > 0:                                                     
            raise ValueError(f"USD에 없는 policy joint가 있습니다: {missing}")     

        # 🚨 [수정 1] Action은 리스트 순서 그대로 (preserve_order=True)
        self.action_to_usd_idx = [usd_dof_names.index(name) for name in G1_29_JOINTS]
        
        # 🚨 [수정 2] Observation은 USD 인덱스 정렬 순서 (preserve_order=False)
        self.obs_to_usd_idx = sorted(self.action_to_usd_idx)

    def reset_policy_state(self):  
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._policy_counter = 0

    def _compute_observation(self, command):
        command = np.asarray(command, dtype=np.float32)  

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0], dtype=np.float32))  

        obs = np.zeros(99, dtype=np.float32)  
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command

        current_joint_pos_usd = self.robot.get_joint_positions()
        current_joint_vel_usd = self.robot.get_joint_velocities()

        # 🚨 [수정 3] 관측값은 obs_to_usd_idx 순서대로 넣어줍니다.
        for i, usd_idx in enumerate(self.obs_to_usd_idx):
            obs[12 + i] = current_joint_pos_usd[usd_idx] - self.default_pos[usd_idx]
            obs[41 + i] = current_joint_vel_usd[usd_idx]

        obs[70:99] = self._previous_action
        return obs

    def forward(self, dt, command):
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            raw_action = self._compute_action(obs).astype(np.float32)  
            
            # 🚨 [방탄 코드 수정] 학습 시 clip_actions: null 이었으므로 제한을 풉니다.
            self.action = raw_action 

            # [디버깅 추가] 첫 번째 제어 스텝에서 네트워크 입출력 상태 확인
            if self._policy_counter == 0:
                print("\n" + "="*50)
                print("[DEBUG] Step 0 - Policy Observation & Action")
                print(f"1. Base Command (명령)       : {obs[9:12]}")
                print(f"2. Proj Gravity (중력 방향)  : {obs[6:9]}")
                print(f"3. Joint Pos Error (관절 오차): \n{np.round(obs[12:41], 3)}")
                print(f"4. Raw Action (네트워크 출력) : \n{np.round(self.action, 3)}")
                print("="*50 + "\n")
            
            self._previous_action = self.action.copy()

        policy_action = self.action * self._action_scale

        usd_target_positions = np.array(self.default_pos, dtype=np.float32)

        for i, usd_idx in enumerate(self.action_to_usd_idx):
            usd_target_positions[usd_idx] += policy_action[i]

        action_cmd = ArticulationAction(joint_positions=usd_target_positions)
        self.robot.apply_action(action_cmd)

        self._policy_counter += 1