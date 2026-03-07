import os
from typing import Optional
import numpy as np

from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from loco.controllers.policy_controller import PolicyController

# 29-DOF 훈련 시 지정했던 관절 순서 (절대 변경 금지)
G1_29_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]

class G1FlatTerrainPolicy(PolicyController):
    def __init__(self, prim_path: str, policy_path: str, usd_path: str, name: str = "G1", position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None) -> None:
        super().__init__(name, prim_path, None, usd_path, position, orientation)
        
        self.load_policy(
            os.path.join(policy_path, "policy.pt"),
            os.path.join(policy_path, "env.yaml")
        )
        self._action_scale = 0.5
        self._previous_action = np.zeros(29)
        self._policy_counter = 0

    def initialize(self):
        super().initialize(set_articulation_props=False)
        
        # [핵심] USD의 관절 순서와 훈련 시 관절 순서(29 DOF) 매핑
        usd_dof_names = self.robot.dof_names
        self.policy_to_usd_idx = [usd_dof_names.index(name) for name in G1_29_JOINTS]

    def _compute_observation(self, command):
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        # 99차원 관측 공간 생성 (3+3+3+3+29+29+29 = 99)
        obs = np.zeros(99)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        
        # 전체 관절 상태 가져오기 (USD 순서)
        current_joint_pos_usd = self.robot.get_joint_positions()
        current_joint_vel_usd = self.robot.get_joint_velocities()
        
        # 29-DOF 정책 순서에 맞게 추출 및 오차 계산
        for i, usd_idx in enumerate(self.policy_to_usd_idx):
            obs[12 + i] = current_joint_pos_usd[usd_idx] - self.default_pos[usd_idx]
            obs[41 + i] = current_joint_vel_usd[usd_idx]
            
        obs[70:99] = self._previous_action
        return obs

    def forward(self, dt, command):
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs) # 29차원 Action 생성
            self._previous_action = self.action.copy()

        # 스케일 적용된 29차원 행동
        policy_action = self.action * self._action_scale
        
        # USD의 전체 관절(43개 등)에 맞게 행동 적용할 빈 배열 생성
        usd_target_positions = np.array(self.default_pos)
        
        # 29개의 제어 관절만 목표값 업데이트
        for i, usd_idx in enumerate(self.policy_to_usd_idx):
            usd_target_positions[usd_idx] += policy_action[i]

        action_cmd = ArticulationAction(joint_positions=usd_target_positions)
        self.robot.apply_action(action_cmd)

        self._policy_counter += 1
