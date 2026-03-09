# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch/g1_catch_policy.py] (연구실)
# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/catch/g1_catch_policy.py] (전산실)

import os
from typing import Optional
import numpy as np

from isaacsim.core.utils.types import ArticulationAction
from catch.controllers.policy_controller import PolicyController

OBS_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

ACTION_JOINT_NAMES = [
    "left_ankle_pitch_joint", "left_hip_pitch_joint", "left_knee_joint", 
    "right_ankle_pitch_joint", "right_hip_pitch_joint", "right_knee_joint",
    "left_ankle_roll_joint", "left_hip_roll_joint", 
    "right_ankle_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint",
    "left_elbow_joint", "left_shoulder_pitch_joint", 
    "right_elbow_joint", "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint", "left_wrist_yaw_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint", "right_wrist_yaw_joint"
]

ACTION_SCALES = np.array([
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.2, 0.2, 0.2, 0.2,
    0.1, 0.1,
    0.2, 0.2, 0.2,
    0.5, 0.5,
    0.5, 0.5,
    0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.3
], dtype=np.float32)

def quat_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_apply(q, v):
    res = quat_mul(quat_mul(q, np.array([0, v[0], v[1], v[2]])), quat_conj(q))
    return res[1:4]

def quat_rotate_inverse(q, v): return quat_apply(quat_conj(q), v)

def quat_to_rot6d(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w),
        2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w)
    ])

class G1CatchPolicy(PolicyController):
    def __init__(
        self, prim_path: str, policy_path: str, usd_path: str,
        name: str = "G1", position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, None, usd_path, position, orientation)
        self.load_policy(os.path.join(policy_path, "policy.pt"), os.path.join(policy_path, "env.yaml"))
        
        self._previous_action = np.zeros(29, dtype=np.float32)   
        self.action = np.zeros(29, dtype=np.float32)             
        self._policy_counter = 0
        self._decimation = 2 

        init_state = self.policy_env_params["scene"]["robot"]["init_state"]     
        self.default_root_pos = np.array(init_state.get("pos", [0.0, 0.0, 0.78]), dtype=np.float32)   
        self.default_root_rot = np.array(init_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)   
        self.default_root_lin_vel = np.array(init_state.get("lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)  
        self.default_root_ang_vel = np.array(init_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)

    def initialize(self):
        super().initialize(set_articulation_props=True)  
        usd_dof_names = self.robot.dof_names
        
        self.obs_to_usd_idx = [usd_dof_names.index(n) for n in OBS_JOINT_NAMES]
        self.action_to_usd_idx = [usd_dof_names.index(n) for n in ACTION_JOINT_NAMES]

        self._env_default_pos = np.zeros(len(usd_dof_names), dtype=np.float32)
        self.default_vel = np.zeros(len(usd_dof_names), dtype=np.float32)

        init_state = self.policy_env_params["scene"]["robot"]["init_state"]
        if "joint_pos" in init_state:
            for j_name, j_val in init_state["joint_pos"].items():
                if j_name == ".*": continue
                if j_name in usd_dof_names:
                    idx = usd_dof_names.index(j_name)
                    self._env_default_pos[idx] = j_val
        self.default_pos = self._env_default_pos

    # 🚨 [생선처럼 굳는 현상 완벽 해결] 에러 유발 코드 제거하고 안전하게 게인 적용
    def apply_pd_gains(self):
        controller = self.robot.get_articulation_controller()
        if controller is None:
            return False # 컨트롤러가 준비 안 되었으면 False 반환 후 대기
            
        usd_dof_names = self.robot.dof_names
        kps = np.zeros(len(usd_dof_names), dtype=np.float32)
        kds = np.zeros(len(usd_dof_names), dtype=np.float32)
        
        for i, name in enumerate(usd_dof_names):
            if name in ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]:
                kps[i] = 120.0; kds[i] = 10.0
            elif name in ["left_shoulder_pitch_joint", "left_elbow_joint", "right_shoulder_pitch_joint", "right_elbow_joint"]:
                kps[i] = 85.0; kds[i] = 12.0
            elif name in ["left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]:
                kps[i] = 55.0; kds[i] = 10.0
            else: 
                kps[i] = 300.0; kds[i] = 60.0
                
        controller.set_gains(kps=kps, kds=kds)
        print("[DEBUG] 짱짱한 PD Gain 강제 주입 완료! 로봇이 힘을 줍니다.")
        return True

    def reset_policy_state(self):  
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._policy_counter = 0

    def _compute_observation(self, toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel):
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        q_IB = self.robot.get_world_pose()[1]

        lin_vel_b = quat_rotate_inverse(q_IB, lin_vel_I)
        ang_vel_b = quat_rotate_inverse(q_IB, ang_vel_I)
        gravity_b = quat_rotate_inverse(q_IB, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        obs = np.zeros(141, dtype=np.float32)  
        obs[0] = toss_signal
        obs[1:4] = gravity_b
        obs[4:7] = lin_vel_b
        obs[7:10] = ang_vel_b

        current_joint_pos_usd = self.robot.get_joint_positions()
        current_joint_vel_usd = self.robot.get_joint_velocities()
        current_joint_efforts = self.robot.get_measured_joint_efforts()
        if current_joint_efforts is None or len(current_joint_efforts) == 0:
            current_joint_efforts = np.zeros_like(current_joint_pos_usd)

        for i, usd_idx in enumerate(self.obs_to_usd_idx):
            obs[10 + i] = current_joint_pos_usd[usd_idx] 
            obs[39 + i] = current_joint_vel_usd[usd_idx]
            obs[68 + i] = np.clip(current_joint_efforts[usd_idx] * 0.0125, -1.0, 1.0) 

        obs[97:126] = self._previous_action

        r_p = self.robot.get_world_pose()[0]
        rel_p_b = quat_rotate_inverse(q_IB, obj_pos - r_p)
        rel_v_b = quat_rotate_inverse(q_IB, obj_lin_vel - lin_vel_I)
        rel_w_b = quat_rotate_inverse(q_IB, obj_ang_vel - ang_vel_I)
        rel_r6 = quat_to_rot6d(quat_mul(quat_conj(q_IB), obj_rot))

        rel_p_b *= toss_signal
        rel_v_b *= toss_signal
        rel_w_b *= toss_signal
        rel_r6 *= toss_signal

        obs[126:129] = rel_p_b
        obs[129:135] = rel_r6
        obs[135:138] = rel_v_b
        obs[138:141] = rel_w_b

        return obs

    def forward(self, dt, toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel):
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel)
            raw_action = self._compute_action(obs).astype(np.float32)  
            self.action = np.clip(raw_action, -1.0, 1.0) 
            self._previous_action = self.action.copy()

        policy_action = self.action * ACTION_SCALES
        usd_target_positions = self._env_default_pos.copy()

        for i, usd_idx in enumerate(self.action_to_usd_idx):
            usd_target_positions[usd_idx] += policy_action[i]

        action_cmd = ArticulationAction(joint_positions=usd_target_positions)
        self.robot.apply_action(action_cmd)
        self._policy_counter += 1