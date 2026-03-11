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
'''
ACTION_JOINT_NAMES = [
    # 1. legs_sagittal (6개)
    "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
    
    # 2. legs_frontal (4개)
    "left_hip_roll_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_ankle_roll_joint",
    
    # 3. legs_yaw (2개)
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    
    # 4. waist (3개)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    
    # 5. left_arm_capture (2개)
    "left_shoulder_pitch_joint", "left_elbow_joint",
    
    # 6. right_arm_capture (2개)
    "right_shoulder_pitch_joint", "right_elbow_joint",
    
    # 7. left_arm_wrap (5개)
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    
    # 8. right_arm_wrap (5개)
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]'''
ACTION_JOINT_NAMES = [
    # 1. legs_sagittal (6개) - USD 인덱스 오름차순 정렬
    "left_hip_pitch_joint", 
    "right_hip_pitch_joint", 
    "left_knee_joint", 
    "right_knee_joint", 
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    
    # 2. legs_frontal (4개) - USD 인덱스 오름차순 정렬
    "left_hip_roll_joint", 
    "right_hip_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    
    # 3. legs_yaw (2개)
    "left_hip_yaw_joint", 
    "right_hip_yaw_joint",
    
    # 4. waist (3개)
    "waist_yaw_joint", 
    "waist_roll_joint", 
    "waist_pitch_joint",
    
    # 5. left_arm_capture (2개)
    "left_shoulder_pitch_joint", 
    "left_elbow_joint",
    
    # 6. right_arm_capture (2개)
    "right_shoulder_pitch_joint", 
    "right_elbow_joint",
    
    # 7. left_arm_wrap (5개)
    "left_shoulder_roll_joint", 
    "left_shoulder_yaw_joint",
    "left_wrist_roll_joint", 
    "left_wrist_pitch_joint", 
    "left_wrist_yaw_joint",
    
    # 8. right_arm_wrap (5개)
    "right_shoulder_roll_joint", 
    "right_shoulder_yaw_joint",
    "right_wrist_roll_joint", 
    "right_wrist_pitch_joint", 
    "right_wrist_yaw_joint"
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

def quat_conj(q): return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

def quat_apply(q, v):
    res = quat_mul(quat_mul(q, np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)), quat_conj(q))
    return res[1:4]

def quat_rotate_inverse(q, v):
    return quat_apply(quat_conj(q), v)

def quat_to_rot6d(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        1 - 2*(y*y + z*z),
        2*(x*y + z*w),
        2*(x*z - y*w),
        2*(x*y - z*w),
        1 - 2*(x*x + z*z),
        2*(y*z + x*w),
    ], dtype=np.float32)


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

        self._quat_raw_is_xyzw = None

        init_state = self.policy_env_params["scene"]["robot"]["init_state"]
        self.default_root_pos = np.array(init_state.get("pos", [0.0, 0.0, 0.78]), dtype=np.float32)
        self.default_root_rot = np.array(init_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.default_root_lin_vel = np.array(init_state.get("lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.default_root_ang_vel = np.array(init_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        # =========================
        # Debug options
        # =========================
        self.debug_enabled = False
        self.debug_print_first_n = 0
        self.debug_print_every = 0
        self.debug_show_full_joint_table = False

        # PROBE 모드에서 0.0으로 두면 policy는 계산하지만 실제 관절에는 적용 안 됨
        self.debug_action_gain = 1.0

        # ZERO 모드에서 True로 두면 clipped action 자체를 0으로 강제
        self.debug_force_zero_action = False

        self._debug_step = 0
        self._last_obs = None
        self._last_raw_action = None
        self._last_clipped_action = None
        self._last_policy_action = None
        self._last_target_positions = None

    def _detect_and_lock_quat_order(self, q_raw):
        if self._quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
        if q.shape[0] != 4:
            self._quat_raw_is_xyzw = True
            return

        # upright 상태면 scalar가 거의 1이어야 함
        if abs(q[3]) > 0.90:
            self._quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90:
            self._quat_raw_is_xyzw = False
        else:
            self._quat_raw_is_xyzw = True

        print(f"[DEBUG] quat raw order locked -> {'xyzw' if self._quat_raw_is_xyzw else 'wxyz'} | raw={q}")

    def _to_wxyz(self, q_raw):
        q = np.asarray(q_raw, dtype=np.float32).copy()
        self._detect_and_lock_quat_order(q)
        if self._quat_raw_is_xyzw:
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        return q

    def initialize(self):
        super().initialize(set_articulation_props=True)
        usd_dof_names = self.robot.dof_names

        self.obs_to_usd_idx = [usd_dof_names.index(n) for n in OBS_JOINT_NAMES]
        self.action_to_usd_idx = [usd_dof_names.index(n) for n in ACTION_JOINT_NAMES]

        # ========================================================
        # [수정된 부분] 수동 파싱 코드를 지우고 부모 클래스가 
        # config_loader.py로 파싱해둔 default_pos를 그대로 복사합니다.
        # ========================================================
        self._env_default_pos = self.default_pos.copy()
        self.default_vel = np.zeros(len(usd_dof_names), dtype=np.float32)

        # quat order lock
        _, q_raw = self.robot.get_world_pose()
        self._detect_and_lock_quat_order(q_raw)

    
    def _fmt(self, x):
        return np.array2string(np.asarray(x), precision=3, suppress_small=True)

    def _debug_should_print(self):
        if not self.debug_enabled:
            return False
        return (
            self._debug_step < self.debug_print_first_n
            or (self.debug_print_every > 0 and self._debug_step % self.debug_print_every == 0)
        )

    def _debug_print_static_mapping(self):
        usd_dof_names = self.robot.dof_names

        print("\n" + "=" * 120)
        print("[DEBUG][STATIC] G1CatchPolicy static mapping dump")
        print(f"[DEBUG][STATIC] policy dt={self._dt}, decimation={self._decimation}, render_interval={self.render_interval}")
        print(f"[DEBUG][STATIC] default_root_pos={self._fmt(self.default_root_pos)}")
        print(f"[DEBUG][STATIC] default_root_rot={self._fmt(self.default_root_rot)}")

        print("\n[DEBUG][STATIC] USD DOF ORDER")
        for i, name in enumerate(usd_dof_names):
            print(f"  usd[{i:02d}] {name}")

        print("\n[DEBUG][STATIC] OBS JOINT -> USD INDEX")
        for i, (name, usd_idx) in enumerate(zip(OBS_JOINT_NAMES, self.obs_to_usd_idx)):
            print(f"  obs[{i:02d}] {name:26s} -> usd[{usd_idx:02d}] {usd_dof_names[usd_idx]:26s}")

        print("\n[DEBUG][STATIC] ACTION JOINT -> USD INDEX / SCALE / DEFAULT")
        for i, (name, usd_idx, scale) in enumerate(zip(ACTION_JOINT_NAMES, self.action_to_usd_idx, ACTION_SCALES)):
            print(
                f"  act[{i:02d}] {name:26s} -> usd[{usd_idx:02d}] {usd_dof_names[usd_idx]:26s} "
                f"scale={scale:5.2f} default={self._env_default_pos[usd_idx]:+7.3f}"
            )

        print("\n[DEBUG][STATIC] ENV ACTION GROUPS FROM env.yaml")
        for group_name, cfg in self.policy_env_params.get("actions", {}).items():
            print(
                f"  {group_name:20s} preserve_order={cfg.get('preserve_order')} "
                f"scale={cfg.get('scale')} joints={cfg.get('joint_names')}"
            )

        print("=" * 120 + "\n")

    def _debug_print_step_dump(
        self,
        obs,
        raw_action,
        clipped_action,
        policy_action,
        usd_target_positions,
        toss_signal,
        obj_pos,
        obj_rot_raw,
        obj_lin_vel,
        obj_ang_vel,
    ):
        usd_pos = self.robot.get_joint_positions()
        usd_vel = self.robot.get_joint_velocities()
        usd_eff = self.robot.get_measured_joint_efforts()
        if usd_eff is None or len(usd_eff) == 0:
            usd_eff = np.zeros_like(usd_pos)

        root_pos, q_raw = self.robot.get_world_pose()
        q_used = self._to_wxyz(q_raw)

        q_raw_as_wxyz = np.asarray(q_raw, dtype=np.float32).copy()
        q_raw_as_xyzw_to_wxyz = np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]], dtype=np.float32)

        g_if_raw_is_wxyz = quat_rotate_inverse(q_raw_as_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        g_if_raw_is_xyzw = quat_rotate_inverse(q_raw_as_xyzw_to_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        g_used = quat_rotate_inverse(q_used, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        rel_p_b = quat_rotate_inverse(q_used, obj_pos - root_pos)

        obs_joint_pos = obs[10:39]
        obs_joint_vel = obs[39:68]
        obs_joint_eff = obs[68:97]

        current_obs_joint_pos = usd_pos[self.obs_to_usd_idx]
        current_obs_joint_vel = usd_vel[self.obs_to_usd_idx]
        current_obs_joint_eff = np.clip(usd_eff[self.obs_to_usd_idx] * 0.0125, -1.0, 1.0)

        pos_obs_err = np.max(np.abs(obs_joint_pos - current_obs_joint_pos))
        vel_obs_err = np.max(np.abs(obs_joint_vel - current_obs_joint_vel))
        eff_obs_err = np.max(np.abs(obs_joint_eff - current_obs_joint_eff))

        joint_err_from_default = usd_pos - self._env_default_pos
        max_joint_err = np.max(np.abs(joint_err_from_default))
        max_joint_vel = np.max(np.abs(usd_vel))
        max_effort = np.max(np.abs(usd_eff))

        sat_count = int((np.abs(clipped_action) >= 0.98).sum())

        print("\n" + "-" * 120)
        print(
            f"[DEBUG][STEP {self._debug_step:04d}] "
            f"policy_counter={self._policy_counter} toss={toss_signal:.1f} "
            f"force_zero={self.debug_force_zero_action} action_gain={self.debug_action_gain:.2f}"
        )
        print(f"[DEBUG][STEP {self._debug_step:04d}] root_pos={self._fmt(root_pos)} root_z={root_pos[2]:.4f}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] q_raw={self._fmt(q_raw)}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] q_used_wxyz={self._fmt(q_used)}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] gravity(if raw=wxyz)={self._fmt(g_if_raw_is_wxyz)}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] gravity(if raw=xyzw)={self._fmt(g_if_raw_is_xyzw)}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] gravity(used)={self._fmt(g_used)}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] lin_vel_I={self._fmt(lin_vel_I)} ang_vel_I={self._fmt(ang_vel_I)}")
        print(f"[DEBUG][STEP {self._debug_step:04d}] obj_pos={self._fmt(obj_pos)} rel_p_b={self._fmt(rel_p_b)}")
        print(
            f"[DEBUG][STEP {self._debug_step:04d}] obs[min,max,mean]="
            f"({obs.min():+.4f}, {obs.max():+.4f}, {obs.mean():+.4f}) "
            f"| raw[min,max]=({raw_action.min():+.4f}, {raw_action.max():+.4f}) "
            f"| sat_count={sat_count}"
        )
        print(
            f"[DEBUG][STEP {self._debug_step:04d}] "
            f"max|joint_err_from_default|={max_joint_err:.4f}, "
            f"max|joint_vel|={max_joint_vel:.4f}, "
            f"max|joint_effort|={max_effort:.4f}"
        )
        print(
            f"[DEBUG][STEP {self._debug_step:04d}] "
            f"obs_reconstruction_err: pos={pos_obs_err:.6f}, vel={vel_obs_err:.6f}, effort={eff_obs_err:.6f}"
        )

        order = np.argsort(-np.abs(policy_action))[:12]
        print(f"[DEBUG][STEP {self._debug_step:04d}] top |scaled action| joints")
        for k in order:
            usd_idx = self.action_to_usd_idx[k]
            print(
                f"  act[{k:02d}] {ACTION_JOINT_NAMES[k]:26s} "
                f"raw={raw_action[k]:+7.3f} clip={clipped_action[k]:+7.3f} "
                f"scale={ACTION_SCALES[k]:4.2f} delta={policy_action[k]:+7.3f} "
                f"default={self._env_default_pos[usd_idx]:+7.3f} "
                f"cur={usd_pos[usd_idx]:+7.3f} target={usd_target_positions[usd_idx]:+7.3f}"
            )

        if self.debug_show_full_joint_table and self._debug_step < 5:
            print(f"[DEBUG][STEP {self._debug_step:04d}] full obs-joint table")
            for i, usd_idx in enumerate(self.obs_to_usd_idx):
                print(
                    f"  obs[{i:02d}] {OBS_JOINT_NAMES[i]:26s} "
                    f"usd[{usd_idx:02d}] "
                    f"cur={usd_pos[usd_idx]:+7.3f} def={self._env_default_pos[usd_idx]:+7.3f} "
                    f"err={usd_pos[usd_idx] - self._env_default_pos[usd_idx]:+7.3f} "
                    f"vel={usd_vel[usd_idx]:+8.3f} eff={usd_eff[usd_idx]:+8.3f} "
                    f"obs_pos={obs_joint_pos[i]:+7.3f} obs_vel={obs_joint_vel[i]:+8.3f} obs_eff={obs_joint_eff[i]:+7.3f}"
                )
        print("-" * 120 + "\n")

    def apply_pd_gains(self):
        controller = self.robot.get_articulation_controller()
        if controller is None:
            return False

        usd_dof_names = self.robot.dof_names
        kps = np.zeros(len(usd_dof_names), dtype=np.float32)
        kds = np.zeros(len(usd_dof_names), dtype=np.float32)

        for i, name in enumerate(usd_dof_names):
            if name in [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
            ]:
                kps[i] = 120.0
                kds[i] = 10.0
            elif name in [
                "left_shoulder_pitch_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_elbow_joint"
            ]:
                kps[i] = 85.0
                kds[i] = 12.0
            elif name in [
                "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint",
                "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint",
                "right_wrist_pitch_joint", "right_wrist_yaw_joint"
            ]:
                kps[i] = 55.0
                kds[i] = 10.0
            else:
                kps[i] = 300.0
                kds[i] = 60.0

        controller.set_gains(kps=kps, kds=kds)
        print("[DEBUG] 짱짱한 PD Gain 강제 주입 완료! 로봇이 힘을 줍니다.")
        return True

    def reset_policy_state(self):
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._policy_counter = 0

    def _compute_observation(self, toss_signal, obj_pos, obj_rot_raw, obj_lin_vel, obj_ang_vel):
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()

        root_pos, q_raw = self.robot.get_world_pose()
        q_IB = self._to_wxyz(q_raw)

        obj_rot = self._to_wxyz(obj_rot_raw)

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

        rel_p_b = quat_rotate_inverse(q_IB, obj_pos - root_pos)
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
        recomputed = False

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel)
            raw_action = self._compute_action(obs).astype(np.float32)
            clipped_action = np.clip(raw_action, -1.0, 1.0)

            if self.debug_force_zero_action:
                clipped_action[:] = 0.0

            self.action = clipped_action
            self._previous_action = self.action.copy()

            self._last_obs = obs.copy()
            self._last_raw_action = raw_action.copy()
            self._last_clipped_action = clipped_action.copy()
            recomputed = True

        policy_action = self.action * ACTION_SCALES * float(self.debug_action_gain)
        usd_target_positions = self._env_default_pos.copy()

        for i, usd_idx in enumerate(self.action_to_usd_idx):
            usd_target_positions[usd_idx] += policy_action[i]

        self._last_policy_action = policy_action.copy()
        self._last_target_positions = usd_target_positions.copy()

        if self._debug_should_print():
            # decimation 중간 step에서는 last_* 사용
            if self._last_obs is not None and self._last_raw_action is not None and self._last_clipped_action is not None:
                self._debug_print_step_dump(
                    self._last_obs,
                    self._last_raw_action,
                    self._last_clipped_action,
                    self._last_policy_action,
                    self._last_target_positions,
                    toss_signal,
                    obj_pos,
                    obj_rot,
                    obj_lin_vel,
                    obj_ang_vel,
                )

        action_cmd = ArticulationAction(joint_positions=usd_target_positions)
        self.robot.apply_action(action_cmd)

        self._policy_counter += 1
        self._debug_step += 1