import io
import os
from typing import Dict, Optional, Tuple

import carb
import numpy as np
import omni
import torch
from omni.physx import get_physx_simulation_interface
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.utils.types import ArticulationAction

from merged.controllers.config_loader import (
    get_articulation_props,
    get_physics_properties,
    get_robot_joint_properties,
    parse_env_config,
)


LOCO_ACTION_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

CATCH_OBS_JOINT_NAMES = [
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

CATCH_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

CATCH_ACTION_SCALES = np.array([
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.2, 0.2, 0.2, 0.2,
    0.1, 0.1,
    0.2, 0.2, 0.2,
    0.5, 0.5,
    0.5, 0.5,
    0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.3,
], dtype=np.float32)

PREPARE_BLEND_JOINTS = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul(quat_mul(q, qq), quat_conj(q))[1:4]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q), v)


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        1 - 2 * (y * y + z * z),
        2 * (x * y + z * w),
        2 * (x * z - y * w),
        2 * (x * y - z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z + x * w),
    ], dtype=np.float32)


class G1LocoCatchMergedPolicy:
    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        loco_policy_dir: str,
        catch_policy_dir: str,
        world_dt: float,
        name: str = "G1",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                carb.log_error("Robot USD path is missing.")

        self.robot = SingleArticulation(prim_path=prim_path, name=name, position=position, orientation=orientation)
        self.world_dt = float(world_dt)

        self.loco_policy = self._load_jit_policy(os.path.join(loco_policy_dir, "policy.pt"))
        self.catch_policy = self._load_jit_policy(os.path.join(catch_policy_dir, "policy.pt"))
        self.loco_env = parse_env_config(os.path.join(loco_policy_dir, "env.yaml"))
        self.catch_env = parse_env_config(os.path.join(catch_policy_dir, "env.yaml"))

        self.loco_decimation_train, self.loco_dt_train, _ = get_physics_properties(self.loco_env)
        self.catch_decimation_train, self.catch_dt_train, _ = get_physics_properties(self.catch_env)
        self.loco_action_period = float(self.loco_decimation_train) * float(self.loco_dt_train)
        self.catch_action_period = float(self.catch_decimation_train) * float(self.catch_dt_train)
        self.loco_runtime_decimation = max(1, int(round(self.loco_action_period / self.world_dt)))
        self.catch_runtime_decimation = max(1, int(round(self.catch_action_period / self.world_dt)))

        self._loco_policy_counter = 0
        self._catch_policy_counter = 0
        self._loco_prev_action = np.zeros(29, dtype=np.float32)
        self._catch_prev_action = np.zeros(29, dtype=np.float32)
        self._loco_action = np.zeros(29, dtype=np.float32)
        self._catch_action = np.zeros(29, dtype=np.float32)

        self._quat_raw_is_xyzw = None
        self._active_profile = "loco"
        self._pending_profile = "loco"
        self.is_initialized = False
        self.debug_enabled = True
        self._debug_printed = False

    def _load_jit_policy(self, policy_file_path: str):
        file_content = omni.client.read_file(policy_file_path)[2]
        return torch.jit.load(io.BytesIO(memoryview(file_content).tobytes()))

    def initialize(self, physics_sim_view=None) -> None:
        self.robot.initialize(physics_sim_view=physics_sim_view)
        controller = self.robot.get_articulation_controller()
        controller.set_effort_modes("force")
        get_physx_simulation_interface().flush_changes()
        controller.switch_control_mode("position")

        self.usd_dof_names = list(self.robot.dof_names)

        (
            self.loco_max_effort,
            self.loco_max_vel,
            self.loco_stiffness,
            self.loco_damping,
            _,
            self.loco_default_pos,
            self.loco_default_vel,
        ) = get_robot_joint_properties(self.loco_env, self.usd_dof_names)

        (
            self.catch_max_effort,
            self.catch_max_vel,
            self.catch_stiffness,
            self.catch_damping,
            _,
            self.catch_default_pos,
            self.catch_default_vel,
        ) = get_robot_joint_properties(self.catch_env, self.usd_dof_names)

        loco_init = self.loco_env["scene"]["robot"]["init_state"]
        catch_init = self.catch_env["scene"]["robot"]["init_state"]

        self.loco_default_root_pos = np.array(loco_init.get("pos", [0.0, 0.0, 0.79]), dtype=np.float32)
        self.loco_default_root_rot = np.array(loco_init.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.loco_default_root_lin_vel = np.array(loco_init.get("lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.loco_default_root_ang_vel = np.array(loco_init.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)

        self.catch_default_root_pos = np.array(catch_init.get("pos", [0.0, 0.0, 0.78]), dtype=np.float32)
        self.catch_default_root_rot = np.array(catch_init.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.catch_default_root_lin_vel = np.array(catch_init.get("lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.catch_default_root_ang_vel = np.array(catch_init.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)

        self.loco_action_to_usd_idx = [self.usd_dof_names.index(name) for name in LOCO_ACTION_JOINTS]
        self.loco_obs_to_usd_idx = sorted(self.loco_action_to_usd_idx)
        self.catch_obs_to_usd_idx = [self.usd_dof_names.index(name) for name in CATCH_OBS_JOINT_NAMES]
        self.catch_action_to_usd_idx = [self.usd_dof_names.index(name) for name in CATCH_ACTION_JOINT_NAMES]
        self.prepare_blend_usd_idx = [self.usd_dof_names.index(name) for name in PREPARE_BLEND_JOINTS]

        self.robot.post_reset()
        _, q_raw = self.robot.get_world_pose()
        self._detect_and_lock_quat_order(q_raw)

        self.is_initialized = True
        self.apply_profile(self._pending_profile)

        if self.debug_enabled and not self._debug_printed:
            self._debug_printed = True
            print("\n" + "=" * 90)
            print("[MERGED] initialized")
            print(f"[MERGED] world_dt={self.world_dt:.6f}")
            print(
                f"[MERGED] loco: train_dt={self.loco_dt_train:.6f}, train_decimation={self.loco_decimation_train}, "
                f"action_period={self.loco_action_period:.6f}, runtime_decimation={self.loco_runtime_decimation}"
            )
            print(
                f"[MERGED] catch: train_dt={self.catch_dt_train:.6f}, train_decimation={self.catch_decimation_train}, "
                f"action_period={self.catch_action_period:.6f}, runtime_decimation={self.catch_runtime_decimation}"
            )
            print("[MERGED] locomotion observation uses USD-sorted indices (preserve_order=False in training).")
            print("[MERGED] locomotion action uses policy joint list order (preserve_order=True in training).")
            print("[MERGED] catch observation/action both use the explicit joint-name orders from g1_catch_policy.py.")
            print("=" * 90 + "\n")

    def _detect_and_lock_quat_order(self, q_raw: np.ndarray) -> None:
        if self._quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
        if q.shape[0] != 4:
            self._quat_raw_is_xyzw = True
            return
        if abs(q[3]) > 0.90:
            self._quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90:
            self._quat_raw_is_xyzw = False
        else:
            self._quat_raw_is_xyzw = True
        print(f"[MERGED] quat raw order locked -> {'xyzw' if self._quat_raw_is_xyzw else 'wxyz'} | raw={q}")

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float32).copy()
        self._detect_and_lock_quat_order(q)
        if self._quat_raw_is_xyzw:
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        return q

    def _compute_action(self, policy, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float()
            action = policy(obs_t).detach().view(-1).cpu().numpy()
        return action.astype(np.float32)

    def apply_profile(self, profile: str) -> bool:
        if profile not in ("loco", "catch"):
            raise ValueError(f"Unknown profile: {profile}")

    # 항상 마지막 요청 profile은 기억
        self._pending_profile = profile

        if not self.is_initialized:
            print(f"[MERGED] profile request before initialize -> deferred: {profile}")
            return False

        art_view = getattr(self.robot, "_articulation_view", None)
        phys_view = getattr(art_view, "_physics_view", None) if art_view is not None else None

        # [MODIFIED] reset/load 직후 physics view가 아직 재생성되지 않은 순간이면 defer
        if art_view is None or phys_view is None:
            print(f"[MERGED] profile request while articulation view is not ready -> deferred: {profile}")
            return False

        if profile == "loco":
            stiffness = self.loco_stiffness
            damping = self.loco_damping
            max_effort = self.loco_max_effort
            max_vel = self.loco_max_vel
            env = self.loco_env
        else:
            stiffness = self.catch_stiffness
            damping = self.catch_damping
            max_effort = self.catch_max_effort
            max_vel = self.catch_max_vel
            env = self.catch_env

        art_view.set_gains(stiffness, damping)
        art_view.set_max_efforts(max_effort)
        get_physx_simulation_interface().flush_changes()
        art_view.set_max_joint_velocities(max_vel)

        self._apply_articulation_props_from_env(env)
        self._active_profile = profile
        print(f"[MERGED] switched articulation profile -> {profile}")
        return True

    def _apply_articulation_props_from_env(self, env_cfg: dict) -> None:
        articulation_prop = get_articulation_props(env_cfg)
        if articulation_prop is None:
            return
        sp = articulation_prop.get("solver_position_iteration_count")
        sv = articulation_prop.get("solver_velocity_iteration_count")
        st = articulation_prop.get("stabilization_threshold")
        esc = articulation_prop.get("enabled_self_collisions")
        sl = articulation_prop.get("sleep_threshold")
        if sp not in [None, float("inf")]:
            self.robot.set_solver_position_iteration_count(sp)
        if sv not in [None, float("inf")]:
            self.robot.set_solver_velocity_iteration_count(sv)
        if st not in [None, float("inf")]:
            self.robot.set_stabilization_threshold(st)
        if isinstance(esc, bool):
            self.robot.set_enabled_self_collisions(esc)
        if sl not in [None, float("inf")]:
            self.robot.set_sleep_threshold(sl)

    def reset_policy_state(self) -> None:
        self._loco_policy_counter = 0
        self._catch_policy_counter = 0
        self._loco_prev_action[:] = 0.0
        self._catch_prev_action[:] = 0.0
        self._loco_action[:] = 0.0
        self._catch_action[:] = 0.0

    def _set_robot_state(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        root_lin_vel: np.ndarray,
        root_ang_vel: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> None:
        self.robot.set_default_state(position=root_pos, orientation=root_rot)
        self.robot.set_joints_default_state(positions=joint_pos, velocities=joint_vel)
        self.robot.post_reset()
        self.robot.set_world_pose(position=root_pos, orientation=root_rot)
        self.robot.set_linear_velocity(root_lin_vel)
        self.robot.set_angular_velocity(root_ang_vel)
        self.robot.set_joint_positions(joint_pos)
        self.robot.set_joint_velocities(joint_vel)
        self.reset_policy_state()

    def reset_robot_to_loco_default(self) -> None:
        if not self.is_initialized:
            return
        self._set_robot_state(
            self.loco_default_root_pos,
            self.loco_default_root_rot,
            self.loco_default_root_lin_vel,
            self.loco_default_root_ang_vel,
            self.loco_default_pos,
            self.loco_default_vel,
        )

    def reset_robot_to_catch_default(self) -> None:
        if not self.is_initialized:
            return
        self._set_robot_state(
            self.catch_default_root_pos,
            self.catch_default_root_rot,
            self.catch_default_root_lin_vel,
            self.catch_default_root_ang_vel,
            self.catch_default_pos,
            self.catch_default_vel,
        )

    def get_root_pose_wxyz(self) -> Tuple[np.ndarray, np.ndarray]:
        root_pos, q_raw = self.robot.get_world_pose()
        return np.asarray(root_pos, dtype=np.float32), self._to_wxyz(q_raw)

    def body_to_world(self, v_body: np.ndarray, q_IB_wxyz: np.ndarray) -> np.ndarray:
        return quat_apply(q_IB_wxyz, np.asarray(v_body, dtype=np.float32))

    def compute_box_relative_kinematics(
        self,
        obj_pos: np.ndarray,
        obj_lin_vel: np.ndarray,
        obj_ang_vel: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        root_pos, q_IB = self.get_root_pose_wxyz()
        robot_lin_I = np.asarray(self.robot.get_linear_velocity(), dtype=np.float32)
        robot_ang_I = np.asarray(self.robot.get_angular_velocity(), dtype=np.float32)
        rel_p_b = quat_rotate_inverse(q_IB, np.asarray(obj_pos, dtype=np.float32) - root_pos)
        rel_v_b = quat_rotate_inverse(q_IB, np.asarray(obj_lin_vel, dtype=np.float32) - robot_lin_I)
        rel_w_b = quat_rotate_inverse(q_IB, np.asarray(obj_ang_vel, dtype=np.float32) - robot_ang_I)
        return {
            "root_pos": root_pos,
            "root_quat_wxyz": q_IB,
            "robot_lin_vel_world": robot_lin_I,
            "rel_p_b": rel_p_b,
            "rel_v_b": rel_v_b,
            "rel_w_b": rel_w_b,
        }

    def _compute_loco_observation(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32)
        lin_vel_I = np.asarray(self.robot.get_linear_velocity(), dtype=np.float32)
        ang_vel_I = np.asarray(self.robot.get_angular_velocity(), dtype=np.float32)
        _, q_IB = self.get_root_pose_wxyz()
        gravity_b = quat_rotate_inverse(q_IB, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        lin_vel_b = quat_rotate_inverse(q_IB, lin_vel_I)
        ang_vel_b = quat_rotate_inverse(q_IB, ang_vel_I)
        joint_pos = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        joint_vel = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)

        obs = np.zeros(99, dtype=np.float32)
        obs[:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command
        for i, usd_idx in enumerate(self.loco_obs_to_usd_idx):
            obs[12 + i] = joint_pos[usd_idx] - self.loco_default_pos[usd_idx]
            obs[41 + i] = joint_vel[usd_idx]
        obs[70:99] = self._loco_prev_action
        return obs

    def _compute_catch_observation(
        self,
        toss_signal: float,
        obj_pos: np.ndarray,
        obj_rot_raw: np.ndarray,
        obj_lin_vel: np.ndarray,
        obj_ang_vel: np.ndarray,
    ) -> np.ndarray:
        lin_vel_I = np.asarray(self.robot.get_linear_velocity(), dtype=np.float32)
        ang_vel_I = np.asarray(self.robot.get_angular_velocity(), dtype=np.float32)
        root_pos, q_IB = self.get_root_pose_wxyz()
        obj_rot = self._to_wxyz(obj_rot_raw)
        gravity_b = quat_rotate_inverse(q_IB, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        lin_vel_b = quat_rotate_inverse(q_IB, lin_vel_I)
        ang_vel_b = quat_rotate_inverse(q_IB, ang_vel_I)

        joint_pos = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        joint_vel = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)
        joint_eff = self.robot.get_measured_joint_efforts()
        if joint_eff is None or len(joint_eff) == 0:
            joint_eff = np.zeros_like(joint_pos)
        else:
            joint_eff = np.asarray(joint_eff, dtype=np.float32)

        obs = np.zeros(141, dtype=np.float32)
        obs[0] = float(toss_signal)
        obs[1:4] = gravity_b
        obs[4:7] = lin_vel_b
        obs[7:10] = ang_vel_b

        for i, usd_idx in enumerate(self.catch_obs_to_usd_idx):
            obs[10 + i] = joint_pos[usd_idx]
            obs[39 + i] = joint_vel[usd_idx]
            obs[68 + i] = np.clip(joint_eff[usd_idx] * 0.0125, -1.0, 1.0)

        obs[97:126] = self._catch_prev_action
        rel_p_b = quat_rotate_inverse(q_IB, np.asarray(obj_pos, dtype=np.float32) - root_pos)
        rel_v_b = quat_rotate_inverse(q_IB, np.asarray(obj_lin_vel, dtype=np.float32) - lin_vel_I)
        rel_w_b = quat_rotate_inverse(q_IB, np.asarray(obj_ang_vel, dtype=np.float32) - ang_vel_I)
        rel_r6 = quat_to_rot6d(quat_mul(quat_conj(q_IB), obj_rot))

        obs[126:129] = rel_p_b * float(toss_signal)
        obs[129:135] = rel_r6 * float(toss_signal)
        obs[135:138] = rel_v_b * float(toss_signal)
        obs[138:141] = rel_w_b * float(toss_signal)
        return obs

    def get_loco_target_positions(self, command: np.ndarray) -> np.ndarray:
        if self._loco_policy_counter % self.loco_runtime_decimation == 0:
            obs = self._compute_loco_observation(command)
            self._loco_action = self._compute_action(self.loco_policy, obs)
            self._loco_prev_action = self._loco_action.copy()
        target = self.loco_default_pos.copy()
        target[self.loco_action_to_usd_idx] += self._loco_action * 0.5
        self._loco_policy_counter += 1
        return target.astype(np.float32)

    def get_catch_target_positions(
        self,
        toss_signal: float,
        obj_pos: np.ndarray,
        obj_rot: np.ndarray,
        obj_lin_vel: np.ndarray,
        obj_ang_vel: np.ndarray,
    ) -> np.ndarray:
        if self._catch_policy_counter % self.catch_runtime_decimation == 0:
            obs = self._compute_catch_observation(toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel)
            raw_action = self._compute_action(self.catch_policy, obs)
            self._catch_action = np.clip(raw_action, -1.0, 1.0)
            self._catch_prev_action = self._catch_action.copy()
        target = self.catch_default_pos.copy()
        target[self.catch_action_to_usd_idx] += self._catch_action * CATCH_ACTION_SCALES
        self._catch_policy_counter += 1
        return target.astype(np.float32)

    def apply_target_positions(self, target_positions: np.ndarray) -> None:
        self.robot.apply_action(ArticulationAction(joint_positions=np.asarray(target_positions, dtype=np.float32)))

    def forward_loco(self, command: np.ndarray) -> np.ndarray:
        target = self.get_loco_target_positions(command)
        self.apply_target_positions(target)
        return target

    def forward_catch(
        self,
        toss_signal: float,
        obj_pos: np.ndarray,
        obj_rot: np.ndarray,
        obj_lin_vel: np.ndarray,
        obj_ang_vel: np.ndarray,
    ) -> np.ndarray:
        target = self.get_catch_target_positions(toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel)
        self.apply_target_positions(target)
        return target

    def blend_prepare_targets(
        self,
        alpha: float,
        loco_target: np.ndarray,
        catch_target: np.ndarray,
    ) -> np.ndarray:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out = np.asarray(loco_target, dtype=np.float32).copy()
        out[self.prepare_blend_usd_idx] = (
            (1.0 - alpha) * out[self.prepare_blend_usd_idx]
            + alpha * np.asarray(catch_target, dtype=np.float32)[self.prepare_blend_usd_idx]
        )
        return out