#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from enum import Enum, auto

import numpy as np
from pynput import keyboard

try:
    import torch
except Exception as e:
    raise RuntimeError("torch is required to run this script") from e

try:
    import yaml
except Exception:
    yaml = None

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

APRILTAG_AVAILABLE = False
try:
    import cv2
    import zmq
    from apriltag_object_state_estimator import (
        AprilTagObjectStateEstimator,
        RobotBaseState,
        make_T,
    )
    APRILTAG_AVAILABLE = True
except Exception:
    cv2 = None
    zmq = None
    AprilTagObjectStateEstimator = None
    RobotBaseState = None
    make_T = None

G1_NUM_MOTOR = 29

MOTOR_ORDER = [
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

POLICY_ACTION_ORDER = [
    ("left_hip_pitch_joint", 0.30),
    ("right_hip_pitch_joint", 0.30),
    ("left_knee_joint", 0.30),
    ("right_knee_joint", 0.30),
    ("left_ankle_pitch_joint", 0.30),
    ("right_ankle_pitch_joint", 0.30),
    ("left_hip_roll_joint", 0.20),
    ("right_hip_roll_joint", 0.20),
    ("left_ankle_roll_joint", 0.20),
    ("right_ankle_roll_joint", 0.20),
    ("left_hip_yaw_joint", 0.10),
    ("right_hip_yaw_joint", 0.10),
    ("waist_yaw_joint", 0.20),
    ("waist_roll_joint", 0.20),
    ("waist_pitch_joint", 0.20),
    ("left_shoulder_pitch_joint", 0.50),
    ("left_elbow_joint", 0.50),
    ("right_shoulder_pitch_joint", 0.50),
    ("right_elbow_joint", 0.50),
    ("left_shoulder_roll_joint", 0.30),
    ("left_shoulder_yaw_joint", 0.30),
    ("left_wrist_roll_joint", 0.30),
    ("left_wrist_pitch_joint", 0.30),
    ("left_wrist_yaw_joint", 0.30),
    ("right_shoulder_roll_joint", 0.30),
    ("right_shoulder_yaw_joint", 0.30),
    ("right_wrist_roll_joint", 0.30),
    ("right_wrist_pitch_joint", 0.30),
    ("right_wrist_yaw_joint", 0.30),
]

ISAAC_CATCH_ACTION_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# 1. 인공지능 학습 자세 (무릎 0.3)
TRAINED_JOINT_POS: Dict[str, float] = {
    "left_hip_pitch_joint": -0.15, "right_hip_pitch_joint": -0.15,
    "left_knee_joint": 0.30, "right_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15, "right_ankle_pitch_joint": -0.15,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.20, "right_shoulder_pitch_joint": 0.20,
    "left_elbow_joint": 0.55, "right_elbow_joint": 0.55,
    "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}
TRAINED_Q = np.array([TRAINED_JOINT_POS[name] for name in MOTOR_ORDER], dtype=np.float64)
CATCH_DEFAULT_Q = TRAINED_Q.copy()

# Legacy alternate squat pose kept only for reference/debug; the active catch path uses CATCH_DEFAULT_Q.
SQUAT_JOINT_POS = TRAINED_JOINT_POS.copy()
SQUAT_JOINT_POS.update({
    "left_hip_pitch_joint": -0.30, "right_hip_pitch_joint": -0.30,
    "left_knee_joint": 0.60, "right_knee_joint": 0.60,
    "left_ankle_pitch_joint": -0.30, "right_ankle_pitch_joint": -0.30,
})
SQUAT_Q = np.array([SQUAT_JOINT_POS[name] for name in MOTOR_ORDER], dtype=np.float64)

KP = np.array([
    120, 120, 120, 120, 120, 120,  
    120, 120, 120, 120, 120, 120,  
    120, 120, 120,                 
    85, 55, 55, 85, 55, 55, 55,    
    85, 55, 55, 85, 55, 55, 55,    
], dtype=np.float64)

KD = np.array([
    10, 10, 10, 10, 10, 10,        
    10, 10, 10, 10, 10, 10,        
    10, 10, 10,                    
    12, 10, 10, 12, 10, 10, 10,    
    12, 10, 10, 12, 10, 10, 10,    
], dtype=np.float64)

ACTION_CLIP = 1.0  
TORQUE_SCALE = 1.0 / 80.0
POLICY_OBS_DIM = 141
POLICY_ACT_DIM = 29

NAME_TO_MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_ORDER)}
POLICY_TO_MOTOR_INDEX = np.array([NAME_TO_MOTOR_INDEX[name] for name, _ in POLICY_ACTION_ORDER], dtype=np.int64)
ACTION_SCALES = np.array([scale for _, scale in POLICY_ACTION_ORDER], dtype=np.float64)


def clamp(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

def quat_apply(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q_wxyz, vq), quat_conj(q_wxyz))[1:4]

def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q_wxyz), v)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(q)
    if q.shape[0] != 4 or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def unitree_body_vec_to_policy_body(v: np.ndarray) -> np.ndarray:
    # Unitree/MuJoCo body frame -> IsaacLab policy body frame.
    return np.array([-v[2], v[1], v[0]], dtype=np.float64)

def get_any_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default

def load_yaml_maybe(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if yaml is None:
        raise RuntimeError("PyYAML is not installed, cannot read YAML config files")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class ObjectRelState:
    valid: bool
    rel_pos_b: np.ndarray
    rel_rot6d_b: np.ndarray
    rel_lin_vel_b: np.ndarray
    rel_ang_vel_b: np.ndarray

    @staticmethod
    def zeros() -> "ObjectRelState":
        return ObjectRelState(
            valid=False,
            rel_pos_b=np.zeros(3, dtype=np.float64),
            rel_rot6d_b=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            rel_lin_vel_b=np.zeros(3, dtype=np.float64),
            rel_ang_vel_b=np.zeros(3, dtype=np.float64),
        )

class ZmqImageReceiver:
    def __init__(self, server_address: str, port: int, image_show: bool = False) -> None:
        pass
    def start(self) -> None:
        pass
    def stop(self) -> None:
        pass
    def latest(self) -> tuple[Optional[np.ndarray], float]:
        return None, 0.0

class G1CatchRealController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.control_dt = float(args.control_dt)
        self.policy_dt = float(args.policy_dt)
        self.blend_duration = float(args.blend_duration)
        self.print_every = float(args.print_every)
        self.hold_default_only = bool(args.hold_default_only)
        self.hold_policy_until_toss = bool(args.hold_policy_until_toss)
        self.zero_vel_obs = bool(args.zero_vel_obs)
        self.force_upright_gravity = bool(args.force_upright_gravity)
        self.action_scale_mult = float(args.action_scale_mult)
        self.target_lowpass_alpha = float(np.clip(args.target_lowpass_alpha, 0.0, 1.0))
        self.use_camera = bool(args.use_camera)

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.mode_machine = 0
        self.mode_machine_ready = False
        self.crc = CRC()

        self.started = False
        self.start_q = None
        self.start_time = 0.0
        self.last_policy_time = -1e9

        self.policy = None
        self.policy_device = torch.device("cpu")
        
        self.prev_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64) 
        self.current_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64) 
        
        self.current_target_q = CATCH_DEFAULT_Q.copy()
        self.current_target_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.last_target_q = CATCH_DEFAULT_Q.copy()
        
        self._toss_signal = 0.0
        self._box_thrown = False
        self.last_debug_print_time = 0.0
        self._imu_quat_raw_is_xyzw: Optional[bool] = None
        self._mapping_debug_printed = False
        self.last_action_max_abs = 0.0
        self.last_target_delta_norm = 0.0
        
        self.virtual_box_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.virtual_box_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def init(self) -> None:
        if not self.hold_default_only:
            self._load_policy()
        self._init_motion_switcher()
        self._init_dds()

    def _load_policy(self) -> None:
        policy_path = Path(self.args.policy)
        if self.args.device == "auto":
            self.policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.policy_device = torch.device(self.args.device)
        self.policy = torch.jit.load(str(policy_path), map_location=self.policy_device)
        self.policy.eval()

    def _init_motion_switcher(self) -> None:
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(2.0)
        self.msc.Init()
        status, result = self.msc.CheckMode()
        if result is None: return
        while result and result.get("name"):
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1.0)

    def _init_dds(self) -> None:
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        if not self.mode_machine_ready:
            self.mode_machine = self.low_state.mode_machine
            self.mode_machine_ready = True

    def on_press(self, key):
        try:
            if key.char == 'k':
                if not self._box_thrown:
                    print("\n[REAL] >>> K pressed: Virtual Toss Signal ON! <<<")
                    self._box_thrown = True
                    self._toss_signal = 1.0
                    if self.hold_policy_until_toss:
                        self.last_policy_time = -1e9
                    self.virtual_box_pos = np.array([1.5, 0.0, 1.2], dtype=np.float64)
                    self.virtual_box_vel = np.array([-2.5, 0.0, 0.0], dtype=np.float64)
            elif key.char == 'r':
                print("\n[REAL] >>> R pressed: Resetting Posture <<<")
                q_now, _, _ = self._read_motor_state()
                self.start_q = q_now.copy()
                self.start_time = time.time()
                self.last_policy_time = self.start_time
                self.prev_policy_action[:] = 0.0
                self.current_policy_action[:] = 0.0
                self.current_target_q = CATCH_DEFAULT_Q.copy()
                self.last_target_q = CATCH_DEFAULT_Q.copy()
                self.current_target_dq[:] = 0.0
                self.last_action_max_abs = 0.0
                self.last_target_delta_norm = 0.0
                self._toss_signal = 0.0
                self._box_thrown = False
        except AttributeError:
            pass

    def start(self) -> None:
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        self.control_thread = RecurrentThread(interval=self.control_dt, target=self.control_step, name="g1_catch_real")
        while not self.mode_machine_ready:
            time.sleep(0.1)
        print("\n[REAL] Auto-Pilot Engaged! Robot will smoothly move to the trained catch pose in 2 seconds.")
        print("[REAL] Keys: 'k' (Toss Virtual Box), 'r' (Reset)")
        print(
            f"[REAL] control_dt={self.control_dt:.4f}s ({1.0 / self.control_dt:.1f} Hz), "
            f"policy_dt={self.policy_dt:.4f}s ({1.0 / self.policy_dt:.1f} Hz)"
        )
        if self.hold_default_only:
            print("[REAL] hold_default_only enabled: policy is bypassed after blend_duration.")
        if self.hold_policy_until_toss:
            print("[REAL] hold_policy_until_toss enabled: policy stays idle at CATCH_DEFAULT_Q until toss_signal becomes 1.")
        if self.zero_vel_obs:
            print("[REAL] zero_vel_obs enabled: base vel / ang vel / dq / tau observations are forced to zero.")
        if self.force_upright_gravity:
            print("[REAL] force_upright_gravity enabled: policy sees gravity=[0, 0, -1], ang_vel=[0, 0, 0].")
        if not self.hold_default_only:
            print(
                f"[REAL] action_scale_mult={self.action_scale_mult:.3f}, "
                f"target_lowpass_alpha={self.target_lowpass_alpha:.3f}"
            )
        self.control_thread.Start()

    def stop(self) -> None:
        if hasattr(self, 'listener'):
            self.listener.stop()

    def _detect_and_lock_imu_quat_order(self, q_raw: np.ndarray) -> None:
        if self._imu_quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            self._imu_quat_raw_is_xyzw = False
            return

        cand_wxyz = normalize_quat(q)
        cand_xyzw = normalize_quat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        g_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if abs(q[3]) > 0.90 and abs(q[0]) < 0.90:
            self._imu_quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90 and abs(q[3]) < 0.90:
            self._imu_quat_raw_is_xyzw = False
        else:
            err_wxyz = np.linalg.norm(quat_rotate_inverse(cand_wxyz, g_ref) - g_ref)
            err_xyzw = np.linalg.norm(quat_rotate_inverse(cand_xyzw, g_ref) - g_ref)
            self._imu_quat_raw_is_xyzw = err_xyzw < err_wxyz
        chosen = "xyzw" if self._imu_quat_raw_is_xyzw else "wxyz"
        print(
            f"[REAL] IMU quaternion order locked -> {chosen} | raw={np.round(q, 5)} | "
            f"g_if_wxyz={np.round(quat_rotate_inverse(cand_wxyz, g_ref), 5)} | "
            f"g_if_xyzw={np.round(quat_rotate_inverse(cand_xyzw, g_ref), 5)}"
        )

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        self._detect_and_lock_imu_quat_order(q)
        if self._imu_quat_raw_is_xyzw:
            return normalize_quat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        return normalize_quat(q)

    def _read_motor_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.low_state is not None
        q = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        dq = np.array([self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        tau = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        for i in range(G1_NUM_MOTOR):
            ms = self.low_state.motor_state[i]
            tau[i] = float(get_any_attr(ms, ["tau_est", "tau", "torque", "tauEst"], 0.0))
        tau = clamp(tau * TORQUE_SCALE, -1.0, 1.0)
        return q, dq, tau

    def _read_base_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.low_state is not None
        imu = self.low_state.imu_state
        q_raw = np.array(get_any_attr(imu, ["quaternion"], [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        quat_wxyz = self._to_wxyz(q_raw)
        gyro_unitree_b = np.array(get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]), dtype=np.float64)
        ang_vel_w = quat_apply(quat_wxyz, gyro_unitree_b)
        lin_vel_w = np.zeros(3, dtype=np.float64)

        gravity_unitree_b = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        gravity_policy_b = unitree_body_vec_to_policy_body(gravity_unitree_b)
        ang_vel_policy_b = unitree_body_vec_to_policy_body(gyro_unitree_b)
        if self.force_upright_gravity:
            gravity_policy_b = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            ang_vel_policy_b = np.zeros(3, dtype=np.float64)
        return (
            q_raw,
            quat_wxyz,
            lin_vel_w,
            ang_vel_w,
            gyro_unitree_b,
            gravity_unitree_b,
            gravity_policy_b,
            ang_vel_policy_b,
        )

    def _estimate_object_rel(self, quat_wxyz: np.ndarray, lin_vel_w: np.ndarray, ang_vel_w: np.ndarray) -> ObjectRelState:
        if self._toss_signal > 0.5:
            root_pos = np.array([0.0, 0.0, 0.8], dtype=np.float64)
            rel_pos_w = self.virtual_box_pos - root_pos
            rel_pos_b = quat_rotate_inverse(quat_wxyz, rel_pos_w)
            rel_lin_vel_b = quat_rotate_inverse(quat_wxyz, self.virtual_box_vel - lin_vel_w)
            
            return ObjectRelState(
                valid=True,
                rel_pos_b=rel_pos_b,
                rel_rot6d_b=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
                rel_lin_vel_b=rel_lin_vel_b,
                rel_ang_vel_b=np.zeros(3, dtype=np.float64)
            )
        return ObjectRelState.zeros()

    def _build_observation(self) -> np.ndarray:
        q, dq, tau = self._read_motor_state()
        (
            _,
            quat_wxyz,
            lin_vel_w,
            ang_vel_w,
            _,
            _,
            gravity_policy_b,
            ang_vel_policy_b,
        ) = self._read_base_state()

        lin_b = quat_rotate_inverse(quat_wxyz, lin_vel_w)
        ang_b = ang_vel_policy_b
        dq_obs = dq.copy()
        tau_obs = tau.copy()
        if self.zero_vel_obs:
            lin_b = np.zeros(3, dtype=np.float64)
            ang_b = np.zeros(3, dtype=np.float64)
            dq_obs = np.zeros_like(dq)
            tau_obs = np.zeros_like(tau)

        obj = self._estimate_object_rel(quat_wxyz, lin_vel_w, ang_vel_w)
        toss_signal = np.array([float(self._toss_signal)], dtype=np.float64)

        proprio = np.concatenate([gravity_policy_b, lin_b, ang_b, q, dq_obs, tau_obs], axis=0)
        obj_rel_arr = np.concatenate([obj.rel_pos_b, obj.rel_rot6d_b, obj.rel_lin_vel_b, obj.rel_ang_vel_b], axis=0)
        obj_rel_arr = obj_rel_arr * float(self._toss_signal)
        
        obs = np.concatenate([toss_signal, proprio, self.prev_policy_action, obj_rel_arr], axis=0)
        return obs.astype(np.float32)

    def _policy_action_to_motor_targets(self, action_policy_order: np.ndarray) -> np.ndarray:
        action = clamp(action_policy_order.astype(np.float64), -ACTION_CLIP, ACTION_CLIP)

        q_des_motor = CATCH_DEFAULT_Q.copy()
        q_des_motor[POLICY_TO_MOTOR_INDEX] = (
            CATCH_DEFAULT_Q[POLICY_TO_MOTOR_INDEX]
            + (ACTION_SCALES * self.action_scale_mult) * action
        )
        return q_des_motor

    def _print_action_mapping_once(self) -> None:
        if self._mapping_debug_printed:
            return
        self._mapping_debug_printed = True
        policy_action_names = [name for name, _ in POLICY_ACTION_ORDER]
        order_match = policy_action_names == ISAAC_CATCH_ACTION_JOINT_NAMES
        print("\n[REAL] Catch action order verification")
        print(f"[REAL] exact match with Isaac Sim deploy: {order_match}")
        for i, (name, scale) in enumerate(POLICY_ACTION_ORDER):
            motor_idx = int(POLICY_TO_MOTOR_INDEX[i])
            print(f"[REAL] act[{i:02d}] -> {name:>26s} | motor[{motor_idx:02d}] | scale={scale:.2f}")

    def _print_runtime_debug(
        self,
        obs: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        q_raw: np.ndarray,
        gyro_unitree_b: np.ndarray,
        gravity_unitree_b: np.ndarray,
        gravity_policy_b: np.ndarray,
        ang_vel_policy_b: np.ndarray,
        raw_action: np.ndarray,
    ) -> None:
        joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        ]
        q_obs_summary = ", ".join(
            f"{name}={obs[10 + i]:+.3f}" for i, name in enumerate(joint_names)
        )
        dq_summary = ", ".join(f"{v:+.3f}" for v in dq[:6])
        tau_obs_summary = ", ".join(f"{v:+.3f}" for v in obs[68:74])
        target_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]
        target_summary = ", ".join(
            f"{name}={self.current_target_q[NAME_TO_MOTOR_INDEX[name]]:+.3f}" for name in target_names
        )
        q_error_norm = float(np.linalg.norm(q - CATCH_DEFAULT_Q))
        print("\n--- [REAL] Catch Debug ---")
        print(f"[REAL] imu_quat_raw      : {np.round(q_raw, 5)}")
        print(f"[REAL] gyro_unitree_b   : {np.round(gyro_unitree_b, 5)}")
        print(f"[REAL] ang_vel_policy_b: {np.round(ang_vel_policy_b, 5)}")
        print(f"[REAL] gravity_unitree_b: {np.round(gravity_unitree_b, 5)}")
        print(f"[REAL] gravity_policy_b : {np.round(gravity_policy_b, 5)}")
        print(f"[REAL] force_upright   : {self.force_upright_gravity}")
        print(f"[REAL] zero_vel_obs    : {self.zero_vel_obs}")
        print(f"[REAL] q_obs first 6    : {q_obs_summary}")
        print(f"[REAL] max_abs_dq      : {np.max(np.abs(dq)):.3f}")
        print(f"[REAL] first six dq    : {dq_summary}")
        print(f"[REAL] max_abs_tau_obs : {np.max(np.abs(obs[68:97])):.3f}")
        print(f"[REAL] first six tau obs: {tau_obs_summary}")
        print(f"[REAL] action max abs  : {self.last_action_max_abs:.3f}")
        print(f"[REAL] target delta norm: {self.last_target_delta_norm:.3f}")
        print(f"[REAL] q error norm    : {q_error_norm:.3f}")
        print(f"[REAL] target hips/legs : {target_summary}")
        print(
            f"[REAL] action samples   : "
            f"hip_pitch(L/R)=({raw_action[0]:+.3f}, {raw_action[1]:+.3f}), "
            f"knee(L/R)=({raw_action[2]:+.3f}, {raw_action[3]:+.3f}), "
            f"ankle_pitch(L/R)=({raw_action[4]:+.3f}, {raw_action[5]:+.3f})"
        )
        print(
            f"[REAL] q_real knees     : "
            f"L={q[NAME_TO_MOTOR_INDEX['left_knee_joint']]:+.3f}, "
            f"R={q[NAME_TO_MOTOR_INDEX['right_knee_joint']]:+.3f}"
        )
        print("---------------------------")

    def _maybe_print_hold_default_debug(self) -> None:
        now = time.time()
        if self.print_every <= 0.0 or (now - self.last_debug_print_time) <= self.print_every:
            return
        self.last_debug_print_time = now
        q, dq, tau = self._read_motor_state()
        q_error_norm = float(np.linalg.norm(q - CATCH_DEFAULT_Q))
        dq_summary = ", ".join(f"{v:+.3f}" for v in dq[:6])
        tau_obs_summary = ", ".join(f"{v:+.3f}" for v in tau[:6])
        print("\n--- [REAL] Hold Default Debug ---")
        print(f"[REAL] q error norm    : {q_error_norm:.3f}")
        print(f"[REAL] max_abs_dq      : {np.max(np.abs(dq)):.3f}")
        print(f"[REAL] first six dq    : {dq_summary}")
        print(f"[REAL] max_abs_tau_obs : {np.max(np.abs(tau)):.3f}")
        print(f"[REAL] first six tau obs: {tau_obs_summary}")
        print("-------------------------------")

    def _hold_default_targets(self) -> tuple[np.ndarray, np.ndarray]:
        self.last_target_q = self.current_target_q.copy()
        self.current_target_q = CATCH_DEFAULT_Q.copy()
        self.current_target_dq[:] = 0.0
        self.current_policy_action[:] = 0.0
        self.prev_policy_action[:] = 0.0
        self.last_action_max_abs = 0.0
        self.last_target_delta_norm = float(np.linalg.norm(self.current_target_q - self.last_target_q))
        return self.current_target_q, self.current_target_dq

    def _run_policy_once(self) -> None:
        obs = self._build_observation()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.policy_device)
        with torch.no_grad():
            act_t = self.policy(obs_t)
        raw_action = clamp(act_t.squeeze(0).detach().cpu().numpy().astype(np.float64), -ACTION_CLIP, ACTION_CLIP)
        
        self.current_policy_action = raw_action
        self.prev_policy_action = self.current_policy_action.copy()

        self.last_target_q = self.current_target_q.copy()
        new_target_q = self._policy_action_to_motor_targets(self.current_policy_action)
        alpha = self.target_lowpass_alpha
        self.current_target_q = (1.0 - alpha) * self.current_target_q + alpha * new_target_q
        self.current_target_dq[:] = 0.0
        self.last_action_max_abs = float(np.max(np.abs(raw_action)))
        self.last_target_delta_norm = float(np.linalg.norm(self.current_target_q - self.last_target_q))

        self._print_action_mapping_once()

        now = time.time()
        if self.print_every > 0.0 and now - self.last_debug_print_time > self.print_every:
            self.last_debug_print_time = now
            q, dq, _ = self._read_motor_state()
            (
                q_raw,
                _,
                _,
                _,
                gyro_unitree_b,
                gravity_unitree_b,
                gravity_policy_b,
                ang_vel_policy_b,
            ) = self._read_base_state()
            self._print_runtime_debug(
                obs,
                q,
                dq,
                q_raw,
                gyro_unitree_b,
                gravity_unitree_b,
                gravity_policy_b,
                ang_vel_policy_b,
                raw_action,
            )


    def _set_motor_commands(self, q_des: np.ndarray, dq_des: np.ndarray) -> None:
        self.low_cmd.mode_pr = 0  
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(KP[i])
            self.low_cmd.motor_cmd[i].kd = float(KD[i])

    def control_step(self) -> None:
        if self.low_state is None:
            return

        now = time.time()
        if not self.started:
            q_now, _, _ = self._read_motor_state()
            self.start_q = q_now.copy()
            self.started = True
            self.start_time = now
            self.last_policy_time = now
            self.prev_policy_action[:] = 0.0
            self.current_policy_action[:] = 0.0
            self.current_target_q = CATCH_DEFAULT_Q.copy()
            self.last_target_q = CATCH_DEFAULT_Q.copy()
            self.current_target_dq[:] = 0.0
            self.last_action_max_abs = 0.0
            self.last_target_delta_norm = 0.0

        elapsed = now - self.start_time
        
        if self._toss_signal > 0.5:
            self.virtual_box_pos += self.virtual_box_vel * self.control_dt

        # First, blend into the trained catch default pose before policy control begins.
        if elapsed < self.blend_duration:
            ratio = np.clip(elapsed / max(self.blend_duration, 1e-6), 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * CATCH_DEFAULT_Q
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
            self.last_policy_time = now 
        
        # Match Isaac deploy more closely: update policy at 50 Hz and hold the target between updates.
        else:
            hold_default_now = self.hold_default_only or (
                self.hold_policy_until_toss and self._toss_signal <= 0.5
            )
            if hold_default_now:
                q_des, dq_des = self._hold_default_targets()
                self._maybe_print_hold_default_debug()
            else:
                if (now - self.last_policy_time) >= self.policy_dt:
                    self._run_policy_once()
                    self.last_policy_time = now

                q_des = self.current_target_q
                dq_des = self.current_target_dq

        self._set_motor_commands(q_des, dq_des)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, required=True)
    p.add_argument("--net-iface", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--control-dt", type=float, default=0.002)
    p.add_argument("--policy-dt", type=float, default=0.020)
    p.add_argument("--blend-duration", type=float, default=2.0)
    p.add_argument("--print-every", type=float, default=5.0)
    p.add_argument("--hold-default-only", action="store_true")
    p.add_argument("--hold-policy-until-toss", action="store_true")
    p.add_argument("--zero-vel-obs", action="store_true")
    p.add_argument("--force-upright-gravity", action="store_true")
    p.add_argument("--action-scale-mult", type=float, default=1.0)
    p.add_argument("--target-lowpass-alpha", type=float, default=1.0)
    p.add_argument("--use-camera", action="store_true")
    p.add_argument("--server-address", type=str, default="192.168.123.164")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--image-show", action="store_true")
    p.add_argument("--intrinsics-yaml", type=str, default=None)
    p.add_argument("--extrinsics-yaml", type=str, default=None)
    p.add_argument("--tag-yaml", type=str, default=None)
    return p

def main() -> None:
    print("WARNING: Keep the robot area clear before enabling low-level control.")
    input("Press Enter to continue...")
    
    args = build_argparser().parse_args()
    if args.net_iface:
        ChannelFactoryInitialize(0, args.net_iface)
    else:
        ChannelFactoryInitialize(0)
    controller = G1CatchRealController(args)
    controller.init()
    controller.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[MERGED] Ctrl+C detected! Relaxing robot joints...")
        for i in range(G1_NUM_MOTOR):
            controller.low_cmd.motor_cmd[i].kp = 0.0
            controller.low_cmd.motor_cmd[i].kd = 0.0
            controller.low_cmd.motor_cmd[i].tau = 0.0
            controller.low_cmd.motor_cmd[i].q = 0.0
            controller.low_cmd.motor_cmd[i].dq = 0.0
        controller.lowcmd_publisher.Write(controller.low_cmd)
        time.sleep(0.1)
        print("[MERGED] Robot is now relaxed. Safe to exit.")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()
