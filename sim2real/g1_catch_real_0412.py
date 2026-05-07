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

# [핵심 수정 1] 아이작심 정책 뇌 구조와 100% 동일하게 신경망 순서 재배열! (다리 좌/우 번갈아가며 나옴)
# [수정] 아이작심 env_cfg.py의 ActionsCfg 순서와 100% 동일하게 재배열!
POLICY_ACTION_ORDER = [
    # legs_sagittal (6)
    ("left_hip_pitch_joint", 0.30),
    ("left_knee_joint", 0.30),
    ("left_ankle_pitch_joint", 0.30),
    ("right_hip_pitch_joint", 0.30),
    ("right_knee_joint", 0.30),
    ("right_ankle_pitch_joint", 0.30),

    # legs_frontal (4)
    ("left_hip_roll_joint", 0.20),
    ("left_ankle_roll_joint", 0.20),
    ("right_hip_roll_joint", 0.20),
    ("right_ankle_roll_joint", 0.20),

    # legs_yaw (2)
    ("left_hip_yaw_joint", 0.10),
    ("right_hip_yaw_joint", 0.10),

    # waist (3)
    ("waist_yaw_joint", 0.20),
    ("waist_roll_joint", 0.20),
    ("waist_pitch_joint", 0.20),

    # left_arm_capture (2)
    ("left_shoulder_pitch_joint", 0.50),
    ("left_elbow_joint", 0.50),

    # right_arm_capture (2)
    ("right_shoulder_pitch_joint", 0.50),
    ("right_elbow_joint", 0.50),

    # left_arm_wrap (5)
    ("left_shoulder_roll_joint", 0.30),
    ("left_shoulder_yaw_joint", 0.30),
    ("left_wrist_roll_joint", 0.30),
    ("left_wrist_pitch_joint", 0.30),
    ("left_wrist_yaw_joint", 0.30),

    # right_arm_wrap (5)
    ("right_shoulder_roll_joint", 0.30),
    ("right_shoulder_yaw_joint", 0.30),
    ("right_wrist_roll_joint", 0.30),
    ("right_wrist_pitch_joint", 0.30),
    ("right_wrist_yaw_joint", 0.30),
]

DEFAULT_JOINT_POS: Dict[str, float] = {
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
DEFAULT_Q = np.array([DEFAULT_JOINT_POS[name] for name in MOTOR_ORDER], dtype=np.float64)

# env.yaml의 actuators 세팅과 완벽하게 일치시킴
KP = np.array([
    120, 120, 120, 120, 120, 120,  # 왼쪽 다리 6개
    120, 120, 120, 120, 120, 120,  # 오른쪽 다리 6개
    120, 120, 120,                 # 허리 3개
    85, 55, 55, 85, 55, 55, 55,    # 왼쪽 팔 7개
    85, 55, 55, 85, 55, 55, 55,    # 오른쪽 팔 7개
], dtype=np.float64)

KD = np.array([
    10, 10, 10, 10, 10, 10,        # 왼쪽 다리 6개
    10, 10, 10, 10, 10, 10,        # 오른쪽 다리 6개
    10, 10, 10,                    # 허리 3개
    12, 10, 10, 12, 10, 10, 10,    # 왼쪽 팔 7개
    12, 10, 10, 12, 10, 10, 10,    # 오른쪽 팔 7개
], dtype=np.float64)

ACTION_CLIP = 1.0  
TORQUE_SCALE = 1.0 / 80.0
POLICY_OBS_DIM = 141
POLICY_ACT_DIM = 29

NAME_TO_MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_ORDER)}
POLICY_TO_MOTOR_INDEX = np.array([NAME_TO_MOTOR_INDEX[name] for name, _ in POLICY_ACTION_ORDER], dtype=np.int64)
ACTION_SCALES = np.array([scale for _, scale in POLICY_ACTION_ORDER], dtype=np.float64)

class Mode(Enum):
    STAND = auto()
    PREPARE_CATCH = auto()
    CATCH_HOLD = auto()

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
        self.current_target_q = DEFAULT_Q.copy()
        self.current_target_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.last_target_q = DEFAULT_Q.copy()
        
        self.mode = Mode.STAND
        self._prepare_elapsed = 0.0
        self._prepare_duration = 0.22
        self._toss_signal = 0.0
        self._box_thrown = False
        self.last_debug_print_time = 0.0
        
        # [핵심 수정 2] 투명 상자 시뮬레이션용 데이터
        self.virtual_box_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.virtual_box_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def init(self) -> None:
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
            if key.char == 'j' and self.mode == Mode.STAND:
                print("\n[REAL] >>> J pressed: Entering PREPARE_CATCH mode <<<")
                self.mode = Mode.PREPARE_CATCH
                self._prepare_elapsed = 0.0
            elif key.char == 'k':
                if self.mode in (Mode.PREPARE_CATCH, Mode.CATCH_HOLD):
                    print("\n[REAL] >>> K pressed: Virtual Toss Signal ON! <<<")
                    self._box_thrown = True
                    self._toss_signal = 1.0
                    # 로봇 앞 1.5m 지점, 높이 1.2m에서 상자가 초속 2.5m로 날아오기 시작!
                    self.virtual_box_pos = np.array([1.5, 0.0, 1.2], dtype=np.float64)
                    self.virtual_box_vel = np.array([-2.5, 0.0, 0.0], dtype=np.float64)
            elif key.char == 'r':
                print("\n[REAL] >>> R pressed: Resetting to STAND mode <<<")
                self.mode = Mode.STAND
                q_now, _, _ = self._read_motor_state()
                self.start_q = q_now.copy()
                self.start_time = time.time()
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
        print("[REAL] Keys: 'j' (Prepare), 'k' (Toss Virtual Box), 'r' (Reset)")
        self.control_thread.Start()

    def stop(self) -> None:
        if hasattr(self, 'listener'):
            self.listener.stop()

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if abs(q[3]) > 0.90:
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        return q

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

    def _read_base_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.low_state is not None
        imu = self.low_state.imu_state
        q_raw = np.array(get_any_attr(imu, ["quaternion"], [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        quat_wxyz = self._to_wxyz(q_raw)
        gyro_b = np.array(get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]), dtype=np.float64)
        ang_vel_w = quat_apply(quat_wxyz, gyro_b)
        lin_vel_w = np.zeros(3, dtype=np.float64)
        
        w, x, y, z = quat_wxyz
        gravity_b = np.array([
            2 * (x * z - w * y) * -1.0,
            2 * (y * z + w * x) * -1.0,
            (1.0 - 2 * (x * x + y * y)) * -1.0
        ], dtype=np.float64)
        return quat_wxyz, lin_vel_w, ang_vel_w, gravity_b

    def _estimate_object_rel(self, quat_wxyz: np.ndarray, lin_vel_w: np.ndarray, ang_vel_w: np.ndarray) -> ObjectRelState:
        # [핵심 수정 3] 던지기 신호가 켜졌을 때, 투명 상자의 좌표를 로봇 관점으로 변환하여 뇌에 주입!
        if self._toss_signal > 0.5:
            # 서 있는 로봇의 중심(Root) 위치를 대략 [0, 0, 0.8]로 가정
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
        quat_wxyz, lin_vel_w, ang_vel_w, gravity_b = self._read_base_state()

        lin_b = quat_rotate_inverse(quat_wxyz, lin_vel_w)
        ang_b = quat_rotate_inverse(quat_wxyz, ang_vel_w)

        obj = self._estimate_object_rel(quat_wxyz, lin_vel_w, ang_vel_w)
        toss_signal = np.array([float(self._toss_signal)], dtype=np.float64)

        # [복구 완료] 가장 순수하고 완벽했던 관측치 배열!
        proprio = np.concatenate([gravity_b, lin_b, ang_b, q, dq, tau], axis=0)
        
        obj_rel_arr = np.concatenate([obj.rel_pos_b, obj.rel_rot6d_b, obj.rel_lin_vel_b, obj.rel_ang_vel_b], axis=0)
        obj_rel_arr = obj_rel_arr * float(self._toss_signal)
        
        obs = np.concatenate([toss_signal, proprio, self.prev_policy_action, obj_rel_arr], axis=0)
        return obs.astype(np.float32)

    def _policy_action_to_motor_targets(self, action_policy_order: np.ndarray) -> np.ndarray:
        action = clamp(action_policy_order.astype(np.float64), -ACTION_CLIP, ACTION_CLIP)
        
        q_des_motor = DEFAULT_Q.copy()
        q_des_motor[POLICY_TO_MOTOR_INDEX] = DEFAULT_Q[POLICY_TO_MOTOR_INDEX] + ACTION_SCALES * action
        return q_des_motor

    def _run_policy_once(self) -> None:
        obs = self._build_observation()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.policy_device)
        with torch.no_grad():
            act_t = self.policy(obs_t)
        raw_action = clamp(act_t.squeeze(0).detach().cpu().numpy().astype(np.float64), -ACTION_CLIP, ACTION_CLIP)
        
        self.current_policy_action = raw_action
        self.prev_policy_action = self.current_policy_action.copy()

        self.last_target_q = self.current_target_q.copy()
        self.current_target_q = self._policy_action_to_motor_targets(self.current_policy_action)
        self.current_target_dq[:] = 0.0

        now = time.time()
        if now - self.last_debug_print_time > 0.5:
            self.last_debug_print_time = now
            
            # [수정] q 값을 여기서 다시 읽어옵니다.
            q, _, _ = self._read_motor_state()
            
            # 무릎과 어깨 인덱스 (실제 모터 배열 기준)
            idx_lk = NAME_TO_MOTOR_INDEX["left_knee_joint"]
            idx_rk = NAME_TO_MOTOR_INDEX["right_knee_joint"]
            idx_l_sh = NAME_TO_MOTOR_INDEX["left_shoulder_pitch_joint"]
            idx_r_sh = NAME_TO_MOTOR_INDEX["right_shoulder_pitch_joint"]

            print(f"\n--- 🐛 DEEP DEBUG LOG ({self.mode.name}) ---")
            print(f"1. Gravity (Body) : {np.round(obs[1:4], 3)}")
            print(f"2. Base Lin Vel   : {np.round(obs[4:7], 3)}")
            print(f"3. Base Ang Vel   : {np.round(obs[7:10], 3)}")
            
            # 인공지능이 뱉은 '순수 Action 값 (-1~1)' (0=L_hip, 1=L_knee, 4=R_knee)
            print(f"4. Net Act (Knee) : L: {raw_action[1]:.3f} | R: {raw_action[4]:.3f}")
            print(f"5. Knee Target(q) : L: {self.current_target_q[idx_lk]:.3f} | R: {self.current_target_q[idx_rk]:.3f}")
            print(f"6. Knee Real(q)   : L: {q[idx_lk]:.3f} | R: {q[idx_rk]:.3f}")
            print("--------------------------------------------")


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

        elapsed = now - self.start_time
        
        # 가상 상자 날리기 시뮬레이션
        if self._toss_signal > 0.5:
            self.virtual_box_pos += self.virtual_box_vel * self.control_dt

        run_policy_this_step = False
        if elapsed >= self.blend_duration and self.mode != Mode.STAND:
            if (now - self.last_policy_time) >= self.policy_dt:
                run_policy_this_step = True
                self.last_policy_time = now

        if run_policy_this_step:
            self._run_policy_once()

        if elapsed < self.blend_duration:
            ratio = np.clip(elapsed / max(self.blend_duration, 1e-6), 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * DEFAULT_Q
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
            self.last_policy_time = now 
        else:
            if self.mode == Mode.STAND:
                q_des = DEFAULT_Q
                dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
                self.last_policy_time = now
            
            elif self.mode == Mode.PREPARE_CATCH:
                self._prepare_elapsed += self.control_dt
                ratio = np.clip((now - self.last_policy_time) / self.policy_dt, 0.0, 1.0)
                q_des = (1.0 - ratio) * self.last_target_q + ratio * self.current_target_q
                dq_des = self.current_target_dq

                if self._prepare_elapsed >= self._prepare_duration:
                    self.mode = Mode.CATCH_HOLD
                    print("\n[REAL] Entered CATCH_HOLD mode")
                    
            elif self.mode == Mode.CATCH_HOLD:
                ratio = np.clip((now - self.last_policy_time) / self.policy_dt, 0.0, 1.0)
                q_des = (1.0 - ratio) * self.last_target_q + ratio * self.current_target_q
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
        pass
    finally:
        controller.stop()

if __name__ == "__main__":
    main()