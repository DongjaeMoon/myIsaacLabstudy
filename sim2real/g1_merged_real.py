#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# =============================================================================
# 1. 하드웨어 모터 순서 (SDK 기준)
# =============================================================================
G1_NUM_MOTOR = 29

MOTOR_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
NAME_TO_MOTOR_IDX = {name: i for i, name in enumerate(MOTOR_ORDER)}

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

# =============================================================================
# 2. 로코모션(LOCO) 뇌 신경망 매핑
# =============================================================================
LOCO_ACT_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
LOCO_OBS_NAMES = sorted(LOCO_ACT_NAMES)
LOCO_ACT_TO_MOTOR = np.array([NAME_TO_MOTOR_IDX[name] for name in LOCO_ACT_NAMES], dtype=np.int64)
LOCO_OBS_TO_MOTOR = np.array([NAME_TO_MOTOR_IDX[name] for name in LOCO_OBS_NAMES], dtype=np.int64)

# =============================================================================
# 3. 캐치(CATCH) 뇌 신경망 매핑
# =============================================================================
CATCH_OBS_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
CATCH_ACT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_elbow_joint", "right_shoulder_pitch_joint", "right_elbow_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# [수정] 억지로 마이너스 부호 넣었던 거 다 빼고 원상복구 (아이작심과 똑같이)
CATCH_ACT_SCALES = np.array([
    0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
    0.2, 0.2, 0.2, 0.2,
    0.1, 0.1,
    0.2, 0.2, 0.2,
    0.5, 0.5, 0.5, 0.5,
    0.3, 0.3, 0.3, 0.3, 0.3,
    0.3, 0.3, 0.3, 0.3, 0.3,
], dtype=np.float64)

CATCH_OBS_TO_MOTOR = np.array([NAME_TO_MOTOR_IDX[name] for name in CATCH_OBS_NAMES], dtype=np.int64)
CATCH_ACT_TO_MOTOR = np.array([NAME_TO_MOTOR_IDX[name] for name in CATCH_ACT_NAMES], dtype=np.int64)

# =============================================================================
# 게인(Gain) 및 상수
# =============================================================================
KP = np.array([120]*15 + [85, 55, 55, 85, 55, 55, 55] + [85, 55, 55, 85, 55, 55, 55], dtype=np.float64)
KD = np.array([10]*15 + [12, 10, 10, 12, 10, 10, 10] + [12, 10, 10, 12, 10, 10, 10], dtype=np.float64)

TORQUE_SCALE = 1.0 / 80.0
ACTION_CLIP = 1.0

class Mode(Enum):
    LOCO_STAND = auto()
    PREPARE_CATCH = auto()
    CATCH_HOLD = auto()

def clamp(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
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
        if hasattr(obj, name): return getattr(obj, name)
    return default

def quat_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        1 - 2*(y*y + z*z), 2*(x*y + z*w), 2*(x*z - y*w),
        2*(x*y - z*w), 1 - 2*(x*x + z*z), 2*(y*z + x*w),
    ], dtype=np.float64)

# =============================================================================
# 메인 컨트롤러
# =============================================================================
class G1MergedController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.control_dt = 0.002
        self.policy_dt = 0.020
        self.blend_duration = 2.0

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.mode_machine = 0
        self.mode_machine_ready = False
        self.crc = CRC()

        self.started = False
        self.start_q = None
        self.start_time = 0.0
        self.last_policy_time = -1e9

        self.device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.loco_policy = None
        self.catch_policy = None
        
        self.loco_prev_action = np.zeros(29, dtype=np.float64)
        self.catch_prev_action = np.zeros(29, dtype=np.float64)
        
        self.current_target_q = DEFAULT_Q.copy()
        self.current_target_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.last_target_q = DEFAULT_Q.copy()
        
        self.mode = Mode.LOCO_STAND
        self._prepare_elapsed = 0.0
        self._prepare_duration = 0.22
        
        self._toss_signal = 0.0
        self.virtual_box_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.virtual_box_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # [키보드 조종석 부활]
        self._pressed_keys = set()
        self.base_command = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def init(self) -> None:
        print(f"[MERGED] Loading LOCO policy: {self.args.policy_loco}")
        self.loco_policy = torch.jit.load(self.args.policy_loco, map_location=self.device)
        self.loco_policy.eval()

        print(f"[MERGED] Loading CATCH policy: {self.args.policy_catch}")
        self.catch_policy = torch.jit.load(self.args.policy_catch, map_location=self.device)
        self.catch_policy.eval()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(2.0)
        self.msc.Init()
        status, result = self.msc.CheckMode()
        if result is not None:
            while result and result.get("name"):
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1.0)

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        if not self.mode_machine_ready:
            self.mode_machine = self.low_state.mode_machine
            self.mode_machine_ready = True

    def _update_command(self):
        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        if 'up' in self._pressed_keys or '8' in self._pressed_keys: cmd[0] += 1.0
        if 'down' in self._pressed_keys or '2' in self._pressed_keys: cmd[0] -= 1.0
        if 'left' in self._pressed_keys or '4' in self._pressed_keys: cmd[1] += 0.5
        if 'right' in self._pressed_keys or '6' in self._pressed_keys: cmd[1] -= 0.5
        if 'q' in self._pressed_keys: cmd[2] += 1.0
        if 'e' in self._pressed_keys: cmd[2] -= 1.0
        self.base_command = cmd

    def on_press(self, key):
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else key.name
            self._pressed_keys.add(k)
            self._update_command()

            if k == 'j' and self.mode == Mode.LOCO_STAND:
                print("\n[MERGED] >>> J pressed: Switching Brain to PREPARE_CATCH <<<")
                self.mode = Mode.PREPARE_CATCH
                self._prepare_elapsed = 0.0
                self.catch_prev_action[:] = 0.0 
            elif k == 'k':
                if self.mode in (Mode.PREPARE_CATCH, Mode.CATCH_HOLD):
                    print("\n[MERGED] >>> K pressed: Toss Signal ON! <<<")
                    self._toss_signal = 1.0
                    self.virtual_box_pos = np.array([1.5, 0.0, 1.2], dtype=np.float64)
                    self.virtual_box_vel = np.array([-2.5, 0.0, 0.0], dtype=np.float64)
            elif k == 'r':
                print("\n[MERGED] >>> R pressed: Switching Brain back to LOCO_STAND <<<")
                self.mode = Mode.LOCO_STAND
                self._toss_signal = 0.0
                self.loco_prev_action[:] = 0.0
        except Exception:
            pass

    def on_release(self, key):
        try:
            k = key.char.lower() if hasattr(key, 'char') and key.char else key.name
            if k in self._pressed_keys:
                self._pressed_keys.remove(k)
            self._update_command()
        except Exception:
            pass

    def start(self) -> None:
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.control_thread = RecurrentThread(interval=self.control_dt, target=self.control_step, name="g1_merged")
        while not self.mode_machine_ready:
            time.sleep(0.1)
        print("\n[MERGED] 🚀 System Ready! Two-Brain Architecture Online.")
        print("[REAL] Keys:")
        print("       - Arrow Keys (or Numpad 8,2,4,6) to Walk")
        print("       - 'j' (Switch to Catch), 'k' (Virtual Toss), 'r' (Back to Loco)")
        self.control_thread.Start()

    def stop(self) -> None:
        if hasattr(self, 'listener'):
            self.listener.stop()

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if abs(q[3]) > 0.90: return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        return q

    def _read_state(self):
        assert self.low_state is not None
        q = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        dq = np.array([self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        tau = np.array([float(get_any_attr(self.low_state.motor_state[i], ["tau_est", "tau", "torque", "tauEst"], 0.0)) for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        tau = clamp(tau * TORQUE_SCALE, -1.0, 1.0)
        
        imu = self.low_state.imu_state
        quat_wxyz = self._to_wxyz(np.array(get_any_attr(imu, ["quaternion"], [1.0, 0.0, 0.0, 0.0]), dtype=np.float64))
        gyro_b = np.array(get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]), dtype=np.float64)
        ang_vel_w = quat_apply(quat_wxyz, gyro_b)
        lin_vel_w = np.zeros(3, dtype=np.float64) # 가상 속도
        
        w, x, y, z = quat_wxyz
        gravity_b = np.array([2*(x*z - w*y)*-1.0, 2*(y*z + w*x)*-1.0, (1.0 - 2*(x*x + y*y))*-1.0], dtype=np.float64)
        lin_b = quat_rotate_inverse(quat_wxyz, lin_vel_w)
        ang_b = quat_rotate_inverse(quat_wxyz, ang_vel_w)
        
        return q, dq, tau, quat_wxyz, lin_vel_w, ang_vel_w, gravity_b, lin_b, ang_b

    def _run_loco_policy(self, q, dq, lin_b, ang_b, gravity_b) -> None:
        obs = np.zeros(99, dtype=np.float64)
        obs[0:3] = lin_b
        obs[3:6] = ang_b
        obs[6:9] = gravity_b
        obs[9:12] = self.base_command
        
        q_obs = q[LOCO_OBS_TO_MOTOR] - DEFAULT_Q[LOCO_OBS_TO_MOTOR]
        dq_obs = dq[LOCO_OBS_TO_MOTOR]
        
        obs[12:41] = q_obs
        obs[41:70] = dq_obs
        obs[70:99] = self.loco_prev_action
        
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            act = self.loco_policy(obs_t).squeeze(0).cpu().numpy().astype(np.float64)
        act = clamp(act, -ACTION_CLIP, ACTION_CLIP)
        
        self.loco_prev_action = act.copy()
        
        q_des_motor = DEFAULT_Q.copy()
        q_des_motor[LOCO_ACT_TO_MOTOR] += act * 0.5
        
        self.last_target_q = self.current_target_q.copy()
        self.current_target_q = q_des_motor
        self.current_target_dq[:] = 0.0

    def _run_catch_policy(self, q, dq, tau, quat_wxyz, lin_vel_w, ang_vel_w, gravity_b, lin_b, ang_b) -> None:
        if self._toss_signal > 0.5:
            root_pos = np.array([0.0, 0.0, 0.8], dtype=np.float64)
            rel_pos_b = quat_rotate_inverse(quat_wxyz, self.virtual_box_pos - root_pos)
            rel_lin_vel_b = quat_rotate_inverse(quat_wxyz, self.virtual_box_vel - lin_vel_w)
            rel_rot6d_b = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
            rel_ang_vel_b = np.zeros(3, dtype=np.float64)
            obj_rel = np.concatenate([rel_pos_b, rel_rot6d_b, rel_lin_vel_b, rel_ang_vel_b], axis=0)
        else:
            obj_rel = np.zeros(15, dtype=np.float64)

        toss_sig = np.array([self._toss_signal], dtype=np.float64)
        proprio = np.concatenate([gravity_b, lin_b, ang_b, q[CATCH_OBS_TO_MOTOR], dq[CATCH_OBS_TO_MOTOR], tau[CATCH_OBS_TO_MOTOR]], axis=0)
        
        obs = np.concatenate([toss_sig, proprio, self.catch_prev_action, obj_rel], axis=0)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            act = self.catch_policy(obs_t).squeeze(0).cpu().numpy().astype(np.float64)
        act = clamp(act, -ACTION_CLIP, ACTION_CLIP)
        
        self.catch_prev_action = act.copy()
        
        q_des_motor = DEFAULT_Q.copy()
        q_des_motor[CATCH_ACT_TO_MOTOR] += act * CATCH_ACT_SCALES
        
        self.last_target_q = self.current_target_q.copy()
        self.current_target_q = q_des_motor
        self.current_target_dq[:] = 0.0

    def control_step(self) -> None:
        if self.low_state is None: return
        now = time.time()
        
        q, dq, tau, quat_wxyz, lin_vel_w, ang_vel_w, gravity_b, lin_b, ang_b = self._read_state()

        if not self.started:
            self.start_q = q.copy()
            self.started = True
            self.start_time = now
            self.last_policy_time = now

        elapsed = now - self.start_time
        
        if self._toss_signal > 0.5:
            self.virtual_box_pos += self.virtual_box_vel * self.control_dt

        run_policy_this_step = False
        if elapsed >= self.blend_duration:
            if (now - self.last_policy_time) >= self.policy_dt:
                run_policy_this_step = True
                self.last_policy_time = now

        if run_policy_this_step:
            if self.mode == Mode.LOCO_STAND:
                self._run_loco_policy(q, dq, lin_b, ang_b, gravity_b)
            elif self.mode in (Mode.PREPARE_CATCH, Mode.CATCH_HOLD):
                self._run_catch_policy(q, dq, tau, quat_wxyz, lin_vel_w, ang_vel_w, gravity_b, lin_b, ang_b)

        if elapsed < self.blend_duration:
            ratio = np.clip(elapsed / max(self.blend_duration, 1e-6), 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * DEFAULT_Q
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
            self.last_policy_time = now 
        else:
            ratio = np.clip((now - self.last_policy_time) / self.policy_dt, 0.0, 1.0)
            q_des = (1.0 - ratio) * self.last_target_q + ratio * self.current_target_q
            dq_des = self.current_target_dq

            if self.mode == Mode.PREPARE_CATCH:
                self._prepare_elapsed += self.control_dt
                if self._prepare_elapsed >= self._prepare_duration:
                    self.mode = Mode.CATCH_HOLD
                    print("\n[MERGED] Entered CATCH_HOLD mode")

        self.low_cmd.mode_pr = 0  
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(KP[i])
            self.low_cmd.motor_cmd[i].kd = float(KD[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy-loco", type=str, required=True)
    p.add_argument("--policy-catch", type=str, required=True)
    p.add_argument("--net-iface", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    print("WARNING: Keep the robot area clear before enabling low-level control.")
    input("Press Enter to continue...")
    
    if args.net_iface: ChannelFactoryInitialize(0, args.net_iface)
    else: ChannelFactoryInitialize(0)
    
    controller = G1MergedController(args)
    controller.init()
    controller.start()
    
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()

if __name__ == "__main__":
    main()