#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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


# ============================================================
# Core constants
# ============================================================

G1_NUM_MOTOR = 29
ACTION_CLIP = 1.0
TORQUE_SCALE = 1.0 / 80.0
POLICY_OBS_DIM = 141
POLICY_ACT_DIM = 29

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
    ("left_knee_joint", 0.30),
    ("left_ankle_pitch_joint", 0.30),
    ("right_hip_pitch_joint", 0.30),
    ("right_knee_joint", 0.30),
    ("right_ankle_pitch_joint", 0.30),

    ("left_hip_roll_joint", 0.20),
    ("left_ankle_roll_joint", 0.20),
    ("right_hip_roll_joint", 0.20),
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

NAME_TO_MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_ORDER)}
POLICY_TO_MOTOR_INDEX = np.array(
    [NAME_TO_MOTOR_INDEX[name] for name, _ in POLICY_ACTION_ORDER],
    dtype=np.int64,
)
ACTION_SCALES = np.array([scale for _, scale in POLICY_ACTION_ORDER], dtype=np.float64)

# Safer gain preset close to sdk_test.py
SDK_SAFE_KP = np.array([
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
], dtype=np.float64)

SDK_SAFE_KD = np.array([
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
], dtype=np.float64)

# More train-like gains from the old catch script
TRAIN_LIKE_KP = np.array([
    120, 120, 120, 120, 120, 120,
    120, 120, 120, 120, 120, 120,
    120, 120, 120,
    85, 55, 55, 85, 55, 55, 55,
    85, 55, 55, 85, 55, 55, 55,
], dtype=np.float64)

TRAIN_LIKE_KD = np.array([
    10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10,
    10, 10, 10,
    12, 10, 10, 12, 10, 10, 10,
    12, 10, 10, 12, 10, 10, 10,
], dtype=np.float64)


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
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
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


class G1CatchTickController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.control_dt = float(args.control_dt)
        self.policy_dt = float(args.policy_dt)
        self.blend_duration = float(args.blend_duration)
        self.prepare_duration = float(args.prepare_duration)

        self.policy_tick_interval = max(1, int(round(self.policy_dt / self.control_dt)))
        self.blend_ticks = max(1, int(round(self.blend_duration / self.control_dt)))
        self.prepare_ticks = max(1, int(round(self.prepare_duration / self.control_dt)))
        self.debug_policy_every = max(1, int(round(self.args.print_every / self.policy_dt)))

        if args.gain_mode == "sdk_safe":
            base_kp = SDK_SAFE_KP
            base_kd = SDK_SAFE_KD
        else:
            base_kp = TRAIN_LIKE_KP
            base_kd = TRAIN_LIKE_KD

        self.kp = base_kp * float(args.gain_scale)
        self.kd = base_kd * float(args.gain_scale)

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.mode_machine = 0
        self.mode_machine_ready = False
        self.crc = CRC()

        self.policy = None
        self.policy_device = torch.device("cpu")

        self.started = False
        self.start_q = DEFAULT_Q.copy()

        self.tick_count = 0
        self.mode_tick_count = 0
        self.policy_update_count = 0

        self.mode = Mode.STAND

        self.prev_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64)
        self.current_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64)
        self.current_target_q = DEFAULT_Q.copy()
        self.current_target_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        self._toss_signal = 0.0
        self._box_thrown = False
        self.virtual_box_pos = np.zeros(3, dtype=np.float64)
        self.virtual_box_vel = np.zeros(3, dtype=np.float64)

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
        if result is None:
            return
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

    def start(self) -> None:
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.control_thread = RecurrentThread(
            interval=self.control_dt,
            target=self.control_step,
            name="g1_catch_real_tick",
        )

        while not self.mode_machine_ready:
            time.sleep(0.1)

        print(f"[REAL-TICK] control_dt={self.control_dt:.4f}s ({1.0/self.control_dt:.1f} Hz)")
        print(f"[REAL-TICK] policy_dt={self.policy_dt:.4f}s ({1.0/self.policy_dt:.1f} Hz)")
        print(f"[REAL-TICK] policy tick interval={self.policy_tick_interval}")
        print(f"[REAL-TICK] gain_mode={self.args.gain_mode}, gain_scale={self.args.gain_scale}")
        print("[REAL-TICK] Keys: 'j' (Prepare), 'k' (Virtual Toss), 'r' (Reset)")
        self.control_thread.Start()

    def stop(self) -> None:
        if hasattr(self, "listener"):
            self.listener.stop()

    def on_press(self, key):
        try:
            if key.char == "j" and self.mode == Mode.STAND:
                print("\n[REAL-TICK] >>> J pressed: Enter PREPARE_CATCH <<<")
                self.mode = Mode.PREPARE_CATCH
                self.mode_tick_count = 0
                self.policy_update_count = 0
                self.prev_policy_action[:] = 0.0
                self.current_policy_action[:] = 0.0
                self.current_target_q = DEFAULT_Q.copy()
                self.current_target_dq[:] = 0.0

            elif key.char == "k" and self.mode in (Mode.PREPARE_CATCH, Mode.CATCH_HOLD):
                print("\n[REAL-TICK] >>> K pressed: Virtual Toss ON <<<")
                self._box_thrown = True
                self._toss_signal = 1.0
                self.virtual_box_pos = np.array([1.5, 0.0, 1.2], dtype=np.float64)
                self.virtual_box_vel = np.array([-2.5, 0.0, 0.0], dtype=np.float64)

            elif key.char == "r":
                print("\n[REAL-TICK] >>> R pressed: Reset to STAND <<<")
                q_now, _, _ = self._read_motor_state()
                self._reset_to_stand(q_now)

        except AttributeError:
            pass

    def _reset_to_stand(self, q_now: np.ndarray) -> None:
        self.mode = Mode.STAND
        self.start_q = q_now.copy()
        self.tick_count = 0
        self.mode_tick_count = 0
        self.policy_update_count = 0
        self.prev_policy_action[:] = 0.0
        self.current_policy_action[:] = 0.0
        self.current_target_q = DEFAULT_Q.copy()
        self.current_target_dq[:] = 0.0
        self._toss_signal = 0.0
        self._box_thrown = False
        self.virtual_box_pos[:] = 0.0
        self.virtual_box_vel[:] = 0.0

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
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
        imu = get_any_attr(self.low_state, ["imu_state", "imu"], None)

        q_raw = np.array(
            get_any_attr(imu, ["quaternion", "quat"], [1.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        quat_wxyz = self._to_wxyz(q_raw)

        gyro_b = np.array(
            get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )

        # 현재 MuJoCo/debug 단계에서는 base linear velocity는 0 고정
        lin_vel_b = np.zeros(3, dtype=np.float64)
        ang_vel_b = gyro_b.copy()

        w, x, y, z = quat_wxyz
        gravity_b = np.array([
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ], dtype=np.float64)

        return quat_wxyz, lin_vel_b, ang_vel_b, gravity_b

    def _estimate_object_rel(self, quat_wxyz: np.ndarray) -> ObjectRelState:
        if self._toss_signal <= 0.5:
            return ObjectRelState.zeros()

        root_pos = np.array([0.0, 0.0, 0.8], dtype=np.float64)
        rel_pos_w = self.virtual_box_pos - root_pos
        rel_pos_b = quat_rotate_inverse(quat_wxyz, rel_pos_w)
        rel_lin_vel_b = quat_rotate_inverse(quat_wxyz, self.virtual_box_vel)

        return ObjectRelState(
            valid=True,
            rel_pos_b=rel_pos_b,
            rel_rot6d_b=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            rel_lin_vel_b=rel_lin_vel_b,
            rel_ang_vel_b=np.zeros(3, dtype=np.float64),
        )

    def _build_observation(self) -> np.ndarray:
        q, dq, tau = self._read_motor_state()
        quat_wxyz, lin_vel_b, ang_vel_b, gravity_b = self._read_base_state()
        obj = self._estimate_object_rel(quat_wxyz)

        toss_signal = np.array([float(self._toss_signal)], dtype=np.float64)
        proprio = np.concatenate([gravity_b, lin_vel_b, ang_vel_b, q, dq, tau], axis=0)
        obj_rel_arr = np.concatenate(
            [obj.rel_pos_b, obj.rel_rot6d_b, obj.rel_lin_vel_b, obj.rel_ang_vel_b],
            axis=0,
        )
        obj_rel_arr *= float(self._toss_signal)

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

        raw_action = clamp(
            act_t.squeeze(0).detach().cpu().numpy().astype(np.float64),
            -ACTION_CLIP,
            ACTION_CLIP,
        )

        self.current_policy_action = raw_action.copy()
        self.prev_policy_action = self.current_policy_action.copy()
        self.current_target_q = self._policy_action_to_motor_targets(self.current_policy_action)
        self.current_target_dq[:] = 0.0
        self.policy_update_count += 1

        if self.policy_update_count % self.debug_policy_every == 0:
            q, _, _ = self._read_motor_state()
            idx_lk = NAME_TO_MOTOR_INDEX["left_knee_joint"]
            idx_rk = NAME_TO_MOTOR_INDEX["right_knee_joint"]
            print(f"\n--- 🐛 TICK DEBUG ({self.mode.name}) ---")
            print(f"policy_update_count : {self.policy_update_count}")
            print(f"tick_count          : {self.tick_count}")
            print(f"Net Act (Knee)      : L {raw_action[1]:+.3f} | R {raw_action[4]:+.3f}")
            print(f"Knee Target(q)      : L {self.current_target_q[idx_lk]:.3f} | R {self.current_target_q[idx_rk]:.3f}")
            print(f"Knee Real(q)        : L {q[idx_lk]:.3f} | R {q[idx_rk]:.3f}")
            print("------------------------------")

    def _set_motor_commands(self, q_des: np.ndarray, dq_des: np.ndarray) -> None:
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.mode_machine

        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kp[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kd[i])

    def control_step(self) -> None:
        if self.low_state is None:
            return

        if not self.started:
            q_now, _, _ = self._read_motor_state()
            self.start_q = q_now.copy()
            self.started = True
            self.tick_count = 0
            print("[REAL-TICK] Controller started. Captured initial joint state.")

        self.tick_count += 1

        if self._toss_signal > 0.5:
            self.virtual_box_pos += self.virtual_box_vel * self.control_dt

        # Stage 1: initial/reset blend current q -> DEFAULT_Q
        if self.tick_count <= self.blend_ticks:
            ratio = np.clip(self.tick_count / max(self.blend_ticks, 1), 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * DEFAULT_Q
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        else:
            if self.mode == Mode.STAND:
                q_des = DEFAULT_Q.copy()
                dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

            else:
                self.mode_tick_count += 1

                # 50 Hz policy update on top of 500 Hz low-level loop
                if (self.mode_tick_count - 1) % self.policy_tick_interval == 0:
                    self._run_policy_once()

                # Isaac Sim-like semantics: hold current target between policy updates
                q_des = self.current_target_q.copy()
                dq_des = self.current_target_dq.copy()

                if self.mode == Mode.PREPARE_CATCH and self.mode_tick_count >= self.prepare_ticks:
                    self.mode = Mode.CATCH_HOLD
                    self.mode_tick_count = 0
                    print("\n[REAL-TICK] Entered CATCH_HOLD mode")

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
    p.add_argument("--blend-duration", type=float, default=1.0)
    p.add_argument("--prepare-duration", type=float, default=0.22)
    p.add_argument("--print-every", type=float, default=0.5)
    p.add_argument("--gain-mode", type=str, choices=["sdk_safe", "train_like"], default="sdk_safe")
    p.add_argument("--gain-scale", type=float, default=1.0)
    return p


def main() -> None:
    print("WARNING: Keep the robot area clear before enabling low-level control.")
    input("Press Enter to continue...")

    args = build_argparser().parse_args()

    if args.net_iface:
        ChannelFactoryInitialize(0, args.net_iface)
    else:
        ChannelFactoryInitialize(0)

    controller = G1CatchTickController(args)
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
