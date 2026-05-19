#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
import select
import termios
import tty
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

try:
    import torch
except Exception as e:
    raise RuntimeError("torch is required to run this script") from e

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = REPO_ROOT / "logs/rsl_rl/UROP_g1_loco_v5/2026-03-06_16-10-35/exported/policy.pt"

G1_NUM_MOTOR = 29
POLICY_OBS_DIM = 99
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

G1_29_JOINTS = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

LOCO_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.2, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.2, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}

LOCO_DEFAULT_Q = np.array([LOCO_DEFAULT_JOINT_POS[name] for name in MOTOR_ORDER], dtype=np.float64)

KP = np.array([
    200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
    200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
    120.0, 120.0, 120.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
], dtype=np.float64)

KD = np.array([
    5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
    5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
    5.0, 5.0, 5.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
    10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
], dtype=np.float64)

NAME_TO_MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_ORDER)}
ACTION_TO_MOTOR_INDEX = np.array([NAME_TO_MOTOR_INDEX[name] for name in G1_29_JOINTS], dtype=np.int64)

USD_DOF_NAMES_SORTED = sorted(G1_29_JOINTS)
OBS_TO_MOTOR_INDEX = np.array([NAME_TO_MOTOR_INDEX[name] for name in USD_DOF_NAMES_SORTED], dtype=np.int64)

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

def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(q)
    if q.shape[0] != 4 or norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm

def get_any_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default

class G1LocoRealController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.control_dt = float(args.control_dt)
        self.policy_dt = float(args.policy_dt)
        self.blend_duration = float(args.blend_duration)
        self.action_scale = float(args.action_scale)
        self.action_clip = float(args.action_clip)
        
        self.startup_q_error_threshold = float(args.startup_q_error_threshold)
        self.startup_dq_threshold = float(args.startup_dq_threshold)
        self.startup_gravity_xy_threshold = float(args.startup_gravity_xy_threshold)
        self.startup_stable_duration = float(args.startup_stable_duration)
        self.target_lowpass_alpha = float(args.target_lowpass_alpha)

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: Optional[LowState_] = None
        self.mode_machine = 0
        self.mode_machine_ready = False
        self.crc = CRC()

        self.started = False
        self.start_q: Optional[np.ndarray] = None
        self.start_time = 0.0
        self.last_policy_time = -1e9
        self.policy_started = False
        self.startup_stable_since: Optional[float] = None

        self.policy = None
        self.policy_device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

        # 조종 명령어: [x속도, y속도, yaw회전속도]
        self.command = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        self.prev_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64)
        self.current_target_q = LOCO_DEFAULT_Q.copy()
        self.current_target_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self._imu_quat_raw_is_xyzw: Optional[bool] = None

    def init(self) -> None:
        self._load_policy()
        self._init_motion_switcher()
        self._init_dds()

    def _load_policy(self) -> None:
        policy_path = Path(self.args.policy)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        self.policy = torch.jit.load(str(policy_path), map_location=self.policy_device)
        self.policy.eval()

    def _init_motion_switcher(self) -> None:
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(2.0)
        self.msc.Init()
        try:
            _, result = self.msc.CheckMode()
            while result and result.get("name"):
                self.msc.ReleaseMode()
                _, result = self.msc.CheckMode()
                time.sleep(0.5)
        except Exception:
            pass # MuJoCo 환경에서는 무시

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
        self.control_thread = RecurrentThread(interval=self.control_dt, target=self.control_step, name="g1_loco_real")
        while not self.mode_machine_ready:
            time.sleep(0.1)
        print("\n[LOCO] Auto-Pilot Engaged. Starting physics control loop...")
        self.control_thread.Start()

    def _detect_and_lock_imu_quat_order(self, q_raw: np.ndarray) -> None:
        if self._imu_quat_raw_is_xyzw is not None: return
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            self._imu_quat_raw_is_xyzw = False
            return
        cand_wxyz = normalize_quat(q)
        cand_xyzw = normalize_quat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        g_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if abs(q[3]) > 0.90 and abs(q[0]) < 0.90: self._imu_quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90 and abs(q[3]) < 0.90: self._imu_quat_raw_is_xyzw = False
        else:
            err_wxyz = np.linalg.norm(quat_rotate_inverse(cand_wxyz, g_ref) - g_ref)
            err_xyzw = np.linalg.norm(quat_rotate_inverse(cand_xyzw, g_ref) - g_ref)
            self._imu_quat_raw_is_xyzw = err_xyzw < err_wxyz

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        self._detect_and_lock_imu_quat_order(q)
        if self._imu_quat_raw_is_xyzw:
            return normalize_quat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        return normalize_quat(q)

    def _read_motor_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        dq = np.array([self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        return q, dq

    def _read_base_observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imu = self.low_state.imu_state
        q_raw = np.array(get_any_attr(imu, ["quaternion"], [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        quat_wxyz = self._to_wxyz(q_raw)
        ang_vel_b = np.array(get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]), dtype=np.float64)

        # 🚨 [중요 트릭] MuJoCo에서는 실제 선속도를 DDS로 주지 않으므로, 네트워크가 혼란스럽지 않게 명령 속도를 선속도로 속임(Cheat)
        lin_vel_b = self.command.copy()
        
        gravity_b = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        return lin_vel_b, ang_vel_b, gravity_b

    def _build_observation(self) -> np.ndarray:
        q, dq = self._read_motor_state()
        lin_vel_b, ang_vel_b, gravity_b = self._read_base_observation()

        q_rel = q[OBS_TO_MOTOR_INDEX] - LOCO_DEFAULT_Q[OBS_TO_MOTOR_INDEX]
        dq_obs = dq[OBS_TO_MOTOR_INDEX]

        obs = np.zeros(POLICY_OBS_DIM, dtype=np.float32)
        obs[0:3] = lin_vel_b.astype(np.float32)
        obs[3:6] = ang_vel_b.astype(np.float32)
        obs[6:9] = gravity_b.astype(np.float32)
        obs[9:12] = self.command.astype(np.float32)
        obs[12:41] = q_rel.astype(np.float32)
        obs[41:70] = dq_obs.astype(np.float32)
        obs[70:99] = self.prev_policy_action.astype(np.float32)
        return obs

    def _policy_action_to_motor_targets(self, raw_action: np.ndarray) -> np.ndarray:
        action = clamp(raw_action.astype(np.float64), -self.action_clip, self.action_clip)
        q_des = LOCO_DEFAULT_Q.copy()
        q_des[ACTION_TO_MOTOR_INDEX] = LOCO_DEFAULT_Q[ACTION_TO_MOTOR_INDEX] + self.action_scale * action
        return q_des

    def _run_policy_once(self) -> None:
        obs = self._build_observation()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.policy_device)

        with torch.no_grad():
            raw_action_t = self.policy(obs_t)

        raw_action = raw_action_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        clipped_action = clamp(raw_action, -self.action_clip, self.action_clip)
        self.prev_policy_action = clipped_action.copy()

        desired_q = self._policy_action_to_motor_targets(raw_action)
        prev_q = self.current_target_q.copy()

        alpha = float(np.clip(self.target_lowpass_alpha, 0.0, 1.0))
        self.current_target_q = (1.0 - alpha) * prev_q + alpha * desired_q
        self.current_target_dq[:] = 0.0

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
        if self.low_state is None: return
        now = time.time()

        if not self.started:
            q_now, _ = self._read_motor_state()
            self.started = True
            self.start_q = q_now.copy()
            self.start_time = now
            self.last_policy_time = now
            self.prev_policy_action[:] = 0.0
            self.current_target_q = LOCO_DEFAULT_Q.copy()
            self.current_target_dq[:] = 0.0
            self.policy_started = False

        q_now, dq_now = self._read_motor_state()
        q_error = float(np.linalg.norm(q_now - LOCO_DEFAULT_Q))
        elapsed = now - self.start_time

        if elapsed < self.blend_duration:
            ratio = np.clip(elapsed / max(self.blend_duration, 1e-6), 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * LOCO_DEFAULT_Q
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
            self.last_policy_time = now

        elif not self.policy_started:
            # 🚨 [수정사항] 시작 조건 대폭 완화! 떨어져도 억지로 서도록 강제 시작
            stable_now = (
                q_error <= self.startup_q_error_threshold
                and float(np.max(np.abs(dq_now))) <= self.startup_dq_threshold
            )
            if stable_now:
                if self.startup_stable_since is None: self.startup_stable_since = now
            else:
                self.startup_stable_since = None

            stable_for = 0.0 if self.startup_stable_since is None else (now - self.startup_stable_since)

            if stable_for < self.startup_stable_duration:
                q_des = LOCO_DEFAULT_Q.copy()
                dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
                self.current_target_q = q_des.copy()
                self.last_policy_time = now
            else:
                print(f"\n✅ [LOCO] Policy START! (q_error={q_error:.3f})")
                self.policy_started = True
                self.last_policy_time = now - self.policy_dt
                q_des = self.current_target_q
                dq_des = self.current_target_dq
        else:
            if (now - self.last_policy_time) >= self.policy_dt:
                self._run_policy_once()
                self.last_policy_time = now
            q_des = self.current_target_q
            dq_des = self.current_target_dq

        self._set_motor_commands(q_des, dq_des)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def stop(self) -> None:
        pass


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--net-iface", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--control-dt", type=float, default=0.002)
    parser.add_argument("--policy-dt", type=float, default=0.020)
    parser.add_argument("--blend-duration", type=float, default=2.0)
    
    parser.add_argument("--action-scale", type=float, default=0.50)
    parser.add_argument("--action-clip", type=float, default=1.0)
    
    # 🚨 [수정사항] 무조코 환경에서 무조건 시작되도록 Hold 기준치 대폭 상향
    parser.add_argument("--startup-q-error-threshold", type=float, default=1.5)
    parser.add_argument("--startup-dq-threshold", type=float, default=2.0)
    parser.add_argument("--startup-gravity-xy-threshold", type=float, default=0.5)
    parser.add_argument("--startup-stable-duration", type=float, default=0.5)
    
    parser.add_argument("--target-lowpass-alpha", type=float, default=0.20)
    return parser

def get_key():
    """리눅스 터미널에서 키 입력을 비동기로 받는 함수"""
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main() -> None:
    args = build_argparser().parse_args()

    if args.net_iface: ChannelFactoryInitialize(0, args.net_iface)
    else: ChannelFactoryInitialize(0)

    controller = G1LocoRealController(args)
    controller.init()
    controller.start()

    print("\n" + "="*50)
    print("🎮 [G1 로봇 조종기 켜짐]")
    print(" W : 전진 (+X)")
    #+#print(" S : 후진 (-X)")
    print(" A : 왼쪽 게걸음 (+Y)")
    print(" D : 오른쪽 게걸음 (-Y)")
    print(" Q : 제자리 왼쪽 회전 (+Yaw)")
    print(" E : 제자리 오른쪽 회전 (-Yaw)")
    print(" Space : 정지 (속도 0)")
    print(" Ctrl+C : 종료 및 안전모드")
    print("="*50 + "\n")

    try:
        while True:
            key = get_key()
            if key == '\x03': # Ctrl+C
                raise KeyboardInterrupt
            elif key == 'w':
                controller.command[0] = np.clip(controller.command[0] + 0.1, -1.0, 1.0)
            elif key == 's':
                controller.command[0] = np.clip(controller.command[0] - 0.1, -1.0, 1.0)
            elif key == 'a':
                controller.command[1] = np.clip(controller.command[1] + 0.1, -1.0, 1.0)
            elif key == 'd':
                controller.command[1] = np.clip(controller.command[1] - 0.1, -1.0, 1.0)
            elif key == 'q':
                controller.command[2] = np.clip(controller.command[2] + 0.2, -1.5, 1.5)
            elif key == 'e':
                controller.command[2] = np.clip(controller.command[2] - 0.2, -1.5, 1.5)
            elif key == ' ':
                controller.command[:] = 0.0
            
            if key in ['w','s','a','d','q','e',' ']:
                sys.stdout.write(f"\r[현재 명령 속도] X: {controller.command[0]:.1f} | Y: {controller.command[1]:.1f} | Yaw: {controller.command[2]:.1f}    ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[LOCO] Ctrl+C 감지! 로봇의 힘을 뺍니다...")
        for i in range(G1_NUM_MOTOR):
            controller.low_cmd.motor_cmd[i].kp = 0.0
            controller.low_cmd.motor_cmd[i].kd = 0.0
            controller.low_cmd.motor_cmd[i].tau = 0.0
            controller.low_cmd.motor_cmd[i].q = 0.0
            controller.low_cmd.motor_cmd[i].dq = 0.0
        controller.lowcmd_publisher.Write(controller.low_cmd)
        time.sleep(0.1)
        print("[LOCO] 로봇이 안전하게 정지되었습니다.")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()