#!/usr/bin/env python3
"""
G1 standing-catch policy runner (first real-robot skeleton).

What this file does now:
- Uses Unitree low-level DDS loop (`rt/lowstate` / `rt/lowcmd`)
- Loads exported TorchScript policy
- Reconstructs the actor observation used by UROP_v12 catch policy
- Sends 29-DOF joint-position targets to G1
- Can run WITHOUT camera first (object_rel = 0, toss_signal = 0)
- Can optionally attach a simple ZMQ image stream + AprilTag estimator later

Recommended first milestone:
1) Run without camera and verify the robot smoothly blends to ready pose.
2) Verify policy inference runs and command publishing is stable.
3) Only then enable camera / AprilTag.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch is required to run this script") from e

try:
    import yaml
except Exception:
    yaml = None

# -----------------------------------------------------------------------------
# Unitree SDK imports
# -----------------------------------------------------------------------------
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

# -----------------------------------------------------------------------------
# Optional camera / AprilTag support
# -----------------------------------------------------------------------------
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


# =============================================================================
# Constants from UROP_v12 training environment
# =============================================================================
G1_NUM_MOTOR = 29

# SDK / motor order (also matches CONTROLLED_JOINT_NAMES observation order)
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

# Policy action concatenation order from env_cfg.ActionsCfg
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

# Ready/default posture from UROP_v12 scene init_state.joint_pos
DEFAULT_JOINT_POS: Dict[str, float] = {
    "left_hip_pitch_joint": -0.15,
    "right_hip_pitch_joint": -0.15,
    "left_knee_joint": 0.30,
    "right_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "right_ankle_pitch_joint": -0.15,
    "left_hip_roll_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.20,
    "right_shoulder_pitch_joint": 0.20,
    "left_elbow_joint": 0.55,
    "right_elbow_joint": 0.55,
    "left_shoulder_roll_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
DEFAULT_Q = np.array([DEFAULT_JOINT_POS[name] for name in MOTOR_ORDER], dtype=np.float64)

# Gains from UROP_v12 actuators
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

ACTION_CLIP = 1.0  # agent.yaml clip_actions
TORQUE_SCALE = 1.0 / 80.0
POLICY_OBS_DIM = 141
POLICY_ACT_DIM = 29

NAME_TO_MOTOR_INDEX = {name: i for i, name in enumerate(MOTOR_ORDER)}
POLICY_TO_MOTOR_INDEX = np.array([NAME_TO_MOTOR_INDEX[name] for name, _ in POLICY_ACTION_ORDER], dtype=np.int64)
ACTION_SCALES = np.array([scale for _, scale in POLICY_ACTION_ORDER], dtype=np.float64)


# =============================================================================
# Utility helpers
# =============================================================================

def clamp(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


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
    """Minimal self-contained receiver for the existing image_server.py PUB stream."""

    def __init__(self, server_address: str, port: int, image_show: bool = False) -> None:
        if zmq is None or cv2 is None:
            raise RuntimeError("opencv-python and pyzmq are required for camera mode")
        self.server_address = server_address
        self.port = port
        self.image_show = image_show
        self._ctx = None
        self._sock = None
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_t = 0.0

    def start(self) -> None:
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(f"tcp://{self.server_address}:{self.port}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[camera] Connected to tcp://{self.server_address}:{self.port}")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close(0)
        if self._ctx is not None:
            self._ctx.term()
        if self.image_show and cv2 is not None:
            cv2.destroyAllWindows()

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                msg = self._sock.recv(flags=0)
                np_img = np.frombuffer(msg, dtype=np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self._lock:
                    self._latest_frame = frame
                    self._latest_t = time.time()
                if self.image_show:
                    view = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                    cv2.imshow("g1_catch_real camera", view)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"[camera] receive error: {e}")
                time.sleep(0.05)

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_t


# =============================================================================
# Main controller
# =============================================================================
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
        self.current_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64)  # policy-order action
        self.current_target_q = DEFAULT_Q.copy()  # motor-order target
        self.current_target_dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        self.last_object_rel = ObjectRelState.zeros()

        self.image_rx: Optional[ZmqImageReceiver] = None
        self.tag_estimator = None

        self._printed_dims = False
        self._state_print_counter = 0
        self.action_lpf_alpha = 0.10
        self.smoothed_policy_action = np.zeros(POLICY_ACT_DIM, dtype=np.float64)

        # Safe debugging gains in POLICY_ACTION_ORDER
        # legs/waist almost frozen, arms only move a little
        self.action_gain_vec = np.array([
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00,   # legs_sagittal (6)
        0.00, 0.00, 0.00, 0.00,               # legs_frontal (4)
        0.00, 0.00,                           # legs_yaw (2)
        0.00, 0.00, 0.00,                     # waist (3)

        0.12, 0.12,                           # left_arm_capture (2)
        0.12, 0.12,                           # right_arm_capture (2)

        0.08, 0.08, 0.06, 0.06, 0.06,         # left_arm_wrap (5)
        0.08, 0.08, 0.06, 0.06, 0.06,         # right_arm_wrap (5)
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def init(self) -> None:
        self._load_policy()
        self._init_motion_switcher()
        self._init_dds()
        self._init_camera_if_needed()

    def _load_policy(self) -> None:
        policy_path = Path(self.args.policy)
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")

        if self.args.device == "auto":
            self.policy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.policy_device = torch.device(self.args.device)

        self.policy = torch.jit.load(str(policy_path), map_location=self.policy_device)
        self.policy.eval()
        print(f"[policy] Loaded: {policy_path}")
        print(f"[policy] Device: {self.policy_device}")
        print(f"[policy] Expected obs/action: {POLICY_OBS_DIM}/{POLICY_ACT_DIM}")

    def _init_motion_switcher(self) -> None:
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            print(f"[motion_switcher] Releasing mode: {result['name']}")
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1.0)

    def _init_dds(self) -> None:
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def _init_camera_if_needed(self) -> None:
        if not self.use_camera:
            print("[camera] disabled: using zero object_rel and toss_signal=0")
            return
        if not APRILTAG_AVAILABLE:
            raise RuntimeError(
                "Camera mode requested, but OpenCV/pyzmq/apriltag estimator import failed. "
                "Place apriltag_object_state_estimator.py next to this script and install dependencies."
            )

        intr = load_yaml_maybe(self.args.intrinsics_yaml)
        extr = load_yaml_maybe(self.args.extrinsics_yaml)
        tag_cfg = load_yaml_maybe(self.args.tag_yaml)
        if intr is None or extr is None or tag_cfg is None:
            raise RuntimeError(
                "Camera mode requires --intrinsics-yaml, --extrinsics-yaml, and --tag-yaml"
            )

        K = np.array(intr["camera_matrix"], dtype=np.float64)
        D = np.array(intr.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)

        t_bc = np.array(extr.get("translation_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        rpy_deg = np.array(extr.get("rpy_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
        rpy = np.deg2rad(rpy_deg)
        R_bc = self._rpy_to_rotmat_xyz(rpy[0], rpy[1], rpy[2])
        T_b_c = make_T(R_bc, t_bc)

        tag_center_in_box = np.array(tag_cfg.get("tag_center_in_box_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        tag_rpy_box = np.deg2rad(np.array(tag_cfg.get("tag_rpy_in_box_deg", [0.0, 0.0, 0.0]), dtype=np.float64))
        R_tag_box = self._rpy_to_rotmat_xyz(tag_rpy_box[0], tag_rpy_box[1], tag_rpy_box[2])
        # tag frame -> object(center) frame; translation is object center expressed in tag frame.
        T_tag_to_object = make_T(R_tag_box, tag_center_in_box)

        self.tag_estimator = AprilTagObjectStateEstimator(
            camera_matrix=K,
            dist_coeffs=D,
            tag_size_m=float(tag_cfg["tag_size_m"]),
            T_b_c=T_b_c,
            T_tag_to_object=T_tag_to_object,
            target_tag_id=tag_cfg.get("target_tag_id", None),
            tag_family=tag_cfg.get("tag_family", "36h11"),
        )
        self.image_rx = ZmqImageReceiver(
            server_address=self.args.server_address,
            port=self.args.port,
            image_show=self.args.image_show,
        )
        self.image_rx.start()

    # ------------------------------------------------------------------
    # Runtime callbacks
    # ------------------------------------------------------------------
    def low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        if not self.mode_machine_ready:
            self.mode_machine = self.low_state.mode_machine
            self.mode_machine_ready = True

        self._state_print_counter += 1
        if self._state_print_counter % int(max(1, self.args.print_every // max(self.control_dt, 1e-6))) == 0:
            self._state_print_counter = 0
            rpy = get_any_attr(self.low_state.imu_state, ["rpy"], [0.0, 0.0, 0.0])
            print(f"[state] imu rpy = {np.round(np.array(rpy, dtype=np.float64), 3)}")

    def start(self) -> None:
        self.control_thread = RecurrentThread(
            interval=self.control_dt,
            target=self.control_step,
            name="g1_catch_real",
        )
        while not self.mode_machine_ready:
            time.sleep(0.1)
        print("[dds] mode_machine ready, starting control thread")
        self.control_thread.Start()

    def stop(self) -> None:
        if self.image_rx is not None:
            self.image_rx.stop()

    # ------------------------------------------------------------------
    # Observation / policy
    # ------------------------------------------------------------------
    def _read_motor_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.low_state is not None
        q = np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        dq = np.array([self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)], dtype=np.float64)
        tau = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        for i in range(G1_NUM_MOTOR):
            ms = self.low_state.motor_state[i]
            tau_val = get_any_attr(ms, ["tau_est", "tau", "torque", "tauEst"], 0.0)
            tau[i] = float(tau_val)
        tau = clamp(tau * TORQUE_SCALE, -1.0, 1.0)
        return q, dq, tau

    def _read_base_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (quat_wxyz, lin_vel_w, ang_vel_w, gravity_b).

        For the first standing-catch version:
        - quat is built from IMU RPY
        - base linear velocity is set to zero (reasonable for standing tests)
        - IMU gyro is treated as body-frame angular velocity and rotated to world
        - gravity_b matches training exactly via quat_rotate_inverse(q, [0,0,-1])
        """
        assert self.low_state is not None
        imu = self.low_state.imu_state

        rpy = np.array(get_any_attr(imu, ["rpy"], [0.0, 0.0, 0.0]), dtype=np.float64)
        quat_wxyz = euler_xyz_to_quat_wxyz(rpy[0], rpy[1], rpy[2])

        gyro_b = np.array(get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]), dtype=np.float64)
        ang_vel_w = quat_apply(quat_wxyz, gyro_b)

        lin_vel_w = np.zeros(3, dtype=np.float64)
        gravity_b = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        return quat_wxyz, lin_vel_w, ang_vel_w, gravity_b

    def _estimate_object_rel(self, quat_wxyz: np.ndarray, lin_vel_w: np.ndarray, ang_vel_w: np.ndarray) -> ObjectRelState:
        if self.tag_estimator is None or self.image_rx is None or RobotBaseState is None:
            return ObjectRelState.zeros()

        frame, frame_t = self.image_rx.latest()
        if frame is None:
            return ObjectRelState.zeros()

        robot = RobotBaseState(
            pos_w=np.zeros(3, dtype=np.float64),
            quat_wxyz=quat_wxyz,
            lin_vel_w=lin_vel_w,
            ang_vel_w=ang_vel_w,
        )
        est = self.tag_estimator.update(frame_bgr=frame, robot=robot, timestamp_s=frame_t if frame_t > 0 else None)
        return ObjectRelState(
            valid=bool(est.valid),
            rel_pos_b=est.rel_pos_b.astype(np.float64),
            rel_rot6d_b=est.rel_rot6d_b.astype(np.float64),
            rel_lin_vel_b=est.rel_lin_vel_b.astype(np.float64),
            rel_ang_vel_b=est.rel_ang_vel_b.astype(np.float64),
        )

    def _compute_toss_signal(self, obj: ObjectRelState) -> float:
        if not obj.valid:
            return 0.0
        # First simple heuristic: tag visible, object in front, moving toward robot.
        x = float(obj.rel_pos_b[0])
        vx = float(obj.rel_lin_vel_b[0])
        z = float(obj.rel_pos_b[2])
        active = (0.20 <= x <= 1.20) and (-2.0 <= vx <= -0.05) and (-0.20 <= z <= 1.20)
        return 1.0 if active else 0.0

    def _build_observation(self) -> np.ndarray:
        q, dq, tau = self._read_motor_state()
        quat_wxyz, lin_vel_w, ang_vel_w, gravity_b = self._read_base_state()

        lin_b = quat_rotate_inverse(quat_wxyz, lin_vel_w)
        ang_b = quat_rotate_inverse(quat_wxyz, ang_vel_w)

        obj = self._estimate_object_rel(quat_wxyz, lin_vel_w, ang_vel_w)
        self.last_object_rel = obj
        toss_signal = np.array([self._compute_toss_signal(obj)], dtype=np.float64)

        proprio = np.concatenate([gravity_b, lin_b, ang_b, q, dq, tau], axis=0)
        obj_rel = np.concatenate([obj.rel_pos_b, obj.rel_rot6d_b, obj.rel_lin_vel_b, obj.rel_ang_vel_b], axis=0)
        obs = np.concatenate([toss_signal, proprio, self.current_policy_action, obj_rel], axis=0)

        if obs.shape[0] != POLICY_OBS_DIM:
            raise RuntimeError(f"Observation dim mismatch: got {obs.shape[0]}, expected {POLICY_OBS_DIM}")
        return obs.astype(np.float32)

    def _policy_action_to_motor_targets(self, action_policy_order: np.ndarray) -> np.ndarray:
        if action_policy_order.shape != (POLICY_ACT_DIM,):
            raise RuntimeError(f"Action dim mismatch: got {action_policy_order.shape}, expected {(POLICY_ACT_DIM,)}")
        action = clamp(action_policy_order.astype(np.float64), -ACTION_CLIP, ACTION_CLIP)
        q_des_motor = DEFAULT_Q.copy()
        # prev_action in observation stays in policy order, but robot target q uses motor order.
        q_des_motor[POLICY_TO_MOTOR_INDEX] = DEFAULT_Q[POLICY_TO_MOTOR_INDEX] + ACTION_SCALES * action
        return q_des_motor

    def _run_policy_once(self) -> None:
        obs = self._build_observation()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.policy_device)
        with torch.no_grad():
            act_t = self.policy(obs_t)
        act = act_t.squeeze(0).detach().cpu().numpy().astype(np.float64)

        if act.shape[0] != POLICY_ACT_DIM:
            raise RuntimeError(f"Policy output dim mismatch: got {act.shape[0]}, expected {POLICY_ACT_DIM}")

        if not self._printed_dims:
            self._printed_dims = True
            print(f"[policy] first obs/action shapes: {obs.shape} -> {act.shape}")

        raw_action = clamp(act, -ACTION_CLIP, ACTION_CLIP)

        # Low-pass filter on policy action
        self.smoothed_policy_action = (
        (1.0 - self.action_lpf_alpha) * self.smoothed_policy_action
        + self.action_lpf_alpha * raw_action
        )

        # Per-action safe scaling
        self.current_policy_action = self.action_gain_vec * self.smoothed_policy_action

        # Debug prints
        print(
        f"[policy] raw_max={np.max(np.abs(raw_action)):.4f} "
        f"scaled_max={np.max(np.abs(self.current_policy_action)):.4f}"
       )

        self.current_target_q = self._policy_action_to_motor_targets(self.current_policy_action)
        self.current_target_dq[:] = 0.0

    # ------------------------------------------------------------------
    # Command publishing
    # ------------------------------------------------------------------
    def _set_motor_commands(self, q_des: np.ndarray, dq_des: np.ndarray) -> None:
        self.low_cmd.mode_pr = 0  # PR mode
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
            print("[control] started; captured current joint state")

        assert self.start_q is not None
        elapsed = now - self.start_time

        # Phase 1: smoothly blend from current pose to UROP_v12 ready pose.
        if elapsed < self.blend_duration:
            ratio = np.clip(elapsed / max(self.blend_duration, 1e-6), 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * DEFAULT_Q
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        else:
            # Policy updates at lower rate; command publication continues at control_dt.
            if (now - self.last_policy_time) >= self.policy_dt:
                self.last_policy_time = now
                try:
                    self._run_policy_once()
                except Exception as e:
                    print(f"[policy] error: {e}")
                    print("[policy] falling back to ready/default posture")
                    self.current_policy_action[:] = 0.0
                    self.current_target_q[:] = DEFAULT_Q
                    self.current_target_dq[:] = 0.0
            q_des = self.current_target_q
            dq_des = self.current_target_dq

        self._set_motor_commands(q_des, dq_des)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    # ------------------------------------------------------------------
    # Math helpers used by config loading
    # ------------------------------------------------------------------
    @staticmethod
    def _rpy_to_rotmat_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        return Rz @ Ry @ Rx


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run UROP_v12 standing catch policy on real G1")
    p.add_argument("--policy", type=str, required=True, help="Path to exported policy.pt")
    p.add_argument("--net-iface", type=str, default=None, help="Network interface for ChannelFactoryInitialize, e.g. enp3s0 or wlan0")
    p.add_argument("--device", type=str, default="auto", help="torch device: auto / cpu / cuda")

    p.add_argument("--control-dt", type=float, default=0.002, help="Low-level command publish period [s]")
    p.add_argument("--policy-dt", type=float, default=0.020, help="Policy inference period [s] (UROP_v12 was 50 Hz)")
    p.add_argument("--blend-duration", type=float, default=2.0, help="Seconds to blend current pose -> ready pose")
    p.add_argument("--print-every", type=float, default=1.0, help="Print IMU RPY every N seconds")

    p.add_argument("--use-camera", action="store_true", help="Enable ZMQ camera + AprilTag object_rel observation")
    p.add_argument("--server-address", type=str, default="192.168.123.164", help="image_server host")
    p.add_argument("--port", type=int, default=5555, help="image_server port")
    p.add_argument("--image-show", action="store_true", help="Show incoming camera frames")
    p.add_argument("--intrinsics-yaml", type=str, default=None, help="Camera intrinsics YAML")
    p.add_argument("--extrinsics-yaml", type=str, default=None, help="Body->camera extrinsics YAML")
    p.add_argument("--tag-yaml", type=str, default=None, help="AprilTag/box config YAML")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    print("WARNING: Keep the robot area clear before enabling low-level control.")
    input("Press Enter to continue...")

    if args.net_iface:
        ChannelFactoryInitialize(0, args.net_iface)
        print(f"[dds] ChannelFactoryInitialize on interface: {args.net_iface}")
    else:
        ChannelFactoryInitialize(0)
        print("[dds] ChannelFactoryInitialize with default interface")

    controller = G1CatchRealController(args)
    controller.init()
    controller.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
