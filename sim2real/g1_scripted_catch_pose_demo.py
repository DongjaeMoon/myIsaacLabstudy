#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import threading
from dataclasses import dataclass

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


G1_NUM_MOTOR = 29
CONTROL_DT = 0.002


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
J = {name: i for i, name in enumerate(JOINT_NAMES)}


class Mode:
    PR = 0
    AB = 1


def smoothstep(u: float) -> tuple[float, float]:
    """Return s(u), ds/du for u in [0,1]."""
    u = float(np.clip(u, 0.0, 1.0))
    s = 3.0 * u * u - 2.0 * u * u * u
    ds_du = 6.0 * u - 6.0 * u * u
    return s, ds_du


def make_demo_gains(kp_scale: float = 1.0, kd_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Conservative gains for a pose demo.

    Lower body and waist hold the measured initial pose. Arms move slowly.
    """
    kp = np.array([
        35, 35, 25, 45, 35, 25,     # left leg
        35, 35, 25, 45, 35, 25,     # right leg
        25, 22, 22,                 # waist
        20, 18, 16, 20, 10, 10, 8,  # left arm
        20, 18, 16, 20, 10, 10, 8,  # right arm
    ], dtype=np.float64)
    kd = np.array([
        1.2, 1.2, 1.0, 1.5, 1.2, 1.0,
        1.2, 1.2, 1.0, 1.5, 1.2, 1.0,
        1.0, 0.9, 0.9,
        0.9, 0.8, 0.8, 0.9, 0.6, 0.6, 0.5,
        0.9, 0.8, 0.8, 0.9, 0.6, 0.6, 0.5,
    ], dtype=np.float64)
    return kp * float(kp_scale), kd * float(kd_scale)


def set_arm_pose(
    q: np.ndarray,
    *,
    l_sp: float,
    l_sr: float,
    l_sy: float,
    l_el: float,
    r_sp: float,
    r_sr: float,
    r_sy: float,
    r_el: float,
    wrist_pitch: float = 0.0,
) -> np.ndarray:
    out = q.copy()

    out[J["left_shoulder_pitch_joint"]] = l_sp
    out[J["left_shoulder_roll_joint"]] = l_sr
    out[J["left_shoulder_yaw_joint"]] = l_sy
    out[J["left_elbow_joint"]] = l_el
    out[J["left_wrist_roll_joint"]] = 0.0
    out[J["left_wrist_pitch_joint"]] = wrist_pitch
    out[J["left_wrist_yaw_joint"]] = 0.0

    out[J["right_shoulder_pitch_joint"]] = r_sp
    out[J["right_shoulder_roll_joint"]] = r_sr
    out[J["right_shoulder_yaw_joint"]] = r_sy
    out[J["right_elbow_joint"]] = r_el
    out[J["right_wrist_roll_joint"]] = 0.0
    out[J["right_wrist_pitch_joint"]] = wrist_pitch
    out[J["right_wrist_yaw_joint"]] = 0.0
    return out


def make_open_pose(base: np.ndarray) -> np.ndarray:
    # Open receiving posture. Lower body/waist are kept at measured initial q.
    return set_arm_pose(
        base,
        l_sp=0.45, l_sr=0.55, l_sy=0.00, l_el=0.10,
        r_sp=0.45, r_sr=-0.55, r_sy=0.00, r_el=0.10,
        wrist_pitch=0.0,
    )


def make_mid_pose(base: np.ndarray) -> np.ndarray:
    # Mild pre-contact cup posture.
    return set_arm_pose(
        base,
        l_sp=0.10, l_sr=0.48, l_sy=0.00, l_el=0.45,
        r_sp=0.10, r_sr=-0.48, r_sy=0.00, r_el=0.45,
        wrist_pitch=0.0,
    )


def make_hug_pose(base: np.ndarray) -> np.ndarray:
    # Conservative hug/hold posture based on the training HOLD_POSE direction.
    return set_arm_pose(
        base,
        l_sp=-0.25, l_sr=0.38, l_sy=0.00, l_el=0.85,
        r_sp=-0.25, r_sr=-0.38, r_sy=0.00, r_el=0.85,
        wrist_pitch=0.0,
    )


@dataclass
class Transition:
    start_q: np.ndarray
    target_q: np.ndarray
    duration: float
    start_time: float


class G1ScriptedCatchPoseDemo:
    def __init__(self, iface: str, kp_scale: float, kd_scale: float):
        self.iface = iface
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_ | None = None
        self.mode_machine = 0
        self.state_ready = threading.Event()
        self.crc = CRC()

        self.kp, self.kd = make_demo_gains(kp_scale=kp_scale, kd_scale=kd_scale)

        self.lock = threading.Lock()
        self.current_target: np.ndarray | None = None
        self.transition: Transition | None = None
        self.running = False
        self.damping_requested = False
        self.control_thread: RecurrentThread | None = None

    def release_mode(self) -> None:
        print("[demo] MotionSwitcher: release high-level mode if active")
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        for i in range(8):
            status, result = msc.CheckMode()
            print(f"[demo] CheckMode {i}: status={status}, result={result}")
            name = result.get("name") if isinstance(result, dict) else None
            if not name:
                print("[demo] high-level mode already released/idle")
                return
            print(f"[demo] active mode={name}; ReleaseMode()")
            ret = msc.ReleaseMode()
            print(f"[demo] ReleaseMode ret={ret}")
            time.sleep(0.8)
        print("[demo] WARNING: high-level mode may still be active")

    def init_dds(self) -> None:
        print(f"[demo] ChannelFactoryInitialize iface={self.iface}")
        ChannelFactoryInitialize(0, self.iface)
        self.release_mode()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        print("[demo] waiting for LowState...")
        if not self.state_ready.wait(timeout=5.0):
            raise RuntimeError("No LowState received. Check Ethernet interface and robot mode.")
        print("[demo] LowState received.")

    def low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        self.mode_machine = msg.mode_machine
        self.state_ready.set()

    def get_q(self) -> np.ndarray:
        if self.low_state is None:
            raise RuntimeError("low_state is None")
        return np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)

    def start(self) -> None:
        q0 = self.get_q()
        with self.lock:
            self.current_target = q0.copy()
            self.transition = None
            self.running = True
            self.damping_requested = False

        self.control_thread = RecurrentThread(
            interval=CONTROL_DT,
            target=self.write_low_cmd,
            name="g1_scripted_catch_pose_demo",
        )
        self.control_thread.Start()
        print("[demo] control thread started; holding current pose.")

    def stop(self) -> None:
        print("[demo] damping for 1.0s then stop")
        self.damping_requested = True
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            time.sleep(0.02)
        with self.lock:
            self.running = False
        if self.control_thread is not None:
            try:
                self.control_thread.Stop()
            except Exception:
                pass

    def set_target(self, q_target: np.ndarray, duration: float) -> None:
        q_now = self.get_q()
        with self.lock:
            self.transition = Transition(
                start_q=q_now.copy(),
                target_q=np.asarray(q_target, dtype=np.float64).copy(),
                duration=max(float(duration), 1e-3),
                start_time=time.monotonic(),
            )
            self.current_target = q_now.copy()
        print(f"[demo] new transition duration={duration:.2f}s")

    def _sample_target(self) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            trans = self.transition
            current = self.current_target

        if trans is None:
            if current is None:
                q = self.get_q()
            else:
                q = current.copy()
            return q, np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        t = time.monotonic() - trans.start_time
        u = np.clip(t / trans.duration, 0.0, 1.0)
        s, ds_du = smoothstep(float(u))
        q = (1.0 - s) * trans.start_q + s * trans.target_q
        dq = (ds_du / trans.duration) * (trans.target_q - trans.start_q)

        if u >= 1.0:
            with self.lock:
                self.current_target = trans.target_q.copy()
                self.transition = None
            q = trans.target_q.copy()
            dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        return q, dq

    def write_low_cmd(self) -> None:
        if not self.running or self.low_state is None:
            return

        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine

        if self.damping_requested:
            for i in range(G1_NUM_MOTOR):
                self.low_cmd.motor_cmd[i].mode = 1
                self.low_cmd.motor_cmd[i].tau = 0.0
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = 2.0
            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_publisher.Write(self.low_cmd)
            return

        q_des, dq_des = self._sample_target()

        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(self.kp[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kd[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Safe scripted arm-only catch pose demo for Unitree G1.")
    p.add_argument("--net-iface", default="enx00e04c0f3e58")
    p.add_argument("--kp-scale", type=float, default=1.0)
    p.add_argument("--kd-scale", type=float, default=1.0)
    p.add_argument("--open-duration", type=float, default=2.0)
    p.add_argument("--mid-duration", type=float, default=0.55)
    p.add_argument("--hug-duration", type=float, default=0.75)
    p.add_argument("--return-duration", type=float, default=1.2)
    return p


def wait_enter(msg: str) -> None:
    input("\n" + msg + "\nPress Enter...")


def main() -> int:
    args = build_argparser().parse_args()

    print("WARNING: scripted pose demo. Keep hands clear; be ready to Ctrl+C.")
    print("This demo keeps lower body/waist at measured initial q and only moves arms.")
    wait_enter("Place G1 in a stable standing/released state.")

    demo = G1ScriptedCatchPoseDemo(args.net_iface, args.kp_scale, args.kd_scale)
    demo.init_dds()
    demo.start()

    base = demo.get_q().copy()
    q_open = make_open_pose(base)
    q_mid = make_mid_pose(base)
    q_hug = make_hug_pose(base)

    try:
        wait_enter("Step 1: move arms to OPEN receiving pose.")
        demo.set_target(q_open, args.open_duration)
        time.sleep(args.open_duration + 0.3)

        while True:
            wait_enter("Step 2: close to MID cup pose. Start moving the box toward the robot now.")
            demo.set_target(q_mid, args.mid_duration)
            time.sleep(args.mid_duration + 0.1)

            wait_enter("Step 3: close to HUG/HOLD pose.")
            demo.set_target(q_hug, args.hug_duration)
            time.sleep(args.hug_duration + 0.5)

            ans = input("\nType 'r' then Enter to reopen, 'q' then Enter to damping/quit, or just Enter to hold: ").strip().lower()
            if ans == "q":
                break
            if ans == "r":
                demo.set_target(q_open, args.return_duration)
                time.sleep(args.return_duration + 0.2)
                continue
            print("[demo] holding hug pose. Ctrl+C to stop.")
    except KeyboardInterrupt:
        print("\n[demo] KeyboardInterrupt.")
    finally:
        demo.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
