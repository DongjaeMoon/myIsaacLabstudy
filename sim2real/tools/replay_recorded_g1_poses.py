#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import threading
from pathlib import Path

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


N = 29
DT = 0.002
ARM_IDX = list(range(15, 29))


class Mode:
    PR = 0


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 3*u*u - 2*u*u*u, 6*u - 6*u*u


def gains(kp_scale=1.0, kd_scale=1.0):
    # Hold legs/waist at current q. Move arms gently.
    kp = np.array([
        25, 25, 18, 35, 25, 18,
        25, 25, 18, 35, 25, 18,
        18, 15, 15,
        18, 16, 14, 18, 8, 8, 6,
        18, 16, 14, 18, 8, 8, 6,
    ], dtype=float)
    kd = np.array([
        1.0, 1.0, 0.8, 1.2, 1.0, 0.8,
        1.0, 1.0, 0.8, 1.2, 1.0, 0.8,
        0.8, 0.7, 0.7,
        0.8, 0.7, 0.7, 0.8, 0.45, 0.45, 0.35,
        0.8, 0.7, 0.7, 0.8, 0.45, 0.45, 0.35,
    ], dtype=float)
    return kp * kp_scale, kd * kd_scale


class Demo:
    def __init__(self, iface, poses, kp_scale, kd_scale, arms_only=True):
        self.iface = iface
        self.poses = poses
        self.arms_only = arms_only
        self.low_state = None
        self.ready = threading.Event()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.kp, self.kd = gains(kp_scale, kd_scale)
        self.mode_machine = 0
        self.lock = threading.Lock()
        self.q_target = None
        self.trans = None
        self.running = False
        self.damping = False

    def cb(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine = msg.mode_machine
        self.ready.set()

    def q(self):
        return np.array([self.low_state.motor_state[i].q for i in range(N)], dtype=float)

    def release(self):
        print("[replay] release high-level mode")
        m = MotionSwitcherClient()
        m.SetTimeout(5.0)
        m.Init()
        for _ in range(6):
            status, result = m.CheckMode()
            name = result.get("name") if isinstance(result, dict) else None
            print(f"[replay] CheckMode status={status} result={result}")
            if not name:
                return
            m.ReleaseMode()
            time.sleep(0.7)

    def init(self):
        ChannelFactoryInitialize(0, self.iface)
        self.release()
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.cb, 10)
        if not self.ready.wait(5):
            raise RuntimeError("No LowState")
        base = self.q()
        self.q_target = base.copy()
        self.base = base.copy()
        print("[replay] ready. Holding current lower body/waist.")

    def start(self):
        self.running = True
        self.th = RecurrentThread(interval=DT, target=self.write, name="recorded_pose_replay")
        self.th.Start()

    def stop(self):
        self.damping = True
        time.sleep(1.0)
        self.running = False
        try:
            self.th.Stop()
        except Exception:
            pass

    def target_from_pose(self, name):
        target = self.base.copy() if self.arms_only else np.asarray(self.poses[name], dtype=float).copy()
        rec = np.asarray(self.poses[name], dtype=float)
        if self.arms_only:
            target[ARM_IDX] = rec[ARM_IDX]
        return target

    def goto(self, name, duration):
        q0 = self.q()
        q1 = self.target_from_pose(name)
        with self.lock:
            self.trans = (q0, q1, time.monotonic(), max(float(duration), 1e-3))
        print(f"[replay] goto {name}, duration={duration:.2f}s")

    def sample(self):
        with self.lock:
            trans = self.trans
        if trans is None:
            return self.q_target.copy(), np.zeros(N)
        q0, q1, t0, dur = trans
        u = (time.monotonic() - t0) / dur
        s, dsdu = smoothstep(u)
        q = (1-s)*q0 + s*q1
        dq = (dsdu/dur)*(q1-q0)
        if u >= 1:
            q = q1.copy()
            dq = np.zeros(N)
            with self.lock:
                self.trans = None
                self.q_target = q.copy()
        return q, dq

    def write(self):
        if not self.running or self.low_state is None:
            return
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine

        if self.damping:
            for i in range(N):
                c = self.low_cmd.motor_cmd[i]
                c.mode = 1; c.tau = 0.0; c.q = 0.0; c.dq = 0.0; c.kp = 0.0; c.kd = 2.0
        else:
            q, dq = self.sample()
            for i in range(N):
                c = self.low_cmd.motor_cmd[i]
                c.mode = 1
                c.tau = 0.0
                c.q = float(q[i])
                c.dq = float(dq[i])
                c.kp = float(self.kp[i])
                c.kd = float(self.kd[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def main():
    ap = argparse.ArgumentParser(description="Replay recorded real G1 poses. Uses arms only by default.")
    ap.add_argument("--net-iface", default="enx00e04c0f3e58")
    ap.add_argument("--poses", default="sim2real/recorded_catch_poses.json")
    ap.add_argument("--sequence", default="open,cup,hug")
    ap.add_argument("--durations", default="1.5,0.7,0.8")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--whole-body", action="store_true", help="Replay all 29 joints instead of arms only. Not recommended.")
    args = ap.parse_args()

    data = json.loads(Path(args.poses).read_text())
    poses = data["poses"]
    seq = [x.strip() for x in args.sequence.split(",") if x.strip()]
    durs = [float(x) for x in args.durations.split(",")]
    if len(seq) != len(durs):
        raise ValueError("sequence and durations lengths differ")
    for name in seq:
        if name not in poses:
            raise KeyError(f"Pose '{name}' not in {args.poses}")

    print("WARNING: recorded pose replay. Keep hands clear. Ctrl+C sends damping.")
    input("Press Enter to start DDS and hold current pose...")

    demo = Demo(args.net_iface, poses, args.kp_scale, args.kd_scale, arms_only=not args.whole_body)
    demo.init()
    demo.start()

    try:
        for name, dur in zip(seq, durs):
            input(f"\nPress Enter to move to '{name}'...")
            demo.goto(name, dur)
            time.sleep(dur + 0.2)
        print("\n[replay] sequence complete. Holding final pose. Ctrl+C to damping/quit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[replay] Ctrl+C")
    finally:
        demo.stop()


if __name__ == "__main__":
    main()
