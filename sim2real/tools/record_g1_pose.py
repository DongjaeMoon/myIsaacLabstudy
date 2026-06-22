#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class LowStateReader:
    def __init__(self):
        self.low_state = None

    def cb(self, msg: LowState_):
        self.low_state = msg

    def q(self):
        if self.low_state is None:
            return None
        return np.array([self.low_state.motor_state[i].q for i in range(29)], dtype=float)

    def dq(self):
        if self.low_state is None:
            return None
        return np.array([self.low_state.motor_state[i].dq for i in range(29)], dtype=float)


def main():
    ap = argparse.ArgumentParser(description="Record current G1 29-DOF joint poses from LowState only. Sends no motor commands.")
    ap.add_argument("--net-iface", default="enx00e04c0f3e58")
    ap.add_argument("--out", default="sim2real/recorded_catch_poses.json")
    ap.add_argument("--names", default="open,cup,hug", help="comma-separated pose names")
    args = ap.parse_args()

    ChannelFactoryInitialize(0, args.net_iface)
    reader = LowStateReader()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(reader.cb, 10)

    print("[record] waiting for lowstate...")
    t0 = time.time()
    while reader.q() is None:
        if time.time() - t0 > 5.0:
            raise RuntimeError("No LowState received. Check interface.")
        time.sleep(0.05)

    poses = {}
    print("[record] This script sends NO commands. Put robot in desired pose, then press Enter.")
    for name in [x.strip() for x in args.names.split(",") if x.strip()]:
        input(f"\nPose '{name}': manually/teleop place robot, then press Enter to record...")
        q = reader.q()
        dq = reader.dq()
        print(f"[record] saved {name}: max|dq|={float(np.max(np.abs(dq))):.4f}")
        print(np.round(q, 4).tolist())
        poses[name] = q.tolist()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "joint_names": JOINT_NAMES,
        "poses": poses,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Recorded from real G1 LowState. Replay script uses arms only by default.",
    }, indent=2))
    print(f"\n[record] wrote {out}")


if __name__ == "__main__":
    main()
