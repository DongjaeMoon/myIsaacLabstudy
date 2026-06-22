#!/usr/bin/env python3
"""
Publish a scripted virtual object observation over UDP for emergency sim2real tests.

Camera/OpenCV frame convention:
  x: right, y: down, z: forward

Run this in a separate terminal, then press Enter to trigger the approach trajectory.
The sim2real config should use policy_runtime.object_source: mujoco_udp.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass


def parse_vec3(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected comma-separated vector: x,y,z")
    return vals


@dataclass
class Trajectory:
    p_start: list[float]
    p_catch: list[float]
    duration: float
    hold_time: float

    def sample(self, elapsed: float) -> tuple[list[float], list[float], float, str]:
        if elapsed < 0.0:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "WAIT"

        if elapsed <= self.duration:
            u = max(0.0, min(1.0, elapsed / self.duration))
            s = 3.0 * u * u - 2.0 * u * u * u
            ds_du = 6.0 * u - 6.0 * u * u
            ds_dt = ds_du / max(self.duration, 1e-6)
            dp = [self.p_catch[i] - self.p_start[i] for i in range(3)]
            pos = [self.p_start[i] + s * dp[i] for i in range(3)]
            vel = [ds_dt * dp[i] for i in range(3)]
            return pos, vel, 1.0, "APPROACH"

        if elapsed <= self.duration + self.hold_time:
            return list(self.p_catch), [0.0, 0.0, 0.0], 1.0, "HOLD"

        return list(self.p_catch), [0.0, 0.0, 0.0], 1.0, "DONE_HOLD"


def make_packet(pos: list[float], vel: list[float], tag: float, status: str) -> bytes:
    now = time.time()
    # Include multiple aliases so this works with slightly different parsers.
    msg = {
        "timestamp": now,
        "time": now,
        "status": status,
        "frame": "camera_opencv_raw",
        "observation_frame": "camera_opencv_raw",
        "tag_visible": float(tag),
        "rel_pos": pos,
        "rel_lin_vel": vel,
        "object_rel_pos": pos,
        "object_rel_lin_vel": vel,
        "rel_pos_b": pos,
        "rel_lin_vel_b": vel,
        "rel_pos_obs": pos,
        "rel_lin_vel_obs": vel,
    }
    return json.dumps(msg, separators=(",", ":")).encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="UDP destination host; normally 127.0.0.1")
    ap.add_argument("--port", type=int, default=5560, help="UDP destination port for sim2real mujoco_udp provider")
    ap.add_argument("--rate", type=float, default=50.0, help="publish rate in Hz")
    ap.add_argument("--start", type=parse_vec3, default=parse_vec3("0.00,0.18,1.00"))
    ap.add_argument("--catch", type=parse_vec3, default=parse_vec3("0.00,0.24,0.42"))
    ap.add_argument("--duration", type=float, default=1.10)
    ap.add_argument("--hold", type=float, default=2.00)
    ap.add_argument("--visible-before-trigger", action="store_true", help="publish start pose with tag=1 before trigger")
    args = ap.parse_args()

    traj = Trajectory(args.start, args.catch, args.duration, args.hold)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.host, args.port)
    dt = 1.0 / max(args.rate, 1e-6)

    print("[scripted_object_udp] destination:", dst)
    print("[scripted_object_udp] frame      : camera_opencv_raw / OpenCV optical (x right, y down, z forward)")
    print("[scripted_object_udp] p_start    :", args.start)
    print("[scripted_object_udp] p_catch    :", args.catch)
    print("[scripted_object_udp] duration   :", args.duration)
    print("[scripted_object_udp] hold       :", args.hold)
    print("[scripted_object_udp] Press Enter to trigger. Ctrl+C to exit.")

    # Non-blocking stdin polling without curses.
    import select

    active = False
    t0 = None
    last_print = 0.0

    try:
        while True:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                active = True
                t0 = time.monotonic()
                print("[scripted_object_udp] TRIGGER")

            if not active:
                if args.visible_before_trigger:
                    pos, vel, tag, status = list(args.start), [0.0, 0.0, 0.0], 1.0, "PREVISIBLE"
                else:
                    pos, vel, tag, status = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "IDLE"
            else:
                pos, vel, tag, status = traj.sample(time.monotonic() - float(t0))

            sock.sendto(make_packet(pos, vel, tag, status), dst)

            now = time.monotonic()
            if now - last_print > 0.2:
                print(f"[scripted_object_udp] {status:<10s} tag={tag:.0f} pos={[round(x,3) for x in pos]} vel={[round(x,3) for x in vel]}")
                last_print = now
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[scripted_object_udp] exit")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
