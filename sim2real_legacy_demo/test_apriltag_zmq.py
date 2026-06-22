#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catch_real.apriltag_zmq_receiver import AprilTagZmqReceiver


DEFAULT_INTRINSICS = THIS_DIR / "calib/head_camera_intrinsics.real.yaml"
DEFAULT_EXTRINSICS = THIS_DIR / "calib/head_camera_extrinsics.real.yaml"
DEFAULT_TAG = THIS_DIR / "calib/box_tag_config.real.yaml"
DEFAULT_BODY_TO_TORSO_URDF = REPO_ROOT / "UROP/UROP_v16/usd/g1_29dof_full_collider_flattened.urdf"
DEFAULT_BODY_LINK_NAME = "g1_29dof_with_hand_rev_1_0"
DEFAULT_TORSO_LINK_NAME = "torso_link"
DEFAULT_WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe standalone AprilTag ZMQ receiver test.")
    parser.add_argument("--server-address", type=str, default="192.168.123.164")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--intrinsics-yaml", type=str, default=str(DEFAULT_INTRINSICS))
    parser.add_argument("--extrinsics-yaml", type=str, default=str(DEFAULT_EXTRINSICS))
    parser.add_argument("--tag-yaml", type=str, default=str(DEFAULT_TAG))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--print-rate", type=float, default=5.0)
    parser.add_argument("--waist-yaw", type=float, default=0.0, help="Standalone waist yaw angle in radians.")
    parser.add_argument("--waist-roll", type=float, default=0.0, help="Standalone waist roll angle in radians.")
    parser.add_argument("--waist-pitch", type=float, default=0.0, help="Standalone waist pitch angle in radians.")
    parser.add_argument("--body-to-torso-urdf", type=str, default=str(DEFAULT_BODY_TO_TORSO_URDF))
    parser.add_argument("--body-link-name", type=str, default=DEFAULT_BODY_LINK_NAME)
    parser.add_argument("--torso-link-name", type=str, default=DEFAULT_TORSO_LINK_NAME)
    parser.add_argument(
        "--dynamic-body-to-camera",
        dest="dynamic_body_to_camera",
        action="store_true",
        help="Enable URDF-backed body->torso->camera FK for torso-mounted camera extrinsics.",
    )
    parser.add_argument(
        "--fixed-body-to-camera",
        dest="dynamic_body_to_camera",
        action="store_false",
        help="Disable dynamic FK and use only fixed body-frame extrinsics.",
    )
    parser.set_defaults(dynamic_body_to_camera=True)
    return parser


def format_age(age_s: float | None) -> str:
    if age_s is None:
        return "n/a"
    return f"{age_s:.2f}s"


def main() -> int:
    args = build_argparser().parse_args()

    receiver = AprilTagZmqReceiver(
        server_address=args.server_address,
        port=args.port,
        intrinsics_yaml=_resolve_path(args.intrinsics_yaml),
        extrinsics_yaml=_resolve_path(args.extrinsics_yaml),
        tag_yaml=_resolve_path(args.tag_yaml),
        policy_body_frame="unitree",
        stale_timeout_s=0.2,
        min_valid_detections=1,
        position_filter_alpha=0.35,
        velocity_filter_alpha=0.35,
        angular_velocity_filter_alpha=0.35,
        status_print_interval_s=1.0,
        controlled_joint_names=DEFAULT_WAIST_JOINT_NAMES,
        dynamic_body_to_camera=args.dynamic_body_to_camera,
        body_to_torso_urdf=_resolve_path(args.body_to_torso_urdf),
        body_link_name=args.body_link_name,
        torso_link_name=args.torso_link_name,
        waist_joint_names=DEFAULT_WAIST_JOINT_NAMES,
        image_show=args.show,
        emit_status_logs=False,
        source_name="apriltag_test",
    )
    receiver.update_robot_state(
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        q=np.array([args.waist_yaw, args.waist_roll, args.waist_pitch], dtype=np.float64),
        controlled_joint_names=DEFAULT_WAIST_JOINT_NAMES,
    )

    print(f"[AprilTagTest] endpoint           : tcp://{args.server_address}:{args.port}")
    print(f"[AprilTagTest] intrinsics yaml    : {_resolve_path(args.intrinsics_yaml)}")
    print(f"[AprilTagTest] extrinsics yaml    : {_resolve_path(args.extrinsics_yaml)}")
    print(f"[AprilTagTest] tag yaml           : {_resolve_path(args.tag_yaml)}")
    print(f"[AprilTagTest] body->torso urdf   : {_resolve_path(args.body_to_torso_urdf)}")
    print(f"[AprilTagTest] receiver init      : {receiver.init_message}")
    print(
        "[AprilTagTest] base-state approx  : "
        "identity/standing assumption only; no DDS and no robot commands are used."
    )
    print(
        "[AprilTagTest] waist radians      : "
        f"[{args.waist_yaw:.4f}, {args.waist_roll:.4f}, {args.waist_pitch:.4f}]"
    )
    for line in receiver.startup_summary_lines():
        print(line.replace("[G1][apriltag_zmq]", "[AprilTagTest][fk]"))

    if not receiver.initialized:
        receiver.close()
        return 1

    print_interval = 1.0 / args.print_rate if args.print_rate > 0.0 else 0.0
    last_print_time = 0.0

    try:
        while True:
            now = time.monotonic()
            snapshot = receiver.get_snapshot()
            camera_debug = receiver.get_camera_pose_debug()
            if print_interval <= 0.0 or (now - last_print_time) >= print_interval:
                rel_pos = np_round(snapshot.rel_pos_b)
                rel_vel = np_round(snapshot.rel_lin_vel_b)
                visible_text = "VISIBLE" if snapshot.tag_visible else "NO_TAG"
                camera_pos_b = (
                    np_round(camera_debug.camera_translation_body_m)
                    if camera_debug.available
                    else "n/a"
                )
                waist_deg = np.round(np.rad2deg(camera_debug.waist_angles_rad), 2).tolist()
                print(
                    f"[AprilTagTest] status={snapshot.status:<11s} "
                    f"valid={str(snapshot.valid).lower():<5s} "
                    f"tag_visible={int(snapshot.tag_visible)}({visible_text}) "
                    f"camera_fk={camera_debug.message} "
                    f"camera_pos_b={camera_pos_b} "
                    f"waist_deg={waist_deg} "
                    f"rel_pos_b={rel_pos} "
                    f"rel_lin_vel_b={rel_vel} "
                    f"fps={snapshot.frame_rate_hz:.1f} "
                    f"last_valid_age={format_age(snapshot.time_since_last_valid_s)} "
                    f"msg={snapshot.message}"
                )
                last_print_time = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[AprilTagTest] interrupted by user.")
    finally:
        receiver.close()

    return 0


def np_round(values) -> list[float]:
    return np.round(np.asarray(values, dtype=float), 3).tolist()


if __name__ == "__main__":
    raise SystemExit(main())
