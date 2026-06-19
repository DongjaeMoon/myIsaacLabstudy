#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.config import load_catch_real_config


DEFAULT_CONFIG_PATH = THIS_DIR / "configs/g1_catch_real_urop_v23_apriltag_obsonly.yaml"
VALID_START_POSES = {"safe_stand", "catch_ready", "hold"}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--net-iface", type=str, default=None)
    parser.add_argument("--no-policy", dest="no_policy", action="store_true")
    parser.add_argument("--use-policy", dest="no_policy", action="store_false")
    parser.set_defaults(no_policy=True)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-pose", type=str, default="safe_stand")
    parser.add_argument("--move-duration", type=float, default=None)
    parser.add_argument("--print-rate", type=float, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config and optional TorchScript policy without importing Unitree SDK/DDS.",
    )
    return parser


def initialize_dds(net_iface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if net_iface:
        ChannelFactoryInitialize(0, net_iface)
    else:
        ChannelFactoryInitialize(0)


def _format_path_status(path: Path | None) -> str:
    if path is None:
        return "not configured"
    return f"{path} ({'exists' if path.exists() else 'MISSING'})"


def run_check_only(args: argparse.Namespace) -> int:
    cfg = load_catch_real_config(args.config, policy_override=args.policy)
    use_policy = not bool(args.no_policy)

    print("[G1][check-only] PASS: config loaded")
    print(f"[G1][check-only] config             : {cfg.config_path}")
    print(f"[G1][check-only] name/version       : {cfg.name} / {cfg.version}")
    print(f"[G1][check-only] urop_version       : {cfg.metadata.urop_version}")
    print(f"[G1][check-only] contract           : {_format_path_status(cfg.metadata.contract_path)}")
    print(f"[G1][check-only] run_safety         : {cfg.metadata.run_safety}")
    print(f"[G1][check-only] obs/action         : {cfg.observation.num_obs} / {cfg.policy.num_actions}")
    print(f"[G1][check-only] policy_dt/control  : {cfg.runtime.policy_dt} / {cfg.runtime.control_dt}")
    print(f"[G1][check-only] policy_body_frame  : {cfg.robot.imu.policy_body_frame}")
    print(f"[G1][check-only] object_source/frame: {cfg.policy_runtime.object_source} / {cfg.policy_runtime.object_observation_frame}")
    print(f"[G1][check-only] gate_visible       : {cfg.policy_runtime.gate_policy_until_object_visible}")
    print(f"[G1][check-only] policy path        : {_format_path_status(cfg.policy.path)}")
    print(f"[G1][check-only] intrinsics         : {_format_path_status(cfg.camera.intrinsics_yaml)}")
    print(f"[G1][check-only] extrinsics         : {_format_path_status(cfg.camera.extrinsics_yaml)}")
    print(f"[G1][check-only] tag yaml           : {_format_path_status(cfg.camera.tag_yaml)}")
    print(f"[G1][check-only] training camera t  : {cfg.camera.training_camera_translation}")
    print(f"[G1][check-only] training camera q  : {cfg.camera.training_camera_quat_wxyz}")
    print(f"[G1][check-only] training convention: {cfg.camera.training_camera_convention or 'n/a'}")

    if use_policy:
        if cfg.policy.path is None:
            raise FileNotFoundError("--use-policy requested but no --policy path or policy.path is configured")
        from sim2real.catch_real.policy_runner import inspect_torchscript_policy

        info = inspect_torchscript_policy(
            cfg.policy.path,
            obs_dim=cfg.observation.num_obs,
            action_dim=cfg.policy.num_actions,
            device="cpu",
        )
        print(f"[G1][check-only] policy output shape: {info['action_shape']}")
        print(f"[G1][check-only] zero action min/max: {info['action_min']:.6f} / {info['action_max']:.6f}")
        print(f"[G1][check-only] normalizer hint    : {info['normalizer']}")
    else:
        print("[G1][check-only] policy dry-run     : skipped (--no-policy)")

    if cfg.robot.imu.policy_body_frame == "unitree":
        print("[G1][check-only] WARNING projected_gravity/base_ang_vel frame must be verified in obs-only.")
    missing_camera = [
        path
        for path in (cfg.camera.intrinsics_yaml, cfg.camera.extrinsics_yaml, cfg.camera.tag_yaml)
        if cfg.camera.enabled and path is not None and not path.exists()
    ]
    if missing_camera:
        print("[G1][check-only] WARNING camera calibration files are missing:")
        for path in missing_camera:
            print(f"[G1][check-only]   MISSING {path}")
    print("[G1][check-only] DONE: no DDS, no Unitree SDK command, no controller start")
    return 0


def main() -> None:
    args = build_argparser().parse_args()

    if args.check_only:
        raise SystemExit(run_check_only(args))

    cfg = load_catch_real_config(args.config, policy_override=args.policy)

    if args.start_pose not in VALID_START_POSES:
        raise ValueError("--start-pose must be one of: safe_stand, catch_ready, hold")

    net_iface = args.net_iface or cfg.communication.net_iface

    if cfg.runtime.require_enter_before_start:
        print("WARNING: Keep the robot area clear before enabling low-level control.")
        input("Press Enter to continue...")

    initialize_dds(net_iface)

    from sim2real.catch_real.controller import G1CatchRealController
    from sim2real.catch_real.keyboard import TerminalKeyReader

    controller = G1CatchRealController(cfg, args)
    controller.init()
    controller.start()

    try:
        with TerminalKeyReader() as key_reader:
            while not controller.exit_requested:
                key = key_reader.read_key(timeout=0.1)
                if key is not None:
                    if key in ("\\x08", "\\x7f", "\b", "backspace", "BACKSPACE", "KEY_BACKSPACE"):
                        print(f"[G1] BACKSPACE pressed in Python terminal at {time.strftime('%H:%M:%S')} key={repr(key)}", flush=True)
                    controller.handle_key(key)
    except KeyboardInterrupt:
        print("\n[G1] KeyboardInterrupt detected; switching to damping.")
    except Exception:
        controller.safe_shutdown()
        controller.stop()
        raise
    finally:
        controller.safe_shutdown()
        controller.stop()
        print("[G1] Controller exited safely.")


if __name__ == "__main__":
    main()
