#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.catch_real.controller import G1CatchRealController
from sim2real.catch_real.keyboard import TerminalKeyReader
from sim2real.config import load_catch_real_config
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


DEFAULT_CONFIG_PATH = THIS_DIR / "configs/g1_catch_real.yaml"
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
    return parser


def initialize_dds(net_iface: str | None) -> None:
    if net_iface:
        ChannelFactoryInitialize(0, net_iface)
    else:
        ChannelFactoryInitialize(0)


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_catch_real_config(args.config, policy_override=args.policy)

    if args.start_pose not in VALID_START_POSES:
        raise ValueError("--start-pose must be one of: safe_stand, catch_ready, hold")

    net_iface = args.net_iface or cfg.communication.net_iface

    if cfg.runtime.require_enter_before_start:
        print("WARNING: Keep the robot area clear before enabling low-level control.")
        input("Press Enter to continue...")

    initialize_dds(net_iface)

    controller = G1CatchRealController(cfg, args)
    controller.init()
    controller.start()

    try:
        with TerminalKeyReader() as key_reader:
            while not controller.exit_requested:
                key = key_reader.read_key(timeout=0.1)
                if key is not None:
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
