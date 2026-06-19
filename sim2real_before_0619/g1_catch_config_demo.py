#!/usr/bin/env python3
"""
g1_catch_config_demo.py

A thin demo entrypoint that uses the SAME config/controller/module structure as
sim2real/g1_catch_real.py.

Purpose:
- Do NOT create a standalone control implementation.
- Reuse sim2real/configs/g1_catch_real.yaml.
- Reuse G1CatchRealController.
- Reuse the same DDS LowState/LowCmd path.
- Reuse the same q_ref pose transition logic.
- Keep policy disabled.
- Provide a simpler test script for scripted or interactive pose transitions.

Place this file at:
    sim2real/g1_catch_config_demo.py

MuJoCo DDS loopback:
    python sim2real/g1_catch_config_demo.py --net-iface lo --sequence

Interactive keyboard:
    python sim2real/g1_catch_config_demo.py --net-iface lo --interactive

Real G1:
    python sim2real/g1_catch_config_demo.py --net-iface <REAL_IFACE> --interactive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.config import load_catch_real_config
from sim2real.catch_real.controller import G1CatchRealController
from sim2real.catch_real.keyboard import TerminalKeyReader
from sim2real.catch_real.modes import ControllerMode
from unitree_sdk2py.core.channel import ChannelFactoryInitialize


DEFAULT_CONFIG_PATH = THIS_DIR / "configs/g1_catch_real.yaml"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Config-driven G1 catch controller demo using the same modules as g1_catch_real.py"
    )

    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--net-iface", type=str, default=None)

    # This demo intentionally disables policy.
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-pose", type=str, default="safe_stand")
    parser.add_argument("--move-duration", type=float, default=None)
    parser.add_argument("--print-rate", type=float, default=None)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sequence",
        action="store_true",
        help="Run scripted sequence: safe_stand -> catch_ready -> hold -> damping -> exit.",
    )
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run keyboard interactive mode using the same controller keys.",
    )

    parser.add_argument("--hold-time", type=float, default=3.0)
    parser.add_argument(
        "--no-enter",
        action="store_true",
        help="Skip the press-Enter safety prompt. Use only in MuJoCo.",
    )

    return parser


def make_controller_args(args: argparse.Namespace) -> SimpleNamespace:
    """
    G1CatchRealController currently expects the same argparse fields as
    g1_catch_real.py. We construct that compatible namespace here.
    """
    return SimpleNamespace(
        config=args.config,
        net_iface=args.net_iface,
        no_policy=True,
        policy=args.policy,
        device=args.device,
        start_pose=args.start_pose,
        move_duration=args.move_duration,
        print_rate=args.print_rate,
    )


def initialize_dds(net_iface: str | None) -> None:
    if net_iface:
        print(f"[DEMO] ChannelFactoryInitialize iface={net_iface}")
        ChannelFactoryInitialize(0, net_iface)
    else:
        print("[DEMO] ChannelFactoryInitialize default")
        ChannelFactoryInitialize(0)


def request_and_wait(
    controller: G1CatchRealController,
    mode: ControllerMode,
    hold_time: float,
) -> None:
    controller.request_mode(mode)
    t0 = time.monotonic()
    while time.monotonic() - t0 < hold_time and not controller.exit_requested:
        time.sleep(0.05)


def run_sequence(controller: G1CatchRealController, hold_time: float) -> None:
    print("[DEMO] Running scripted sequence.")
    print("[DEMO] Sequence: SAFE_STAND -> CATCH_READY -> HOLD -> DAMPING -> exit")

    request_and_wait(controller, ControllerMode.SAFE_STAND, hold_time)
    request_and_wait(controller, ControllerMode.CATCH_READY, hold_time)
    request_and_wait(controller, ControllerMode.HOLD, hold_time)

    print("[DEMO] Sequence complete. Switching to damping.")
    controller.request_mode(ControllerMode.DAMPING)
    time.sleep(0.5)
    controller.exit_requested = True


def run_interactive(controller: G1CatchRealController) -> None:
    print("[DEMO] Interactive mode.")
    print("[DEMO] Keys are the same as g1_catch_real.py:")
    print("[DEMO] S=safe_stand, P/C=catch_ready, H=hold, D=damping, K=catch(policy disabled), Q/ESC=exit")

    with TerminalKeyReader() as key_reader:
        while not controller.exit_requested:
            key = key_reader.read_key(timeout=0.1)
            if key is not None:
                controller.handle_key(key)


def main() -> None:
    args = build_argparser().parse_args()

    if args.start_pose not in {"safe_stand", "catch_ready", "hold"}:
        raise ValueError("--start-pose must be one of: safe_stand, catch_ready, hold")

    cfg = load_catch_real_config(args.config, policy_override=args.policy)
    net_iface = args.net_iface or cfg.communication.net_iface

    if not args.no_enter and cfg.runtime.require_enter_before_start:
        print("[DEMO] WARNING: This uses the SAME low-level controller path as g1_catch_real.py.")
        print("[DEMO] Keep the robot area clear before enabling low-level control.")
        input("Press Enter to continue...")

    initialize_dds(net_iface)

    controller_args = make_controller_args(args)
    controller = G1CatchRealController(cfg, controller_args)

    controller.init()
    controller.start()

    try:
        if args.sequence:
            run_sequence(controller, args.hold_time)
        else:
            # Default to interactive if neither option is supplied.
            run_interactive(controller)
    except KeyboardInterrupt:
        print("\n[DEMO] KeyboardInterrupt. Switching to damping.")
    except Exception:
        controller.safe_shutdown()
        controller.stop()
        raise
    finally:
        controller.safe_shutdown()
        controller.stop()
        print("[DEMO] Demo exited safely.")


if __name__ == "__main__":
    main()
