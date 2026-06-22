#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from config import load_catch_real_config


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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate config and optional TorchScript policy without DDS, LowCmd, or controller start.",
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


def _inspect_torchscript_policy(
    policy_path: Path,
    *,
    obs_dim: int,
    action_dim: int,
    device: str,
) -> dict[str, Any]:
    import torch

    model = torch.jit.load(str(policy_path), map_location=device)
    model.eval()
    obs = torch.zeros((1, int(obs_dim)), dtype=torch.float32, device=device)
    with torch.no_grad():
        action = model(obs)
    action_flat = action.detach().cpu().numpy().reshape(-1)
    if action_flat.size != int(action_dim):
        raise RuntimeError(
            f"Policy output mismatch: expected action_dim={action_dim}, got shape={tuple(action.shape)}"
        )
    return {
        "obs_shape": tuple(obs.shape),
        "action_shape": tuple(action.shape),
        "action_min": float(action_flat.min()) if action_flat.size else 0.0,
        "action_max": float(action_flat.max()) if action_flat.size else 0.0,
    }


def run_check_only(args: argparse.Namespace) -> int:
    cfg = load_catch_real_config(args.config, policy_override=args.policy)
    use_policy = not bool(args.no_policy)

    print("[G1][legacy-check] PASS: config loaded")
    print(f"[G1][legacy-check] config             : {cfg.config_path}")
    print(f"[G1][legacy-check] name/version       : {cfg.name} / {cfg.version}")
    print(f"[G1][legacy-check] obs/action         : {cfg.observation.num_obs} / {cfg.policy.num_actions}")
    print(f"[G1][legacy-check] policy_dt/control  : {cfg.runtime.policy_dt} / {cfg.runtime.control_dt}")
    print(f"[G1][legacy-check] release_high_level : {cfg.communication.release_high_level_mode}")
    print(f"[G1][legacy-check] object_source      : {cfg.policy_runtime.object_source}")
    print(f"[G1][legacy-check] fake/gate          : {cfg.policy_runtime.fake_object_debug} / {cfg.policy_runtime.gate_policy_until_object_visible}")
    print(f"[G1][legacy-check] upper_body_only    : {cfg.policy_runtime.gantry_upper_body_only}")
    print(f"[G1][legacy-check] policy path        : {_format_path_status(cfg.policy.path)}")
    print(f"[G1][legacy-check] intrinsics         : {_format_path_status(cfg.camera.intrinsics_yaml)}")
    print(f"[G1][legacy-check] extrinsics         : {_format_path_status(cfg.camera.extrinsics_yaml)}")
    print(f"[G1][legacy-check] tag yaml           : {_format_path_status(cfg.camera.tag_yaml)}")
    print(f"[G1][legacy-check] kp min/max         : {cfg.control.kp.min():.3f} / {cfg.control.kp.max():.3f}")
    print(f"[G1][legacy-check] kd min/max         : {cfg.control.kd.min():.3f} / {cfg.control.kd.max():.3f}")
    print(f"[G1][legacy-check] target deltas      : ctrl={cfg.safety.max_target_delta_per_control_step}, policy={cfg.safety.max_target_delta_per_policy_step}")

    if use_policy:
        if cfg.policy.path is None:
            raise FileNotFoundError("--use-policy requested but no policy.path or --policy is configured")
        device = args.device or "cpu"
        if device == "auto":
            device = "cpu"
        info = _inspect_torchscript_policy(
            cfg.policy.path,
            obs_dim=cfg.observation.num_obs,
            action_dim=cfg.policy.num_actions,
            device=device,
        )
        print(f"[G1][legacy-check] policy obs shape   : {info['obs_shape']}")
        print(f"[G1][legacy-check] policy out shape   : {info['action_shape']}")
        print(f"[G1][legacy-check] zero action min/max: {info['action_min']:.6f} / {info['action_max']:.6f}")
    else:
        print("[G1][legacy-check] policy dry-run     : skipped (--no-policy)")

    print("[G1][legacy-check] DONE: no DDS, no LowCmd publish, no controller start")
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

    from catch_real.controller import G1CatchRealController
    from catch_real.keyboard import TerminalKeyReader

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
