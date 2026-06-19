#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.config import load_catch_real_config
from sim2real.catch_real.policy_runner import inspect_torchscript_policy, validate_policy_path


def _status_line(kind: str, message: str) -> None:
    print(f"[{kind}] {message}")


def _almost_equal(a: float, b: float, tol: float = 1.0e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _compare_pose(cfg_pose: np.ndarray, contract_pose: dict[str, float], joint_order: list[str]) -> bool:
    expected = np.array([float(contract_pose[name]) for name in joint_order], dtype=np.float64)
    return bool(np.allclose(cfg_pose, expected, atol=1.0e-9, rtol=0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=["v21", "v22", "v23", "v24"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--policy", type=Path, default=None)
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    contract_path = REPO_ROOT / "sim2real/configs/contracts" / f"urop_{args.version}_contract.yaml"
    if not contract_path.exists():
        failures.append(f"missing contract yaml: {contract_path}")
        contract = {}
    else:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        _status_line("OK", f"contract loaded: {contract_path.relative_to(REPO_ROOT)}")

    try:
        cfg = load_catch_real_config(args.config, policy_override=str(args.policy) if args.policy else None)
        _status_line("OK", f"config loaded: {cfg.config_path.relative_to(REPO_ROOT)}")
    except Exception as exc:
        _status_line("FAIL", f"config loader failed: {exc!r}")
        return 1

    if cfg.metadata.urop_version != args.version:
        warnings.append(f"metadata.urop_version={cfg.metadata.urop_version}, requested {args.version}")

    if contract:
        if cfg.observation.num_obs != int(contract["obs"]["num_obs"]):
            failures.append(f"obs_dim mismatch cfg={cfg.observation.num_obs} contract={contract['obs']['num_obs']}")
        if cfg.policy.num_actions != int(contract["policy"]["num_actions"]):
            failures.append(f"action_dim mismatch cfg={cfg.policy.num_actions} contract={contract['policy']['num_actions']}")

        cfg_terms = [(term.name, term.dim, float(term.scale)) for term in cfg.observation.terms]
        contract_terms = [
            (term["name"], int(term["dim"]), float(term.get("scale", 1.0)))
            for term in contract["obs"]["terms"]
        ]
        if cfg_terms != contract_terms:
            failures.append(f"observation terms mismatch cfg={cfg_terms} contract={contract_terms}")

        joint_order = list(contract["policy"]["joint_order"])
        if cfg.policy.action_joint_names != joint_order:
            failures.append("action_order mismatch")
        expected_scales = np.array([float(contract["policy"]["action_scale_by_joint"][name]) for name in joint_order])
        if not np.allclose(cfg.policy.action_scales, expected_scales, atol=1.0e-9, rtol=0.0):
            failures.append("action_scale mismatch")
        if not _compare_pose(cfg.poses["catch_ready"], contract["poses"]["policy_reference_pose"], joint_order):
            failures.append("policy reference/catch_ready pose mismatch")
        if not _almost_equal(cfg.runtime.policy_dt, contract["timing"]["policy_dt"]):
            failures.append("policy_dt mismatch")
        joint_vel_scale = next((term.scale for term in cfg.observation.terms if term.name == "joint_vel"), None)
        contract_joint_vel_scale = next(
            (term["scale"] for term in contract["obs"]["terms"] if term["name"] == "joint_vel"),
            None,
        )
        if not _almost_equal(joint_vel_scale, contract_joint_vel_scale):
            failures.append("joint_vel scale mismatch")
        expected_frame = contract["camera"].get("runtime_object_observation_frame")
        if cfg.policy_runtime.object_observation_frame != expected_frame:
            failures.append(
                f"object frame mismatch cfg={cfg.policy_runtime.object_observation_frame} contract={expected_frame}"
            )
        if contract["camera"].get("training_camera_translation") is None:
            warnings.append("contract has no training_camera_translation")

    for path in (cfg.camera.intrinsics_yaml, cfg.camera.extrinsics_yaml, cfg.camera.tag_yaml):
        if cfg.camera.enabled and path is not None and not path.exists():
            warnings.append(f"missing camera calibration path: {path}")

    if cfg.robot.imu.policy_body_frame == "unitree":
        warnings.append("policy_body_frame=unitree; verify IMU/base axes in obs-only before real policy")

    if args.policy is not None:
        try:
            validate_policy_path(args.policy)
            info = inspect_torchscript_policy(
                args.policy,
                obs_dim=cfg.observation.num_obs,
                action_dim=cfg.policy.num_actions,
                device="cpu",
            )
            _status_line("OK", f"policy dry-load output shape {info['action_shape']}")
            if contract and contract["runtime_notes"].get("empirical_normalization") and info["normalizer"] == "identity":
                warnings.append("empirical_normalization=True but exported JIT normalizer looks like Identity")
        except Exception as exc:
            failures.append(f"policy dry-load failed: {exc!r}")
    else:
        warnings.append("no policy path provided; skipped TorchScript dry-load")

    for warning in warnings:
        _status_line("WARN", warning)
    for failure in failures:
        _status_line("FAIL", failure)

    if failures:
        print("RESULT: FAIL")
        return 1
    if warnings:
        print("RESULT: PASS_WITH_WARNINGS")
        return 0
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
