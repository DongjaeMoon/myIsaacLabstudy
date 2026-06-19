#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.catch_real.policy_runner import inspect_torchscript_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    info = inspect_torchscript_policy(
        args.policy,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        device=args.device,
    )
    module_text = str(info["module"])
    print(f"load success: {info['path']}")
    print(f"input shape : {info['obs_shape']}")
    print(f"output shape: {info['action_shape']}")
    print(f"zero action min/max: {info['action_min']:.6f} / {info['action_max']:.6f}")
    print(f"normalizer hint: {info['normalizer']}")
    print("module contains normalizer:", "normalizer" in module_text.lower())
    print("module preview:")
    print(module_text[:2000])
    if len(module_text) > 2000:
        print("... <module text truncated>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
