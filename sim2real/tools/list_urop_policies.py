#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim2real.catch_real.policy_runner import inspect_torchscript_policy


def main() -> int:
    versions = ("v21", "v22", "v23", "v24")
    any_found = False
    for version in versions:
        root = REPO_ROOT / "logs/rsl_rl" / f"UROP_{version}"
        policies = sorted(
            root.glob("*/exported/policy.pt"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
            reverse=True,
        )
        print(f"\n[{version}] {len(policies)} exported policies")
        if not policies:
            continue
        any_found = True
        for index, path in enumerate(policies):
            rel = path.relative_to(REPO_ROOT)
            latest = " latest" if index == 0 else ""
            try:
                info = inspect_torchscript_policy(path, obs_dim=100, action_dim=29, device="cpu")
                print(
                    f"  OK{latest}: {rel} "
                    f"out={info['action_shape']} minmax={info['action_min']:.4f}/{info['action_max']:.4f} "
                    f"normalizer={info['normalizer']}"
                )
            except Exception as exc:
                print(f"  FAIL{latest}: {rel} {exc!r}")
    return 0 if any_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
