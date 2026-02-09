# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility functions for versioning."""

from __future__ import annotations

import functools
from packaging.version import Version


@functools.lru_cache(maxsize=1)
def get_isaac_sim_version() -> Version:
    """Get the Isaac Sim version as a Version object, cached for performance."""
    try:
        from isaacsim.core.version import get_version
        version_tuple = get_version()
        # Isaac Sim 5.1.0 fix: Check if tuple is valid, otherwise fallback
        if not version_tuple or version_tuple[2] == "":
            return Version("5.1.0")
            
        return Version(f"{version_tuple[2]}.{version_tuple[3]}.{version_tuple[4]}")
    except Exception:
        # Fallback for any error
        return Version("5.1.0")


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings."""
    ver1 = Version(v1)
    ver2 = Version(v2)

    if ver1 > ver2:
        return 1
    elif ver1 < ver2:
        return -1
    else:
        return 0
