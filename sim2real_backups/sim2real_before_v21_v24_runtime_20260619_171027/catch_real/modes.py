from __future__ import annotations

from enum import Enum


class ControllerMode(Enum):
    DAMPING = "damping"
    SAFE_STAND = "safe_stand"
    CATCH_READY = "catch_ready"
    CATCH = "catch"
    HOLD = "hold"
