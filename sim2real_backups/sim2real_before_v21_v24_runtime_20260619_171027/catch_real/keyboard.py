from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Any


class TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self._fd: int | None = None
        self._settings: Any = None

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            print("[G1] stdin is not a TTY; keyboard pose switching is disabled.")
            return self

        self.enabled = True
        self._fd = sys.stdin.fileno()
        self._settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._fd is not None and self._settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)

    def read_key(self, timeout: float = 0.1) -> str | None:
        if not self.enabled:
            time.sleep(timeout)
            return None

        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            return None

        key = sys.stdin.read(1)
        if key == "\x1b":
            return "esc"
        return key.lower()
