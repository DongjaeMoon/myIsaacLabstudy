from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time

import numpy as np

from .config_schema import CatchRealConfig


@dataclass(frozen=True)
class ObjectObservation:
    rel_pos: np.ndarray
    rel_lin_vel: np.ndarray
    tag_visible: np.ndarray

    @staticmethod
    def zeros() -> "ObjectObservation":
        return ObjectObservation(
            rel_pos=np.zeros(3, dtype=np.float64),
            rel_lin_vel=np.zeros(3, dtype=np.float64),
            tag_visible=np.zeros(1, dtype=np.float64),
        )

    @staticmethod
    def fake_debug() -> "ObjectObservation":
        return ObjectObservation(
            rel_pos=np.array([0.8, 0.0, 0.4], dtype=np.float64),
            rel_lin_vel=np.array([-0.5, 0.0, 0.0], dtype=np.float64),
            tag_visible=np.ones(1, dtype=np.float64),
        )


class MujocoUdpReceiver:
    def __init__(self, port: int = 5560, stale_timeout_s: float = 0.2):
        self.port = int(port)
        self.stale_timeout_s = float(stale_timeout_s)
        self.sock: socket.socket | None = None
        self.last_observation = ObjectObservation.zeros()
        self.last_packet_time: float | None = None
        self.was_fresh = False
        self._bind_socket()

    def _bind_socket(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.port))
            sock.setblocking(False)
        except OSError as exc:
            print(f"[G1] mujoco_udp bind failed on port {self.port}: {exc}")
            self.sock = None
            return

        self.sock = sock
        print(f"[G1] mujoco_udp listening on UDP {self.port}")

    def _parse_packet(self, payload: bytes) -> ObjectObservation | None:
        text = payload.decode("utf-8", errors="ignore").strip()
        if not text:
            return None

        values: list[float]
        if text[0] in "{[":
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                return None

            if isinstance(raw, dict):
                rel_pos = raw.get("rel_pos") or raw.get("object_rel_pos")
                rel_vel = raw.get("rel_lin_vel") or raw.get("object_rel_lin_vel")
                tag_visible = raw.get("tag_visible", 0.0)
                if rel_pos is None or rel_vel is None:
                    return None
                values = [
                    float(rel_pos[0]),
                    float(rel_pos[1]),
                    float(rel_pos[2]),
                    float(rel_vel[0]),
                    float(rel_vel[1]),
                    float(rel_vel[2]),
                    float(tag_visible[0] if isinstance(tag_visible, (list, tuple)) else tag_visible),
                ]
            elif isinstance(raw, list) and len(raw) >= 7:
                values = [float(value) for value in raw[:7]]
            else:
                return None
        else:
            parts = text.split()
            if len(parts) < 7:
                return None
            try:
                values = [float(part) for part in parts[:7]]
            except ValueError:
                return None

        return ObjectObservation(
            rel_pos=np.array(values[0:3], dtype=np.float64),
            rel_lin_vel=np.array(values[3:6], dtype=np.float64),
            tag_visible=np.array([values[6]], dtype=np.float64),
        )

    def get(self) -> ObjectObservation:
        if self.sock is not None:
            while True:
                try:
                    payload, _ = self.sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except OSError:
                    break

                observation = self._parse_packet(payload)
                if observation is None:
                    continue
                self.last_observation = observation
                self.last_packet_time = time.monotonic()

        now = time.monotonic()
        is_fresh = self.last_packet_time is not None and (now - self.last_packet_time) <= self.stale_timeout_s
        if is_fresh and not self.was_fresh:
            print("[G1] mujoco_udp object packets -> FRESH")
        elif not is_fresh and self.was_fresh:
            print("[G1] mujoco_udp object packets -> STALE")
        self.was_fresh = is_fresh

        if is_fresh:
            return self.last_observation
        return ObjectObservation.zeros()


class ObjectObservationProvider:
    def __init__(self, cfg: CatchRealConfig):
        self.cfg = cfg
        self.fake_enabled = bool(cfg.policy_runtime.fake_object_debug)
        self.object_source = str(cfg.policy_runtime.object_source).lower()
        self.mujoco_udp: MujocoUdpReceiver | None = None
        if self.object_source == "mujoco_udp":
            self.mujoco_udp = MujocoUdpReceiver(port=5560, stale_timeout_s=0.2)

    def get(self) -> ObjectObservation:
        if self.fake_enabled:
            return ObjectObservation.fake_debug()
        if self.object_source == "mujoco_udp" and self.mujoco_udp is not None:
            return self.mujoco_udp.get()
        return ObjectObservation.zeros()

    def toggle_fake(self) -> bool:
        self.fake_enabled = not self.fake_enabled
        return self.fake_enabled

    def status_label(self) -> str:
        if self.fake_enabled:
            return "fake"
        if self.object_source == "zeros":
            return "zeros"
        if self.object_source == "mujoco_udp":
            return "mujoco_udp"
        return f"{self.object_source} later (stub->zeros)"
