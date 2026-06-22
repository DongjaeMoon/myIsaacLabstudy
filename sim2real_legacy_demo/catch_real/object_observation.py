from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time

import numpy as np

from .apriltag_zmq_receiver import AprilTagDetectionSnapshot, AprilTagZmqReceiver
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

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

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
        self.apriltag_zmq: AprilTagZmqReceiver | None = None
        self.last_apriltag_snapshot = AprilTagDetectionSnapshot.zeros(
            initialized=False,
            status="DISABLED",
            message="AprilTag source not selected",
        )
        if self.object_source == "mujoco_udp":
            self.mujoco_udp = MujocoUdpReceiver(port=5560, stale_timeout_s=0.2)
        elif self.object_source == "apriltag_zmq":
            if not self.cfg.camera.enabled:
                self.last_apriltag_snapshot = AprilTagDetectionSnapshot.zeros(
                    initialized=False,
                    status="DISABLED",
                    message="camera.enabled is false",
                )
            else:
                self.apriltag_zmq = AprilTagZmqReceiver(
                    server_address=self.cfg.camera.server_address,
                    port=self.cfg.camera.port,
                    intrinsics_yaml=self.cfg.camera.intrinsics_yaml,
                    extrinsics_yaml=self.cfg.camera.extrinsics_yaml,
                    tag_yaml=self.cfg.camera.tag_yaml,
                    policy_body_frame=self.cfg.robot.imu.policy_body_frame,
                    stale_timeout_s=self.cfg.camera.estimator.lost_timeout_s,
                    min_valid_detections=self.cfg.camera.estimator.min_valid_detections,
                    position_filter_alpha=self.cfg.camera.estimator.position_filter_alpha,
                    velocity_filter_alpha=self.cfg.camera.estimator.velocity_filter_alpha,
                    angular_velocity_filter_alpha=self.cfg.camera.estimator.angular_velocity_filter_alpha,
                    status_print_interval_s=self.cfg.camera.estimator.status_print_interval_s,
                    controlled_joint_names=self.cfg.robot.controlled_joint_names,
                    extrinsics_parent_frame=self.cfg.camera.extrinsics_parent_frame,
                    dynamic_body_to_camera=self.cfg.camera.dynamic_body_to_camera,
                    body_to_torso_urdf=self.cfg.camera.body_to_torso_urdf,
                    body_link_name=self.cfg.camera.body_link_name,
                    torso_link_name=self.cfg.camera.torso_link_name,
                    waist_joint_names=self.cfg.camera.waist_joint_names,
                    image_show=self.cfg.camera.image_show,
                    emit_status_logs=True,
                    source_name="apriltag_zmq",
                )
                self.last_apriltag_snapshot = self.apriltag_zmq.get_snapshot()

    def get(self) -> ObjectObservation:
        if self.fake_enabled:
            return ObjectObservation.fake_debug()
        if self.object_source == "mujoco_udp" and self.mujoco_udp is not None:
            return self.mujoco_udp.get()
        if self.object_source == "apriltag_zmq" and self.apriltag_zmq is not None:
            self.last_apriltag_snapshot = self.apriltag_zmq.get_snapshot()
            if self.last_apriltag_snapshot.valid:
                return ObjectObservation(
                    rel_pos=self.last_apriltag_snapshot.rel_pos_b.copy(),
                    rel_lin_vel=self.last_apriltag_snapshot.rel_lin_vel_b.copy(),
                    tag_visible=np.ones(1, dtype=np.float64),
                )
            return ObjectObservation.zeros()
        return ObjectObservation.zeros()

    def update_robot_state(
        self,
        *,
        quat_wxyz: np.ndarray,
        q: np.ndarray | None = None,
        controlled_joint_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        if self.apriltag_zmq is not None:
            self.apriltag_zmq.update_robot_state(
                quat_wxyz=quat_wxyz,
                q=q,
                controlled_joint_names=controlled_joint_names,
            )

    def toggle_fake(self) -> bool:
        self.fake_enabled = not self.fake_enabled
        return self.fake_enabled

    def close(self) -> None:
        if self.apriltag_zmq is not None:
            self.apriltag_zmq.close()
        if self.mujoco_udp is not None:
            self.mujoco_udp.close()

    def status_label(self) -> str:
        if self.fake_enabled:
            return "fake"
        if self.object_source == "zeros":
            return "zeros"
        if self.object_source == "mujoco_udp":
            return "mujoco_udp"
        if self.object_source == "apriltag_zmq":
            snapshot = self.last_apriltag_snapshot
            return f"apriltag_zmq:{snapshot.status.lower()}"
        return f"{self.object_source} later (stub->zeros)"

    def source_name(self) -> str:
        if self.fake_enabled:
            return "fake"
        return self.object_source

    def source_status(self) -> str:
        if self.fake_enabled:
            return "fake"
        if self.object_source == "apriltag_zmq":
            snapshot = self.last_apriltag_snapshot
            debug_state = self.apriltag_zmq.get_camera_pose_debug() if self.apriltag_zmq is not None else None
            waist_deg = (
                np.rad2deg(debug_state.waist_angles_rad)
                if debug_state is not None
                else np.zeros(len(self.cfg.camera.waist_joint_names), dtype=np.float64)
            )
            camera_pos_b = (
                np.round(debug_state.camera_translation_body_m, 3).tolist()
                if debug_state is not None and debug_state.available
                else "n/a"
            )
            fk_status = debug_state.message if debug_state is not None else "camera debug unavailable"
            return (
                f"{snapshot.status} "
                f"tag_visible={int(snapshot.tag_visible)} "
                f"fps={snapshot.frame_rate_hz:.1f} "
                f"waist_deg={np.round(waist_deg, 1).tolist()} "
                f"camera_pos_b={camera_pos_b} "
                f"camera_fk={fk_status}"
            )
        return self.status_label()

    def initialization_status(self) -> str:
        if self.object_source != "apriltag_zmq":
            return "not requested"
        if self.apriltag_zmq is None:
            snapshot = self.last_apriltag_snapshot
            return f"{snapshot.status.lower()}: {snapshot.message}"
        state = "ok" if self.apriltag_zmq.initialized else "failed"
        return f"{state}: {self.apriltag_zmq.init_message}"

    def stale_timeout_s(self) -> float | None:
        if self.object_source == "mujoco_udp":
            return self.mujoco_udp.stale_timeout_s if self.mujoco_udp is not None else None
        if self.object_source == "apriltag_zmq":
            if self.apriltag_zmq is not None:
                return self.apriltag_zmq.stale_timeout_s
            return self.cfg.camera.estimator.lost_timeout_s
        return None

    def startup_summary_lines(self) -> list[str]:
        if self.object_source == "apriltag_zmq" and self.apriltag_zmq is not None:
            return self.apriltag_zmq.startup_summary_lines()
        return []
