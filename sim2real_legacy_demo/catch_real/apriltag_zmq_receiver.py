from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    cv2 = None
    _CV2_IMPORT_ERROR: Exception | None = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import yaml
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    yaml = None
    _YAML_IMPORT_ERROR: Exception | None = exc
else:
    _YAML_IMPORT_ERROR = None

try:
    import zmq
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    zmq = None
    _ZMQ_IMPORT_ERROR: Exception | None = exc
else:
    _ZMQ_IMPORT_ERROR = None

try:
    from apriltag_object_state_estimator import (
        AprilTagObjectStateEstimator,
        RobotBaseState,
        make_T,
    )
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    AprilTagObjectStateEstimator = None
    RobotBaseState = Any
    make_T = None
    _ESTIMATOR_IMPORT_ERROR: Exception | None = exc
else:
    _ESTIMATOR_IMPORT_ERROR = None


def normalize_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def rpy_to_rotmat_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def quat_to_rotmat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat_wxyz(q_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        diag = np.diag(R)
        index = int(np.argmax(diag))
        if index == 0:
            s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (R[2, 1] - R[1, 2]) / s,
                    0.25 * s,
                    (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (R[0, 2] - R[2, 0]) / s,
                    (R[0, 1] + R[1, 0]) / s,
                    0.25 * s,
                    (R[1, 2] + R[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1.0e-12)) * 2.0
            quat = np.array(
                [
                    (R[1, 0] - R[0, 1]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return normalize_quat_wxyz(quat)


def rotmat_to_rpy_xyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = float(np.clip(R[0, 2], -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(np.cos(pitch)) > 1.0e-6:
        roll = np.arctan2(-R[1, 2], R[2, 2])
        yaw = np.arctan2(-R[0, 1], R[0, 0])
    else:
        roll = np.arctan2(R[2, 1], R[1, 1])
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def axis_angle_to_rotmat(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )


def map_body_vector_to_policy_frame(vec_unitree_b: np.ndarray, policy_body_frame: str) -> np.ndarray:
    vec_unitree_b = np.asarray(vec_unitree_b, dtype=np.float64).reshape(3)
    if str(policy_body_frame).lower() == "isaaclab":
        return np.array([-vec_unitree_b[2], vec_unitree_b[1], vec_unitree_b[0]], dtype=np.float64)
    return vec_unitree_b.copy()


def format_vec3(values: np.ndarray) -> str:
    return np.array2string(np.round(np.asarray(values, dtype=np.float64), 3), precision=3, separator=",")


def _parse_xyz_triplet(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text in (None, ""):
        return np.array(default, dtype=np.float64)
    parts = [float(part) for part in str(text).split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 values, got {parts}")
    return np.array(parts, dtype=np.float64)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(f"PyYAML import failed: {_YAML_IMPORT_ERROR}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in '{path}', got {type(raw).__name__}")
    return raw


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy_rad: np.ndarray
    axis_xyz: np.ndarray


@dataclass(frozen=True)
class CameraPoseDebugState:
    available: bool
    dynamic_enabled: bool
    parent_frame: str
    message: str
    camera_translation_body_m: np.ndarray
    camera_rpy_body_deg: np.ndarray
    waist_joint_names: tuple[str, ...]
    waist_angles_rad: np.ndarray
    chain_description: str

    @staticmethod
    def zeros(
        *,
        dynamic_enabled: bool,
        parent_frame: str,
        message: str,
        waist_joint_names: Sequence[str],
    ) -> "CameraPoseDebugState":
        return CameraPoseDebugState(
            available=False,
            dynamic_enabled=dynamic_enabled,
            parent_frame=parent_frame,
            message=message,
            camera_translation_body_m=np.zeros(3, dtype=np.float64),
            camera_rpy_body_deg=np.zeros(3, dtype=np.float64),
            waist_joint_names=tuple(str(name) for name in waist_joint_names),
            waist_angles_rad=np.zeros(len(tuple(waist_joint_names)), dtype=np.float64),
            chain_description="",
        )


@dataclass(frozen=True)
class AprilTagCalibrationBundle:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    parent_frame: str
    static_translation_m: np.ndarray
    static_quat_wxyz: np.ndarray
    static_rpy_deg: np.ndarray | None
    T_parent_camera: np.ndarray
    T_tag_to_object: np.ndarray
    tag_size_m: float
    tag_family: str
    target_tag_id: int | None


@dataclass(frozen=True)
class AprilTagDetectionSnapshot:
    initialized: bool
    valid: bool
    tag_visible: bool
    rel_pos_b: np.ndarray
    rel_lin_vel_b: np.ndarray
    frame_rate_hz: float
    status: str
    message: str
    last_frame_timestamp_s: float | None
    last_valid_timestamp_s: float | None
    last_frame_age_s: float | None
    time_since_last_valid_s: float | None

    @staticmethod
    def zeros(
        *,
        initialized: bool,
        status: str,
        message: str,
        frame_rate_hz: float = 0.0,
        last_frame_timestamp_s: float | None = None,
        last_valid_timestamp_s: float | None = None,
        last_frame_age_s: float | None = None,
        time_since_last_valid_s: float | None = None,
    ) -> "AprilTagDetectionSnapshot":
        return AprilTagDetectionSnapshot(
            initialized=initialized,
            valid=False,
            tag_visible=False,
            rel_pos_b=np.zeros(3, dtype=np.float64),
            rel_lin_vel_b=np.zeros(3, dtype=np.float64),
            frame_rate_hz=float(frame_rate_hz),
            status=status,
            message=message,
            last_frame_timestamp_s=last_frame_timestamp_s,
            last_valid_timestamp_s=last_valid_timestamp_s,
            last_frame_age_s=last_frame_age_s,
            time_since_last_valid_s=time_since_last_valid_s,
        )


def load_apriltag_calibration_bundle(
    intrinsics_yaml: Path,
    extrinsics_yaml: Path,
    tag_yaml: Path,
    *,
    extrinsics_parent_frame_override: str | None = None,
) -> AprilTagCalibrationBundle:
    if make_T is None:
        raise RuntimeError(f"AprilTag estimator import failed: {_ESTIMATOR_IMPORT_ERROR}")

    intrinsics = _load_yaml_mapping(intrinsics_yaml)
    extrinsics = _load_yaml_mapping(extrinsics_yaml)
    tag_cfg = _load_yaml_mapping(tag_yaml)

    camera_matrix = np.array(intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)

    parent_frame = str(
        extrinsics_parent_frame_override
        if extrinsics_parent_frame_override not in (None, "")
        else extrinsics.get("parent_frame", "body")
    ).lower()
    translation_m = np.array(extrinsics["translation_m"], dtype=np.float64)

    quat_raw = extrinsics.get("quat_wxyz")
    rpy_raw = extrinsics.get("rpy_deg")
    if quat_raw not in (None, ""):
        quat_wxyz = normalize_quat_wxyz(np.array(quat_raw, dtype=np.float64))
        R_parent_camera = quat_to_rotmat_wxyz(quat_wxyz)
        rpy_deg = None if rpy_raw in (None, "") else np.array(rpy_raw, dtype=np.float64)
    elif rpy_raw not in (None, ""):
        rpy_deg = np.array(rpy_raw, dtype=np.float64)
        R_parent_camera = rpy_to_rotmat_xyz(*np.deg2rad(rpy_deg))
        quat_wxyz = rotmat_to_quat_wxyz(R_parent_camera)
    else:
        raise ValueError(
            f"Extrinsics YAML '{extrinsics_yaml}' must provide either quat_wxyz or rpy_deg."
        )
    T_parent_camera = make_T(R_parent_camera, translation_m)

    tag_center_in_object = np.array(tag_cfg["tag_center_in_box_m"], dtype=np.float64)
    tag_rpy_in_object = np.deg2rad(np.array(tag_cfg["tag_rpy_in_box_deg"], dtype=np.float64))
    T_tag_to_object = make_T(
        rpy_to_rotmat_xyz(tag_rpy_in_object[0], tag_rpy_in_object[1], tag_rpy_in_object[2]),
        tag_center_in_object,
    )

    target_tag_id_raw = tag_cfg.get("target_tag_id")
    target_tag_id = None if target_tag_id_raw in (None, "") else int(target_tag_id_raw)

    return AprilTagCalibrationBundle(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        parent_frame=parent_frame,
        static_translation_m=translation_m,
        static_quat_wxyz=quat_wxyz,
        static_rpy_deg=rpy_deg,
        T_parent_camera=T_parent_camera,
        T_tag_to_object=T_tag_to_object,
        tag_size_m=float(tag_cfg["tag_size_m"]),
        tag_family=str(tag_cfg.get("tag_family", "36h11")),
        target_tag_id=target_tag_id,
    )


def _parse_urdf_joint_specs(urdf_path: Path) -> dict[str, UrdfJointSpec]:
    root = ET.parse(urdf_path).getroot()
    joint_specs: dict[str, UrdfJointSpec] = {}
    for joint_elem in root.findall("joint"):
        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            raise ValueError(f"URDF joint '{name}' is missing parent/child link metadata")
        joint_specs[name] = UrdfJointSpec(
            name=name,
            joint_type=joint_type,
            parent_link=parent_elem.attrib["link"],
            child_link=child_elem.attrib["link"],
            origin_xyz=_parse_xyz_triplet(origin_elem.attrib.get("xyz") if origin_elem is not None else None, [0.0, 0.0, 0.0]),
            origin_rpy_rad=np.deg2rad(np.zeros(3, dtype=np.float64))
            if origin_elem is None or origin_elem.attrib.get("rpy") in (None, "")
            else _parse_xyz_triplet(origin_elem.attrib.get("rpy"), [0.0, 0.0, 0.0]),
            axis_xyz=_parse_xyz_triplet(axis_elem.attrib.get("xyz") if axis_elem is not None else None, [0.0, 0.0, 0.0]),
        )
    return joint_specs


class BodyToCameraKinematics:
    BODY_PARENT_ALIASES = {"body", "base", "pelvis", "root"}

    def __init__(
        self,
        calibration: AprilTagCalibrationBundle,
        *,
        dynamic_body_to_camera: bool,
        body_to_torso_urdf: Path | None,
        body_link_name: str | None,
        torso_link_name: str,
        waist_joint_names: Sequence[str],
    ) -> None:
        self.parent_frame = str(calibration.parent_frame).lower()
        self.dynamic_enabled = bool(dynamic_body_to_camera and self.parent_frame == "torso")
        self.body_to_torso_urdf = body_to_torso_urdf
        self.body_link_name = None if body_link_name in (None, "") else str(body_link_name)
        self.torso_link_name = str(torso_link_name)
        self.waist_joint_names = tuple(str(name) for name in waist_joint_names)
        self.T_parent_camera = calibration.T_parent_camera.copy()
        self.chain_specs: list[UrdfJointSpec] = []
        self.chain_description = ""
        self.initialization_message = "fixed body-to-camera transform"
        self.available = True

        if self.parent_frame in self.BODY_PARENT_ALIASES:
            self.parent_frame = "body"
            self.dynamic_enabled = False
            self.initialization_message = "using fixed body-frame camera extrinsics"
            return

        if self.parent_frame != "torso":
            self.available = False
            self.initialization_message = (
                f"unsupported extrinsics parent_frame '{self.parent_frame}'; expected body or torso"
            )
            return

        if not dynamic_body_to_camera:
            self.available = False
            self.initialization_message = (
                "camera parent_frame is torso but dynamic_body_to_camera is false; "
                "refusing to treat torso-mounted camera as fixed body camera"
            )
            return

        if body_to_torso_urdf is None or not body_to_torso_urdf.exists():
            self.available = False
            self.initialization_message = (
                f"body_to_torso_urdf is missing or does not exist: {body_to_torso_urdf}"
            )
            return

        try:
            joint_specs = _parse_urdf_joint_specs(body_to_torso_urdf)
            self.chain_specs = [joint_specs[name] for name in self.waist_joint_names]
        except Exception as exc:
            self.available = False
            self.initialization_message = f"failed to parse URDF waist chain: {exc}"
            return

        try:
            self._validate_chain()
        except Exception as exc:
            self.available = False
            self.initialization_message = f"invalid URDF waist chain: {exc}"
            return

        segments = [self.chain_specs[0].parent_link]
        for joint in self.chain_specs:
            segments.append(f"--{joint.name}--> {joint.child_link}")
        self.chain_description = " ".join(segments)
        self.initialization_message = "dynamic torso-mounted camera FK enabled"

    def _validate_chain(self) -> None:
        if not self.chain_specs:
            raise ValueError("waist chain is empty")
        for index, joint in enumerate(self.chain_specs):
            if joint.joint_type not in {"revolute", "continuous", "fixed"}:
                raise ValueError(
                    f"joint '{joint.name}' has unsupported type '{joint.joint_type}'"
                )
            if index == 0:
                if self.body_link_name is None:
                    self.body_link_name = joint.parent_link
                elif joint.parent_link != self.body_link_name:
                    raise ValueError(
                        f"first waist joint parent is '{joint.parent_link}', expected '{self.body_link_name}'"
                    )
            else:
                prev = self.chain_specs[index - 1]
                if joint.parent_link != prev.child_link:
                    raise ValueError(
                        f"joint '{joint.name}' parent '{joint.parent_link}' does not follow "
                        f"previous child '{prev.child_link}'"
                    )
        if self.chain_specs[-1].child_link != self.torso_link_name:
            raise ValueError(
                f"final waist joint child is '{self.chain_specs[-1].child_link}', "
                f"expected torso link '{self.torso_link_name}'"
            )

    def _joint_transform(self, joint: UrdfJointSpec, q_rad: float) -> np.ndarray:
        T_origin = make_T(rpy_to_rotmat_xyz(*joint.origin_rpy_rad), joint.origin_xyz)
        if joint.joint_type in {"revolute", "continuous"}:
            return T_origin @ make_T(axis_angle_to_rotmat(joint.axis_xyz, q_rad), np.zeros(3, dtype=np.float64))
        return T_origin

    def _waist_angles(self, joint_positions_by_name: dict[str, float]) -> tuple[np.ndarray, str | None]:
        waist_angles = np.zeros(len(self.waist_joint_names), dtype=np.float64)
        for index, joint_name in enumerate(self.waist_joint_names):
            if joint_name not in joint_positions_by_name:
                return waist_angles, f"missing joint position for '{joint_name}'"
            waist_angles[index] = float(joint_positions_by_name[joint_name])
        return waist_angles, None

    def compute(
        self,
        joint_positions_by_name: dict[str, float],
    ) -> tuple[np.ndarray | None, CameraPoseDebugState]:
        waist_angles, missing_joint_message = self._waist_angles(joint_positions_by_name)
        if self.parent_frame == "body":
            debug_state = CameraPoseDebugState(
                available=True,
                dynamic_enabled=False,
                parent_frame="body",
                message=self.initialization_message,
                camera_translation_body_m=self.T_parent_camera[:3, 3].copy(),
                camera_rpy_body_deg=rotmat_to_rpy_xyz(self.T_parent_camera[:3, :3]),
                waist_joint_names=self.waist_joint_names,
                waist_angles_rad=waist_angles,
                chain_description=self.chain_description,
            )
            return self.T_parent_camera.copy(), debug_state

        if not self.available:
            return None, CameraPoseDebugState.zeros(
                dynamic_enabled=self.dynamic_enabled,
                parent_frame=self.parent_frame,
                message=self.initialization_message,
                waist_joint_names=self.waist_joint_names,
            )

        if missing_joint_message is not None:
            return None, CameraPoseDebugState.zeros(
                dynamic_enabled=True,
                parent_frame=self.parent_frame,
                message=missing_joint_message,
                waist_joint_names=self.waist_joint_names,
            )

        T_body_torso = np.eye(4, dtype=np.float64)
        for joint, q_rad in zip(self.chain_specs, waist_angles):
            T_body_torso = T_body_torso @ self._joint_transform(joint, q_rad)
        T_body_camera = T_body_torso @ self.T_parent_camera
        debug_state = CameraPoseDebugState(
            available=True,
            dynamic_enabled=True,
            parent_frame=self.parent_frame,
            message="dynamic torso FK active",
            camera_translation_body_m=T_body_camera[:3, 3].copy(),
            camera_rpy_body_deg=rotmat_to_rpy_xyz(T_body_camera[:3, :3]),
            waist_joint_names=self.waist_joint_names,
            waist_angles_rad=waist_angles,
            chain_description=self.chain_description,
        )
        return T_body_camera, debug_state

    def startup_summary_lines(self) -> list[str]:
        static_rpy = (
            "n/a (quat_wxyz preferred)"
            if self.T_parent_camera is None
            else format_vec3(rotmat_to_rpy_xyz(self.T_parent_camera[:3, :3]))
        )
        lines = [
            f"[G1][apriltag_zmq] camera extrinsics parent_frame : {self.parent_frame}",
            f"[G1][apriltag_zmq] static parent->camera xyz [m]  : {format_vec3(self.T_parent_camera[:3, 3])}",
            f"[G1][apriltag_zmq] static parent->camera rpy [deg]: {static_rpy}",
            f"[G1][apriltag_zmq] dynamic torso FK enabled       : {self.dynamic_enabled}",
            f"[G1][apriltag_zmq] waist joint names             : {list(self.waist_joint_names)}",
        ]
        if self.body_to_torso_urdf is not None:
            lines.append(f"[G1][apriltag_zmq] body->torso URDF            : {self.body_to_torso_urdf}")
        if self.body_link_name is not None:
            lines.append(f"[G1][apriltag_zmq] body link name              : {self.body_link_name}")
        lines.append(f"[G1][apriltag_zmq] torso link name             : {self.torso_link_name}")
        if self.chain_description:
            lines.append(f"[G1][apriltag_zmq] body->torso chain           : {self.chain_description}")
        lines.append(f"[G1][apriltag_zmq] camera FK init              : {self.initialization_message}")
        T_body_camera_zero, debug_state_zero = self.compute(
            {joint_name: 0.0 for joint_name in self.waist_joint_names}
        )
        if T_body_camera_zero is not None and debug_state_zero.available:
            lines.append(
                f"[G1][apriltag_zmq] initial T_body_camera xyz [m]: "
                f"{format_vec3(debug_state_zero.camera_translation_body_m)}"
            )
            lines.append(
                f"[G1][apriltag_zmq] initial T_body_camera rpy[deg]: "
                f"{format_vec3(debug_state_zero.camera_rpy_body_deg)}"
            )
        else:
            lines.append(
                f"[G1][apriltag_zmq] initial T_body_camera       : unavailable ({debug_state_zero.message})"
            )
        lines.append(
            "[G1][apriltag_zmq] camera frame note            : "
            "solvePnP assumes this extrinsics frame matches the incoming image camera frame; "
            "verify optical-vs-USD-frame alignment manually."
        )
        return lines


class AprilTagZmqReceiver:
    def __init__(
        self,
        *,
        server_address: str,
        port: int,
        intrinsics_yaml: Path | None,
        extrinsics_yaml: Path | None,
        tag_yaml: Path | None,
        policy_body_frame: str,
        stale_timeout_s: float,
        min_valid_detections: int,
        position_filter_alpha: float,
        velocity_filter_alpha: float,
        angular_velocity_filter_alpha: float,
        status_print_interval_s: float,
        controlled_joint_names: Sequence[str] | None = None,
        extrinsics_parent_frame: str | None = None,
        dynamic_body_to_camera: bool = False,
        body_to_torso_urdf: Path | None = None,
        body_link_name: str | None = None,
        torso_link_name: str = "torso_link",
        waist_joint_names: Sequence[str] = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
        image_show: bool = False,
        emit_status_logs: bool = True,
        source_name: str = "apriltag_zmq",
    ) -> None:
        self.server_address = str(server_address)
        self.port = int(port)
        self.socket_url = f"tcp://{self.server_address}:{self.port}"
        self.intrinsics_yaml = intrinsics_yaml
        self.extrinsics_yaml = extrinsics_yaml
        self.tag_yaml = tag_yaml
        self.policy_body_frame = str(policy_body_frame)
        self.stale_timeout_s = float(stale_timeout_s)
        self.min_valid_detections = max(int(min_valid_detections), 1)
        self.position_filter_alpha = float(position_filter_alpha)
        self.velocity_filter_alpha = float(velocity_filter_alpha)
        self.angular_velocity_filter_alpha = float(angular_velocity_filter_alpha)
        self.status_print_interval_s = max(float(status_print_interval_s), 0.0)
        self.extrinsics_parent_frame_override = (
            None if extrinsics_parent_frame in (None, "") else str(extrinsics_parent_frame)
        )
        self.dynamic_body_to_camera = bool(dynamic_body_to_camera)
        self.body_to_torso_urdf = body_to_torso_urdf
        self.body_link_name = None if body_link_name in (None, "") else str(body_link_name)
        self.torso_link_name = str(torso_link_name)
        self.waist_joint_names = tuple(str(name) for name in waist_joint_names)
        self.image_show = bool(image_show)
        self.emit_status_logs = bool(emit_status_logs)
        self.source_name = source_name

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._context = None
        self._socket = None
        self._window_name = "AprilTag ZMQ Receiver"

        self._initialized = False
        self._init_message = "not initialized"
        self._latest_status = "NO_FRAME"
        self._latest_message = "waiting for first frame"
        self._robot_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        controlled_joint_names = (
            tuple(str(name) for name in controlled_joint_names)
            if controlled_joint_names is not None
            else self.waist_joint_names
        )
        self._joint_name_to_index = {name: index for index, name in enumerate(controlled_joint_names)}
        self._joint_positions = np.zeros(len(controlled_joint_names), dtype=np.float64)
        self._fps_hz = 0.0
        self._last_frame_timestamp_s: float | None = None
        self._last_valid_timestamp_s: float | None = None
        self._consecutive_valid_detections = 0
        self._last_status_log_time = 0.0
        self._last_logged_status: str | None = None
        self._last_snapshot = AprilTagDetectionSnapshot.zeros(
            initialized=False,
            status="INIT_PENDING",
            message="initializing",
        )
        self._calibration: AprilTagCalibrationBundle | None = None
        self._camera_kinematics: BodyToCameraKinematics | None = None
        self._last_camera_debug_state = CameraPoseDebugState.zeros(
            dynamic_enabled=self.dynamic_body_to_camera,
            parent_frame=self.extrinsics_parent_frame_override or "body",
            message="camera FK not initialized",
            waist_joint_names=self.waist_joint_names,
        )
        self._startup_summary_lines: list[str] = []
        self._estimator = None

        self._initialize()
        if self._initialized:
            self._running = True
            self._thread = threading.Thread(target=self._run, name="apriltag_zmq_receiver", daemon=True)
            self._thread.start()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def init_message(self) -> str:
        return self._init_message

    def startup_summary_lines(self) -> list[str]:
        return list(self._startup_summary_lines)

    def update_robot_state(
        self,
        *,
        quat_wxyz: np.ndarray,
        q: np.ndarray | None = None,
        controlled_joint_names: Sequence[str] | None = None,
    ) -> None:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        with self._lock:
            self._robot_quat_wxyz = quat_wxyz.copy()
            if controlled_joint_names is not None:
                names = tuple(str(name) for name in controlled_joint_names)
                self._joint_name_to_index = {name: index for index, name in enumerate(names)}
                if q is not None:
                    self._joint_positions = np.asarray(q, dtype=np.float64).reshape(-1).copy()
                elif len(self._joint_positions) != len(names):
                    self._joint_positions = np.zeros(len(names), dtype=np.float64)
            elif q is not None:
                self._joint_positions = np.asarray(q, dtype=np.float64).reshape(-1).copy()

    def get_snapshot(self) -> AprilTagDetectionSnapshot:
        with self._lock:
            base_snapshot = self._clone_snapshot(self._last_snapshot)

        if not base_snapshot.initialized:
            return base_snapshot

        now = time.monotonic()
        if base_snapshot.last_frame_timestamp_s is None:
            return AprilTagDetectionSnapshot.zeros(
                initialized=True,
                status="NO_FRAME",
                message=self._latest_message,
                frame_rate_hz=base_snapshot.frame_rate_hz,
                last_valid_timestamp_s=base_snapshot.last_valid_timestamp_s,
                time_since_last_valid_s=self._age(now, base_snapshot.last_valid_timestamp_s),
            )

        last_frame_age_s = self._age(now, base_snapshot.last_frame_timestamp_s)
        last_valid_age_s = self._age(now, base_snapshot.last_valid_timestamp_s)
        if last_frame_age_s is not None and last_frame_age_s > self.stale_timeout_s:
            return AprilTagDetectionSnapshot.zeros(
                initialized=True,
                status="STALE",
                message=(
                    f"last frame age {last_frame_age_s:.3f}s exceeds stale timeout "
                    f"{self.stale_timeout_s:.3f}s"
                ),
                frame_rate_hz=base_snapshot.frame_rate_hz,
                last_frame_timestamp_s=base_snapshot.last_frame_timestamp_s,
                last_valid_timestamp_s=base_snapshot.last_valid_timestamp_s,
                last_frame_age_s=last_frame_age_s,
                time_since_last_valid_s=last_valid_age_s,
            )

        if base_snapshot.valid:
            return AprilTagDetectionSnapshot(
                initialized=True,
                valid=True,
                tag_visible=True,
                rel_pos_b=base_snapshot.rel_pos_b.copy(),
                rel_lin_vel_b=base_snapshot.rel_lin_vel_b.copy(),
                frame_rate_hz=base_snapshot.frame_rate_hz,
                status="FRESH",
                message=base_snapshot.message,
                last_frame_timestamp_s=base_snapshot.last_frame_timestamp_s,
                last_valid_timestamp_s=base_snapshot.last_valid_timestamp_s,
                last_frame_age_s=last_frame_age_s,
                time_since_last_valid_s=last_valid_age_s,
            )

        return AprilTagDetectionSnapshot.zeros(
            initialized=True,
            status=base_snapshot.status,
            message=base_snapshot.message,
            frame_rate_hz=base_snapshot.frame_rate_hz,
            last_frame_timestamp_s=base_snapshot.last_frame_timestamp_s,
            last_valid_timestamp_s=base_snapshot.last_valid_timestamp_s,
            last_frame_age_s=last_frame_age_s,
            time_since_last_valid_s=last_valid_age_s,
        )

    def get_camera_pose_debug(self) -> CameraPoseDebugState:
        with self._lock:
            joint_positions_by_name = self._joint_positions_by_name_locked()
        if self._camera_kinematics is None:
            return self._last_camera_debug_state
        _, debug_state = self._camera_kinematics.compute(joint_positions_by_name)
        with self._lock:
            self._last_camera_debug_state = debug_state
        return debug_state

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_socket()
        if self.image_show and cv2 is not None:
            try:
                cv2.destroyWindow(self._window_name)
            except Exception:
                pass

    def _initialize(self) -> None:
        if cv2 is None:
            self._set_init_failed(f"OpenCV import failed: {_CV2_IMPORT_ERROR}")
            return
        if zmq is None:
            self._set_init_failed(f"pyzmq import failed: {_ZMQ_IMPORT_ERROR}")
            return
        if yaml is None:
            self._set_init_failed(f"PyYAML import failed: {_YAML_IMPORT_ERROR}")
            return
        if AprilTagObjectStateEstimator is None or make_T is None:
            self._set_init_failed(f"AprilTag estimator import failed: {_ESTIMATOR_IMPORT_ERROR}")
            return
        if self.intrinsics_yaml is None or self.extrinsics_yaml is None or self.tag_yaml is None:
            self._set_init_failed("camera calibration YAML path is missing")
            return
        for path in (self.intrinsics_yaml, self.extrinsics_yaml, self.tag_yaml):
            if not path.exists():
                self._set_init_failed(f"camera calibration file does not exist: {path}")
                return

        try:
            self._calibration = load_apriltag_calibration_bundle(
                self.intrinsics_yaml,
                self.extrinsics_yaml,
                self.tag_yaml,
                extrinsics_parent_frame_override=self.extrinsics_parent_frame_override,
            )
            self._camera_kinematics = BodyToCameraKinematics(
                self._calibration,
                dynamic_body_to_camera=self.dynamic_body_to_camera,
                body_to_torso_urdf=self.body_to_torso_urdf,
                body_link_name=self.body_link_name,
                torso_link_name=self.torso_link_name,
                waist_joint_names=self.waist_joint_names,
            )
            self._last_camera_debug_state = self._camera_kinematics.compute(
                self._joint_positions_by_name_locked()
            )[1]
            self._startup_summary_lines = [
                f"[G1][apriltag_zmq] static parent->camera quat   : "
                f"{format_vec3(self._calibration.static_quat_wxyz)}"
            ]
            self._startup_summary_lines.extend(self._camera_kinematics.startup_summary_lines())
            self._estimator = AprilTagObjectStateEstimator(
                camera_matrix=self._calibration.camera_matrix,
                dist_coeffs=self._calibration.dist_coeffs,
                tag_size_m=self._calibration.tag_size_m,
                T_b_c=self._camera_kinematics.T_parent_camera,
                T_tag_to_object=self._calibration.T_tag_to_object,
                tag_family=self._calibration.tag_family,
                target_tag_id=self._calibration.target_tag_id,
                pos_alpha=self.position_filter_alpha,
                vel_alpha=self.velocity_filter_alpha,
                ang_alpha=self.angular_velocity_filter_alpha,
            )
        except Exception as exc:
            self._set_init_failed(f"AprilTag receiver initialization failed: {exc}")
            return

        self._initialized = True
        fk_state = "ok" if self._last_camera_debug_state.available else f"degraded ({self._last_camera_debug_state.message})"
        self._init_message = (
            f"connected to {self.socket_url} with tag_family={self._calibration.tag_family} "
            f"target_tag_id={self._calibration.target_tag_id} camera_fk={fk_state}"
        )
        self._last_snapshot = AprilTagDetectionSnapshot.zeros(
            initialized=True,
            status="NO_FRAME",
            message="initialized; waiting for first frame",
        )

    def _set_init_failed(self, message: str) -> None:
        self._initialized = False
        self._init_message = message
        self._last_snapshot = AprilTagDetectionSnapshot.zeros(
            initialized=False,
            status="INIT_FAILED",
            message=message,
        )
        self._latest_status = "INIT_FAILED"
        self._latest_message = message
        self._log_status(self._last_snapshot)

    def _run(self) -> None:
        assert zmq is not None
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.RCVTIMEO, 100)
            self._socket.connect(self.socket_url)
        except Exception as exc:
            with self._lock:
                self._last_snapshot = AprilTagDetectionSnapshot.zeros(
                    initialized=False,
                    status="INIT_FAILED",
                    message=f"ZMQ connect failed: {exc}",
                )
            self._initialized = False
            self._init_message = f"ZMQ connect failed: {exc}"
            self._log_status(self._last_snapshot)
            self._close_socket()
            return

        while self._running:
            try:
                message = self._socket.recv()
            except zmq.Again:
                self._maybe_mark_stale_or_waiting()
                continue
            except Exception as exc:
                err_time = time.monotonic()
                self._publish_zero_status(
                    "STALE",
                    f"ZMQ receive failed: {exc}",
                    now=err_time,
                    frame_timestamp_s=self._last_frame_timestamp_s,
                )
                time.sleep(0.05)
                continue

            now = time.monotonic()
            self._update_fps(now)
            frame = self._decode_frame(message)
            if frame is None:
                self._consecutive_valid_detections = 0
                self._publish_zero_status(
                    "DECODE_FAIL",
                    "received frame payload but JPEG decode failed",
                    now=now,
                )
                continue

            robot = self._build_robot_base_state()
            T_body_camera, camera_debug = self._compute_current_T_body_camera()
            if T_body_camera is None:
                self._consecutive_valid_detections = 0
                self._publish_zero_status("NO_FK", camera_debug.message, now=now)
                if self.image_show:
                    self._maybe_show_frame(frame, False)
                continue

            estimate = self._estimator.update(
                frame_bgr=frame,
                robot=robot,
                timestamp_s=now,
                T_b_c_override=T_body_camera,
            )

            if estimate.valid:
                self._consecutive_valid_detections += 1
                if self._consecutive_valid_detections >= self.min_valid_detections:
                    rel_pos_b = map_body_vector_to_policy_frame(estimate.rel_pos_b, self.policy_body_frame)
                    rel_lin_vel_b = map_body_vector_to_policy_frame(estimate.rel_lin_vel_b, self.policy_body_frame)
                    self._publish_snapshot(
                        AprilTagDetectionSnapshot(
                            initialized=True,
                            valid=True,
                            tag_visible=True,
                            rel_pos_b=rel_pos_b,
                            rel_lin_vel_b=rel_lin_vel_b,
                            frame_rate_hz=self._fps_hz,
                            status="FRESH",
                            message="tag visible",
                            last_frame_timestamp_s=now,
                            last_valid_timestamp_s=now,
                            last_frame_age_s=0.0,
                            time_since_last_valid_s=0.0,
                        )
                    )
                else:
                    self._publish_zero_status(
                        "WARMUP",
                        (
                            f"valid detections warming up "
                            f"({self._consecutive_valid_detections}/{self.min_valid_detections})"
                        ),
                        now=now,
                    )
            else:
                self._consecutive_valid_detections = 0
                self._publish_zero_status("NO_TAG", "no target tag detected in current frame", now=now)

            if self.image_show:
                self._maybe_show_frame(frame, estimate.valid)

        self._close_socket()

    def _build_robot_base_state(self) -> RobotBaseState:
        with self._lock:
            quat_wxyz = self._robot_quat_wxyz.copy()
        return RobotBaseState(
            pos_w=np.zeros(3, dtype=np.float64),
            quat_wxyz=quat_wxyz,
            lin_vel_w=np.zeros(3, dtype=np.float64),
            ang_vel_w=np.zeros(3, dtype=np.float64),
        )

    def _joint_positions_by_name_locked(self) -> dict[str, float]:
        return {
            name: float(self._joint_positions[index])
            for name, index in self._joint_name_to_index.items()
            if index < len(self._joint_positions)
        }

    def _compute_current_T_body_camera(self) -> tuple[np.ndarray | None, CameraPoseDebugState]:
        with self._lock:
            joint_positions_by_name = self._joint_positions_by_name_locked()
        if self._camera_kinematics is None:
            return None, self._last_camera_debug_state
        T_body_camera, debug_state = self._camera_kinematics.compute(joint_positions_by_name)
        with self._lock:
            self._last_camera_debug_state = debug_state
        return T_body_camera, debug_state

    def _publish_snapshot(self, snapshot: AprilTagDetectionSnapshot) -> None:
        with self._lock:
            self._last_snapshot = snapshot
            self._last_frame_timestamp_s = snapshot.last_frame_timestamp_s
            self._last_valid_timestamp_s = snapshot.last_valid_timestamp_s
            self._latest_status = snapshot.status
            self._latest_message = snapshot.message
        self._log_status(snapshot)

    def _publish_zero_status(
        self,
        status: str,
        message: str,
        *,
        now: float | None = None,
        frame_timestamp_s: float | None = None,
    ) -> None:
        timestamp_s = time.monotonic() if now is None else float(now)
        last_frame_timestamp_s = frame_timestamp_s
        if last_frame_timestamp_s is None and status != "NO_FRAME":
            last_frame_timestamp_s = timestamp_s
        snapshot = AprilTagDetectionSnapshot.zeros(
            initialized=self._initialized,
            status=status,
            message=message,
            frame_rate_hz=self._fps_hz,
            last_frame_timestamp_s=last_frame_timestamp_s,
            last_valid_timestamp_s=self._last_valid_timestamp_s,
            last_frame_age_s=self._age(timestamp_s, last_frame_timestamp_s),
            time_since_last_valid_s=self._age(timestamp_s, self._last_valid_timestamp_s),
        )
        self._publish_snapshot(snapshot)

    def _maybe_mark_stale_or_waiting(self) -> None:
        now = time.monotonic()
        if self._last_frame_timestamp_s is None:
            snapshot = AprilTagDetectionSnapshot.zeros(
                initialized=True,
                status="NO_FRAME",
                message="waiting for first frame",
                frame_rate_hz=self._fps_hz,
                last_valid_timestamp_s=self._last_valid_timestamp_s,
                time_since_last_valid_s=self._age(now, self._last_valid_timestamp_s),
            )
            self._publish_snapshot(snapshot)
            return

        last_frame_age_s = self._age(now, self._last_frame_timestamp_s)
        if last_frame_age_s is not None and last_frame_age_s > self.stale_timeout_s:
            snapshot = AprilTagDetectionSnapshot.zeros(
                initialized=True,
                status="STALE",
                message=(
                    f"last frame age {last_frame_age_s:.3f}s exceeds stale timeout "
                    f"{self.stale_timeout_s:.3f}s"
                ),
                frame_rate_hz=self._fps_hz,
                last_frame_timestamp_s=self._last_frame_timestamp_s,
                last_valid_timestamp_s=self._last_valid_timestamp_s,
                last_frame_age_s=last_frame_age_s,
                time_since_last_valid_s=self._age(now, self._last_valid_timestamp_s),
            )
            self._publish_snapshot(snapshot)

    def _decode_frame(self, message: bytes) -> np.ndarray | None:
        if cv2 is None:
            return None
        candidates = [message]
        if len(message) > 12:
            candidates.append(message[12:])
        for payload in candidates:
            np_img = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
        return None

    def _maybe_show_frame(self, frame_bgr: np.ndarray, tag_visible: bool) -> None:
        if cv2 is None:
            return
        frame_to_show = frame_bgr.copy()
        text = f"{self.source_name}: {'VISIBLE' if tag_visible else 'NO_TAG'}"
        color = (0, 255, 0) if tag_visible else (0, 0, 255)
        cv2.putText(frame_to_show, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(self._window_name, frame_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self._running = False

    def _update_fps(self, now: float) -> None:
        if self._last_frame_timestamp_s is not None:
            dt = now - self._last_frame_timestamp_s
            if dt > 1.0e-6:
                inst_fps = 1.0 / dt
                if self._fps_hz <= 0.0:
                    self._fps_hz = inst_fps
                else:
                    self._fps_hz = 0.85 * self._fps_hz + 0.15 * inst_fps
        self._last_frame_timestamp_s = now

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None

    def _log_status(self, snapshot: AprilTagDetectionSnapshot) -> None:
        if not self.emit_status_logs:
            return
        now = time.monotonic()
        should_print = False
        if snapshot.status != self._last_logged_status:
            should_print = True
        elif self.status_print_interval_s > 0.0 and (now - self._last_status_log_time) >= self.status_print_interval_s:
            should_print = True
        if not should_print:
            return

        camera_debug = self.get_camera_pose_debug()
        rel_pos = format_vec3(snapshot.rel_pos_b)
        rel_vel = format_vec3(snapshot.rel_lin_vel_b)
        waist_deg = np.rad2deg(camera_debug.waist_angles_rad)
        age_text = "n/a" if snapshot.time_since_last_valid_s is None else f"{snapshot.time_since_last_valid_s:.2f}s"
        camera_pos_text = (
            "n/a"
            if not camera_debug.available
            else format_vec3(camera_debug.camera_translation_body_m)
        )
        print(
            f"[G1][{self.source_name}] status={snapshot.status} "
            f"tag_visible={int(snapshot.tag_visible)} "
            f"fps={snapshot.frame_rate_hz:.1f} "
            f"waist_deg={format_vec3(waist_deg)} "
            f"camera_pos_b={camera_pos_text} "
            f"rel_pos_b={rel_pos} "
            f"rel_lin_vel_b={rel_vel} "
            f"last_valid_age={age_text} "
            f"msg={snapshot.message if snapshot.message else camera_debug.message}"
        )
        self._last_logged_status = snapshot.status
        self._last_status_log_time = now

    @staticmethod
    def _clone_snapshot(snapshot: AprilTagDetectionSnapshot) -> AprilTagDetectionSnapshot:
        return AprilTagDetectionSnapshot(
            initialized=bool(snapshot.initialized),
            valid=bool(snapshot.valid),
            tag_visible=bool(snapshot.tag_visible),
            rel_pos_b=np.asarray(snapshot.rel_pos_b, dtype=np.float64).copy(),
            rel_lin_vel_b=np.asarray(snapshot.rel_lin_vel_b, dtype=np.float64).copy(),
            frame_rate_hz=float(snapshot.frame_rate_hz),
            status=str(snapshot.status),
            message=str(snapshot.message),
            last_frame_timestamp_s=snapshot.last_frame_timestamp_s,
            last_valid_timestamp_s=snapshot.last_valid_timestamp_s,
            last_frame_age_s=snapshot.last_frame_age_s,
            time_since_last_valid_s=snapshot.time_since_last_valid_s,
        )

    @staticmethod
    def _age(now: float, timestamp_s: float | None) -> float | None:
        if timestamp_s is None:
            return None
        return max(now - timestamp_s, 0.0)
