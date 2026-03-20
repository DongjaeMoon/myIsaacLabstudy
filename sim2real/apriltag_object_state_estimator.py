# sim2real/apriltag_object_state_estimator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import cv2


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_apply(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    return quat_mul(quat_mul(q_wxyz, vq), quat_conj(q_wxyz))[1:4]


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q_wxyz), v)


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    q = normalize(np.asarray(q_wxyz, dtype=np.float64))
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return normalize(q)


def quat_to_rot6d(q_wxyz: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q_wxyz)
    return np.array([R[0, 0], R[1, 0], R[2, 0], R[0, 1], R[1, 1], R[2, 1]], dtype=np.float64)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def angvel_from_quats(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 1e-6:
        return np.zeros(3, dtype=np.float64)

    dq = quat_mul(q_curr, quat_conj(q_prev))
    dq = normalize(dq)

    angle = 2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0))
    if angle > np.pi:
        angle -= 2.0 * np.pi

    s = np.sqrt(max(1.0 - dq[0] * dq[0], 0.0))
    if s < 1e-6:
        axis = np.zeros(3, dtype=np.float64)
    else:
        axis = dq[1:4] / s

    return axis * (angle / dt)


@dataclass
class RobotBaseState:
    pos_w: np.ndarray           # (3,)
    quat_wxyz: np.ndarray       # body->world quaternion
    lin_vel_w: np.ndarray       # (3,)
    ang_vel_w: np.ndarray       # (3,)


@dataclass
class ObjectStateEstimate:
    valid: bool
    timestamp_s: float
    pos_w: np.ndarray
    quat_wxyz: np.ndarray
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray
    rel_pos_b: np.ndarray
    rel_rot6d_b: np.ndarray
    rel_lin_vel_b: np.ndarray
    rel_ang_vel_b: np.ndarray


class AprilTagObjectStateEstimator:
    """
    Assumptions:
      - camera is rigidly attached to the robot body frame
      - T_b_c = transform from robot body frame to camera frame
      - T_tag_to_object = transform from tag frame to object-center frame
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size_m: float,
        T_b_c: np.ndarray,
        T_tag_to_object: Optional[np.ndarray] = None,
        tag_family: str = "36h11",
        target_tag_id: Optional[int] = None,
        pos_alpha: float = 0.35,
        vel_alpha: float = 0.25,
        ang_alpha: float = 0.25,
    ) -> None:
        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.D = np.asarray(dist_coeffs, dtype=np.float64)
        self.tag_size_m = float(tag_size_m)
        self.T_b_c = np.asarray(T_b_c, dtype=np.float64)
        self.T_tag_to_object = np.eye(4, dtype=np.float64) if T_tag_to_object is None else np.asarray(T_tag_to_object, dtype=np.float64)
        self.target_tag_id = target_tag_id

        aruco_map = {
            "16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "36h11": cv2.aruco.DICT_APRILTAG_36h11,
        }
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_map[tag_family])
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())

        self.pos_alpha = float(pos_alpha)
        self.vel_alpha = float(vel_alpha)
        self.ang_alpha = float(ang_alpha)

        self._last_t: Optional[float] = None
        self._last_pos_w: Optional[np.ndarray] = None
        self._last_quat_wxyz: Optional[np.ndarray] = None
        self._last_lin_vel_w = np.zeros(3, dtype=np.float64)
        self._last_ang_vel_w = np.zeros(3, dtype=np.float64)

    def _detect_target(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        corners, ids, _ = self.detector.detectMarkers(frame_gray)
        if ids is None or len(ids) == 0:
            return None

        ids = ids.flatten().tolist()

        if self.target_tag_id is None:
            idx = 0
        else:
            if self.target_tag_id not in ids:
                return None
            idx = ids.index(self.target_tag_id)

        # OpenCV corner order: top-left, top-right, bottom-right, bottom-left
        return corners[idx].reshape(4, 2).astype(np.float64)

    def _estimate_T_c_tag(self, corners_2d: np.ndarray) -> Optional[np.ndarray]:
        half = self.tag_size_m / 2.0
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts,
            corners_2d,
            self.K,
            self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None

        R_c_tag, _ = cv2.Rodrigues(rvec)
        return make_T(R_c_tag, tvec.reshape(3))

    def update(
        self,
        frame_bgr: np.ndarray,
        robot: RobotBaseState,
        timestamp_s: Optional[float] = None,
    ) -> ObjectStateEstimate:
        t_now = time.time() if timestamp_s is None else float(timestamp_s)

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners = self._detect_target(frame_gray)
        if corners is None:
            return ObjectStateEstimate(
                valid=False,
                timestamp_s=t_now,
                pos_w=np.zeros(3, dtype=np.float64),
                quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                lin_vel_w=np.zeros(3, dtype=np.float64),
                ang_vel_w=np.zeros(3, dtype=np.float64),
                rel_pos_b=np.zeros(3, dtype=np.float64),
                rel_rot6d_b=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
                rel_lin_vel_b=np.zeros(3, dtype=np.float64),
                rel_ang_vel_b=np.zeros(3, dtype=np.float64),
            )

        T_c_tag = self._estimate_T_c_tag(corners)
        if T_c_tag is None:
            return ObjectStateEstimate(
                valid=False,
                timestamp_s=t_now,
                pos_w=np.zeros(3, dtype=np.float64),
                quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                lin_vel_w=np.zeros(3, dtype=np.float64),
                ang_vel_w=np.zeros(3, dtype=np.float64),
                rel_pos_b=np.zeros(3, dtype=np.float64),
                rel_rot6d_b=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
                rel_lin_vel_b=np.zeros(3, dtype=np.float64),
                rel_ang_vel_b=np.zeros(3, dtype=np.float64),
            )

        R_w_b = quat_to_rotmat(robot.quat_wxyz)
        T_w_b = make_T(R_w_b, robot.pos_w)
        T_w_c = T_w_b @ self.T_b_c
        T_w_o = T_w_c @ T_c_tag @ self.T_tag_to_object

        pos_w_raw = T_w_o[:3, 3].copy()
        quat_wxyz_raw = rotmat_to_quat_wxyz(T_w_o[:3, :3])

        if self._last_t is None:
            pos_w = pos_w_raw
            quat_wxyz = quat_wxyz_raw
            lin_vel_w = np.zeros(3, dtype=np.float64)
            ang_vel_w = np.zeros(3, dtype=np.float64)
        else:
            dt = max(t_now - self._last_t, 1e-6)

            pos_w = self.pos_alpha * pos_w_raw + (1.0 - self.pos_alpha) * self._last_pos_w
            quat_wxyz = quat_wxyz_raw  # orientation smoothing은 처음엔 생략

            lin_vel_raw = (pos_w - self._last_pos_w) / dt
            ang_vel_raw = angvel_from_quats(self._last_quat_wxyz, quat_wxyz, dt)

            lin_vel_w = self.vel_alpha * lin_vel_raw + (1.0 - self.vel_alpha) * self._last_lin_vel_w
            ang_vel_w = self.ang_alpha * ang_vel_raw + (1.0 - self.ang_alpha) * self._last_ang_vel_w

        rel_pos_b = quat_rotate_inverse(robot.quat_wxyz, pos_w - robot.pos_w)
        rel_lin_vel_b = quat_rotate_inverse(robot.quat_wxyz, lin_vel_w - robot.lin_vel_w)
        rel_ang_vel_b = quat_rotate_inverse(robot.quat_wxyz, ang_vel_w - robot.ang_vel_w)
        rel_quat_b = quat_mul(quat_conj(robot.quat_wxyz), quat_wxyz)
        rel_rot6d_b = quat_to_rot6d(rel_quat_b)

        self._last_t = t_now
        self._last_pos_w = pos_w.copy()
        self._last_quat_wxyz = quat_wxyz.copy()
        self._last_lin_vel_w = lin_vel_w.copy()
        self._last_ang_vel_w = ang_vel_w.copy()

        return ObjectStateEstimate(
            valid=True,
            timestamp_s=t_now,
            pos_w=pos_w,
            quat_wxyz=quat_wxyz,
            lin_vel_w=lin_vel_w,
            ang_vel_w=ang_vel_w,
            rel_pos_b=rel_pos_b,
            rel_rot6d_b=rel_rot6d_b,
            rel_lin_vel_b=rel_lin_vel_b,
            rel_ang_vel_b=rel_ang_vel_b,
        )
