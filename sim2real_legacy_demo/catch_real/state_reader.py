from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config_schema import CatchRealConfig
from .math_utils import normalize_quat, quat_rotate_inverse


def get_any_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


class StateReader:
    def __init__(self, cfg: CatchRealConfig):
        self.cfg = cfg
        self._imu_quat_raw_is_xyzw: bool | None = None

    def read_motor_state(self, low_state) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros(self.cfg.robot.num_motors, dtype=np.float64)
        dq = np.zeros(self.cfg.robot.num_motors, dtype=np.float64)
        for slot, motor_index in enumerate(self.cfg.robot.motor_indices):
            motor_state = low_state.motor_state[motor_index]
            q[slot] = float(get_any_attr(motor_state, ["q"], 0.0))
            dq[slot] = float(get_any_attr(motor_state, ["dq"], 0.0))
        return q, dq

    def read_base_state(self, low_state) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        imu = low_state.imu_state
        q_raw = np.array(get_any_attr(imu, ["quaternion"], [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        quat_wxyz = self._to_wxyz(q_raw)
        gyro_unitree_b = np.array(get_any_attr(imu, ["gyroscope", "gyro", "omega"], [0.0, 0.0, 0.0]), dtype=np.float64)
        gravity_unitree_b = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        gravity_policy_b = self._map_body_vector(gravity_unitree_b)
        base_ang_vel_b = self._map_body_vector(gyro_unitree_b)
        return q_raw, quat_wxyz, gravity_policy_b, base_ang_vel_b

    def _detect_and_lock_imu_quat_order(self, q_raw: np.ndarray) -> None:
        order = self.cfg.robot.imu.quaternion_order.lower()
        if order in {"wxyz", "xyzw"}:
            if self._imu_quat_raw_is_xyzw is None:
                self._imu_quat_raw_is_xyzw = (order == "xyzw")
                print(f"[G1] IMU quaternion order fixed from config -> {order}")
            return

        if self._imu_quat_raw_is_xyzw is not None:
            return

        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        cand_wxyz = normalize_quat(q)
        cand_xyzw = normalize_quat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        g_ref = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if abs(q[3]) > 0.90 and abs(q[0]) < 0.90:
            self._imu_quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90 and abs(q[3]) < 0.90:
            self._imu_quat_raw_is_xyzw = False
        else:
            err_wxyz = np.linalg.norm(quat_rotate_inverse(cand_wxyz, g_ref) - g_ref)
            err_xyzw = np.linalg.norm(quat_rotate_inverse(cand_xyzw, g_ref) - g_ref)
            self._imu_quat_raw_is_xyzw = err_xyzw < err_wxyz
        chosen = "xyzw" if self._imu_quat_raw_is_xyzw else "wxyz"
        print(f"[G1] IMU quaternion order auto-locked -> {chosen} | raw={np.round(q, 5)}")

    def _to_wxyz(self, q_raw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float64).reshape(-1)
        self._detect_and_lock_imu_quat_order(q)
        if self._imu_quat_raw_is_xyzw:
            return normalize_quat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))
        return normalize_quat(q)

    def _map_body_vector(self, vec_unitree_b: np.ndarray) -> np.ndarray:
        if self.cfg.robot.imu.policy_body_frame.lower() == "isaaclab":
            return np.array([-vec_unitree_b[2], vec_unitree_b[1], vec_unitree_b[0]], dtype=np.float64)
        return np.asarray(vec_unitree_b, dtype=np.float64)
