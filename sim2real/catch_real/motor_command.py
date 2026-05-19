from __future__ import annotations

import numpy as np

from .config_schema import CatchRealConfig


class MotorCommandWriter:
    def __init__(self, cfg: CatchRealConfig, low_cmd, publisher, crc, command_slots: int):
        self.cfg = cfg
        self.low_cmd = low_cmd
        self.publisher = publisher
        self.crc = crc
        self.command_slots = command_slots

    def set_position_command(self, mode_machine: int, q_des: np.ndarray) -> None:
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = mode_machine

        for motor_index in range(self.command_slots):
            cmd = self.low_cmd.motor_cmd[motor_index]
            cmd.mode = 1
            cmd.tau = 0.0
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.kp = 0.0
            cmd.kd = 0.0

        for slot, motor_index in enumerate(self.cfg.robot.motor_indices):
            cmd = self.low_cmd.motor_cmd[motor_index]
            cmd.mode = 1
            cmd.tau = float(self.cfg.control.tau_ff)
            cmd.q = float(q_des[slot])
            cmd.dq = float(self.cfg.control.dq_des)
            cmd.kp = float(self.cfg.control.kp[slot])
            cmd.kd = float(self.cfg.control.kd[slot])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)

    def send_damping_command(self, mode_machine: int) -> None:
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = mode_machine
        for motor_index in range(self.command_slots):
            cmd = self.low_cmd.motor_cmd[motor_index]
            cmd.mode = 1
            cmd.tau = 0.0
            cmd.q = 0.0
            cmd.dq = 0.0
            cmd.kp = 0.0
            cmd.kd = float(self.cfg.safety.damping_kd)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
