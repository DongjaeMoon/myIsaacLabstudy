import sys
import time
import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


# =============================================================================
# USER-TUNABLE PARAMETERS
# =============================================================================

G1_NUM_MOTOR = 29
CONTROL_DT = 0.002          # 2 ms
BLEND_TIME = 1.0            # current pose -> WAYPOINT_0
SEGMENT_DURATIONS = [1.0, 1.0, 1.0]   # W0->W1, W1->W2, W2->W3
PRINT_EVERY_N = 250         # print motor states every N low-state callbacks

# Gains: length must be 29
KP = [
    60, 60, 60, 100, 40, 40,   # legs
    60, 60, 60, 100, 40, 40,   # legs
    60, 40, 40,                # waist
    40, 40, 40, 40, 40, 40, 40,  # left arm
    40, 40, 40, 40, 40, 40, 40   # right arm
]

KD = [
    1, 1, 1, 2, 1, 1,          # legs
    1, 1, 1, 2, 1, 1,          # legs
    1, 1, 1,                   # waist
    1, 1, 1, 1, 1, 1, 1,       # left arm
    1, 1, 1, 1, 1, 1, 1        # right arm
]

# 4 waypoint lists, each length = 29
# Replace these with your actual action values.
WAYPOINT_0 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0
]

WAYPOINT_1 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0
]

WAYPOINT_2 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    -0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0
]

WAYPOINT_3 = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0
]


# =============================================================================
# SAME JOINT INDEX ORDER AS UNITREE OFFICIAL EXAMPLE
# =============================================================================

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13
    WaistA = 13
    WaistPitch = 14
    WaistB = 14

    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


class Mode:
    PR = 0
    AB = 1


# =============================================================================
# SPLINE HELPERS
# =============================================================================

def validate_config():
    assert len(KP) == G1_NUM_MOTOR, f"KP must have length {G1_NUM_MOTOR}"
    assert len(KD) == G1_NUM_MOTOR, f"KD must have length {G1_NUM_MOTOR}"
    assert len(WAYPOINT_0) == G1_NUM_MOTOR
    assert len(WAYPOINT_1) == G1_NUM_MOTOR
    assert len(WAYPOINT_2) == G1_NUM_MOTOR
    assert len(WAYPOINT_3) == G1_NUM_MOTOR
    assert len(SEGMENT_DURATIONS) == 3
    assert all(d > 0.0 for d in SEGMENT_DURATIONS)


def compute_natural_cubic_second_derivatives(y, x):
    """
    Natural cubic spline second derivatives for 4 points.
    y: shape [4]
    x: shape [4]
    returns M: shape [4]
    """
    n = 4
    h = np.diff(x)  # [3]

    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    # natural boundary
    A[0, 0] = 1.0
    A[-1, -1] = 1.0

    # interior equations
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    M = np.linalg.solve(A, b)
    return M


def eval_natural_cubic(y, x, M, t):
    """
    Evaluate cubic spline position and velocity at time t.
    y: knot positions shape [4]
    x: knot times shape [4]
    M: second derivatives shape [4]
    """
    if t <= x[0]:
        return y[0], 0.0
    if t >= x[-1]:
        return y[-1], 0.0

    i = np.searchsorted(x, t) - 1
    i = max(0, min(i, len(x) - 2))

    xi, xi1 = x[i], x[i + 1]
    yi, yi1 = y[i], y[i + 1]
    Mi, Mi1 = M[i], M[i + 1]
    h = xi1 - xi
    a = (xi1 - t) / h
    b = (t - xi) / h

    q = (
        Mi * (a ** 3) * h * h / 6.0
        + Mi1 * (b ** 3) * h * h / 6.0
        + (yi - Mi * h * h / 6.0) * a
        + (yi1 - Mi1 * h * h / 6.0) * b
    )

    dq = (
        -Mi * (a ** 2) * h / 2.0
        + Mi1 * (b ** 2) * h / 2.0
        + (yi1 - yi) / h
        - (Mi1 - Mi) * h / 6.0
    )

    return q, dq


class JointSplineTrajectory:
    def __init__(self, waypoints, segment_durations):
        """
        waypoints: np.ndarray shape [4, 29]
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.segment_durations = np.asarray(segment_durations, dtype=np.float64)

        self.knot_times = np.array(
            [
                0.0,
                self.segment_durations[0],
                self.segment_durations[0] + self.segment_durations[1],
                self.segment_durations[0] + self.segment_durations[1] + self.segment_durations[2],
            ],
            dtype=np.float64,
        )
        self.total_time = self.knot_times[-1]

        self.second_derivatives = np.zeros_like(self.waypoints)
        for j in range(G1_NUM_MOTOR):
            self.second_derivatives[:, j] = compute_natural_cubic_second_derivatives(
                self.waypoints[:, j], self.knot_times
            )

    def sample(self, t):
        q = np.zeros(G1_NUM_MOTOR, dtype=np.float64)
        dq = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        for j in range(G1_NUM_MOTOR):
            q[j], dq[j] = eval_natural_cubic(
                self.waypoints[:, j],
                self.knot_times,
                self.second_derivatives[:, j],
                t,
            )
        return q, dq


# =============================================================================
# MAIN CONTROLLER
# =============================================================================

class G1SplineActionController:
    def __init__(self):
        self.control_dt = CONTROL_DT
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.mode_machine = 0
        self.update_mode_machine = False
        self.state_counter = 0
        self.control_time = 0.0
        self.started = False
        self.start_q = None

        self.crc = CRC()
        self.trajectory = JointSplineTrajectory(
            waypoints=np.stack([WAYPOINT_0, WAYPOINT_1, WAYPOINT_2, WAYPOINT_3], axis=0),
            segment_durations=SEGMENT_DURATIONS,
        )

    def init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result["name"]:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def start(self):
        self.control_thread = RecurrentThread(
            interval=self.control_dt,
            target=self.low_cmd_write,
            name="g1_spline_control",
        )

        while not self.update_mode_machine:
            time.sleep(0.1)

        self.control_thread.Start()

    def low_state_handler(self, msg: LowState_):
        self.low_state = msg

        if not self.update_mode_machine:
            self.mode_machine = self.low_state.mode_machine
            self.update_mode_machine = True

        self.state_counter += 1
        if self.state_counter % PRINT_EVERY_N == 0:
            self.state_counter = 0
            q_now = [self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            dq_now = [self.low_state.motor_state[i].dq for i in range(G1_NUM_MOTOR)]
            print("Current q:", np.round(q_now, 4).tolist())
            print("Current dq:", np.round(dq_now, 4).tolist())
            print("-" * 80)

    def get_current_q(self):
        return np.array([self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)], dtype=np.float64)

    def set_motor_commands(self, q_des, dq_des):
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine

        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].q = float(q_des[i])
            self.low_cmd.motor_cmd[i].dq = float(dq_des[i])
            self.low_cmd.motor_cmd[i].kp = float(KP[i])
            self.low_cmd.motor_cmd[i].kd = float(KD[i])

    def low_cmd_write(self):
        if self.low_state is None:
            return

        if not self.started:
            self.start_q = self.get_current_q()
            self.started = True
            self.control_time = 0.0
            print("Controller started. Captured initial joint state.")

        self.control_time += self.control_dt

        # Stage 1: blend from current measured q to WAYPOINT_0
        if self.control_time < BLEND_TIME:
            ratio = np.clip(self.control_time / BLEND_TIME, 0.0, 1.0)
            q_des = (1.0 - ratio) * self.start_q + ratio * np.array(WAYPOINT_0, dtype=np.float64)
            dq_des = np.zeros(G1_NUM_MOTOR, dtype=np.float64)

        # Stage 2: run 4-waypoint spline
        else:
            traj_t = self.control_time - BLEND_TIME
            q_des, dq_des = self.trajectory.sample(traj_t)

        self.set_motor_commands(q_des, dq_des)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


if __name__ == "__main__":
    validate_config()

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    controller = G1SplineActionController()
    controller.init()
    controller.start()

    while True:
        time.sleep(1)