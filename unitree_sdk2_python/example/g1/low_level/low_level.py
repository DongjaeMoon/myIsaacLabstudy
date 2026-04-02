import time
import sys
import os

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

G1_NUM_MOTOR = 29

Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]

Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]

class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleRoll = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleRoll = 11
    WaistYaw = 12
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26

class Mode:
    PR = 0  
    AB = 1  

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.002  # 500Hz
        self.duration_ = 3.0      # 3초 대기
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()

    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        # 내부 균형 제어 모드를 강제로 해제
        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        # 프로그램을 처음 시작할 때의 관절 각도를 안전한 초기값으로 영구 기록
        if not hasattr(self, 'initial_q'):
            self.initial_q = [self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)]

        # 기본적으로 모든 관절을 초기 자세로 빳빳하게 고정
        self.low_cmd.mode_pr = Mode.PR
        self.low_cmd.mode_machine = self.mode_machine_
        for i in range(G1_NUM_MOTOR):
            self.low_cmd.motor_cmd[i].mode = 1 
            self.low_cmd.motor_cmd[i].q = self.initial_q[i]
            self.low_cmd.motor_cmd[i].dq = 0. 
            self.low_cmd.motor_cmd[i].tau = 0. 
            self.low_cmd.motor_cmd[i].kp = Kp[i] 
            self.low_cmd.motor_cmd[i].kd = Kd[i]

        if self.time_ < self.duration_ :
            # [Stage 1] 0 ~ 3초: 시작 자세 그대로 빳빳하게 굳어서 대기
            pass

        else :
            # [Stage 2] 3초 이후: 어깨를 옆으로 크게 들고 팔꿈치를 굽혀 하체와의 충돌을 원천 차단
            t = self.time_ - self.duration_
            
            # 주파수 (1초에 0.5번 왕복 = 2초 주기)
            freq = 0.5       
            
            # 1. 어깨 벌리기 진폭: 0.6 (1.0 - cos 파형을 거치면 최대 1.2 라디안, 약 68도까지 크게 벌어짐)
            roll_amp = 0.6 
            roll_wave = roll_amp * (1.0 - np.cos(2.0 * np.pi * freq * t))

            # 2. 팔꿈치 굽히기 진폭: 0.5 (최대 1.0 라디안, 약 57도까지 굽어지며 손이 몸통에서 멀어짐)
            elbow_amp = 0.5
            elbow_wave = elbow_amp * (1.0 - np.cos(2.0 * np.pi * freq * t))

            # 어깨 (옆으로 들어올리기)
            self.low_cmd.motor_cmd[G1JointIndex.LeftShoulderRoll].q = self.initial_q[G1JointIndex.LeftShoulderRoll] + roll_wave
            self.low_cmd.motor_cmd[G1JointIndex.RightShoulderRoll].q = self.initial_q[G1JointIndex.RightShoulderRoll] - roll_wave

            # 팔꿈치 (안으로 굽혀서 위로 올리기)
            self.low_cmd.motor_cmd[G1JointIndex.LeftElbow].q = self.initial_q[G1JointIndex.LeftElbow] + elbow_wave
            self.low_cmd.motor_cmd[G1JointIndex.RightElbow].q = self.initial_q[G1JointIndex.RightElbow] + elbow_wave

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

if __name__ == '__main__':
    print("==========================================================")
    print(" 1. 로봇이 크레인에 단단히 매달려 있는지 확인하세요.")
    print(" 2. 초기 자세에서 팔을 살짝 바깥쪽으로 빼준 상태로 세팅하세요.")
    print("==========================================================")
    input("모두 확인하셨다면 Enter 키를 눌러 넒은 범위의 안전 테스트를 시작합니다...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    try:
        while True:        
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[안전 종료] 통신 스레드를 포함하여 모든 프로세스를 즉시 종료합니다.")
        os._exit(0)
