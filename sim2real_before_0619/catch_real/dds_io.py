from __future__ import annotations

import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from .config_schema import CommunicationConfig


class DDSInterface:
    def __init__(self, communication: CommunicationConfig):
        self.communication = communication
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_ | None = None
        self.low_state_time = 0.0
        self.mode_machine = 0
        self.mode_machine_ready = False
        self.crc = CRC()
        self.command_slots = len(self.low_cmd.motor_cmd)
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.msc = None

    def init_motion_switcher(self) -> None:
        if not self.communication.release_high_level_mode:
            return

        try:
            self.msc = MotionSwitcherClient()
            self.msc.SetTimeout(2.0)
            self.msc.Init()
            _, result = self.msc.CheckMode()
            while result and result.get("name"):
                self.msc.ReleaseMode()
                _, result = self.msc.CheckMode()
                time.sleep(0.5)
        except Exception as exc:
            print(f"[G1] MotionSwitcher release skipped: {exc}")

    def init_channels(self) -> None:
        self.lowcmd_publisher = ChannelPublisher(self.communication.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(self.communication.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_) -> None:
        self.low_state = msg
        self.low_state_time = time.monotonic()
        self.mode_machine = int(getattr(msg, "mode_machine", 0))
        self.mode_machine_ready = True
