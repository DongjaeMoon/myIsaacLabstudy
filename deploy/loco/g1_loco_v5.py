import os
import carb
import numpy as np
import omni
import omni.appwindow

from isaacsim.examples.interactive.base_sample import BaseSample
from loco.g1 import G1FlatTerrainPolicy

class G1LocoV5Deploy(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0
        self._world_settings["rendering_dt"] = 8.0 / 200.0
        self._base_command = [0.0, 0.0, 0.0]
        
        # 키보드 매핑 (조작키)
        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],       # 전진
            "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],   # 후진
            "NUMPAD_4": [0.0, 0.5, 0.0], "LEFT": [0.0, 0.5, 0.0],     # 좌로 걷기
            "NUMPAD_6": [0.0, -0.5, 0.0], "RIGHT": [0.0, -0.5, 0.0],  # 우로 걷기
            "Q": [0.0, 0.0, 1.0], "E": [0.0, 0.0, -1.0]               # 제자리 회전
        }

    def setup_scene(self) -> None:
        self.get_world().scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        )
        
        ROOT_DIR = "/home/idim5080-2/mdj/myIsaacLabstudy"
        ROBOT_USD_PATH = os.path.join(ROOT_DIR, "UROP/UROP_g1_loco_v5/g1_29dof_full_collider_flattened.usd")
        POLICY_DIR = os.path.join(ROOT_DIR, "logs/rsl_rl/UROP_g1_loco_v5/2026-03-06_16-10-35/exported")

        self.g1 = G1FlatTerrainPolicy(
            prim_path="/World/G1",
            name="G1",
            usd_path=ROBOT_USD_PATH,
            policy_path=POLICY_DIR,
            position=np.array([0.0, 0.0, 0.79]), 
        )
        
        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)
        self._physics_ready = False
        
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    def on_physics_step(self, step_size) -> None:
        if self._physics_ready:
            self.g1.forward(step_size, self._base_command)
        else:
            self._physics_ready = True
            self.g1.initialize()
            self.g1.post_reset()
            self.g1.robot.set_joints_default_state(self.g1.default_pos)
            
            # [핵심 수정!!] 물리 엔진 시작 첫 프레임에 로봇 관절을 default_pos로 순간이동 시킵니다.
            # 이 코드가 없으면 로봇이 0도(일자 다리)에서 시작해 어마어마한 토크 충격을 받고 폭발합니다.
            self.g1.robot.set_joint_positions(self.g1.default_pos)
            self.g1.robot.set_joint_velocities(self.g1.default_vel)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.g1:
            self._physics_ready = False
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")