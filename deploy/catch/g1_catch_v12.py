#[/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch/g1_catch_v12.py]
import os
import carb
import numpy as np
import omni
import omni.appwindow

from isaacsim.examples.interactive.base_sample import BaseSample
from loco.g1_loco_policy import G1FlatTerrainPolicy


class G1LocoV5Deploy(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0
        self._world_settings["rendering_dt"] = 8.0 / 200.0

        self._base_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  
        self.g1 = None  
        self._physics_ready = False  

        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],       
            "NUMPAD_4": [0.0, 0.5, 0.0], "LEFT": [0.0, 0.5, 0.0],     
            "NUMPAD_6": [0.0, -0.5, 0.0], "RIGHT": [0.0, -0.5, 0.0],  
            "Q": [0.0, 0.0, 1.0], "E": [0.0, 0.0, -1.0],              
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
            position=np.array([0.0, 0.0, 0.79], dtype=np.float32),  
        )

        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), self._timeline_timer_callback_fn
        )

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._physics_ready = False

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._physics_ready = False
        self._base_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)  

        if self.g1 is not None:  
            self.g1.reset_policy_state()  

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    def _apply_trained_default_state(self):  
        self.g1.robot.set_default_state(position=self.g1.default_root_pos, orientation=self.g1.default_root_rot)
        self.g1.robot.set_joints_default_state(positions=self.g1.default_pos, velocities=self.g1.default_vel)
        self.g1.post_reset()

        self.g1.robot.set_world_pose(position=self.g1.default_root_pos, orientation=self.g1.default_root_rot)
        self.g1.robot.set_linear_velocity(self.g1.default_root_lin_vel)
        self.g1.robot.set_angular_velocity(self.g1.default_root_ang_vel)
        self.g1.robot.set_joint_positions(self.g1.default_pos)
        self.g1.robot.set_joint_velocities(self.g1.default_vel)
        self.g1.reset_policy_state()

    def on_physics_step(self, step_size) -> None:
        if not self._physics_ready:
            self.g1.initialize()
            self._apply_trained_default_state()   
            self._physics_ready = True
            
            # 🚨 불필요한 코드 제거: policy_controller에서 이미 세팅 완료됨
            return

        # 🚨 [가장 중요한 핵심 수정 2] 삭제: 매 스텝 Gain 주입을 지워야 물리 솔버가 작동합니다!
        # self.g1.robot._articulation_view.set_gains(self.g1.stiffness, self.g1.damping) (이 줄 삭제)

        self.g1.forward(step_size, self._base_command)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name], dtype=np.float32)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name], dtype=np.float32)
        # =================================================================
        # 🚨 [추가된 방탄 코드] 속도 무한 증식 방지 (Command Clipping)
        # env.yaml에서 학습한 커맨드 범위(X: 0~1, Y:-0.5~0.5, Z:-1~1)로 잘라냅니다.
        # =================================================================
        self._base_command[0] = np.clip(self._base_command[0], -0.5, 1.0)  # X축 (후진 -0.5 ~ 전진 1.0)
        self._base_command[1] = np.clip(self._base_command[1], -0.5, 0.5)  # Y축 (좌우 게걸음)
        self._base_command[2] = np.clip(self._base_command[2], -1.0, 1.0)  # Z축 (제자리 회전)

        # 부동소수점 오차로 인해 미세하게 떨리는 것을 방지 (0에 가까우면 완벽한 0으로 고정)
        self._base_command = np.where(np.abs(self._base_command) < 0.01, 0.0, self._base_command)
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.g1 is not None:  
            self._physics_ready = False
            self.g1.reset_policy_state()  

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_sub_keyboard"): 
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)  
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")
