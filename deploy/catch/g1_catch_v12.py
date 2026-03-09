# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch/g1_catch_v12.py] (연구실)
# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/catch/g1_catch_v12.py] (전산실)

import os
import carb
import numpy as np
import omni
import omni.appwindow

from omni.isaac.core.objects import DynamicCuboid
from isaacsim.examples.interactive.base_sample import BaseSample
from catch.g1_catch_policy import G1CatchPolicy

class G1CatchV12Deploy(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 100.0
        self._world_settings["rendering_dt"] = 2.0 / 100.0

        self.g1 = None  
        self.box = None
        self._physics_ready = False  
        self._gains_applied = False
        
        self._toss_signal = 0.0
        self._trigger_toss = False 

        self._input_keyboard_mapping = {
            "K": "TOSS"  
        }

    def setup_scene(self) -> None:
        self.get_world().scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0
        )

        pc_dir = "/home/dongjae/isaaclab/myIsaacLabstudy"
        lab_dir = "/home/idim5080-2/mdj/myIsaacLabstudy"
        
        if os.path.exists(pc_dir):
            ROOT_DIR = pc_dir
            print(f"[INFO] 전산실 컴퓨터 경로 인식됨: {ROOT_DIR}")
        else:
            ROOT_DIR = lab_dir
            print(f"[INFO] 연구실 컴퓨터 경로 인식됨: {ROOT_DIR}")

        ROBOT_USD_PATH = os.path.join(ROOT_DIR, "UROP/UROP_v12/usd/g1_29dof_full_collider_flattened.usd")
        POLICY_DIR = os.path.join(ROOT_DIR, "logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported")

        self.g1 = G1CatchPolicy(
            prim_path="/World/G1",
            name="G1",
            usd_path=ROBOT_USD_PATH,
            policy_path=POLICY_DIR,
            position=np.array([0.0, 0.0, 0.85], dtype=np.float32),  
        )

        self.box = DynamicCuboid(
            prim_path="/World/Object",
            name="box",
            position=np.array([1.5, 0.0, 0.5]), 
            scale=np.array([0.32, 0.24, 0.24]),
            color=np.array([0.0, 0.8, 0.0]),
            mass=3.0,
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
        
        self.box.initialize()
        self.g1.initialize()
        self._apply_trained_default_state()
        
        self._physics_ready = True
        self._gains_applied = False
        self._trigger_toss = False

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._toss_signal = 0.0
        self._trigger_toss = False
        self._gains_applied = False
        
        self.box.set_world_pose(position=np.array([1.5, 0.0, 0.5]))
        self.box.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        if self.g1 is not None:  
            self._apply_trained_default_state()
            self.g1.reset_policy_state()  

        self._physics_ready = True

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    def _apply_trained_default_state(self):  
        self.g1.robot.set_world_pose(position=self.g1.default_root_pos, orientation=self.g1.default_root_rot)
        self.g1.robot.set_linear_velocity(self.g1.default_root_lin_vel)
        self.g1.robot.set_angular_velocity(self.g1.default_root_ang_vel)
        self.g1.robot.set_joint_positions(self.g1.default_pos)
        self.g1.robot.set_joint_velocities(self.g1.default_vel)

    def on_physics_step(self, step_size) -> None:
        if not self._physics_ready or not self.get_world().is_playing():
            return

        # 🚨 [굳음 차단] 물리 엔진이 준비된 후, 한 번만 게인을 강제로 꽂아 넣습니다.
        if not self._gains_applied:
            success = self.g1.apply_pd_gains()
            if success:
                self._gains_applied = True

        obj_pos, obj_rot = self.box.get_world_pose()
        obj_lin_vel = self.box.get_linear_velocity()
        obj_ang_vel = self.box.get_angular_velocity()
        robot_lin_vel = self.g1.robot.get_linear_velocity()

        # 🚨 [리셋 크래시 완벽 차단] 물리 엔진이 값을 아직 계산하지 않아 None을 뱉으면 즉시 대기
        if obj_pos is None or obj_lin_vel is None or robot_lin_vel is None or np.isnan(obj_pos).any():
            return

        if self._trigger_toss:
            self._toss_signal = 1.0
            
            # 🚨 [가슴 높이 동적 투척] 로봇의 현재 골반 높이(Z)를 기준으로 0.35m 위에 스폰합니다.
            robot_pos, _ = self.g1.robot.get_world_pose()
            if robot_pos is None: robot_pos = np.array([0.0, 0.0, 0.78])
            
            toss_pos = np.array([robot_pos[0] + 0.65, robot_pos[1], robot_pos[2] + 0.35], dtype=np.float32)
            toss_lin_vel = np.array([-1.20, 0.0, 0.0], dtype=np.float32)
            
            self.box.set_world_pose(position=toss_pos)
            self.box.set_linear_velocity(toss_lin_vel)
            self.box.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
            self._trigger_toss = False

        self.g1.forward(step_size, self._toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "K":
                print("[DEBUG] 'K' Pressed! Triggering Toss (가슴 높이로 투척!)...")
                self._trigger_toss = True
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        if self.g1 is not None:  
            self.g1.reset_policy_state()  
            self._physics_ready = True
            self._gains_applied = False # 리셋 시 다시 게인을 주입하기 위해 초기화

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_sub_keyboard"): 
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)  
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")