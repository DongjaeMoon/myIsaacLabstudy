# [/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch/g1_catch_v12.py] (연구실)
# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/catch/g1_catch_v12.py] (전산실)

import os
import carb
import numpy as np
import omni
import omni.appwindow

from omni.isaac.core.objects import DynamicCuboid
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.types import ArticulationAction
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

        # training reset_object_parked에 맞춘 nominal parked pose
        self._park_pos = np.array([1.50, 0.00, 0.20], dtype=np.float32)

        # [추가] 웜업 카운터와 지속 시간 (100Hz 기준 60스텝 = 0.6초)
        self._warmup_steps = 0
        self._warmup_duration = 0

    def setup_scene(self) -> None:
        self.get_world().scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        )

        pc_dir = "/home/dongjae/isaaclab/myIsaacLabstudy"
        lab_dir = "/home/idim5080-2/mdj/myIsaacLabstudy"

        if os.path.exists(pc_dir):
            root_dir = pc_dir
            print(f"[INFO] 전산실 컴퓨터 경로 인식됨: {root_dir}")
        else:
            root_dir = lab_dir
            print(f"[INFO] 연구실 컴퓨터 경로 인식됨: {root_dir}")

        robot_usd_path = os.path.join(root_dir, "UROP/UROP_v12/usd/g1_29dof_full_collider_flattened.usd")
        policy_dir = os.path.join(root_dir, "logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported")

        self.g1 = G1CatchPolicy(
            prim_path="/World/G1",
            name="G1",
            usd_path=robot_usd_path,
            policy_path=policy_dir,
            position=np.array([0.0, 0.0, 0.78], dtype=np.float32),
        )

        self.box = DynamicCuboid(
            prim_path="/World/Object",
            name="box",
            position=self._park_pos.copy(),
            scale=np.array([0.32, 0.24, 0.24]),
            color=np.array([0.0, 0.8, 0.0]),
            mass=3.0,
        )

        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY),
            self._timeline_timer_callback_fn,
        )

    def _park_box(self):
        self.box.set_world_pose(position=self._park_pos.copy())
        self.box.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.box.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def _apply_trained_default_state(self):
        self.g1.robot.set_world_pose(
            position=self.g1.default_root_pos,
            orientation=self.g1.default_root_rot,
        )
        self.g1.robot.set_linear_velocity(self.g1.default_root_lin_vel)
        self.g1.robot.set_angular_velocity(self.g1.default_root_ang_vel)
        self.g1.robot.set_joint_positions(self.g1.default_pos)
        self.g1.robot.set_joint_velocities(self.g1.default_vel)

    def _reset_deploy_state(self):
        self._toss_signal = 0.0
        self._trigger_toss = False
        self._gains_applied = False
        self._physics_ready = False

        self._apply_trained_default_state()
        self.g1.reset_policy_state()
        self.g1.robot.apply_action(ArticulationAction(joint_positions=self.g1.default_pos.copy()))
        self._park_box()

        self._physics_ready = True

        # [추가] 리셋 시 웜업 카운터 초기화
        self._warmup_steps = 0

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )

        self.box.initialize()
        self.g1.initialize()

        # 디버그 출력 꺼두기
        self.g1.debug_enabled = True
        self.g1.debug_print_first_n = 5
        self.g1.debug_show_full_joint_table = True
        self.g1.debug_force_zero_action = False
        self.g1.debug_action_gain = 1.0

        self._reset_deploy_state()

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        self._reset_deploy_state()

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

        await self.get_world().play_async()

    def on_physics_step(self, step_size) -> None:
        if not self._physics_ready or not self.get_world().is_playing():
            return
        
        #if not self._gains_applied:
        #    success = self.g1.apply_pd_gains()
        #    if success:
        #        self._gains_applied = True

        obj_pos, obj_rot = self.box.get_world_pose()
        obj_lin_vel = self.box.get_linear_velocity()
        obj_ang_vel = self.box.get_angular_velocity()

        # ========================================================
        # [핵심 추가] 웜업(Warm-up) 페이즈: 초기 물리 튕김 안정화
        # ========================================================
        if self._warmup_steps < self._warmup_duration:
            # Policy 연산 없이 기본 자세 유지
            self.g1.robot.apply_action(ArticulationAction(joint_positions=self.g1.default_pos.copy()))
            self._warmup_steps += 1
            return

        if obj_pos is None or obj_lin_vel is None or np.isnan(obj_pos).any():
            return

        if self._trigger_toss:
            self._toss_signal = 1.0

            robot_pos, _ = self.g1.robot.get_world_pose()
            if robot_pos is None:
                robot_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32)

            toss_pos = np.array(
                [robot_pos[0] + 0.60, robot_pos[1], robot_pos[2] + 0.36],
                dtype=np.float32,
            )
            toss_lin_vel = np.array([-1.20, 0.0, 0.0], dtype=np.float32)

            self.box.set_world_pose(position=toss_pos)
            self.box.set_linear_velocity(toss_lin_vel)
            self.box.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self._trigger_toss = False

        self.g1.forward(step_size, self._toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "K":
                print("[INFO] Toss triggered.")
                self._trigger_toss = True
        return True

    def _timeline_timer_callback_fn(self, event) -> None:
        # ========================================================
        # [수정된 부분] initialize가 완료되어 default_pos가 존재하는지 확인
        # ========================================================
        if self.g1 is not None and hasattr(self.g1, "default_pos"):
            self._reset_deploy_state()

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

    def world_cleanup(self):
        world = self.get_world()
        self._event_timer_callback = None
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_sub_keyboard"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub_keyboard)
        if world.physics_callback_exists("physics_step"):
            world.remove_physics_callback("physics_step")