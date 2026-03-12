import os
from enum import Enum, auto

import carb
import numpy as np
import omni
import omni.appwindow

from omni.isaac.core.objects import DynamicCuboid
from isaacsim.examples.interactive.base_sample import BaseSample

from merged.g1_loco_catch_merged_policy import G1LocoCatchMergedPolicy


class Mode(Enum):
    WALK = auto()
    PREPARE_CATCH = auto()
    CATCH_HOLD = auto()


class LocoCatchMergedDeployV0(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._world_settings["stage_units_in_meters"] = 1.0
        self._world_settings["physics_dt"] = 1.0 / 200.0   # 0.005
        self._world_settings["rendering_dt"] = 8.0 / 200.0

        self.controller = None
        self.box = None
        self._physics_ready = False

        self.mode = Mode.WALK
        self._prepare_elapsed = 0.0
        self._prepare_duration = 0.22
        self._catch_success_elapsed = 0.0
        self._success_reported = False

        self._base_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._toss_requested = False
        self._box_thrown = False
        self._toss_signal = 0.0

        self._park_pos = np.array([1.50, 0.00, 0.20], dtype=np.float32)

        # [MODIFIED] 키 입력 꼬임 방지용. press/release 누적 대신 현재 눌린 키 집합으로 명령 재구성
        self._pressed_move_keys = set()

        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP": [1.0, 0.0, 0.0],
            "NUMPAD_4": [0.0, 0.5, 0.0], "LEFT": [0.0, 0.5, 0.0],
            "NUMPAD_6": [0.0, -0.5, 0.0], "RIGHT": [0.0, -0.5, 0.0],
            "Q": [0.0, 0.0, 1.0], "E": [0.0, 0.0, -1.0],
        }

        # [MODIFIED] cleanup/load 안전성용 핸들 초기화
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        self._event_timer_callback = None

    def _resolve_root_dir(self) -> str:
        pc_dir = "/home/dongjae/isaaclab/myIsaacLabstudy"
        lab_dir = "/home/idim5080-2/mdj/myIsaacLabstudy"
        if os.path.exists(pc_dir):
            print(f"[INFO] 전산실 컴퓨터 경로 인식됨: {pc_dir}")
            return pc_dir
        print(f"[INFO] 연구실 컴퓨터 경로 인식됨: {lab_dir}")
        return lab_dir

    def setup_scene(self) -> None:
        self.get_world().scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        root_dir = self._resolve_root_dir()
        robot_usd_path = os.path.join(
            root_dir, "UROP/UROP_g1_loco_v5/g1_29dof_full_collider_flattened.usd"
        )
        loco_policy_dir = os.path.join(
            root_dir, "logs/rsl_rl/UROP_g1_loco_v5/2026-03-06_16-10-35/exported"
        )
        catch_policy_dir = os.path.join(
            root_dir, "logs/rsl_rl/UROP_v12/2026-03-07_03-20-52/exported"
        )

        self.controller = G1LocoCatchMergedPolicy(
            prim_path="/World/G1",
            usd_path=robot_usd_path,
            loco_policy_dir=loco_policy_dir,
            catch_policy_dir=catch_policy_dir,
            world_dt=self._world_settings["physics_dt"],
            name="G1",
            position=np.array([0.0, 0.0, 0.79], dtype=np.float32),
        )

        self.box = DynamicCuboid(
            prim_path="/World/Object",
            name="box",
            position=self._park_pos.copy(),
            scale=np.array([0.32, 0.24, 0.24], dtype=np.float32),
            color=np.array([0.0, 0.8, 0.0], dtype=np.float32),
            mass=3.0,
        )

        timeline = omni.timeline.get_timeline_interface()
        self._event_timer_callback = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), self._timeline_timer_callback_fn
        )

    def _park_box(self) -> None:
        self.box.set_world_pose(position=self._park_pos.copy())
        self.box.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.box.set_angular_velocity(np.zeros(3, dtype=np.float32))

    # [MODIFIED] 현재 눌린 키 집합으로 locomotion command 재계산
    def _recompute_base_command_from_keys(self) -> None:
        cmd = np.zeros(3, dtype=np.float32)
        for key in self._pressed_move_keys:
            if key in self._input_keyboard_mapping:
                cmd += np.array(self._input_keyboard_mapping[key], dtype=np.float32)

        self._base_command[:] = cmd
        self._clip_command()

    # [MODIFIED] reset/load/cleanup 시 입력 상태 완전 초기화
    def _clear_input_state(self) -> None:
        self._pressed_move_keys.clear()
        self._base_command[:] = 0.0

    def _reset_deploy_state(self) -> None:
        # [MODIFIED] reset 도중 physics callback이 들어오지 않게 먼저 막음
        self._physics_ready = False

        self.mode = Mode.WALK
        self._prepare_elapsed = 0.0
        self._catch_success_elapsed = 0.0
        self._success_reported = False
        self._toss_requested = False
        self._box_thrown = False
        self._toss_signal = 0.0

        # [MODIFIED]
        self._clear_input_state()

        self.controller.apply_profile("loco")
        self.controller.reset_policy_state()
        self.controller.reset_robot_to_loco_default()
        self._park_box()

        self._physics_ready = True
        print("[MERGED] reset complete -> WALK mode")

    async def setup_post_load(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # [MODIFIED] 혹시 이전 구독 핸들이 남아있으면 안전하게 정리
        try:
            if self._sub_keyboard is not None and self._input is not None and self._keyboard is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
        except Exception as e:
            print(f"[MERGED] keyboard unsubscribe before reload skipped: {e}")
        self._sub_keyboard = None

        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )

        self.box.initialize()
        self.controller.initialize()
        self._reset_deploy_state()

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        await self.get_world().play_async()

    async def setup_post_reset(self) -> None:
        # [MODIFIED] reset 직후에는 기존 physics handle이 무효일 수 있으니 먼저 막아둠
        self._physics_ready = False

        # [MODIFIED] reset 후 fresh stage에 다시 initialize
        self.box.initialize()
        self.controller.initialize()

        # 이제 새 physics handle 기준으로 초기 상태 복원
        self._reset_deploy_state()

        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

        await self.get_world().play_async()

    def _clip_command(self) -> None:
        self._base_command[0] = np.clip(self._base_command[0], -0.5, 1.0)
        self._base_command[1] = np.clip(self._base_command[1], -0.5, 0.5)
        self._base_command[2] = np.clip(self._base_command[2], -1.0, 1.0)
        self._base_command[:] = np.where(np.abs(self._base_command) < 0.01, 0.0, self._base_command)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping and self.mode == Mode.WALK:
                # [MODIFIED] 증가식이 아니라 set 기반
                self._pressed_move_keys.add(event.input.name)
                self._recompute_base_command_from_keys()

            elif event.input.name == "J":
                if self.mode == Mode.WALK:
                    print("[MERGED] manual prepare requested")
                    self._enter_prepare_mode()
                else:
                    print(f"[MERGED] J ignored: current mode = {self.mode.name}")

            elif event.input.name == "K":
                if self.mode == Mode.WALK:
                    print("[MERGED] K ignored: press J first to enter catch-prep mode")
                elif self.mode == Mode.PREPARE_CATCH and self._prepare_elapsed < self._prepare_duration:
                    print(
                        f"[MERGED] K ignored: still preparing "
                        f"({self._prepare_elapsed:.3f}/{self._prepare_duration:.3f}s)"
                    )
                elif self._box_thrown:
                    print("[MERGED] K ignored: box already thrown")
                else:
                    print("[MERGED] toss requested")
                    self._toss_requested = True

            elif event.input.name == "R":
                print("[MERGED] manual reset requested")
                self._reset_deploy_state()

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping and self.mode == Mode.WALK:
                # [MODIFIED] 증가식이 아니라 set 기반
                self._pressed_move_keys.discard(event.input.name)
                self._recompute_base_command_from_keys()

        return True

    # [MODIFIED] 로봇 yaw 계산용 helper
    @staticmethod
    def _quat_wxyz_to_yaw(q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        w, x, y, z = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _spawn_relative_toss(self) -> None:
        # [MODIFIED]
        # 기존: world frame x축 기준으로 toss
        # 수정: 로봇의 현재 yaw를 반영하여 "현재 정면"에서 동일 거리/높이/속도로 toss
        root_pos, root_quat_wxyz = self.controller.get_root_pose_wxyz()
        yaw = self._quat_wxyz_to_yaw(root_quat_wxyz)

        forward_xy = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)

        # 거리 0.60, 높이 +0.36은 그대로 유지
        toss_pos_w = np.asarray(root_pos, dtype=np.float32) + 0.60 * forward_xy
        toss_pos_w[2] = np.asarray(root_pos, dtype=np.float32)[2] + 0.36

        # 속도 크기 1.20은 그대로 유지, 방향만 로봇 정면 반대(-forward)로
        toss_vel_w = -1.20 * forward_xy

        self.box.set_world_pose(position=toss_pos_w)
        self.box.set_linear_velocity(toss_vel_w)
        self.box.set_angular_velocity(np.zeros(3, dtype=np.float32))

        self._box_thrown = True
        self._toss_requested = False
        self._toss_signal = 1.0
        self._catch_success_elapsed = 0.0
        self._success_reported = False

        print(
            f"[MERGED] toss spawned | "
            f"yaw={yaw:.3f} "
            f"pos_w={np.round(toss_pos_w, 3)} "
            f"vel_w={np.round(toss_vel_w, 3)}"
        )

    def _enter_prepare_mode(self) -> None:
        if self.mode != Mode.WALK:
            return

        self.mode = Mode.PREPARE_CATCH
        self._prepare_elapsed = 0.0
        self._toss_signal = 0.0   # 아직 박스 안 날림
        self._box_thrown = False
        self._toss_requested = False
        self._catch_success_elapsed = 0.0
        self._success_reported = False

        # [MODIFIED] prepare 진입 시 locomotion 입력도 완전히 초기화
        self._clear_input_state()

        self.controller.apply_profile("catch")
        self.controller.reset_policy_state()

        print("[MERGED] WALK -> PREPARE_CATCH")

    def _enter_catch_hold_mode(self) -> None:
        self.mode = Mode.CATCH_HOLD
        self._toss_signal = 1.0 if self._box_thrown else 0.0
        self._catch_success_elapsed = 0.0
        print("[MERGED] PREPARE_CATCH -> CATCH_HOLD")

    def _is_hold_stable(self, obj_pos, obj_lin_vel, obj_ang_vel, dt: float) -> bool:
        rel = self.controller.compute_box_relative_kinematics(obj_pos, obj_lin_vel, obj_ang_vel)
        rel_p_b = rel["rel_p_b"]
        rel_v_b = rel["rel_v_b"]

        near_torso = (
            (0.10 < rel_p_b[0] < 0.55)
            and (abs(rel_p_b[1]) < 0.38)
            and (0.05 < rel_p_b[2] < 0.85)
        )
        slow_box = np.linalg.norm(obj_lin_vel) < 0.70 and np.linalg.norm(rel_v_b) < 0.75
        not_dropped = obj_pos[2] > 0.25

        if near_torso and slow_box and not_dropped:
            self._catch_success_elapsed += dt
        else:
            self._catch_success_elapsed = 0.0

        return self._catch_success_elapsed > 0.60

    def _should_abort_catch(self, obj_pos) -> bool:
        root_pos, _ = self.controller.get_root_pose_wxyz()
        too_low = float(obj_pos[2]) < 0.10
        too_far = np.linalg.norm(np.asarray(obj_pos, dtype=np.float32) - root_pos) > 3.0
        return bool(too_low or too_far)

    def on_physics_step(self, step_size) -> None:
        if not self._physics_ready or not self.get_world().is_playing():
            return

        obj_pos, obj_rot = self.box.get_world_pose()
        obj_lin_vel = self.box.get_linear_velocity()
        obj_ang_vel = self.box.get_angular_velocity()

        obj_pos = np.asarray(obj_pos, dtype=np.float32)
        obj_rot = np.asarray(obj_rot, dtype=np.float32)
        obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32)
        obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32)

        if self._toss_requested:
            self._spawn_relative_toss()

            obj_pos, obj_rot = self.box.get_world_pose()
            obj_lin_vel = self.box.get_linear_velocity()
            obj_ang_vel = self.box.get_angular_velocity()

            obj_pos = np.asarray(obj_pos, dtype=np.float32)
            obj_rot = np.asarray(obj_rot, dtype=np.float32)
            obj_lin_vel = np.asarray(obj_lin_vel, dtype=np.float32)
            obj_ang_vel = np.asarray(obj_ang_vel, dtype=np.float32)

        if self.mode == Mode.WALK:
            self.controller.forward_loco(self._base_command)
            return

        if self.mode == Mode.PREPARE_CATCH:
            self._prepare_elapsed += step_size
            self.controller.forward_catch(
                self._toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel
            )

            if self._prepare_elapsed >= self._prepare_duration:
                self._enter_catch_hold_mode()
            return

        if self.mode == Mode.CATCH_HOLD:
            self.controller.forward_catch(
                self._toss_signal, obj_pos, obj_rot, obj_lin_vel, obj_ang_vel
            )

            if self._box_thrown and self._is_hold_stable(obj_pos, obj_lin_vel, obj_ang_vel, step_size):
                if not self._success_reported:
                    self._success_reported = True
                    print("[MERGED] catch success: box stabilized near torso")

            if self._box_thrown and self._should_abort_catch(obj_pos):
                print("[MERGED] catch aborted -> reset to WALK")
                self._reset_deploy_state()
            return

    def _timeline_timer_callback_fn(self, event) -> None:
    # [MODIFIED]
    # PLAY 이벤트에서 _reset_deploy_state()를 또 호출하면
    # setup_post_load/setup_post_reset와 중복 reset이 발생해서 타이밍이 꼬일 수 있음.
    # 여기서는 physics callback 존재만 보장.
        if not self.get_world().physics_callback_exists("physics_step"):
            self.get_world().add_physics_callback("physics_step", callback_fn=self.on_physics_step)

    def world_cleanup(self):
        # [MODIFIED] cleanup 실패 때문에 reload 후 입력/콜백이 꼬이는 문제 방지
        self._physics_ready = False
        self._clear_input_state()

        # [MODIFIED] 올바른 API 이름 사용: unsubscribe_to_keyboard_events
        try:
            if self._input is not None and self._keyboard is not None and self._sub_keyboard is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
        except Exception as e:
            print(f"[MERGED] keyboard cleanup skipped: {e}")

        self._sub_keyboard = None
        self._keyboard = None
        self._input = None
        self._appwindow = None
        self._event_timer_callback = None

        try:
            world = self.get_world()
            if world is not None and world.physics_callback_exists("physics_step"):
                world.remove_physics_callback("physics_step")
        except Exception as e:
            print(f"[MERGED] physics callback cleanup skipped: {e}")