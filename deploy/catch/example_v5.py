# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example_v5.py]

import math
import time
import traceback
import numpy as np
import torch
import carb
import omni.appwindow
import omni.timeline

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.physics import ContactSensor


# --------------------------------------------------------------------------
# [설정] 경로 (너 환경에 맞게 수정)
# --------------------------------------------------------------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v5/2026-02-21_22-40-07/exported/policy.pt"
# POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v6/2026-02-23_23-58-09/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v5/usd/G1_23DOF_UROP.usd"
ROBOT_PRIM_PATH = "/World/G1"

# [MOD] physics callback name 고정
PHYSICS_CALLBACK_NAME = "policy_physics_step_v5"


# env_cfg.py Observation contact 순서(11개)와 동일해야 함
CONTACT_SENSOR_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_rubber_hand"
]

# ActionsCfg(policy action order 23) — 학습과 동일
POLICY_JOINT_ORDER = [
    # legs_sagittal (scale=0.5)
    "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
    # legs_frontal (scale=0.22)
    "left_hip_roll_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_ankle_roll_joint",
    # legs_yaw (scale=0.10)
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    # waist (scale=0.25)
    "waist_yaw_joint",
    # left_arm_capture (scale=1.2)
    "left_shoulder_pitch_joint", "left_elbow_joint",
    # right_arm_capture (scale=1.2)
    "right_shoulder_pitch_joint", "right_elbow_joint",
    # left_arm_wrap (scale=0.7)
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint",
    # right_arm_wrap (scale=0.7)
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint"
]

# joint->scale (학습 ActionsCfg와 동일)
JOINT_SCALE_MAP = {
    # legs_sagittal
    "left_hip_pitch_joint": 0.5, "left_knee_joint": 0.5, "left_ankle_pitch_joint": 0.5,
    "right_hip_pitch_joint": 0.5, "right_knee_joint": 0.5, "right_ankle_pitch_joint": 0.5,
    # legs_frontal
    "left_hip_roll_joint": 0.22, "left_ankle_roll_joint": 0.22,
    "right_hip_roll_joint": 0.22, "right_ankle_roll_joint": 0.22,
    # legs_yaw
    "left_hip_yaw_joint": 0.10, "right_hip_yaw_joint": 0.10,
    # waist
    "waist_yaw_joint": 0.25,
    # arm capture
    "left_shoulder_pitch_joint": 1.2, "left_elbow_joint": 1.2,
    "right_shoulder_pitch_joint": 1.2, "right_elbow_joint": 1.2,
    # arm wrap
    "left_shoulder_roll_joint": 0.7, "left_shoulder_yaw_joint": 0.7, "left_wrist_roll_joint": 0.7,
    "right_shoulder_roll_joint": 0.7, "right_shoulder_yaw_joint": 0.7, "right_wrist_roll_joint": 0.7,
}

# v5 training init_state (scene_objects_cfg.py와 동일)
TRAIN_INIT_JOINT_POS = {
    # legs/waist: 0
    "left_hip_pitch_joint": 0.0, "right_hip_pitch_joint": 0.0,
    "left_knee_joint": 0.0, "right_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0, "right_ankle_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.2, "right_shoulder_pitch_joint": 0.2,
    "left_elbow_joint": 0.5, "right_elbow_joint": 0.5,
    "left_shoulder_roll_joint": 0.0, "right_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "left_wrist_roll_joint": 0.0, "right_wrist_roll_joint": 0.0,
}

# v5 parking pose
PARK_REL_POS = (0.0, 1.30, -0.60)  # (x,y,z) in robot yaw frame

# [MOD] v5 env.yaml(stage2) 분포와 맞춤
THROW_REL_POS_RANGE = ((0.3, 0.5), (-0.1, 0.1), (0.30, 0.40))
THROW_REL_VEL_RANGE = ((-2.0, -0.8), (-0.1, 0.1), (-0.20, 0.20))


class ExampleV5(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 학습: sim.dt=1/120, decimation=2 -> policy 60Hz
        self.physics_hz_target = 120.0
        self.policy_hz = 60.0
        self._accum_dt = 0.0

        self.debug_mode = True
        self.debug_print_every_n_policy_steps = 120
        self.debug_print_every_n_physics_steps = 240

        self._policy_step_count = 0
        self._physics_step_count = 0

        self._quat_raw_is_xyzw = None
        self._callback_registered = False
        self._is_running = False

        # toss signal (학습의 env._urop_toss_active 역할)
        self._toss_active = False

        # box parked mode
        self._box_hold_mode = True
        self.follow_parked_box_every_step = False

        # IsaacLab applied_torque vs IsaacSim effort mismatch 가능성
        self.use_torque_obs = False  # [MOD]

        # reset 직후 settle
        self.reset_settle_sec = 0.35
        self._sim_time_since_reset = 0.0

        # torque low-pass
        self._torque_lp = None
        self._torque_lp_alpha = 0.2

        # policy on/off (사용자 토글 상태는 reset/stop-play에서도 유지)
        self._use_policy = True

        # prev action (policy order 23)
        self.prev_action_policy_order = None
        self._last_action_policy_order = None

        self._printed_torque_hint = False
        self._t0_wall = time.time()

        # [MOD] STOP->PLAY 이후 runtime 재초기화 플래그
        self._pending_play_reinit = False

        # [MOD] callback/input 상태 추적
        self._physics_callback_name = PHYSICS_CALLBACK_NAME
        self._sub_keyboard = None
        self._input = None
        self._keyboard = None

        # handles
        self._world = None
        self._robot = None
        self._box = None
        self.contact_sensors = {}

    # ---------------- Quaternion utils (training uses wxyz) ----------------
    def _detect_and_lock_quat_order(self, q_raw):
        if self._quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float32).copy()
        if q.shape[0] != 4:
            self._quat_raw_is_xyzw = True
            return
        # upright 초기에서 w~1이면 xyzw일 확률 큼 / q[0]~1이면 wxyz
        if abs(q[3]) > 0.90:
            self._quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90:
            self._quat_raw_is_xyzw = False
        else:
            self._quat_raw_is_xyzw = True
        if self.debug_mode:
            print(
                f"[DEBUG] quat raw order locked: "
                f"{'xyzw(x,y,z,w)' if self._quat_raw_is_xyzw else 'wxyz(w,x,y,z)'} raw={q_raw}"
            )

    def _to_wxyz(self, q_raw):
        q = np.asarray(q_raw, dtype=np.float32).copy()
        self._detect_and_lock_quat_order(q_raw)
        if self._quat_raw_is_xyzw:
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        return q

    def _quat_conj(self, q):
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _quat_apply(self, q, v):
        zeros = torch.zeros_like(v[..., :1])
        vq = torch.cat([zeros, v], dim=-1)
        return self._quat_mul(self._quat_mul(q, vq), self._quat_conj(q))[..., 1:4]

    def _quat_rotate_inverse(self, q, v):
        return self._quat_apply(self._quat_conj(q), v)

    def _quat_to_rot6d(self, q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - z * w)
        r10 = 2 * (x * y + z * w)
        r11 = 1 - 2 * (x * x + z * z)
        r20 = 2 * (x * z - y * w)
        r21 = 2 * (y * z + x * w)
        return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)

    def _yaw_from_wxyz(self, q_wxyz):
        w, x, y, z = q_wxyz
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _rotate_vec_by_yaw(self, yaw, v3):
        c = math.cos(yaw)
        s = math.sin(yaw)
        x, y, z = float(v3[0]), float(v3[1]), float(v3[2])
        return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float32)

    # ---------------- Scene setup ----------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.78], dtype=np.float32))

        # box: training과 동일하게 size(0.4,0.3,0.3), mass=2.0
        self.box = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name="box",
                position=np.array([3.0, 0.0, 1.0], dtype=np.float32),
                scale=np.array([0.4, 0.3, 0.3], dtype=np.float32),
                mass=2.0,
                color=np.array([0.0, 0.8, 0.0], dtype=np.float32),
            )
        )

        # contact sensors
        self.contact_sensors = {}
        for link_name in CONTACT_SENSOR_LINKS:
            sensor_prim_path = f"{ROBOT_PRIM_PATH}/{link_name}/contact_sensor"
            try:
                self.contact_sensors[link_name] = world.scene.add(
                    ContactSensor(
                        prim_path=sensor_prim_path,
                        name=f"contact_{link_name}",
                        min_threshold=0.0,
                        max_threshold=100000.0,
                        radius=0.05,
                        translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    )
                )
            except Exception:
                self.contact_sensors[link_name] = None

    # ---------------- Runtime callback / input helpers ----------------
    def _safe_remove_physics_callback(self):
        # [MOD] STOP/PLAY 후 callback stale 방지
        if self._world is None:
            return
        try:
            if hasattr(self._world, "remove_physics_callback"):
                self._world.remove_physics_callback(self._physics_callback_name)
                if self.debug_mode:
                    print(f"[DEBUG] removed physics callback: {self._physics_callback_name}")
        except Exception:
            pass
        self._callback_registered = False

    def _ensure_physics_callback_registered(self, force_rebind=False):
        # [MOD] PLAY 때 callback 재등록 강제
        try:
            self._world = self.get_world()
        except Exception:
            pass

        if self._world is None:
            return False

        if force_rebind:
            self._safe_remove_physics_callback()

        if self._callback_registered:
            return True

        try:
            self._world.add_physics_callback(self._physics_callback_name, callback_fn=self._robot_control)
            self._callback_registered = True
            if self.debug_mode:
                print(f"[DEBUG] physics callback registered: {self._physics_callback_name}")
            return True
        except Exception as e:
            print(f"[WARN] add_physics_callback failed: {e}")
            self._callback_registered = False
            return False

    def _ensure_keyboard_subscription(self):
        # [MOD] keyboard subscription 재생성 (필요시)
        try:
            if self._sub_keyboard is not None:
                return True
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
            if self.debug_mode:
                print("[DEBUG] keyboard subscription registered")
            return True
        except Exception as e:
            print(f"[WARN] keyboard subscribe failed: {e}")
            self._sub_keyboard = None
            return False

    def _unsubscribe_keyboard(self):
        try:
            if self._input is not None and self._sub_keyboard is not None:
                self._input.unsubscribe_to_keyboard_events(self._sub_keyboard)
        except Exception:
            pass
        self._sub_keyboard = None

    # ---------------- Post load ----------------
    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")
        self._box = self._world.scene.get_object("box")

        self._try_set_physics_dt(1.0 / self.physics_hz_target)

        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        self._robot.initialize()
        _, q_raw = self._robot.get_world_pose()
        self._detect_and_lock_quat_order(q_raw)

        print(f">>> Loading Policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        self.num_dof = self._robot.num_dof
        self.robot_dof_names = list(self._robot.dof_names)

        # map policy joints -> robot dof indices
        self.policy_to_robot_indices = []
        for pj in POLICY_JOINT_ORDER:
            if pj in self.robot_dof_names:
                self.policy_to_robot_indices.append(self.robot_dof_names.index(pj))
            else:
                carb.log_error(f"[FATAL] Joint {pj} not found in robot USD DOF names!")

        if len(self.policy_to_robot_indices) != 23:
            raise RuntimeError(f"policy_to_robot_indices len={len(self.policy_to_robot_indices)} (expected 23)")

        self.policy_to_robot_indices = torch.tensor(
            self.policy_to_robot_indices, device=self.device, dtype=torch.long
        )

        # buffers
        self.default_pos = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self.action_scale = torch.ones(self.num_dof, device=self.device, dtype=torch.float32)

        # set joint scales (robot order)
        for i, jn in enumerate(self.robot_dof_names):
            if jn in JOINT_SCALE_MAP:
                self.action_scale[i] = float(JOINT_SCALE_MAP[jn])

        # prev action (policy order)
        self.prev_action_policy_order = torch.zeros(23, device=self.device, dtype=torch.float32)
        self._last_action_policy_order = torch.zeros(23, device=self.device, dtype=torch.float32)

        # configure drives like training
        self._configure_joint_drives_like_training()

        # [MOD] callback 등록 helper 사용
        self._ensure_physics_callback_registered(force_rebind=True)

        # [MOD] keyboard 등록 helper 사용
        self._ensure_keyboard_subscription()

        # hard reset
        self._hard_reset_all()

        # policy I/O dim check
        try:
            obs = self._get_observation()
            with torch.no_grad():
                out = self.policy(obs.unsqueeze(0))
                if isinstance(out, (tuple, list)):
                    out = out[0]
            expected_obs_dim = 128  # v5: 1 + 78 + 23 + 15 + 11
            print(f">>> obs_dim={obs.numel()}  policy_out_dim={out.numel()}  (expected obs_dim={expected_obs_dim}, act_dim=23)")
            if obs.numel() != expected_obs_dim:
                print("[FATAL] Observation dim mismatch. Do NOT run deploy until fixed.")
        except Exception:
            print("[WARN] policy I/O dim check failed:")
            traceback.print_exc()

        print(">>> ExampleV5 setup_post_load done.")
        print(">>> Controls: [Play], K=throw, R=hard reset, P=policy on/off, H=park hold on/off, T=torque obs toggle")

    async def setup_post_reset(self):
        # GUI reset 계열 대비
        self._ensure_physics_callback_registered(force_rebind=True)  # [MOD]
        self._ensure_keyboard_subscription()  # [MOD]
        self._hard_reset_all()

    def _hard_reset_all(self):
        self._accum_dt = 0.0
        self._physics_step_count = 0
        self._policy_step_count = 0
        self._sim_time_since_reset = 0.0
        self._torque_lp = None
        self._toss_active = False
        self._box_hold_mode = True
        self._pending_play_reinit = False  # [MOD]

        if self.prev_action_policy_order is not None:
            self.prev_action_policy_order.zero_()
        if self._last_action_policy_order is not None:
            self._last_action_policy_order.zero_()

        # reset robot to training init pose
        if self._robot and self._robot.is_valid():
            try:
                self._robot.initialize()
            except Exception:
                pass

            self._configure_joint_drives_like_training()
            self._robot.set_world_pose(position=np.array([0.0, 0.0, 0.78], dtype=np.float32))

            init = np.zeros(self.num_dof, dtype=np.float32)
            for i, jn in enumerate(self.robot_dof_names):
                if jn in TRAIN_INIT_JOINT_POS:
                    init[i] = float(TRAIN_INIT_JOINT_POS[jn])

            self._robot.set_joint_positions(init)
            self._robot.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float32))

            # default_pos는 학습의 default_joint_pos와 동일
            self.default_pos = torch.tensor(init, device=self.device, dtype=torch.float32)

            # [MOD] reset 직후 target 1회 적용 (drive initialize 안정화)
            try:
                self._robot.apply_action(ArticulationAction(joint_positions=init.astype(np.float32)))
            except Exception:
                pass

        # park box (1회 배치)
        self._place_box_parked()

    # ---------------- Timeline ----------------
    def _on_timeline_event(self, event):
        """[MOD] STOP/PLAY 이후 callback/handle stale 문제 robust 처리."""
        etype = int(event.type)
        play_type = int(omni.timeline.TimelineEventType.PLAY)
        stop_type = int(omni.timeline.TimelineEventType.STOP)

        pause_type = None
        if hasattr(omni.timeline.TimelineEventType, "PAUSE"):
            pause_type = int(omni.timeline.TimelineEventType.PAUSE)

        if etype == play_type:
            self._is_running = True

            # [MOD] PLAY마다 callback 재바인딩 시도 (핵심)
            self._ensure_physics_callback_registered(force_rebind=True)
            self._ensure_keyboard_subscription()

            # [MOD] 첫 physics tick에서 runtime handle 재초기화 + hard reset 하도록 예약
            self._pending_play_reinit = True

            if self.debug_mode:
                print("[DEBUG] Timeline PLAY -> callback rebind + pending runtime reinit")

        elif etype == stop_type:
            self._is_running = False

            # [MOD] STOP 시 callback 제거 (old physics scene에 매달린 callback 제거)
            self._safe_remove_physics_callback()

            # 다음 PLAY에서 재초기화
            self._pending_play_reinit = True
            self._accum_dt = 0.0
            self._sim_time_since_reset = 0.0
            self._toss_active = False
            self._box_hold_mode = True

            if self.debug_mode:
                print("[DEBUG] Timeline STOP -> callback removed, pending reinit set")

        elif (pause_type is not None) and (etype == pause_type):
            self._is_running = False
            if self.debug_mode:
                print("[DEBUG] Timeline PAUSE")

    def _reacquire_runtime_handles(self):
        """STOP/PLAY 이후 stale handle 방지용."""
        self._world = self.get_world()

        try:
            self._robot = self._world.scene.get_object("g1")
        except Exception:
            self._robot = None

        try:
            self._box = self._world.scene.get_object("box")
        except Exception:
            self._box = None

        # contact sensors도 scene에서 다시 가져오기
        for link_name in CONTACT_SENSOR_LINKS:
            sensor_name = f"contact_{link_name}"
            try:
                self.contact_sensors[link_name] = self._world.scene.get_object(sensor_name)
            except Exception:
                self.contact_sensors[link_name] = self.contact_sensors.get(link_name, None)

    def _ensure_runtime_initialized(self):
        """physics 재시작 후 articulation handle 재생성."""
        self._reacquire_runtime_handles()

        if self._robot is None:
            return False

        try:
            self._robot.initialize()
        except Exception as e:
            print(f"[WARN] robot.initialize() failed: {e}")
            return False

        try:
            self._configure_joint_drives_like_training()
        except Exception as e:
            print(f"[WARN] set gains failed after reinit: {e}")

        # physics dt 재적용
        try:
            self._try_set_physics_dt(1.0 / self.physics_hz_target)
        except Exception:
            pass

        ok = bool(getattr(self._robot, "handles_initialized", False))
        if self.debug_mode:
            print(f"[DEBUG] _ensure_runtime_initialized -> handles_initialized={ok}")
        return ok

    # ---------------- Keyboard ----------------
    # K: throw (toss_active=True)
    # R: reset
    # P: policy on/off
    # H: park hold on/off
    # T: torque obs on/off
    def _on_key(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return

        if event.input == carb.input.KeyboardInput.K:
            self.throw_box_stage2_like_training()

        elif event.input == carb.input.KeyboardInput.R:
            print(">>> Hard reset")
            # [MOD] handles stale일 수 있으니 재초기화 시도 후 reset
            if self._is_running:
                self._ensure_runtime_initialized()
            self._hard_reset_all()

        elif event.input == carb.input.KeyboardInput.P:
            self._use_policy = not self._use_policy
            print(f">>> Toggle policy: {'ON' if self._use_policy else 'OFF'}")

        elif event.input == carb.input.KeyboardInput.H:
            self._box_hold_mode = not self._box_hold_mode
            print(f">>> Toggle box hold mode: {'ON(park)' if self._box_hold_mode else 'OFF(dynamic)'}")
            if self._box_hold_mode:
                self._place_box_parked()

        elif event.input == carb.input.KeyboardInput.T:
            self.use_torque_obs = not self.use_torque_obs
            self._torque_lp = None
            print(f">>> Toggle torque obs: {'ON' if self.use_torque_obs else 'OFF'}")

    # ---------------- Box placement ----------------
    def _place_box_parked(self):
        if (self._box is None) or (self._robot is None):
            return
        base_pos, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)
        yaw = self._yaw_from_wxyz(base_quat)

        rel_p = np.array(PARK_REL_POS, dtype=np.float32)
        p_w = np.asarray(base_pos, dtype=np.float32) + self._rotate_vec_by_yaw(yaw, rel_p)

        self._box.set_world_pose(position=p_w)
        self._box.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        try:
            self._box.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        except Exception:
            pass

    def throw_box_stage2_like_training(self):
        if (self._box is None) or (self._robot is None):
            return
        if not self._is_running:
            print("[WARN] not running (press Play first)")
            return

        # [MOD] PLAY 직후 stale handle 방어
        if not bool(getattr(self._robot, "handles_initialized", False)):
            ok = self._ensure_runtime_initialized()
            if not ok:
                print("[WARN] runtime not initialized yet. Try Play and wait a moment.")
                return

        self._box_hold_mode = False
        self._toss_active = True

        base_pos, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)
        yaw = self._yaw_from_wxyz(base_quat)

        rel_p = np.array(
            [
                np.random.uniform(*THROW_REL_POS_RANGE[0]),
                np.random.uniform(*THROW_REL_POS_RANGE[1]),
                np.random.uniform(*THROW_REL_POS_RANGE[2]),
            ],
            dtype=np.float32,
        )
        rel_v = np.array(
            [
                np.random.uniform(*THROW_REL_VEL_RANGE[0]),
                np.random.uniform(*THROW_REL_VEL_RANGE[1]),
                np.random.uniform(*THROW_REL_VEL_RANGE[2]),
            ],
            dtype=np.float32,
        )

        p_w = np.asarray(base_pos, dtype=np.float32) + self._rotate_vec_by_yaw(yaw, rel_p)
        v_w = np.asarray(self._robot.get_linear_velocity(), dtype=np.float32) + self._rotate_vec_by_yaw(yaw, rel_v)

        self._box.set_world_pose(position=p_w)
        self._box.set_linear_velocity(v_w)
        try:
            self._box.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        except Exception:
            pass

        if self.debug_mode:
            print(f">>> THROW(stage2-like): rel_p={rel_p}, rel_v={rel_v} -> pos_w={p_w}, vel_w={v_w}")

    # ---------------- Observation (must match UROP_v5 env_cfg.py) ----------------
    def _try_get_joint_torque_like_training(self, j_pos_t: torch.Tensor) -> torch.Tensor:
        # deploy 초기 검증은 torque obs OFF 권장
        if not self.use_torque_obs:
            return torch.zeros_like(j_pos_t)

        jt = torch.zeros_like(j_pos_t)

        cand = None
        try:
            if hasattr(self._robot, "get_joint_efforts"):
                cand = self._robot.get_joint_efforts()
            elif hasattr(self._robot, "get_measured_joint_efforts"):
                cand = self._robot.get_measured_joint_efforts()
        except Exception:
            cand = None

        if cand is not None:
            try:
                cand_t = torch.tensor(cand, device=self.device, dtype=torch.float32)
                if cand_t.shape == jt.shape:
                    jt = cand_t
                    # low-pass filter
                    if self._torque_lp is None:
                        self._torque_lp = jt.clone()
                    else:
                        a = self._torque_lp_alpha
                        self._torque_lp = (1.0 - a) * self._torque_lp + a * jt
                    jt = self._torque_lp

                    if self.debug_mode and (not self._printed_torque_hint):
                        self._printed_torque_hint = True
                        print("[DEBUG] torque obs: using IsaacSim efforts + low-pass filter")
                else:
                    jt = torch.zeros_like(j_pos_t)
            except Exception:
                jt = torch.zeros_like(j_pos_t)
        else:
            if self.debug_mode and (not self._printed_torque_hint):
                self._printed_torque_hint = True
                print("[DEBUG] torque obs: not available -> zeros")

        # training torque_scale = 1/80, clamp [-1, 1]
        jt = torch.clamp(jt * (1.0 / 80.0), -1.0, 1.0)
        return jt

    def _read_contact_11(self) -> torch.Tensor:
        # training: contact_forces(...) * (1/300), toss_active로 gate
        forces = []
        for name in CONTACT_SENSOR_LINKS:
            sensor = self.contact_sensors.get(name, None)
            val = 0.0
            if sensor and self._is_running:
                try:
                    reading = sensor.get_current_frame()
                    f = None
                    if isinstance(reading, dict):
                        for k in ["net_force", "net_forces", "force", "forces"]:
                            if k in reading:
                                f = reading[k]
                                break
                    if f is not None:
                        val = float(np.linalg.norm(np.asarray(f)))
                except Exception:
                    pass
            forces.append(val)

        contact = torch.tensor(forces, device=self.device, dtype=torch.float32) * (1.0 / 300.0)
        if not self._toss_active:
            contact *= 0.0
        return contact

    def _get_observation(self):
        # toss_signal (1)
        toss_signal = torch.tensor([1.0 if self._toss_active else 0.0], device=self.device, dtype=torch.float32)

        base_pos, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)

        base_lin_w = torch.tensor(self._robot.get_linear_velocity(), device=self.device, dtype=torch.float32)
        base_ang_w = torch.tensor(self._robot.get_angular_velocity(), device=self.device, dtype=torch.float32)

        q = torch.tensor(base_quat, device=self.device, dtype=torch.float32)

        g_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        g_b = self._quat_rotate_inverse(q, g_world)
        lin_b = self._quat_rotate_inverse(q, base_lin_w)
        ang_b = self._quat_rotate_inverse(q, base_ang_w)

        # [MOD] v5 training observations.py는 joint_pos absolute 사용 (rel 아님)
        j_pos = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
        j_torque = self._try_get_joint_torque_like_training(j_pos)

        # training: [g_b(3), lin_b(3), ang_b(3), joint_pos(23), joint_vel(23), applied_torque(23)] = 78
        proprio = torch.cat([g_b, lin_b, ang_b, j_pos, j_vel, j_torque], dim=-1)

        prev_act = self.prev_action_policy_order  # (23)

        # object_rel (15)
        obj_pos, obj_quat_raw = self._box.get_world_pose()
        obj_quat = self._to_wxyz(obj_quat_raw)

        obj_lin_w = torch.tensor(self._box.get_linear_velocity(), device=self.device, dtype=torch.float32)
        try:
            obj_ang_w = torch.tensor(self._box.get_angular_velocity(), device=self.device, dtype=torch.float32)
        except Exception:
            obj_ang_w = torch.zeros(3, device=self.device, dtype=torch.float32)

        r_pos = torch.tensor(base_pos, device=self.device, dtype=torch.float32)
        rel_p_b = self._quat_rotate_inverse(q, torch.tensor(obj_pos, device=self.device, dtype=torch.float32) - r_pos)
        rel_v_b = self._quat_rotate_inverse(q, obj_lin_w - base_lin_w)
        rel_w_b = self._quat_rotate_inverse(q, obj_ang_w - base_ang_w)

        b_q = torch.tensor(obj_quat, device=self.device, dtype=torch.float32)
        rel_q = self._quat_mul(self._quat_conj(q), b_q)
        rel_r6 = self._quat_to_rot6d(rel_q)

        obj_rel = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)

        # contact (11)
        contact = self._read_contact_11()

        # final obs = [toss(1), proprio(78), prev_action(23), obj_rel(15), contact(11)] = 128
        obs = torch.cat([toss_signal, proprio, prev_act, obj_rel, contact], dim=-1)

        # [MOD] 방어 체크
        if obs.numel() != 128:
            raise RuntimeError(f"obs_dim mismatch: got {obs.numel()}, expected 128")
        if not torch.isfinite(obs).all():
            raise RuntimeError("non-finite value in observation")
        return obs

    # ---------------- Control loop ----------------
    def _robot_control(self, step_size: float):
        # [MOD] callback이 실제 살아있는지 확인용 (낮은 빈도 로그)
        dt = float(step_size)
        if dt <= 0.0:
            return

        # [MOD] PLAY 후 callback은 살아있지만 handle stale일 수 있으므로 먼저 복구
        need_reinit = False
        if self._robot is None:
            need_reinit = True
        else:
            if not bool(getattr(self._robot, "handles_initialized", False)):
                need_reinit = True
        if self._pending_play_reinit:
            need_reinit = True

        if need_reinit and self._is_running:
            ok = self._ensure_runtime_initialized()
            if not ok:
                return

            if self._pending_play_reinit:
                if self.debug_mode:
                    print("[DEBUG] runtime reinit complete -> hard reset after PLAY")
                self._hard_reset_all()
                # [MOD] hard reset 후에도 playback 상태는 유지
                self._is_running = True

        # 여기까지 왔는데 robot 없음/handle 미초기화면 제어 불가
        if (self._robot is None) or (not bool(getattr(self._robot, "handles_initialized", False))):
            return

        self._physics_step_count += 1

        # parked mode: 옵션으로만 follow (기본 False)
        if self._is_running and self._box_hold_mode and self.follow_parked_box_every_step:
            self._place_box_parked()

        if self.debug_mode and (self._physics_step_count % self.debug_print_every_n_physics_steps == 0):
            try:
                p, qraw = self._robot.get_world_pose()
                q = self._to_wxyz(qraw)
                print(
                    f"[DEBUG] t={time.time()-self._t0_wall:6.2f}s "
                    f"phys={self._physics_step_count:06d} "
                    f"policy={'ON' if self._use_policy else 'OFF'} "
                    f"toss={self._toss_active} hold={self._box_hold_mode} "
                    f"callback={'ON' if self._callback_registered else 'OFF'}"
                )
                print(f"        base_pos={np.array(p).round(3)} base_quat_wxyz={np.array(q).round(3)}")
            except Exception:
                pass

        if not self._is_running:
            return

        self._sim_time_since_reset += dt
        if self._sim_time_since_reset < self.reset_settle_sec:
            self._apply_policy_action(torch.zeros(23, device=self.device))
            return

        # policy off -> zero action (default pose 유지)
        if not self._use_policy:
            self._apply_policy_action(torch.zeros(23, device=self.device))
            return

        self._accum_dt += dt

        # policy 60Hz, 사이사이에는 이전 action hold
        if self._accum_dt < (1.0 / self.policy_hz):
            self._apply_policy_action(self._last_action_policy_order)
            return

        while self._accum_dt >= (1.0 / self.policy_hz):
            self._accum_dt -= (1.0 / self.policy_hz)

        try:
            obs = self._get_observation()
        except Exception:
            if self.debug_mode:
                print("[DEBUG] _get_observation failed:")
                traceback.print_exc()
            return

        with torch.no_grad():
            out = self.policy(obs.unsqueeze(0))
            if isinstance(out, (tuple, list)):
                out = out[0]
            action_policy_order = out.squeeze(0)

        action_policy_order = torch.clamp(action_policy_order, -1.0, 1.0)

        if not torch.isfinite(action_policy_order).all():
            print("[FATAL] non-finite action from policy.")
            return

        # training 의미와 맞게 obs 계산 후 prev_action 갱신
        self.prev_action_policy_order = action_policy_order.clone()
        self._last_action_policy_order = action_policy_order.clone()

        self._policy_step_count += 1
        if self.debug_mode and (self._policy_step_count % self.debug_print_every_n_policy_steps == 0):
            a_min = float(action_policy_order.min().item())
            a_max = float(action_policy_order.max().item())
            print(
                f"[DEBUG] policy_step={self._policy_step_count:06d} "
                f"action(min,max)=({a_min:+.3f},{a_max:+.3f}) obs_dim={obs.numel()}"
            )

        self._apply_policy_action(action_policy_order)

    def _apply_policy_action(self, action_policy_order: torch.Tensor):
        if self._robot is None:
            return
        if not bool(getattr(self._robot, "handles_initialized", False)):
            return

        # robot order action vector
        action_robot_order = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        action_robot_order[self.policy_to_robot_indices] = action_policy_order

        # target_pos = default_pos + scale * action
        target_pos = self.default_pos + (self.action_scale * action_robot_order)
        target_np = target_pos.detach().cpu().numpy().astype(np.float32)

        self._robot.apply_action(ArticulationAction(joint_positions=target_np))

    # ---------------- Gains / dt ----------------
    def _configure_joint_drives_like_training(self):
        if not self._robot:
            return
        try:
            ctrl = self._robot.get_articulation_controller()
        except Exception:
            ctrl = None

        if ctrl is None:
            print("[WARN] Could not get articulation controller -> cannot set gains.")
            return

        kp = np.zeros(self.num_dof, dtype=np.float32)
        kd = np.zeros(self.num_dof, dtype=np.float32)

        for i, name in enumerate(self.robot_dof_names):
            # training(scene_objects_cfg):
            # legs+waist: stiffness=120, damping=5
            # arms_load(shoulder_pitch, elbow): stiffness=80, damping=4
            # arms_pos(other arms): stiffness=60, damping=3
            if ("hip_" in name) or ("knee" in name) or ("ankle_" in name) or ("waist_" in name):
                kp[i] = 120.0
                kd[i] = 5.0
            elif ("shoulder_pitch" in name) or ("elbow" in name):
                kp[i] = 80.0
                kd[i] = 4.0
            elif ("shoulder_" in name) or ("wrist_" in name):
                kp[i] = 60.0
                kd[i] = 3.0
            else:
                kp[i] = 60.0
                kd[i] = 3.0

        ok = False
        try:
            ctrl.set_gains(kps=kp, kds=kd)
            ok = True
        except Exception:
            try:
                ctrl.set_gains(kp, kd)
                ok = True
            except Exception:
                ok = False

        if self.debug_mode:
            print(f"[DEBUG] set_gains ok={ok}")

    def _try_set_physics_dt(self, dt: float):
        ok = False
        try:
            if hasattr(self._world, "set_physics_dt"):
                self._world.set_physics_dt(dt)
                ok = True
        except Exception:
            ok = False

        if not ok:
            try:
                pc = self._world.get_physics_context()
                if hasattr(pc, "set_physics_dt"):
                    pc.set_physics_dt(dt)
                    ok = True
            except Exception:
                ok = False

        if self.debug_mode:
            print(f"[DEBUG] try_set_physics_dt({dt:.6f}) ok={ok} target={self.physics_hz_target:.1f}Hz")

    # ---------------- Cleanup ----------------
    def world_cleanup(self):
        self._timeline_sub = None
        self._unsubscribe_keyboard()
        self._safe_remove_physics_callback()