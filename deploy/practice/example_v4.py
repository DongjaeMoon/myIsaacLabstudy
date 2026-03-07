# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example_v3.py]

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
# [설정] 경로
# --------------------------------------------------------------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v4/2026-02-17_01-02-56/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v4/usd/G1_23DOF_UROP.usd"
ROBOT_PRIM_PATH = "/World/G1"

# env_cfg.py Observation contact 순서 (11개)
CONTACT_SENSOR_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_rubber_hand"
]


class ExampleV4(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------
        # 디버그 옵션
        # -----------------------------
        self.debug_mode = True
        self.debug_print_first_n_policy_steps = 30
        self.debug_print_every_n_policy_steps = 60
        self.debug_print_every_n_physics_steps = 120  # 120Hz면 1초마다
        self._printed_torque_hint = False

        # -----------------------------
        # ActionsCfg에 정의된 policy action order (23)
        # -----------------------------
        self.policy_joint_order = [
            # legs_sagittal
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
            # legs_frontal
            "left_hip_roll_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_ankle_roll_joint",
            # legs_yaw
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            # waist
            "waist_yaw_joint",
            # left_arm_capture
            "left_shoulder_pitch_joint", "left_elbow_joint",
            # right_arm_capture
            "right_shoulder_pitch_joint", "right_elbow_joint",
            # left_arm_wrap
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint",
            # right_arm_wrap
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint"
        ]

        # 학습(IsaacLab): sim.dt=1/120, decimation=2 => policy 60Hz
        self.physics_hz_target = 120.0
        self.policy_hz = 60.0
        self._accum_dt = 0.0

        self._policy_step_count = 0
        self._physics_step_count = 0

        # quat order는 "처음에 한 번"만 판별해서 고정
        self._quat_raw_is_xyzw = None  # True면 raw=(x,y,z,w), False면 raw=(w,x,y,z)

        # callback 등록은 한 번만
        self._callback_registered = False
        self._is_running = False

        # box hold 모드: 기본적으로 고정(낙하/충돌 방지) -> K 누르면 throw하면서 hold 해제
        self._box_hold_mode = True

        # policy 토글(디버그): P 누르면 policy on/off
        self._use_policy = True

        # 마지막 action hold
        self._last_action_policy_order = None
        self.prev_action_policy_order = None

        # 디버그 타이머
        self._t0_wall = time.time()

    # ----------------------------------------------------------------------
    # Quaternion utils (학습 코드와 동일: wxyz)
    # ----------------------------------------------------------------------
    def _detect_and_lock_quat_order(self, q_raw):
        """초기 자세에서 w가 1에 가까운 위치를 보고 raw order를 '한 번만' 결정."""
        if self._quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float32).copy()
        if q.shape[0] != 4:
            self._quat_raw_is_xyzw = True  # fallback
            return

        # 초기 upright에서 w ~ 1.0 이어야 함
        if abs(q[3]) > 0.90:
            self._quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90:
            self._quat_raw_is_xyzw = False
        else:
            # 애매하면 IsaacSim 쪽이 보통 xyzw를 주는 경우가 많아서 그쪽으로
            self._quat_raw_is_xyzw = True

        if self.debug_mode:
            print(f"[DEBUG] quat raw order locked: "
                  f"{'xyzw(x,y,z,w)' if self._quat_raw_is_xyzw else 'wxyz(w,x,y,z)'}  raw={q_raw}")

    def _to_wxyz(self, q_raw):
        q = np.asarray(q_raw, dtype=np.float32).copy()
        if q.shape[0] != 4:
            return q
        self._detect_and_lock_quat_order(q_raw)
        if self._quat_raw_is_xyzw:
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        else:
            return q

    def _quat_conj(self, q):
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack(
            [
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
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
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)
        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)
        r20 = 2*(x*z - y*w)
        r21 = 2*(y*z + x*w)
        return torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)

    def _yaw_from_wxyz(self, q_wxyz):
        w, x, y, z = q_wxyz
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def _rotate_vec_by_yaw(self, yaw, v3):
        c = math.cos(yaw)
        s = math.sin(yaw)
        x, y, z = float(v3[0]), float(v3[1]), float(v3[2])
        return np.array([c*x - s*y, s*x + c*y, z], dtype=np.float32)

    # ----------------------------------------------------------------------
    # Scene setup
    # ----------------------------------------------------------------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.78], dtype=np.float32))

        self.box = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name="box",
                position=np.array([2.5, 0.0, 1.2], dtype=np.float32),  # 초기에는 멀리 둠(hold 모드에서 계속 위치 잡아줄 거라 큰 의미 없음)
                scale=np.array([0.4, 0.3, 0.3], dtype=np.float32),
                mass=5.0,
                color=np.array([0.0, 0.8, 0.0], dtype=np.float32),
            )
        )

        # contact sensors (없어도 정책이 당장 폭주하진 않아야 함 -> 실패하면 0으로 대체)
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

        return

    # ----------------------------------------------------------------------
    # Post load / reset hooks
    # ----------------------------------------------------------------------
    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")
        self._box = self._world.scene.get_object("box")

        self._try_set_physics_dt(1.0 / self.physics_hz_target)

        # timeline events
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        # init robot handles
        self._robot.initialize()

        # lock quat order immediately using current pose
        _, q_raw = self._robot.get_world_pose()
        self._detect_and_lock_quat_order(q_raw)

        # load policy
        print(f">>> Loading Policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        self.num_dof = self._robot.num_dof
        self.robot_dof_names = list(self._robot.dof_names)

        print(f">>> Robot DOFs: {self.num_dof}")
        if self.debug_mode:
            print("[DEBUG] robot_dof_names:")
            for i, n in enumerate(self.robot_dof_names):
                print(f"  - {i:02d}: {n}")

        # Joint Mapping (Policy Order -> Robot Order)
        self.policy_to_robot_indices = []
        for pj in self.policy_joint_order:
            if pj in self.robot_dof_names:
                self.policy_to_robot_indices.append(self.robot_dof_names.index(pj))
            else:
                carb.log_error(f"[ERROR] Joint {pj} not found in robot USD DOF names!")

        if len(self.policy_to_robot_indices) != 23:
            print(f"[FATAL] policy_to_robot_indices len={len(self.policy_to_robot_indices)} (expected 23)")
        else:
            print(">>> policy_to_robot_indices OK (23)")

        self.policy_to_robot_indices = torch.tensor(
            self.policy_to_robot_indices, device=self.device, dtype=torch.long
        )

        # Buffers (Robot Order 기준)
        self.default_pos = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self.action_scale = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)

        # Buffers (Policy Order 기준)
        self.prev_action_policy_order = torch.zeros(23, device=self.device, dtype=torch.float32)
        self._last_action_policy_order = torch.zeros(23, device=self.device, dtype=torch.float32)

        # default pose & scale (scene_objects_cfg/env_cfg와 동일)
        init_pos_dict = {
            "hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2,
            "shoulder_pitch": 0.2, "elbow": 0.5
        }
        scale_dict = {
            "hip_pitch": 0.35, "knee": 0.35, "ankle_pitch": 0.35,
            "hip_roll": 0.22, "ankle_roll": 0.22,
            "hip_yaw": 0.10,
            "waist_yaw": 0.25,
            "shoulder_pitch": 1.2, "elbow": 1.2,
            "shoulder_roll": 0.7, "shoulder_yaw": 0.7, "wrist_roll": 0.7
        }

        for i, name in enumerate(self.robot_dof_names):
            # Scale
            found_scale = False
            for key, val in scale_dict.items():
                if key in name:
                    self.action_scale[i] = float(val)
                    found_scale = True
                    break
            if not found_scale:
                self.action_scale[i] = 1.0

            # Default Pose
            for key, val in init_pos_dict.items():
                if key in name:
                    self.default_pos[i] = float(val)

        # Actuator gains (controller gains)
        self._configure_joint_drives_like_training()

        # callback은 "한 번만" 등록
        if not self._callback_registered:
            self._world.add_physics_callback("policy_physics_step", callback_fn=self._robot_control)
            self._callback_registered = True
            if self.debug_mode:
                print("[DEBUG] physics callback registered once: policy_physics_step")

        # keyboard
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

        # 첫 상태 리셋
        self._hard_reset_all()

        # 정책 입출력 차원 체크
        try:
            test_obs = self._get_observation()
            with torch.no_grad():
                test_out = self.policy(test_obs.unsqueeze(0))
                if isinstance(test_out, (tuple, list)):
                    test_out = test_out[0]
            print(f"[DEBUG] obs_dim={int(test_obs.numel())}, policy_out_dim={int(test_out.numel())}")
        except Exception:
            print("[DEBUG] policy I/O dim check failed:")
            traceback.print_exc()

        print(">>> ExampleV3 setup_post_load done.")
        return

    async def setup_post_reset(self):
        if self.debug_mode:
            print("[DEBUG] setup_post_reset called")
        self._hard_reset_all()
        return

    def _hard_reset_all(self):
        # flags / counters
        self._accum_dt = 0.0
        self._physics_step_count = 0
        self._policy_step_count = 0
        if self.prev_action_policy_order is not None:
            self.prev_action_policy_order.zero_()
        if self._last_action_policy_order is not None:
            self._last_action_policy_order.zero_()

        # robot reset
        if self._robot and self._robot.is_valid():
            self._robot.initialize()
            self._configure_joint_drives_like_training()
            self._robot.set_world_pose(position=np.array([0.0, 0.0, 0.78], dtype=np.float32))
            self._robot.set_joint_positions(self.default_pos.detach().cpu().numpy())
            self._robot.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float32))

        # box reset: hold 모드 ON
        self._box_hold_mode = True
        self._place_box_hold_pose()

    # ----------------------------------------------------------------------
    # Timeline events
    # ----------------------------------------------------------------------
    def _on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._is_running = True
            if self.debug_mode:
                print("[DEBUG] Timeline PLAY -> hard reset (no callback re-register)")
            self._hard_reset_all()

        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._is_running = False
            if self.debug_mode:
                print("[DEBUG] Timeline STOP")

    # ----------------------------------------------------------------------
    # Keyboard
    #   K: throw (stage2-like) + hold 해제
    #   P: policy on/off 토글
    #   H: box hold 모드 토글(hold면 박스 고정, off면 그냥 물리)
    #   R: hard reset (로봇/박스/카운터)
    # ----------------------------------------------------------------------
    def _on_key(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return

        if event.input == carb.input.KeyboardInput.K:
            self.throw_box_stage2_like_training()

        elif event.input == carb.input.KeyboardInput.P:
            self._use_policy = not self._use_policy
            print(f">>> Toggle policy: {'ON' if self._use_policy else 'OFF'}")

        elif event.input == carb.input.KeyboardInput.H:
            self._box_hold_mode = not self._box_hold_mode
            print(f">>> Toggle box hold mode: {'ON(hold)' if self._box_hold_mode else 'OFF(dynamic)'}")

        elif event.input == carb.input.KeyboardInput.R:
            print(">>> Hard reset requested")
            self._hard_reset_all()

    # ----------------------------------------------------------------------
    # Box placement utilities
    # ----------------------------------------------------------------------
    def _place_box_hold_pose(self):
        """학습 stage0 느낌의 상대 위치로 박스를 '고정 배치'"""
        if not self._box or not self._robot:
            return
        base_pos, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)
        yaw = self._yaw_from_wxyz(base_quat)

        rel_p = np.array([1.55, 0.05, 0.26], dtype=np.float32)
        p_w = np.asarray(base_pos, dtype=np.float32) + self._rotate_vec_by_yaw(yaw, rel_p)

        self._box.set_world_pose(position=p_w)
        self._box.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        try:
            self._box.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        except Exception:
            pass

        if self.debug_mode:
            print(f"[DEBUG] box hold pose: pos={p_w}")

    def throw_box_stage2_like_training(self):
        """
        env_cfg stage2 분포:
          pos_x: (0.3, 0.5), pos_y: (-0.1, 0.1), pos_z: (0.3, 0.4)
          vel_x: (-2.0, -0.8), vel_y: (-0.1, 0.1), vel_z: (-0.2, 0.2)
        """
        if not self._box or not self._robot:
            return
        if not self._is_running:
            print("[WARN] not running (press Play first)")
            return

        print(">>> Throwing Box! (stage2-like)  [hold OFF]")
        self._box_hold_mode = False

        base_pos, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)
        yaw = self._yaw_from_wxyz(base_quat)

        rel_p = np.array(
            [np.random.uniform(0.3, 0.5),
             np.random.uniform(-0.1, 0.1),
             np.random.uniform(0.3, 0.4)], dtype=np.float32
        )
        rel_v = np.array(
            [np.random.uniform(-2.0, -0.8),
             np.random.uniform(-0.1, 0.1),
             np.random.uniform(-0.2, 0.2)], dtype=np.float32
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
            print(f"[DEBUG] throw rel_p={rel_p}, rel_v={rel_v} => pos_w={p_w}, vel_w={v_w}")

    # ----------------------------------------------------------------------
    # Observation (Shoulde be same structure with UROP_v4/mdp/observations.py)
    # ----------------------------------------------------------------------
    def _try_get_joint_torque_like_training(self, j_pos):
        jt = torch.zeros_like(j_pos)

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
                    if self.debug_mode and (not self._printed_torque_hint):
                        self._printed_torque_hint = True
                        print("[DEBUG] Using joint efforts from IsaacSim API for torque obs.")
            except Exception:
                pass
        else:
            if self.debug_mode and (not self._printed_torque_hint):
                self._printed_torque_hint = True
                print("[DEBUG] Torque obs not available -> using zeros.")

        jt = torch.clamp(jt * (1.0/80.0), -1.0, 1.0)
        return jt

    def _get_observation(self):
        base_pos, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)

        base_lin_w = torch.tensor(self._robot.get_linear_velocity(), device=self.device, dtype=torch.float32)
        base_ang_w = torch.tensor(self._robot.get_angular_velocity(), device=self.device, dtype=torch.float32)

        q = torch.tensor(base_quat, device=self.device, dtype=torch.float32)

        g_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        g_b = self._quat_rotate_inverse(q, g_world)
        lin_b = self._quat_rotate_inverse(q, base_lin_w)
        ang_b = self._quat_rotate_inverse(q, base_ang_w)

        j_pos = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
        j_torque = self._try_get_joint_torque_like_training(j_pos)

        proprio = torch.cat([g_b, lin_b, ang_b, j_pos, j_vel, j_torque], dim=-1)

        prev_act = self.prev_action_policy_order

        obj_pos, obj_quat_raw = self._box.get_world_pose()
        obj_quat = self._to_wxyz(obj_quat_raw)

        obj_lin_w = torch.tensor(self._box.get_linear_velocity(), device=self.device, dtype=torch.float32)
        obj_ang_w = torch.tensor(self._box.get_angular_velocity(), device=self.device, dtype=torch.float32)

        r_pos = torch.tensor(base_pos, device=self.device, dtype=torch.float32)
        rel_p_b = self._quat_rotate_inverse(q, torch.tensor(obj_pos, device=self.device, dtype=torch.float32) - r_pos)
        rel_v_b = self._quat_rotate_inverse(q, obj_lin_w - base_lin_w)
        rel_w_b = self._quat_rotate_inverse(q, obj_ang_w - base_ang_w)

        b_q = torch.tensor(obj_quat, device=self.device, dtype=torch.float32)
        rel_q = self._quat_mul(self._quat_conj(q), b_q)
        rel_r6 = self._quat_to_rot6d(rel_q)
        obj_rel = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)

        # contact (11 dims, scale=1/300)
        forces = []
        for name in CONTACT_SENSOR_LINKS:
            sensor = self.contact_sensors.get(name, None)
            val = 0.0
            if sensor and self._is_running:
                try:
                    reading = sensor.get_current_frame()
                    f = None
                    if isinstance(reading, dict):
                        if "force" in reading:
                            f = reading["force"]
                        elif "forces" in reading:
                            f = reading["forces"]
                    if f is not None:
                        val = float(np.linalg.norm(np.asarray(f)))
                except Exception:
                    pass
            forces.append(val)
        contact = torch.tensor(forces, device=self.device, dtype=torch.float32) * (1.0/300.0)

        obs = torch.cat([proprio, prev_act, obj_rel, contact], dim=-1)
        return obs

    # ----------------------------------------------------------------------
    # Control loop
    # ----------------------------------------------------------------------
    def _robot_control(self, step_size):
        if (not self._robot) or (not self._robot.handles_initialized):
            return

        self._physics_step_count += 1
        dt = float(step_size)
        if dt <= 0.0:
            return

        # box hold 모드면 매 physics step마다 박스를 고정 위치로 다시 세팅(=kinematic처럼)
        if self._is_running and self._box_hold_mode:
            self._place_box_hold_pose()

        # 1초마다 physics 상태 요약
        if self.debug_mode and (self._physics_step_count % self.debug_print_every_n_physics_steps == 0):
            try:
                p, qraw = self._robot.get_world_pose()
                q = self._to_wxyz(qraw)
                jp = self._robot.get_joint_positions()
                jv = self._robot.get_joint_velocities()
                print(f"[DEBUG] t~{time.time()-self._t0_wall:6.2f}s  phys_step={self._physics_step_count:06d}  running={self._is_running}  policy={'ON' if self._use_policy else 'OFF'}")
                print(f"        base_pos={np.array(p).round(3)}  base_quat_wxyz={np.array(q).round(3)}")
                print(f"        jp(min,max)=({float(np.min(jp)):+.3f},{float(np.max(jp)):+.3f})  jv(min,max)=({float(np.min(jv)):+.3f},{float(np.max(jv)):+.3f})")
            except Exception:
                pass

        if not self._is_running:
            return

        # policy off면 default pose만 유지(이 상태에서 로봇이 안정적으로 서야 "actuation/drive"가 정상)
        if not self._use_policy:
            self._apply_policy_action(torch.zeros(23, device=self.device))
            return

        self._accum_dt += dt

        # policy 60Hz
        if self._accum_dt < (1.0 / self.policy_hz):
            self._apply_policy_action(self._last_action_policy_order)
            return

        while self._accum_dt >= (1.0 / self.policy_hz):
            self._accum_dt -= (1.0 / self.policy_hz)

        try:
            obs = self._get_observation()
        except Exception:
            if self.debug_mode:
                print("[DEBUG] _get_observation exception:")
                traceback.print_exc()
            return

        # NaN 체크
        if torch.isnan(obs).any():
            print("[FATAL] NaN detected in observation -> policy will explode. Printing obs stats:")
            print("obs(min,max)=", float(torch.nanmin(obs)), float(torch.nanmax(obs)))
            return

        with torch.no_grad():
            out = self.policy(obs.unsqueeze(0))
            if isinstance(out, (tuple, list)):
                out = out[0]
            action_policy_order = out.squeeze(0)

        action_policy_order = torch.clamp(action_policy_order, -1.0, 1.0)

        self.prev_action_policy_order = action_policy_order.clone()
        self._last_action_policy_order = action_policy_order.clone()

        self._policy_step_count += 1

        # 디버그: 초반/주기적으로 action 포화 여부 출력
        if self.debug_mode:
            do_print = (self._policy_step_count <= self.debug_print_first_n_policy_steps) or \
                       (self._policy_step_count % self.debug_print_every_n_policy_steps == 0)
            if do_print:
                a_min = float(action_policy_order.min().item())
                a_max = float(action_policy_order.max().item())
                sat = (abs(a_min) > 0.98) or (abs(a_max) > 0.98)
                print(f"[DEBUG] policy_step={self._policy_step_count:05d}  phys_step={self._physics_step_count:06d}  dt={dt:.6f}  sat={sat}")
                print(f"        action(min,max)=({a_min:+.3f},{a_max:+.3f})  obs(min,max)=({float(obs.min()):+.3f},{float(obs.max()):+.3f})")

        self._apply_policy_action(action_policy_order)

    def _apply_policy_action(self, action_policy_order: torch.Tensor):
        if self.policy_to_robot_indices.numel() != 23:
            return

        action_robot_order = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        action_robot_order[self.policy_to_robot_indices] = action_policy_order

        target_pos = action_robot_order * self.action_scale + self.default_pos
        target_np = target_pos.detach().cpu().numpy().astype(np.float32)

        self._robot.apply_action(ArticulationAction(joint_positions=target_np))

        if self.debug_mode and (self._policy_step_count <= self.debug_print_first_n_policy_steps):
            tmin = float(target_pos.min().item())
            tmax = float(target_pos.max().item())
            print(f"        target_pos(min,max)=({tmin:+.3f},{tmax:+.3f})")

    # ----------------------------------------------------------------------
    # Gains / dt
    # ----------------------------------------------------------------------
    def _configure_joint_drives_like_training(self):
        if not self._robot:
            return

        ctrl = None
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
            is_leg = ("hip_" in name) or ("knee" in name) or ("ankle_" in name) or ("waist_" in name)
            is_arm = ("shoulder_" in name) or ("elbow" in name) or ("wrist_" in name)
            if is_leg:
                kp[i] = 120.0
                kd[i] = 6.0
            elif is_arm:
                kp[i] = 60.0
                kd[i] = 3.0
            else:
                kp[i] = 60.0
                kd[i] = 3.0

        ok_gain = False
        try:
            ctrl.set_gains(kps=kp, kds=kd)
            ok_gain = True
        except Exception:
            try:
                ctrl.set_gains(kp, kd)
                ok_gain = True
            except Exception:
                ok_gain = False

        if self.debug_mode:
            print(f"[DEBUG] set_gains ok={ok_gain}")
            if not ok_gain:
                print("[WARN] set_gains failed -> joint PD may be wrong.")

    def _try_set_physics_dt(self, dt):
        ok = False
        try:
            if hasattr(self._world, "set_physics_dt"):
                self._world.set_physics_dt(dt)
                ok = True
        except Exception:
            ok = False

        if not ok:
            try:
                if hasattr(self._world, "get_physics_context"):
                    pc = self._world.get_physics_context()
                    if hasattr(pc, "set_physics_dt"):
                        pc.set_physics_dt(dt)
                        ok = True
            except Exception:
                ok = False

        if self.debug_mode:
            print(f"[DEBUG] try_set_physics_dt({dt:.6f}) ok={ok} (target {self.physics_hz_target:.1f}Hz)")

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    def world_cleanup(self):
        self._timeline_sub = None
        try:
            self._input.unsubscribe_to_keyboard_events(self._sub_keyboard)
        except Exception:
            pass
        return
