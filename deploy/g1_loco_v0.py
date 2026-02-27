# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/g1_loco_v0.py]

import math
import time
import traceback
import numpy as np
import torch
import carb
import carb.input
import omni.appwindow
import omni.timeline

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction

# --------------------------------------------------------------------------
# [설정] 경로 지정
# --------------------------------------------------------------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_g1_loco_v0/2026-02-27_17-13-18/exported/policy.pt"
ROBOT_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd"
ROBOT_PRIM_PATH = "/World/G1"

PHYSICS_CALLBACK_NAME = "policy_physics_step_loco_v0"


# --------------------------------------------------------------------------
# [수학 유틸] IsaacLab 의존성 없이 순수 PyTorch로 쿼터니언 변환 구현
# --------------------------------------------------------------------------
def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply the inverse of a quaternion to a vector.
    q: [w, x, y, z] 형태의 쿼터니언 (Isaac Sim 표준)
    v: [x, y, z] 형태의 3차원 벡터
    """
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:4]
    q_xyz_inv = -q_xyz
    t = 2.0 * torch.cross(q_xyz_inv, v, dim=-1)
    v_rotated = v + q_w * t + torch.cross(q_xyz_inv, t, dim=-1)
    return v_rotated


class G1LocoV0(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 학습 시 설정: dt=0.008333(120Hz), decimation=2 -> policy 60Hz
        self.physics_hz_target = 120.0
        self.policy_hz = 60.0
        self._accum_dt = 0.0

        self.debug_mode = True
        self.debug_print_every_n_policy_steps = 120

        self._policy_step_count = 0
        self._physics_step_count = 0

        self._callback_registered = False
        self._is_running = False

        self._sim_time_since_reset = 0.0
        self.reset_settle_sec = 0.35

        # 정책 활성화 토글 (P키)
        self._use_policy = True

        # 명령 벡터 [lin_vel_x, lin_vel_y, ang_vel_z]
        self.command = torch.zeros(3, device=self.device)

        # 액션/옵저베이션 기록용
        self.prev_action = None
        self._last_action = None

        self._pending_play_reinit = False
        self._physics_callback_name = PHYSICS_CALLBACK_NAME
        self._sub_keyboard = None
        self._input = None
        self._keyboard = None

        self._world = None
        self._robot = None

        # [MOD] URDF의 29 DOF + 양손(14) = 총 43 DOF 이름 목록 및 매핑
        self.robot_dof_names = []
        self.num_dof = 43

    # ---------------- Scene setup ----------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.8], dtype=np.float32))

    # ---------------- Runtime callback / input helpers ----------------
    def _safe_remove_physics_callback(self):
        if self._world is None:
            return
        try:
            if hasattr(self._world, "remove_physics_callback"):
                self._world.remove_physics_callback(self._physics_callback_name)
        except Exception:
            pass
        self._callback_registered = False

    def _ensure_physics_callback_registered(self, force_rebind=False):
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
            return True
        except Exception as e:
            print(f"[WARN] add_physics_callback failed: {e}")
            self._callback_registered = False
            return False

    def _ensure_keyboard_subscription(self):
        try:
            if self._sub_keyboard is not None:
                return True
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
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

        self._try_set_physics_dt(1.0 / self.physics_hz_target)

        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        self._robot.initialize()

        print(f">>> Loading Policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        self.num_dof = self._robot.num_dof
        self.robot_dof_names = list(self._robot.dof_names)

        # 버퍼 초기화
        self.default_pos = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self.action_scale = torch.ones(self.num_dof, device=self.device, dtype=torch.float32) * 0.5  # default scale 0.5

        # init state 세팅
        init_pos_dict = {
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_pitch_joint": 0.2,
            "left_elbow_joint": 0.5,
            "right_elbow_joint": 0.5
        }

        for i, name in enumerate(self.robot_dof_names):
            if name in init_pos_dict:
                self.default_pos[i] = init_pos_dict[name]

        self.prev_action = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self._last_action = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)

        self._configure_joint_drives_like_training()
        self._ensure_physics_callback_registered(force_rebind=True)
        self._ensure_keyboard_subscription()
        self._hard_reset_all()

        print(">>> G1 Loco v0 setup_post_load done.")
        print(">>> Controls: [Play], Arrow Keys=Move, R=Hard reset, P=Policy on/off")

    def _hard_reset_all(self):
        self._accum_dt = 0.0
        self._physics_step_count = 0
        self._policy_step_count = 0
        self._sim_time_since_reset = 0.0
        self._pending_play_reinit = False
        self.command.zero_()

        if self.prev_action is not None:
            self.prev_action.zero_()
        if self._last_action is not None:
            self._last_action.zero_()

        if self._robot and self._robot.is_valid():
            try:
                self._robot.initialize()
            except Exception:
                pass

            self._configure_joint_drives_like_training()
            self._robot.set_world_pose(position=np.array([0.0, 0.0, 0.8], dtype=np.float32), 
                                       orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

            init_pos_np = self.default_pos.cpu().numpy()
            self._robot.set_joint_positions(init_pos_np)
            self._robot.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float32))

            try:
                self._robot.apply_action(ArticulationAction(joint_positions=init_pos_np))
            except Exception:
                pass

    # ---------------- Timeline ----------------
    def _on_timeline_event(self, event):
        etype = int(event.type)
        play_type = int(omni.timeline.TimelineEventType.PLAY)
        stop_type = int(omni.timeline.TimelineEventType.STOP)

        if etype == play_type:
            self._is_running = True
            self._ensure_physics_callback_registered(force_rebind=True)
            self._ensure_keyboard_subscription()
            self._pending_play_reinit = True

        elif etype == stop_type:
            self._is_running = False
            self._safe_remove_physics_callback()
            self._pending_play_reinit = True
            self._accum_dt = 0.0
            self._sim_time_since_reset = 0.0

    def _ensure_runtime_initialized(self):
        self._world = self.get_world()
        try:
            self._robot = self._world.scene.get_object("g1")
        except Exception:
            self._robot = None

        if self._robot is None:
            return False

        try:
            self._robot.initialize()
            self._configure_joint_drives_like_training()
            self._try_set_physics_dt(1.0 / self.physics_hz_target)
        except Exception:
            return False

        return bool(getattr(self._robot, "handles_initialized", False))

    # ---------------- Keyboard ----------------
    def _on_key(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return

        if event.input == carb.input.KeyboardInput.R:
            if self._is_running:
                self._ensure_runtime_initialized()
            self._hard_reset_all()

        elif event.input == carb.input.KeyboardInput.P:
            self._use_policy = not self._use_policy
            print(f">>> Toggle policy: {'ON' if self._use_policy else 'OFF'}")

    def _update_command_from_keyboard(self):
        lin_vel_x = 0.0
        ang_vel_z = 0.0
        
        # 키 입력으로 목표 속도 설정
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.UP):
            lin_vel_x += 1.0
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.DOWN):
            lin_vel_x -= 1.0
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.LEFT):
            ang_vel_z += 1.0
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.RIGHT):
            ang_vel_z -= 1.0
            
        self.command[0] = lin_vel_x
        self.command[1] = 0.0
        self.command[2] = ang_vel_z

    # ---------------- Observation ----------------
    def _get_observation(self):
        # 1. 상태 가져오기
        j_pos_np = self._robot.get_joint_positions()
        j_vel_np = self._robot.get_joint_velocities()
        _, base_quat_np = self._robot.get_world_pose()  # (w, x, y, z)
        base_lin_np = self._robot.get_linear_velocity()  
        base_ang_np = self._robot.get_angular_velocity() 

        j_pos = torch.tensor(j_pos_np, device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(j_vel_np, device=self.device, dtype=torch.float32)
        
        base_quat = torch.tensor(base_quat_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_lin_world = torch.tensor(base_lin_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_ang_world = torch.tensor(base_ang_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)

        # 2. 좌표 변환 (World -> Base)
        base_lin_vel = quat_apply_inverse(base_quat, base_lin_world).squeeze(0)
        base_ang_vel = quat_apply_inverse(base_quat, base_ang_world).squeeze(0)
        projected_gravity = quat_apply_inverse(base_quat, gravity_world).squeeze(0)

        # 3. 조인트 상대 위치
        joint_pos_rel = j_pos - self.default_pos

        # 4. 결합 (141차원)
        # obs = [lin_vel(3), ang_vel(3), proj_g(3), command(3), joint_pos(43), joint_vel(43), prev_action(43)]
        obs = torch.cat([
            base_lin_vel,       # 3
            base_ang_vel,       # 3
            projected_gravity,  # 3
            self.command,       # 3
            joint_pos_rel,      # 43
            j_vel,              # 43
            self.prev_action    # 43
        ])
        
        if obs.numel() != 141:
            raise RuntimeError(f"obs_dim mismatch: got {obs.numel()}, expected 141")
            
        return obs

    # ---------------- Control loop ----------------
    def _robot_control(self, step_size: float):
        dt = float(step_size)
        if dt <= 0.0: return

        need_reinit = (self._robot is None) or (not getattr(self._robot, "handles_initialized", False)) or self._pending_play_reinit
        if need_reinit and self._is_running:
            if not self._ensure_runtime_initialized(): return
            if self._pending_play_reinit:
                self._hard_reset_all()
                self._is_running = True

        if (self._robot is None) or (not getattr(self._robot, "handles_initialized", False)) or (not self._is_running):
            return

        self._physics_step_count += 1
        self._update_command_from_keyboard()

        self._sim_time_since_reset += dt
        if self._sim_time_since_reset < self.reset_settle_sec or not self._use_policy:
            self._apply_policy_action(torch.zeros(self.num_dof, device=self.device))
            return

        self._accum_dt += dt

        # Policy Hz (60Hz) 에 맞춰서만 추론 실행
        if self._accum_dt < (1.0 / self.policy_hz):
            self._apply_policy_action(self._last_action)
            return

        while self._accum_dt >= (1.0 / self.policy_hz):
            self._accum_dt -= (1.0 / self.policy_hz)

        try:
            obs = self._get_observation()
        except Exception as e:
            return

        with torch.no_grad():
            out = self.policy(obs.unsqueeze(0))
            if isinstance(out, (tuple, list)): out = out[0]
            action = out.squeeze(0)

        action = torch.clamp(action, -1.0, 1.0)
        self.prev_action = action.clone()
        self._last_action = action.clone()

        self._policy_step_count += 1
        self._apply_policy_action(action)

    def _apply_policy_action(self, action: torch.Tensor):
        if self._robot is None or not getattr(self._robot, "handles_initialized", False): return

        target_pos = self.default_pos + (self.action_scale * action)
        target_np = target_pos.detach().cpu().numpy().astype(np.float32)

        self._robot.apply_action(ArticulationAction(joint_positions=target_np))

    # ---------------- Gains / dt ----------------
    def _configure_joint_drives_like_training(self):
        if not self._robot: return
        try:
            ctrl = self._robot.get_articulation_controller()
        except: return
        if ctrl is None: return

        kp = np.zeros(self.num_dof, dtype=np.float32)
        kd = np.zeros(self.num_dof, dtype=np.float32)

        for i, name in enumerate(self.robot_dof_names):
            if ("hip_" in name) or ("knee" in name) or ("ankle_" in name) or ("waist_" in name):
                kp[i], kd[i] = 100.0, 5.0
            else:
                kp[i], kd[i] = 40.0, 2.0

        try:
            ctrl.set_gains(kps=kp, kds=kd)
        except:
            pass

    def _try_set_physics_dt(self, dt: float):
        try:
            self._world.set_physics_dt(dt)
        except Exception:
            pass

    # ---------------- Cleanup ----------------
    def world_cleanup(self):
        self._timeline_sub = None
        self._unsubscribe_keyboard()
        self._safe_remove_physics_callback()