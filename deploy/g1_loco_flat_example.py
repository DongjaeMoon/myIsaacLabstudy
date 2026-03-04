# Robust to Isaac Sim UI "World Controls -> RESET" and keyboard focus issues.
#
# Controls:
#   UP: forward
#   LEFT/RIGHT: yaw
#   R: reset (recommended)
#   P: policy on/off

import asyncio
import time
import re
import numpy as np
import torch
import carb
import carb.input
import omni.appwindow
import omni.timeline
import omni.kit.app

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/g1_flat/2026-03-02_21-32-28/exported/policy.pt"
ROBOT_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/Unitree/G1/g1_minimal.usd"
ROBOT_PRIM_PATH = "/World/G1"
PHYSICS_CALLBACK_NAME = "policy_physics_step_g1_loco_flat"

# --------------------------------------------------------------------------
# Joint Setup based on env.yaml
# --------------------------------------------------------------------------
DEFAULT_POS_DICT = {
    "left_hip_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.2,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_elbow_pitch_joint": 0.87,
    "right_elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}

def build_default_pos(joint_names, device):
    return torch.tensor([DEFAULT_POS_DICT.get(n, 0.0) for n in joint_names],
                        device=device, dtype=torch.float32)

def build_action_scale(joint_names, device):
    return torch.full((len(joint_names),), 0.5, device=device, dtype=torch.float32)

def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:4]
    q_xyz_inv = -q_xyz
    t = 2.0 * torch.cross(q_xyz_inv, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz_inv, t, dim=-1)

def infer_policy_io(policy, device):
    for obs_dim in range(10, 300):
        try:
            x = torch.zeros(1, obs_dim, device=device, dtype=torch.float32)
            with torch.no_grad():
                y = policy(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return obs_dim, int(y.shape[1])
        except Exception:
            pass
    raise RuntimeError("Could not infer policy IO dims.")

class G1LocoFlatExample(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.physics_hz_target = 200.0 
        self.policy_hz = 50.0 
        self._accum_dt = 0.0

        self.spawn_z = 0.74 
        self.reset_settle_sec = 0.8

        self.max_vx = 1.0
        self.max_wz = 1.0
        self.cmd_smooth = 0.15

        self._use_policy = True
        self._is_running = False
        self._ready = False
        self._pending_reset = False

        self._keys = {"UP": False, "LEFT": False, "RIGHT": False}
        
        self.cmd_dim = 3
        self.command = torch.zeros(self.cmd_dim, device=self.device)

        self.obs_dim = None
        self.act_dim = None
        self.controlled_joint_names = None

        self.default_pos_ctrl = None
        self.action_scale_ctrl = None
        self.prev_action = None
        self._last_action = None

        self._world = None
        self._robot = None
        self._timeline = None
        self._timeline_sub = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        self.policy = None

        self.dof_names = []
        self.dof_name_to_index = {}
        self.ctrl_dof_ids = None
        self.ctrl_dof_ids_np = None

        self._reinit_task = None
        self._reinit_requested = False
        self._sim_time_since_reset = 0.0
        self._last_debug_t = 0.0

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, self.spawn_z], dtype=np.float32))

    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")

        try:
            self._world.set_physics_dt(1.0 / self.physics_hz_target)
        except Exception:
            pass

        self._timeline = omni.timeline.get_timeline_interface()
        # 중복 구독 방지 (이벤트 폭주 원인 해결)
        if self._timeline_sub is None:
            self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
                self._on_timeline_event
            )

        self._world.add_physics_callback(PHYSICS_CALLBACK_NAME, callback_fn=self._robot_control)
        self._ensure_keyboard_subscription()

        print(f">>> Loading policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        self.obs_dim, self.act_dim = infer_policy_io(self.policy, self.device)
        print(f">>> Policy expects obs_dim={self.obs_dim}, action_dim={self.act_dim}")

        self.cmd_dim = self.obs_dim - 9 - (self.act_dim * 3)
        if self.cmd_dim < 1:
            self.cmd_dim = 3
        
        self.command = torch.zeros(self.cmd_dim, device=self.device)

        self._ready = False
        self._reinit_requested = True
        print(">>> setup_post_load done. Press Play.")

    def _on_timeline_event(self, event):
        etype = int(event.type)
        play_type = int(omni.timeline.TimelineEventType.PLAY)
        stop_type = int(omni.timeline.TimelineEventType.STOP)

        if etype == play_type:
            self._is_running = True
            self._ready = False
            self._pending_reset = False
            self._sim_time_since_reset = 0.0
            self._accum_dt = 0.0
            self._request_reinit("PLAY")

        elif etype == stop_type:
            self._is_running = False
            self._ready = False
            self._pending_reset = False
            self._sim_time_since_reset = 0.0
            self._accum_dt = 0.0

    def _ensure_keyboard_subscription(self):
        try:
            if self._sub_keyboard is not None:
                return True
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key_event)
            return True
        except Exception as e:
            print(f"[WARN] keyboard subscribe failed: {e}")
            self._sub_keyboard = None
            return False

    def _on_key_event(self, event, *args, **kwargs):
        if event.input == carb.input.KeyboardInput.R and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pending_reset = True
            return
        if event.input == carb.input.KeyboardInput.P and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._use_policy = not self._use_policy
            print(f">>> Policy: {'ON' if self._use_policy else 'OFF'}")
            return

        def set_key(name, down):
            self._keys[name] = bool(down)

        if event.input == carb.input.KeyboardInput.UP:
            set_key("UP", event.type == carb.input.KeyboardEventType.KEY_PRESS)
        elif event.input == carb.input.KeyboardInput.LEFT:
            set_key("LEFT", event.type == carb.input.KeyboardEventType.KEY_PRESS)
        elif event.input == carb.input.KeyboardInput.RIGHT:
            set_key("RIGHT", event.type == carb.input.KeyboardEventType.KEY_PRESS)

    def _update_command(self):
        vx_t = self.max_vx if self._keys["UP"] else 0.0
        wz_t = 0.0
        if self._keys["LEFT"]:
            wz_t += self.max_wz
        if self._keys["RIGHT"]:
            wz_t -= self.max_wz

        self.command[0] = (1.0 - self.cmd_smooth) * self.command[0] + self.cmd_smooth * vx_t
        self.command[1] = 0.0
        self.command[2] = (1.0 - self.cmd_smooth) * self.command[2] + self.cmd_smooth * wz_t
        
        if self.cmd_dim > 3:
            self.command[3:] = 0.0

    def _request_reinit(self, reason: str):
        self._reinit_requested = True
        if self._reinit_task is None or self._reinit_task.done():
            self._reinit_task = asyncio.ensure_future(self._reinit_async(reason))

    async def _reinit_async(self, reason: str):
        app = omni.kit.app.get_app()
        await app.next_update_async()
        await app.next_update_async()

        if not self._is_running:
            return

        self._world = self.get_world()
        try:
            self._robot = self._world.scene.get_object("g1")
        except Exception:
            self._robot = None

        if self._robot is None:
            return

        # 리셋 먹통 원인 해결: 이미 초기화된 로봇에 initialize() 중복 호출 금지
        try:
            if not self._robot.initialized:
                self._robot.initialize()
            elif hasattr(self._robot, "post_reset"):
                self._robot.post_reset()
        except Exception:
            pass

        self.dof_names = list(self._robot.dof_names)
        
        # [핵심] Isaac Lab 설정(preserve_order: false)에 맞춰 관절을 반드시 알파벳 순으로 정렬!
        self.controlled_joint_names = sorted(self.dof_names)
        
        self.dof_name_to_index = {n: i for i, n in enumerate(self.dof_names)}
        
        # 정렬된 정책 출력값 -> USD의 원래 관절 순서로 맵핑해주는 인덱스 배열
        self.ctrl_dof_ids = torch.tensor(
            [self.dof_name_to_index[n] for n in self.controlled_joint_names],
            device=self.device, dtype=torch.long
        )
        self.ctrl_dof_ids_np = self.ctrl_dof_ids.detach().cpu().numpy()

        self.default_pos_ctrl = build_default_pos(self.controlled_joint_names, self.device)
        self.action_scale_ctrl = build_action_scale(self.controlled_joint_names, self.device)

        self.prev_action = torch.zeros(self.act_dim, device=self.device, dtype=torch.float32)
        self._last_action = torch.zeros(self.act_dim, device=self.device, dtype=torch.float32)

        self._configure_joint_drives_like_training()
        self._hard_reset_pose_only()

        self._ready = True
        self._reinit_requested = False
        print(f">>> Ready ({reason}) | joints={len(self.controlled_joint_names)}")

    def _try_zero_root_vel(self):
        try:
            if hasattr(self._robot, "set_linear_velocity"):
                self._robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            if hasattr(self._robot, "set_angular_velocity"):
                self._robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        except Exception:
            pass

    def _hard_reset_pose_only(self):
        self._sim_time_since_reset = 0.0
        self._accum_dt = 0.0
        self.prev_action.zero_()
        self._last_action.zero_()
        self.command.zero_()

        try:
            self._robot.set_world_pose(
                position=np.array([0.0, 0.0, self.spawn_z], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        except Exception:
            pass

        self._try_zero_root_vel()

        try:
            q_full = np.array(self._robot.get_joint_positions(), dtype=np.float32)
            q_full[self.ctrl_dof_ids_np] = self.default_pos_ctrl.detach().cpu().numpy().astype(np.float32)
            
            # 명시적으로 물리 엔진에 타겟을 쏴줍니다.
            self._robot.set_joint_positions(q_full)
            self._robot.set_joint_velocities(np.zeros_like(q_full, dtype=np.float32))
            self._robot.get_articulation_controller().set_joint_position_targets(q_full)
        except Exception:
            pass

    def _get_observation(self) -> torch.Tensor | None:
        try:
            q_full = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
            qd_full = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
            _, base_quat_np = self._robot.get_world_pose()
            base_lin_np = self._robot.get_linear_velocity()
            base_ang_np = self._robot.get_angular_velocity()
        except Exception as e:
            self._ready = False
            self._request_reinit(f"GETTERS-FAIL: {type(e).__name__}")
            return None

        # 정책(알파벳 정렬 순서)에 맞게 관절 상태 매핑
        q = q_full[self.ctrl_dof_ids]
        qd = qd_full[self.ctrl_dof_ids]

        base_quat = torch.tensor(base_quat_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_lin_world = torch.tensor(base_lin_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_ang_world = torch.tensor(base_ang_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)

        base_lin_vel = quat_apply_inverse(base_quat, base_lin_world).squeeze(0)
        base_ang_vel = quat_apply_inverse(base_quat, base_ang_world).squeeze(0)
        projected_gravity = quat_apply_inverse(base_quat, gravity_world).squeeze(0)

        joint_pos_rel = q - self.default_pos_ctrl

        obs = torch.cat(
            [base_lin_vel, base_ang_vel, projected_gravity, self.command, joint_pos_rel, qd, self.prev_action],
            dim=0,
        )
        if obs.numel() != self.obs_dim:
            self._ready = False
            self._request_reinit("OBS-DIM-MISMATCH")
            return None
        return obs

    def _apply_action(self, action_ctrl: torch.Tensor):
        try:
            q_full = np.array(self._robot.get_joint_positions(), dtype=np.float32)
        except Exception:
            self._ready = False
            self._request_reinit("APPLY-get_joint_positions-fail")
            return

        q_target_ctrl = self.default_pos_ctrl + (self.action_scale_ctrl * action_ctrl)
        # 정책(알파벳 정렬) 출력값을 원래 USD 관절 순서로 다시 되돌려서 물리엔진에 적용
        q_full[self.ctrl_dof_ids_np] = q_target_ctrl.detach().cpu().numpy().astype(np.float32)

        try:
            self._robot.get_articulation_controller().set_joint_position_targets(q_full)
        except Exception:
            pass

    def _robot_control(self, step_size: float):
        if not self._is_running or self.policy is None:
            return

        if self._robot is None or (hasattr(self._robot, "is_valid") and not self._robot.is_valid()):
            if not self._reinit_requested:
                self._ready = False
                self._request_reinit("ROBOT-INVALID")
            return

        if not self._ready:
            if not self._reinit_requested:
                self._request_reinit("NOT-READY")
            return

        if self._pending_reset:
            self._pending_reset = False
            self._ready = False
            self._request_reinit("R-RESET")
            return

        dt = float(step_size)
        if dt <= 0.0:
            return

        self._update_command()

        self._sim_time_since_reset += dt
        if self._sim_time_since_reset < self.reset_settle_sec:
            return
        if not self._use_policy:
            return

        self._accum_dt += dt
        if self._accum_dt < (1.0 / self.policy_hz):
            self._apply_action(self._last_action)
            return
        while self._accum_dt >= (1.0 / self.policy_hz):
            self._accum_dt -= (1.0 / self.policy_hz)

        obs = self._get_observation()
        if obs is None:
            return

        with torch.no_grad():
            y = self.policy(obs.unsqueeze(0))
            if isinstance(y, (tuple, list)):
                y = y[0]
            action = y.squeeze(0)

        action = torch.clamp(action, -1.0, 1.0)
        
        self.prev_action = action.clone()
        self._last_action = action.clone()
        self._apply_action(action)

    def _configure_joint_drives_like_training(self):
        try:
            ctrl = self._robot.get_articulation_controller()
        except Exception:
            return
        if ctrl is None:
            return

        n = len(self.dof_names)
        kp = np.zeros(n, dtype=np.float32)
        kd = np.zeros(n, dtype=np.float32)

        for name, i in self.dof_name_to_index.items():
            if re.match(r".*_hip_yaw_joint", name) or re.match(r".*_hip_roll_joint", name):
                kp[i], kd[i] = 150.0, 5.0
            elif re.match(r".*_hip_pitch_joint", name) or re.match(r".*_knee_joint", name) or name == "torso_joint":
                kp[i], kd[i] = 200.0, 5.0
            elif re.match(r".*_ankle_pitch_joint", name) or re.match(r".*_ankle_roll_joint", name):
                kp[i], kd[i] = 20.0, 2.0
            else: 
                kp[i], kd[i] = 40.0, 10.0

        try:
            ctrl.set_gains(kps=kp, kds=kd)
        except Exception:
            pass

    def world_cleanup(self):
        self._timeline_sub = None
        try:
            if self._input is not None and self._sub_keyboard is not None:
                self._input.unsubscribe_to_keyboard_events(self._sub_keyboard)
        except Exception:
            pass
        self._sub_keyboard = None