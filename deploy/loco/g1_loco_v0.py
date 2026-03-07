# g1_loco_v0.py — Isaac Sim deploy for Isaac Lab TorchScript policy
# Robust to Isaac Sim UI "World Controls -> RESET" and keyboard focus issues.
#
# Controls:
#   UP: forward
#   LEFT/RIGHT: yaw
#   R: reset (recommended)
#   P: policy on/off
#
# Debug prints once per second:
#   cmd=[vx,vy,wz], key_state, ready, robot_valid

import asyncio
import time
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
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_g1_loco_v0/2026-03-02_03-40-35/exported/policy.pt"
ROBOT_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd"
ROBOT_PRIM_PATH = "/World/G1"
PHYSICS_CALLBACK_NAME = "policy_physics_step_g1_loco_v0"

# --------------------------------------------------------------------------
# Joint name sets
# 29: legs(12)+arms&waist(17)
# 43: + hands(14)
# --------------------------------------------------------------------------
LEGS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
ARMS_WAIST = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HANDS = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
JOINTS_29 = LEGS + ARMS_WAIST
JOINTS_43 = LEGS + ARMS_WAIST + HANDS

# Training-ish default pose (used as "default offset" in Isaac Lab)
DEFAULT_POS_DICT = {
    # legs
    "left_hip_pitch_joint": -0.2, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.4, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.2, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.4, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    # arms (nonzero in your init)
    "left_shoulder_pitch_joint": 0.2, "left_elbow_joint": 0.5,
    "right_shoulder_pitch_joint": 0.2, "right_elbow_joint": 0.5,
}

def build_default_pos(joint_names, device):
    return torch.tensor([DEFAULT_POS_DICT.get(n, 0.0) for n in joint_names],
                        device=device, dtype=torch.float32)

def build_action_scale(joint_names, device):
    # match your training: legs 0.5, arms+waist 0.05, hands 0.0
    s = torch.zeros(len(joint_names), device=device, dtype=torch.float32)
    for i, n in enumerate(joint_names):
        if n in LEGS:
            s[i] = 0.5
        elif n in ARMS_WAIST:
            s[i] = 0.05
        elif n in HANDS:
            s[i] = 0.0
        else:
            s[i] = 0.0
    return s

def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: [w,x,y,z], v: (...,3)
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:4]
    q_xyz_inv = -q_xyz
    t = 2.0 * torch.cross(q_xyz_inv, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz_inv, t, dim=-1)

def infer_policy_io(policy, device):
    # try the two common IsaacLab layouts we discussed
    for obs_dim in (99, 141):
        try:
            x = torch.zeros(1, obs_dim, device=device, dtype=torch.float32)
            y = policy(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return obs_dim, int(y.shape[1])
        except Exception:
            pass
    raise RuntimeError("Could not infer policy IO dims (expected 99 or 141).")

class G1LocoV0(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # physics/policy rates
        self.physics_hz_target = 120.0
        self.policy_hz = 60.0
        self._accum_dt = 0.0

        self.spawn_z = 0.78
        self.reset_settle_sec = 0.8

        # safer commands (you can increase later)
        self.max_vx = 0.35
        self.max_wz = 0.8
        self.cmd_smooth = 0.15

        # runtime flags
        self._use_policy = True
        self._is_running = False
        self._ready = False
        self._pending_reset = False

        # keyboard pressed state (robust after UI clicks)
        self._keys = {
            "UP": False, "LEFT": False, "RIGHT": False
        }

        # command [vx, vy, wz]
        self.command = torch.zeros(3, device=self.device)

        # will be set after policy infer
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

        # DOF mapping
        self.dof_names = []
        self.dof_name_to_index = {}
        self.ctrl_dof_ids = None
        self.ctrl_dof_ids_np = None

        # async reinit
        self._reinit_task = None
        self._reinit_requested = False
        self._sim_time_since_reset = 0.0
        self._last_debug_t = 0.0

    # ---------------- scene ----------------
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
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        # callback once
        self._world.add_physics_callback(PHYSICS_CALLBACK_NAME, callback_fn=self._robot_control)

        # keyboard
        self._ensure_keyboard_subscription()

        # load policy
        print(f">>> Loading policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        self.obs_dim, self.act_dim = infer_policy_io(self.policy, self.device)
        print(f">>> Policy expects obs_dim={self.obs_dim}, action_dim={self.act_dim}")

        if self.obs_dim == 99 and self.act_dim == 29:
            self.controlled_joint_names = JOINTS_29
        elif self.obs_dim == 141 and self.act_dim == 43:
            self.controlled_joint_names = JOINTS_43
        else:
            raise RuntimeError(f"Unexpected policy IO: obs={self.obs_dim}, act={self.act_dim}")

        self.default_pos_ctrl = build_default_pos(self.controlled_joint_names, self.device)
        self.action_scale_ctrl = build_action_scale(self.controlled_joint_names, self.device)

        self.prev_action = torch.zeros(self.act_dim, device=self.device, dtype=torch.float32)
        self._last_action = torch.zeros(self.act_dim, device=self.device, dtype=torch.float32)

        self._ready = False
        self._reinit_requested = True

        print(">>> setup_post_load done. Press Play.")
        print(">>> Controls: UP/LEFT/RIGHT, R=Reset, P=Policy toggle")

    # ---------------- timeline ----------------
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

    # ---------------- keyboard (event-based pressed state) ----------------
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

        # pressed-state tracking for arrows
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

    # ---------------- async reinit ----------------
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

        # after UI reset, old handle may be invalid: reacquire
        try:
            self._robot = self._world.scene.get_object("g1")
        except Exception:
            self._robot = None

        if self._robot is None:
            print("[WARN] Could not reacquire robot 'g1'.")
            return

        try:
            self._robot.initialize()
        except Exception as e:
            print(f"[WARN] robot.initialize failed: {e}")
            return

        self.dof_names = list(self._robot.dof_names)
        self.dof_name_to_index = {n: i for i, n in enumerate(self.dof_names)}
        missing = [n for n in self.controlled_joint_names if n not in self.dof_name_to_index]
        if missing:
            raise RuntimeError(f"[FATAL] USD missing joints: {missing}")

        self.ctrl_dof_ids = torch.tensor(
            [self.dof_name_to_index[n] for n in self.controlled_joint_names],
            device=self.device, dtype=torch.long
        )
        self.ctrl_dof_ids_np = self.ctrl_dof_ids.detach().cpu().numpy()

        self._configure_joint_drives_like_training()
        self._hard_reset_pose_only()

        self._ready = True
        self._reinit_requested = False
        print(f">>> Ready ({reason}) | joints={len(self.controlled_joint_names)} obs={self.obs_dim} act={self.act_dim}")

    def _try_zero_root_vel(self):
        try:
            if hasattr(self._robot, "set_linear_velocity"):
                self._robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        except Exception:
            pass
        try:
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
            self.policy.reset()
        except Exception:
            pass

        if self._robot is None or (not self._robot.is_valid()) or self.ctrl_dof_ids_np is None:
            return

        try:
            self._robot.set_world_pose(
                position=np.array([0.0, 0.0, self.spawn_z], dtype=np.float32),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
        except Exception:
            pass

        self._try_zero_root_vel()

        # set joints
        try:
            q_full = np.array(self._robot.get_joint_positions(), dtype=np.float32)
        except Exception:
            return
        q_full[self.ctrl_dof_ids_np] = self.default_pos_ctrl.detach().cpu().numpy().astype(np.float32)

        try:
            self._robot.set_joint_positions(q_full)
            self._robot.set_joint_velocities(np.zeros_like(q_full, dtype=np.float32))
            self._robot.apply_action(ArticulationAction(joint_positions=q_full))
        except Exception:
            pass

    # ---------------- obs/action ----------------
    def _get_observation(self) -> torch.Tensor | None:
        try:
            q_full = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
            qd_full = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
            _, base_quat_np = self._robot.get_world_pose()
            base_lin_np = self._robot.get_linear_velocity()
            base_ang_np = self._robot.get_angular_velocity()
        except Exception as e:
            # common after UI reset: schedule reinit
            self._ready = False
            self._request_reinit(f"GETTERS-FAIL: {type(e).__name__}")
            return None

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
        q_full[self.ctrl_dof_ids_np] = q_target_ctrl.detach().cpu().numpy().astype(np.float32)

        try:
            self._robot.apply_action(ArticulationAction(joint_positions=q_full))
        except Exception:
            pass

    # ---------------- main loop ----------------
    def _robot_control(self, step_size: float):
        if not self._is_running or self.policy is None:
            return

        # if UI reset invalidated robot, we will auto-reinit
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

        # debug once per second
        now = time.time()
        if now - self._last_debug_t > 1.0:
            self._last_debug_t = now
            print(f"[DBG] ready={self._ready} robot_valid={self._robot.is_valid() if hasattr(self._robot,'is_valid') else True} "
                  f"keys={self._keys} cmd={self.command.detach().cpu().numpy().round(3)}")

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
        if action.numel() != self.act_dim:
            self._ready = False
            self._request_reinit("ACTION-DIM-MISMATCH")
            return

        self.prev_action = action.clone()
        self._last_action = action.clone()
        self._apply_action(action)

    def _configure_joint_drives_like_training(self):
        # approximate training gains:
        # legs+waist: 120/5, arms: 40/2, hands: 2/0.1
        try:
            ctrl = self._robot.get_articulation_controller()
        except Exception:
            return
        if ctrl is None:
            return

        n = len(self.dof_names)
        kp = np.zeros(n, dtype=np.float32)
        kd = np.zeros(n, dtype=np.float32)

        waist = {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
        legs_set = set(LEGS)
        arms_set = set(ARMS_WAIST)
        hands_set = set(HANDS)

        for name, i in self.dof_name_to_index.items():
            if name in legs_set or name in waist:
                kp[i], kd[i] = 120.0, 5.0
            elif name in arms_set:
                kp[i], kd[i] = 40.0, 2.0
            elif name in hands_set:
                kp[i], kd[i] = 2.0, 0.1
            else:
                kp[i], kd[i] = 0.0, 0.0

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