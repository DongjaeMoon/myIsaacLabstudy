# g1_loco_v3_deploy.py — Isaac Sim deploy for Isaac Lab TorchScript policy (G1 locomotion v3)
# v3.3 (ported lessons from your successful example_v5 deploy)
#
# Why v5 deploy "just worked":
#   - It uses a USD whose PhysX drives / limits are already sane for position control.
#   - It robustly re-initializes runtime handles after STOP/PLAY (handles_initialized) and re-binds callbacks.
#   - It uses keyboard EVENT subscription (reliable in Isaac Sim) and stores pressed-state.
#
# This v3.3 applies the same patterns:
#   1) Robust STOP/PLAY: remove/re-add physics callback, re-init articulation after PLAY, then hard-reset pose.
#   2) Keyboard events: pressed-state for Arrow + WASD; prints on first key event so you KNOW it's received.
#   3) Actuation hardening:
#        - set articulation controller gains
#        - set DOF properties (stiffness/damping/maxEffort/armature) if available
#        - (best-effort) apply PhysX rigid-body & articulation-root properties similar to Isaac Lab spawn overrides
#   4) Observation + action ordering matches your env_cfg.py / scene_objects_cfg.py exactly.
#
# Controls:
#   Arrow UP / W : faster forward
#   Arrow DOWN / S : stop (vx=0, may fall because off-distribution)
#   Arrow LEFT / A : yaw left
#   Arrow RIGHT / D : yaw right
#   R : hard reset pose
#   P : policy on/off

import math
import time
import asyncio
from typing import Dict, Optional

import numpy as np
import torch
import carb
import omni.appwindow
import omni.timeline
import omni.kit.app
import omni.usd

from pxr import Usd, UsdPhysics, Gf, PhysxSchema

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction


# ------------------ Paths (EDIT) ------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_g1_loco_v3/2026-03-04_01-20-37/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_g1_loco_v3/g1_29dof.usd"
ROBOT_PRIM_PATH = "/World/G1"
PHYSICS_CALLBACK_NAME = "policy_physics_step_g1_loco_v3"


# ------------------ Training joint order (scene_objects_cfg.G1_29_JOINTS) ------------------
JOINTS_29 = [
    # Legs (12): (yaw, roll, pitch, knee, ankle_pitch, ankle_roll) x 2
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Arms (14)
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

# Training init_state joint_pos (scene_objects_cfg.py)
DEFAULT_POS_DICT = {
    # Legs
    "left_hip_pitch_joint": -0.20, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.42, "left_ankle_pitch_joint": -0.23, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.20, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.42, "right_ankle_pitch_joint": -0.23, "right_ankle_roll_joint": 0.0,
    # Waist
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
    # Arms
    "left_shoulder_pitch_joint": 0.35, "left_shoulder_roll_joint": 0.0, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.85, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35, "right_shoulder_roll_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.85, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
    **{j: 0.0 for j in HANDS},
}

ACTION_SCALE = 0.5  # ActionsCfg.joint_pos.scale


# ------------------ Quaternion utils (like your v5) ------------------
def _detect_quat_order(q_raw: np.ndarray) -> bool:
    """Return True if raw is xyzw, False if wxyz. Lock based on upright pose."""
    q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        return True
    if abs(q[3]) > 0.90:
        return True   # xyzw
    if abs(q[0]) > 0.90:
        return False  # wxyz
    # fallback: assume xyzw in Isaac Sim
    return True

def _to_wxyz(q_raw: np.ndarray, raw_is_xyzw: bool) -> np.ndarray:
    q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if raw_is_xyzw:
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return q

def quat_apply_inverse(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q_wxyz[..., 0:1]
    q_xyz = q_wxyz[..., 1:4]
    q_xyz_inv = -q_xyz
    t = 2.0 * torch.cross(q_xyz_inv, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz_inv, t, dim=-1)


def infer_policy_io(policy, device):
    for obs_dim in (99,):
        x = torch.zeros(1, obs_dim, device=device, dtype=torch.float32)
        y = policy(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return obs_dim, int(y.shape[1])
    raise RuntimeError("Could not infer policy IO.")


class G1LocoV3Deploy(BaseSample):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Training timing: dt=0.005, decimation=4 -> policy 50 Hz
        self.physics_dt = 0.005
        self.decimation = 4
        self.physics_hz = 1.0 / self.physics_dt
        self.policy_hz = self.physics_hz / self.decimation
        self._accum_dt = 0.0

        # Spawn
        self.spawn_z = 0.78
        self.reset_settle_sec = 0.6
        self._sim_time_since_reset = 0.0

        # Commands: training had lin_vel_x in [0,1], standing_envs=0.02
        self.vx_idle = 0.25
        self.max_vx = 0.55
        self.max_wz = 0.8
        self.cmd_smooth = 0.12

        # state
        self._is_running = False
        self._use_policy = True
        self._pending_play_reinit = False
        self._callback_registered = False

        # key pressed state (event-based)
        self._keys = {"UP": False, "DOWN": False, "LEFT": False, "RIGHT": False, "W": False, "A": False, "S": False, "D": False}
        self._seen_any_key = False

        # handles
        self._world = None
        self._robot = None
        self._timeline = None
        self._timeline_sub = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None

        # policy
        self.policy = None
        self.obs_dim = None
        self.act_dim = None

        # quat order lock
        self._quat_raw_is_xyzw = None

        # DOF mapping
        self.dof_names = []
        self.dof_name_to_index = {}
        self.ctrl_dof_ids = None
        self.ctrl_dof_ids_np = None
        self.lock_dof_ids = None
        self.lock_dof_ids_np = None

        # buffers
        self.default_pos_ctrl = None
        self.action_scale_ctrl = None
        self.prev_action = None
        self._last_action = None

        # IMPORTANT: full default target for ALL dofs (prevents drift on uncontrolled dofs)
        self.default_pos_all = None   # torch (num_dof,)
        self.default_pos_all_np = None

        self.command = torch.zeros(3, device=self.device)

        self._last_dbg_t = 0.0

    # ---------------- Scene ----------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, self.spawn_z], dtype=np.float32))

    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")

        # dt
        try:
            self._world.set_physics_dt(self.physics_dt)
        except Exception:
            pass

        # timeline events
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(self._on_timeline_event)

        # policy
        print(f">>> Loading policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()
        self.obs_dim, self.act_dim = infer_policy_io(self.policy, self.device)
        print(f">>> Policy expects obs_dim={self.obs_dim}, action_dim={self.act_dim}")

        # buffers
        self.default_pos_ctrl = torch.tensor([DEFAULT_POS_DICT.get(n, 0.0) for n in JOINTS_29], device=self.device, dtype=torch.float32)
        self.action_scale_ctrl = torch.full((len(JOINTS_29),), float(ACTION_SCALE), device=self.device, dtype=torch.float32)
        self.prev_action = torch.zeros(self.act_dim, device=self.device, dtype=torch.float32)
        self._last_action = torch.zeros(self.act_dim, device=self.device, dtype=torch.float32)

        # Register callback + keyboard (like v5)
        self._ensure_physics_callback_registered(force_rebind=True)
        self._ensure_keyboard_subscription()

        # Pre-initialize + reset pose BEFORE play
        await self._ensure_runtime_initialized(also_reset=True)

        print(">>> setup_post_load done. Press Play.")
        print(">>> Controls: Arrow or WASD, R=Reset, P=Policy toggle")
        print(f">>> Rates: physics_dt={self.physics_dt} ({self.physics_hz:.1f} Hz), policy={self.policy_hz:.1f} Hz")
        print(f">>> vx_idle={self.vx_idle}")

    # ---------------- Timeline (ported from v5) ----------------
    def _on_timeline_event(self, event):
        etype = int(event.type)
        play_type = int(omni.timeline.TimelineEventType.PLAY)
        stop_type = int(omni.timeline.TimelineEventType.STOP)

        if etype == play_type:
            self._is_running = True
            self._ensure_physics_callback_registered(force_rebind=True)
            self._ensure_keyboard_subscription()
            self._pending_play_reinit = True
            self._accum_dt = 0.0
            self._sim_time_since_reset = 0.0
            if not self._seen_any_key:
                print(">>> (Hint) If keys don't respond, click the viewport once to give it focus.")
        elif etype == stop_type:
            self._is_running = False
            self._safe_remove_physics_callback()
            self._pending_play_reinit = True
            self._accum_dt = 0.0
            self._sim_time_since_reset = 0.0

    # ---------------- Callback / keyboard helpers (from v5) ----------------
    def _safe_remove_physics_callback(self):
        if self._world is None:
            return
        try:
            if hasattr(self._world, "remove_physics_callback"):
                self._world.remove_physics_callback(PHYSICS_CALLBACK_NAME)
        except Exception:
            pass
        self._callback_registered = False

    def _ensure_physics_callback_registered(self, force_rebind: bool = False):
        try:
            self._world = self.get_world()
        except Exception:
            return False

        if self._world is None:
            return False

        if force_rebind:
            self._safe_remove_physics_callback()

        if self._callback_registered:
            return True

        try:
            self._world.add_physics_callback(PHYSICS_CALLBACK_NAME, callback_fn=self._robot_control)
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
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key_event)
            return True
        except Exception as e:
            print(f"[WARN] keyboard subscribe failed: {e}")
            self._sub_keyboard = None
            return False

    def _on_key_event(self, event, *args, **kwargs):
        # record that we are receiving key events
        if not self._seen_any_key and event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._seen_any_key = True
            print(">>> Keyboard events OK (received first key press).")

        # toggles
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self._pending_play_reinit = True
                return
            if event.input == carb.input.KeyboardInput.P:
                self._use_policy = not self._use_policy
                print(f">>> Policy: {'ON' if self._use_policy else 'OFF'}")
                return

        # pressed state (both press/release)
        down = (event.type == carb.input.KeyboardEventType.KEY_PRESS)

        if event.input == carb.input.KeyboardInput.UP:
            self._keys["UP"] = down
        elif event.input == carb.input.KeyboardInput.DOWN:
            self._keys["DOWN"] = down
        elif event.input == carb.input.KeyboardInput.LEFT:
            self._keys["LEFT"] = down
        elif event.input == carb.input.KeyboardInput.RIGHT:
            self._keys["RIGHT"] = down
        elif event.input == carb.input.KeyboardInput.W:
            self._keys["W"] = down
        elif event.input == carb.input.KeyboardInput.S:
            self._keys["S"] = down
        elif event.input == carb.input.KeyboardInput.A:
            self._keys["A"] = down
        elif event.input == carb.input.KeyboardInput.D:
            self._keys["D"] = down

    # ---------------- Runtime init (ported from v5) ----------------
    def _reacquire_robot_handle(self):
        self._world = self.get_world()
        try:
            self._robot = self._world.scene.get_object("g1")
        except Exception:
            self._robot = None

    async def _ensure_runtime_initialized(self, also_reset: bool):
        # wait a couple frames to allow scene to be ready
        app = omni.kit.app.get_app()
        await app.next_update_async()
        await app.next_update_async()

        self._reacquire_robot_handle()
        if self._robot is None:
            return False

        # initialize articulation handles
        try:
            self._robot.initialize()
        except Exception as e:
            print(f"[WARN] robot.initialize() failed: {e}")
            return False

        # lock quat order once
        try:
            _, q_raw = self._robot.get_world_pose()
            if self._quat_raw_is_xyzw is None:
                self._quat_raw_is_xyzw = _detect_quat_order(q_raw)
                print(f">>> quat raw order locked: {'xyzw(x,y,z,w)' if self._quat_raw_is_xyzw else 'wxyz(w,x,y,z)'} raw={np.array(q_raw).round(3)}")
        except Exception:
            if self._quat_raw_is_xyzw is None:
                self._quat_raw_is_xyzw = True

        # dof mapping
        self.dof_names = list(self._robot.dof_names)
        self.dof_name_to_index = {n: i for i, n in enumerate(self.dof_names)}

        missing = [n for n in JOINTS_29 if n not in self.dof_name_to_index]
        if missing:
            raise RuntimeError(f"[FATAL] USD missing joints: {missing}")

        self.ctrl_dof_ids = torch.tensor([self.dof_name_to_index[n] for n in JOINTS_29], device=self.device, dtype=torch.long)
        self.ctrl_dof_ids_np = self.ctrl_dof_ids.detach().cpu().numpy()

        lock_names = [n for n in HANDS if n in self.dof_name_to_index]
        if lock_names:
            self.lock_dof_ids = torch.tensor([self.dof_name_to_index[n] for n in lock_names], device=self.device, dtype=torch.long)
            self.lock_dof_ids_np = self.lock_dof_ids.detach().cpu().numpy()
        else:
            self.lock_dof_ids = None
            self.lock_dof_ids_np = None

        # Apply spawn-like PhysX overrides + PD drives (THIS is the big difference vs your failing deploys)
        self._apply_spawn_like_physx_overrides()
        self._configure_joint_drives_like_training()

        # Build full default pose (ALL dofs)
        q0 = np.array(self._robot.get_joint_positions(), dtype=np.float32)
        # overwrite with training defaults where specified
        for name, idx in self.dof_name_to_index.items():
            if name in DEFAULT_POS_DICT:
                q0[idx] = float(DEFAULT_POS_DICT[name])
        self.default_pos_all_np = q0.astype(np.float32)
        self.default_pos_all = torch.tensor(self.default_pos_all_np, device=self.device, dtype=torch.float32)

        if also_reset:
            self._hard_reset_all()

        return bool(getattr(self._robot, "handles_initialized", True))

    def _hard_reset_all(self):
        self._accum_dt = 0.0
        self._sim_time_since_reset = 0.0
        self.prev_action.zero_()
        self._last_action.zero_()
        self.command.zero_()
        self.command[0] = float(self.vx_idle)

        # pose reset
        try:
            self._robot.set_world_pose(position=np.array([0.0, 0.0, self.spawn_z], dtype=np.float32))
        except Exception:
            pass

        try:
            self._robot.set_joint_positions(self.default_pos_all_np)
            self._robot.set_joint_velocities(np.zeros_like(self.default_pos_all_np, dtype=np.float32))
            self._robot.apply_action(ArticulationAction(joint_positions=self.default_pos_all_np))
        except Exception:
            pass

    # ---------------- PhysX overrides (approx Isaac Lab spawn overrides) ----------------
    def _apply_spawn_like_physx_overrides(self):
        """Best-effort: apply rigid/articulation root properties similar to scene_objects_cfg spawn."""
        try:
            stage = omni.usd.get_context().get_stage()
            robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
            if not robot_prim.IsValid():
                return

            # Articulation root
            try:
                PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
                art_api = PhysxSchema.PhysxArticulationAPI.Get(stage, ROBOT_PRIM_PATH)
                # solver iterations (scene_objects_cfg: 16/1)
                for name, val in [("CreateSolverPositionIterationCountAttr", 16), ("CreateSolverVelocityIterationCountAttr", 1)]:
                    if hasattr(art_api, name):
                        try:
                            getattr(art_api, name)().Set(int(val))
                        except Exception:
                            pass
            except Exception:
                pass

            # Rigid bodies under robot
            for prim in Usd.PrimRange(robot_prim):
                # only apply to rigid bodies
                try:
                    rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
                    if not rb_api:
                        continue
                except Exception:
                    continue

                # match scene_objects_cfg rigid_props
                for attr_name, val in [
                    ("CreateDisableGravityAttr", False),
                    ("CreateLinearDampingAttr", 0.0),
                    ("CreateAngularDampingAttr", 0.0),
                    ("CreateMaxLinearVelocityAttr", 1000.0),
                    ("CreateMaxAngularVelocityAttr", 1000.0),
                    ("CreateMaxDepenetrationVelocityAttr", 1.0),
                    ("CreateRetainAccelerationsAttr", False),
                ]:
                    if hasattr(rb_api, attr_name):
                        try:
                            getattr(rb_api, attr_name)().Set(val)
                        except Exception:
                            pass
        except Exception:
            pass

    # ---------------- PD drives (the other big difference) ----------------
    def _configure_joint_drives_like_training(self):
        """Set gains and effort caps similar to scene_objects_cfg actuators."""
        if self._robot is None:
            return
        # Prepare arrays in dof order
        n = len(self.dof_names)
        kp = np.zeros(n, dtype=np.float32)
        kd = np.zeros(n, dtype=np.float32)
        max_eff = np.zeros(n, dtype=np.float32)
        arm = np.zeros(n, dtype=np.float32)

        def set_group(name: str, i: int):
            # legs + waist
            if name.endswith("hip_yaw_joint"):
                kp[i], kd[i], max_eff[i], arm[i] = 150.0, 5.0, 160.0, 0.05
            elif name.endswith("hip_roll_joint"):
                kp[i], kd[i], max_eff[i], arm[i] = 150.0, 5.0, 160.0, 0.05
            elif name.endswith("hip_pitch_joint"):
                kp[i], kd[i], max_eff[i], arm[i] = 200.0, 5.0, 160.0, 0.05
            elif name.endswith("knee_joint"):
                kp[i], kd[i], max_eff[i], arm[i] = 200.0, 5.0, 160.0, 0.05
            elif name.startswith("waist_"):
                kp[i], kd[i], max_eff[i], arm[i] = 120.0, 5.0, 160.0, 0.05
            # ankles
            elif name.endswith("ankle_pitch_joint") or name.endswith("ankle_roll_joint"):
                kp[i], kd[i], max_eff[i], arm[i] = 20.0, 2.0, 50.0, 0.05
            # arms
            elif ("shoulder_" in name) or name.endswith("elbow_joint") or ("wrist_" in name):
                kp[i], kd[i], max_eff[i], arm[i] = 40.0, 10.0, 30.0, 0.05
            # hands lock
            elif name.startswith("left_hand_") or name.startswith("right_hand_"):
                kp[i], kd[i], max_eff[i], arm[i] = 200.0, 10.0, 5.0, 0.001
            else:
                kp[i], kd[i], max_eff[i], arm[i] = 0.0, 0.0, 0.0, 0.0

        for name, i in self.dof_name_to_index.items():
            set_group(name, i)

        # (1) articulation controller gains
        try:
            ctrl = self._robot.get_articulation_controller()
        except Exception:
            ctrl = None
        if ctrl is not None:
            try:
                ctrl.set_gains(kps=kp, kds=kd)
            except Exception:
                pass
            # effort caps (API differs)
            for fn in ("set_effort_limits", "set_max_efforts", "set_max_effort"):
                if hasattr(ctrl, fn):
                    try:
                        getattr(ctrl, fn)(max_eff)
                        break
                    except Exception:
                        pass

        # (2) DOF properties (often the most effective)
        try:
            if hasattr(self._robot, "get_dof_properties") and hasattr(self._robot, "set_dof_properties"):
                props = self._robot.get_dof_properties()
                if isinstance(props, np.ndarray) and props.dtype.names:
                    names = set(props.dtype.names)
                    if "stiffness" in names:
                        props["stiffness"] = kp
                    if "damping" in names:
                        props["damping"] = kd
                    if "armature" in names:
                        props["armature"] = arm
                    if "maxEffort" in names:
                        props["maxEffort"] = max_eff
                    if "max_effort" in names:
                        props["max_effort"] = max_eff
                    self._robot.set_dof_properties(props)
                    # print a small sanity sample
                    sample = ["left_knee_joint", "left_ankle_pitch_joint", "waist_pitch_joint", "left_elbow_joint"]
                    s = []
                    for j in sample:
                        if j in self.dof_name_to_index:
                            i = self.dof_name_to_index[j]
                            s.append(f"{j}:kp={kp[i]:.0f},kd={kd[i]:.0f},eff={max_eff[i]:.0f}")
                    print(">>> DOF props applied. " + " | ".join(s))
        except Exception as e:
            print(f"[WARN] set_dof_properties failed: {e}")

    # ---------------- Commands ----------------
    def _update_command(self):
        up = self._keys["UP"] or self._keys["W"]
        down = self._keys["DOWN"] or self._keys["S"]
        left = self._keys["LEFT"] or self._keys["A"]
        right = self._keys["RIGHT"] or self._keys["D"]

        vx_t = 0.0 if down else (self.max_vx if up else self.vx_idle)
        wz_t = 0.0
        if left:
            wz_t += self.max_wz
        if right:
            wz_t -= self.max_wz

        self.command[0] = (1.0 - self.cmd_smooth) * self.command[0] + self.cmd_smooth * vx_t
        self.command[1] = 0.0
        self.command[2] = (1.0 - self.cmd_smooth) * self.command[2] + self.cmd_smooth * wz_t

    # ---------------- Observation ----------------
    def _get_observation(self) -> Optional[torch.Tensor]:
        try:
            q_full = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
            qd_full = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
            _, quat_raw = self._robot.get_world_pose()
            lin_w = self._robot.get_linear_velocity()
            ang_w = self._robot.get_angular_velocity()
        except Exception:
            return None

        q = q_full[self.ctrl_dof_ids]
        qd = qd_full[self.ctrl_dof_ids]

        quat_wxyz = _to_wxyz(quat_raw, bool(self._quat_raw_is_xyzw))
        base_quat = torch.tensor(quat_wxyz, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_lin_world = torch.tensor(lin_w, device=self.device, dtype=torch.float32).unsqueeze(0)
        base_ang_world = torch.tensor(ang_w, device=self.device, dtype=torch.float32).unsqueeze(0)
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
            return None
        if not torch.isfinite(obs).all():
            return None
        return obs

    # ---------------- Action apply (full dof target like v5) ----------------
    def _apply_action(self, action_29: torch.Tensor):
        # full target = default_all, then override controlled joints
        tgt = self.default_pos_all.clone()
        tgt[self.ctrl_dof_ids] = self.default_pos_ctrl + (self.action_scale_ctrl * action_29)
        if self.lock_dof_ids is not None:
            tgt[self.lock_dof_ids] = 0.0

        tgt_np = tgt.detach().cpu().numpy().astype(np.float32)
        try:
            self._robot.apply_action(ArticulationAction(joint_positions=tgt_np))
        except Exception:
            pass

    # ---------------- Main control loop ----------------
    def _robot_control(self, step_size: float):
        if not self._is_running or self.policy is None:
            return

        dt = float(step_size)
        if dt <= 0.0:
            return

        # On first ticks after PLAY, reinit runtime handles + reset (like v5)
        if self._pending_play_reinit:
            self._pending_play_reinit = False
            asyncio.ensure_future(self._ensure_runtime_initialized(also_reset=True))
            return

        # Update command from pressed state
        self._update_command()

        # settle time
        self._sim_time_since_reset += dt
        if self._sim_time_since_reset < self.reset_settle_sec:
            self._apply_action(self._last_action)
            return

        if not self._use_policy:
            self._apply_action(torch.zeros(self.act_dim, device=self.device))
            return

        # policy rate
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
            out = self.policy(obs.unsqueeze(0))
            if isinstance(out, (tuple, list)):
                out = out[0]
            act = out.squeeze(0)

        act = torch.clamp(act, -1.0, 1.0)
        self.prev_action = act.clone()
        self._last_action = act.clone()

        # debug once per second: verify policy output and actuation error
        now = time.time()
        if now - self._last_dbg_t > 1.0:
            self._last_dbg_t = now
            a = act.detach().cpu().numpy()
            try:
                q_now = np.array(self._robot.get_joint_positions(), dtype=np.float32)
                err = q_now[self.ctrl_dof_ids_np] - (self.default_pos_ctrl + self.action_scale_ctrl * act).detach().cpu().numpy()
                err_norm = float(np.linalg.norm(err))
            except Exception:
                err_norm = float("nan")
            print(f"[DBG] cmd={self.command.detach().cpu().numpy().round(3)}  a_std={a.std():.3f}  a_max={np.max(np.abs(a)):.3f}  track_err_norm={err_norm:.3f}")

        self._apply_action(act)

    def world_cleanup(self):
        self._timeline_sub = None
        try:
            if self._input is not None and self._sub_keyboard is not None:
                self._input.unsubscribe_to_keyboard_events(self._sub_keyboard)
        except Exception:
            pass
        self._sub_keyboard = None
        self._safe_remove_physics_callback()