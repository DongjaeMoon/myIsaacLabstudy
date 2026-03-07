# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/UROP_v8_deploy.py]
# G1 box-catching deployment (UROP v8) for Isaac Sim
#
# Key updates vs old v5-style deploy:
#   - Policy action space: 29 dims (waist roll/pitch + wrist pitch/yaw added)
#   - Robot USD assumed: official Unitree G1 (many DOFs) but policy controls a 29-DOF subset
#   - PD gains match env.yaml (legs/waist 120/10, arms_load 85/8, arms_pose 55/6, fingers_lock 220/10)
#   - Observations use ONLY the 29 controlled joints (not all USD DOFs), consistent with Isaac Lab training pattern

import math
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import carb
import omni.appwindow
import omni.timeline
import omni.usd
from pxr import UsdGeom, Usd

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.physics import ContactSensor


# --------------------------------------------------------------------------
# Paths (edit to your environment)
# --------------------------------------------------------------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v8/2026-03-01_02-07-26/exported/policy.pt"
ROBOT_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd"
ROBOT_PRIM_PATH = "/World/G1"
OBJECT_PRIM_PATH = "/World/Object"

PHYSICS_CALLBACK_NAME = "policy_physics_step_v8"


# --------------------------------------------------------------------------
# Contact sensor link list (11) - must match training contact observation order
# --------------------------------------------------------------------------
CONTACT_SENSOR_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]


# --------------------------------------------------------------------------
# Policy action joint order (29) - MUST match ActionsCfg order used in training
# If your ActionsCfg ordering differs, change this list accordingly.
# --------------------------------------------------------------------------
POLICY_JOINT_ORDER: List[str] = [
    # legs_sagittal
    "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
    # legs_frontal
    "left_hip_roll_joint", "left_ankle_roll_joint",
    "right_hip_roll_joint", "right_ankle_roll_joint",
    # legs_yaw
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    # waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # arms_load (capture)
    "left_shoulder_pitch_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint",
    # arms_pose (wrap)
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# --------------------------------------------------------------------------
# Action scaling (default_pos + scale * action) - MUST match training ActionsCfg scales
# If you changed action scales in training, update these numbers.
# --------------------------------------------------------------------------
JOINT_SCALE_MAP: Dict[str, float] = {
    # legs_sagittal (scale=0.3)
    "left_hip_pitch_joint": 0.30, "left_knee_joint": 0.30, "left_ankle_pitch_joint": 0.30,
    "right_hip_pitch_joint": 0.30, "right_knee_joint": 0.30, "right_ankle_pitch_joint": 0.30,
    # legs_frontal (scale=0.15)
    "left_hip_roll_joint": 0.15, "left_ankle_roll_joint": 0.15,
    "right_hip_roll_joint": 0.15, "right_ankle_roll_joint": 0.15,
    # legs_yaw (scale=0.07)
    "left_hip_yaw_joint": 0.07, "right_hip_yaw_joint": 0.07,
    # waist (scale=0.14)
    "waist_yaw_joint": 0.14, "waist_roll_joint": 0.14, "waist_pitch_joint": 0.14,
    # left_arm_capture (scale=0.60)
    "left_shoulder_pitch_joint": 0.60, "left_elbow_joint": 0.60,
    # right_arm_capture (scale=0.50)
    "right_shoulder_pitch_joint": 0.50, "right_elbow_joint": 0.50,
    # left_arm_wrap (scale=0.30)
    "left_shoulder_roll_joint": 0.30, "left_shoulder_yaw_joint": 0.30,
    "left_wrist_roll_joint": 0.30, "left_wrist_pitch_joint": 0.30, "left_wrist_yaw_joint": 0.30,
    # right_arm_wrap (scale=0.30)
    "right_shoulder_roll_joint": 0.30, "right_shoulder_yaw_joint": 0.30,
    "right_wrist_roll_joint": 0.30, "right_wrist_pitch_joint": 0.30, "right_wrist_yaw_joint": 0.30,
}
# --------------------------------------------------------------------------
# Training init pose (from your env.yaml)
# We set all DOFs present in the USD; non-listed DOFs default to 0.
# --------------------------------------------------------------------------
TRAIN_INIT_JOINT_POS: Dict[str, float] = {
    "left_hip_pitch_joint": -0.15,
    "right_hip_pitch_joint": -0.15,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.15,
    "right_ankle_pitch_joint": -0.15,
    "left_hip_roll_joint": 0,
    "right_hip_roll_joint": 0,
    "left_hip_yaw_joint": 0,
    "right_hip_yaw_joint": 0,
    "left_ankle_roll_joint": 0,
    "right_ankle_roll_joint": 0,
    "waist_yaw_joint": 0,
    "waist_roll_joint": 0,
    "waist_pitch_joint": 0,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_elbow_joint": 0.55,
    "right_elbow_joint": 0.55,
    "left_shoulder_roll_joint": 0,
    "right_shoulder_roll_joint": 0,
    "left_shoulder_yaw_joint": 0,
    "right_shoulder_yaw_joint": 0,
    "left_wrist_roll_joint": 0,
    "right_wrist_roll_joint": 0,
    "left_wrist_pitch_joint": 0,
    "right_wrist_pitch_joint": 0,
    "left_wrist_yaw_joint": 0,
    "right_wrist_yaw_joint": 0,
    "left_hand_index_0_joint": 0,
    "left_hand_index_1_joint": 0,
    "left_hand_middle_0_joint": 0,
    "left_hand_middle_1_joint": 0,
    "left_hand_thumb_0_joint": 0,
    "left_hand_thumb_1_joint": 0,
    "left_hand_thumb_2_joint": 0,
    "right_hand_index_0_joint": 0,
    "right_hand_index_1_joint": 0,
    "right_hand_middle_0_joint": 0,
    "right_hand_middle_1_joint": 0,
    "right_hand_thumb_0_joint": 0,
    "right_hand_thumb_1_joint": 0,
    "right_hand_thumb_2_joint": 0,
}


# --------------------------------------------------------------------------
# Box properties (from env.yaml)
# --------------------------------------------------------------------------
BOX_SIZE = (0.32, 0.24, 0.24)   # (x,y,z) scale of DynamicCuboid
BOX_MASS = 3.0

# Box material params for obs 'obj_params' (training randomizes these; here we keep nominal values)
BOX_FRICTION_NOMINAL = 0.70
BOX_RESTITUTION_NOMINAL = 0.02
BOX_COLOR = (0.0, 0.8, 0.0)

# Park position: mimic env.yaml init_state object pos (world frame)
BOX_PARK_WORLD_POS = (2.0, 0.0, 1.6)

# Throw distribution (tune to your v8 training event config)
# Interpreted in robot yaw frame (x forward, y left, z up) then rotated into world.
THROW_REL_POS_RANGE = ((0.45, 0.62), (-0.08, 0.08), (0.36, 0.50))
THROW_REL_VEL_RANGE = ((-1.15, -0.70), (-0.10, 0.10), (-0.06, 0.10))


# --------------------------------------------------------------------------
# Gains (from env.yaml actuators)
# --------------------------------------------------------------------------
LEGS_AND_WAIST_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
ARMS_LOAD_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_elbow_joint",
]
ARMS_POSE_JOINTS = [
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
FINGERS_LOCK_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

KP_KD = {
    "legs_and_waist": (120.0, 10.0),
    "arms_load": (85.0, 8.0),
    "arms_pose": (55.0, 6.0),
    "fingers_lock": (220.0, 10.0),
}


class G1DeployV8(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Training: sim.dt=1/120, decimation=2 -> policy 60Hz
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

        # toss signal (like env._urop_toss_active)
        self._toss_active = False

        # hold latch (like env._urop_hold_latched) + anchor (xy)
        self._hold_latched = False
        self._hold_steps = 0
        self._hold_anchor_xy = np.zeros(2, dtype=np.float32)

        # box domain params for obs (like env._urop_box_*)
        self._box_size = np.array(BOX_SIZE, dtype=np.float32)
        self._box_mass = float(BOX_MASS)
        self._box_friction = float(BOX_FRICTION_NOMINAL)
        self._box_restitution = float(BOX_RESTITUTION_NOMINAL)

        # box parked mode
        self._box_hold_mode = True
        self.follow_parked_box_every_step = True

        # torque obs is optional; if IsaacSim cannot provide efforts, we feed zeros
        self.use_torque_obs = True

        # Use IsaacLab-like implicit PD (compute torques) instead of relying on USD joint drives.
        self.use_effort_pd = True
        self._kp_full_t = None  # torch (num_dof,)
        self._kd_full_t = None  # torch (num_dof,)
        self._tau_limit_full_t = None  # torch (num_dof,)
        self._last_target_pos_full = None  # torch (num_dof,)
        self._last_tau_full = None  # torch (num_dof,)

        # reset settle
        self.reset_settle_sec = 0.35
        self._sim_time_since_reset = 0.0

        # torque low-pass
        self._torque_lp = None
        self._torque_lp_alpha = 0.2
        self._printed_torque_hint = False

        # policy on/off
        self._use_policy = True

        # runtime flags
        self._pending_play_reinit = False
        self._physics_callback_name = PHYSICS_CALLBACK_NAME
        self._sub_keyboard = None
        self._input = None
        self._keyboard = None
        self._t0_wall = time.time()

        # handles
        self._world = None
        self._robot = None
        self._box = None
        self.contact_sensors = {}

        # policy I/O (inferred)
        self.policy_obs_dim: Optional[int] = None
        self.policy_act_dim: Optional[int] = None
        self.obs_mode: str = "torque"  # chosen after loading policy

        # buffers (set after loading)
        self.num_dof = 0
        self.robot_dof_names: List[str] = []
        self.policy_to_robot_indices: Optional[torch.Tensor] = None
        self.default_pos: Optional[torch.Tensor] = None
        self.action_scale: Optional[torch.Tensor] = None
        self.prev_action_policy_order: Optional[torch.Tensor] = None
        self._last_action_policy_order: Optional[torch.Tensor] = None

    # ---------------- Quaternion utils (training uses wxyz) ----------------
    def _detect_and_lock_quat_order(self, q_raw):
        if self._quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float32).copy()
        if q.shape[0] != 4:
            self._quat_raw_is_xyzw = True
            return
        # if upright, w is near 1 (could be xyzw or wxyz depending on API)
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

        # box (match env.yaml)
        self.box = world.scene.add(
            DynamicCuboid(
                prim_path=OBJECT_PRIM_PATH,
                name="box",
                position=np.array(BOX_PARK_WORLD_POS, dtype=np.float32),
                scale=np.array(BOX_SIZE, dtype=np.float32),
                mass=float(BOX_MASS),
                color=np.array(BOX_COLOR, dtype=np.float32),
            )
        )

        # contact sensors (optional; failure-safe)
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

        self.policy_obs_dim, self.policy_act_dim = self._infer_policy_io_dims()
        print(f">>> policy inferred: obs_dim={self.policy_obs_dim}, act_dim={self.policy_act_dim}")

        self.num_dof = self._robot.num_dof
        self.robot_dof_names = list(self._robot.dof_names)

        # map policy joints -> robot dof indices (policy order)
        self.policy_to_robot_indices = []
        for pj in POLICY_JOINT_ORDER:
            if pj in self.robot_dof_names:
                self.policy_to_robot_indices.append(self.robot_dof_names.index(pj))
            else:
                carb.log_error(f"[FATAL] Joint {pj} not found in robot USD DOF names!")

        if self.policy_act_dim is not None and len(self.policy_to_robot_indices) != int(self.policy_act_dim):
            raise RuntimeError(
                f"policy_to_robot_indices len={len(self.policy_to_robot_indices)} "
                f"(expected {self.policy_act_dim}). Check POLICY_JOINT_ORDER."
            )

        self.policy_to_robot_indices = torch.tensor(
            self.policy_to_robot_indices, device=self.device, dtype=torch.long
        )

        # buffers
        self.default_pos = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self.action_scale = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)

        # set joint scales (robot order); non-controlled joints remain 0 scale (hold default)
        for i, jn in enumerate(self.robot_dof_names):
            if jn in JOINT_SCALE_MAP:
                self.action_scale[i] = float(JOINT_SCALE_MAP[jn])

        # prev action (policy order)
        act_dim = int(self.policy_act_dim or len(POLICY_JOINT_ORDER))
        self.prev_action_policy_order = torch.zeros(act_dim, device=self.device, dtype=torch.float32)
        self._last_action_policy_order = torch.zeros(act_dim, device=self.device, dtype=torch.float32)

        # configure drives like training
        self._configure_joint_drives_like_training()

        # callbacks / keyboard
        self._ensure_physics_callback_registered(force_rebind=True)
        self._ensure_keyboard_subscription()

        # hard reset
        self._hard_reset_all()

        # select obs_mode that matches policy input dim
        self.obs_mode = self._select_obs_mode()
        print(f">>> obs_mode selected: {self.obs_mode}")

        # policy I/O sanity check
        try:
            obs = self._get_observation()
            with torch.no_grad():
                out = self.policy(obs.unsqueeze(0))
                if isinstance(out, (tuple, list)):
                    out = out[0]
            print(f">>> obs_dim={obs.numel()}  policy_out_dim={out.numel()}")
        except Exception:
            print("[WARN] policy I/O check failed:")
            traceback.print_exc()

        print(">>> G1DeployV8 setup_post_load done.")
        print(">>> Controls: [Play], K=throw, R=hard reset, P=policy on/off, H=park hold on/off, T=torque obs toggle")

    async def setup_post_reset(self):
        self._ensure_physics_callback_registered(force_rebind=True)
        self._ensure_keyboard_subscription()
        self._hard_reset_all()

    def _hard_reset_all(self):
        self._accum_dt = 0.0
        self._physics_step_count = 0
        self._policy_step_count = 0
        self._sim_time_since_reset = 0.0
        self._torque_lp = None
        self._toss_active = False
        self._hold_latched = False
        self._hold_steps = 0
        self._hold_anchor_xy[:] = 0.0
        self._box_hold_mode = True
        self._pending_play_reinit = False

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

            # default_pos must match training default_joint_pos (for all DOFs we control)
            self.default_pos = torch.tensor(init, device=self.device, dtype=torch.float32)

            # apply once to stabilize
            try:
                self._robot.apply_action(ArticulationAction(joint_positions=init.astype(np.float32)))
            except Exception:
                pass

        # park box
        self._place_box_parked()

    # ---------------- Timeline ----------------
    def _on_timeline_event(self, event):
        etype = int(event.type)
        play_type = int(omni.timeline.TimelineEventType.PLAY)
        stop_type = int(omni.timeline.TimelineEventType.STOP)

        pause_type = None
        if hasattr(omni.timeline.TimelineEventType, "PAUSE"):
            pause_type = int(omni.timeline.TimelineEventType.PAUSE)

        if etype == play_type:
            self._is_running = True
            self._ensure_physics_callback_registered(force_rebind=True)
            self._ensure_keyboard_subscription()
            self._pending_play_reinit = True
            if self.debug_mode:
                print("[DEBUG] Timeline PLAY -> callback rebind + pending runtime reinit")

        elif etype == stop_type:
            self._is_running = False
            self._safe_remove_physics_callback()
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
        self._world = self.get_world()
        try:
            self._robot = self._world.scene.get_object("g1")
        except Exception:
            self._robot = None
        try:
            self._box = self._world.scene.get_object("box")
        except Exception:
            self._box = None

        for link_name in CONTACT_SENSOR_LINKS:
            sensor_name = f"contact_{link_name}"
            try:
                self.contact_sensors[link_name] = self._world.scene.get_object(sensor_name)
            except Exception:
                self.contact_sensors[link_name] = self.contact_sensors.get(link_name, None)

    def _ensure_runtime_initialized(self):
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

        try:
            self._try_set_physics_dt(1.0 / self.physics_hz_target)
        except Exception:
            pass

        ok = bool(getattr(self._robot, "handles_initialized", False))
        if self.debug_mode:
            print(f"[DEBUG] _ensure_runtime_initialized -> handles_initialized={ok}")
        return ok

    # ---------------- Keyboard ----------------
    # K: throw
    # R: reset
    # P: policy on/off
    # H: park hold on/off
    # T: torque obs toggle
    def _on_key(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return

        if event.input == carb.input.KeyboardInput.K:
            self.throw_box_stage2_like_training()

        elif event.input == carb.input.KeyboardInput.R:
            print(">>> Hard reset")
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

    # ---------------- Box placement / throw ----------------
    def _place_box_parked(self):
        if self._box is None:
            return
        self._box.set_world_pose(position=np.array(BOX_PARK_WORLD_POS, dtype=np.float32))
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

        if not bool(getattr(self._robot, "handles_initialized", False)):
            ok = self._ensure_runtime_initialized()
            if not ok:
                print("[WARN] runtime not initialized yet. Try Play and wait a moment.")
                return

        self._box_hold_mode = False
        self._toss_active = True
        self._hold_latched = False
        self._hold_steps = 0
        self._hold_anchor_xy[:] = 0.0

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

        # --- spawn safety clamp (prevents immediate leg/foot collisions) ---
        base_pos_np = np.asarray(base_pos, dtype=np.float32)

        # Ensure minimum horizontal clearance from robot root
        min_xy = 0.80  # meters (tune 0.7~1.0)
        dxy = float(np.linalg.norm((p_w - base_pos_np)[:2]))
        if dxy < min_xy:
            # Put the box straight in front of the robot yaw with guaranteed clearance
            rel_p_safe = np.array([min_xy, 0.0, rel_p[2]], dtype=np.float32)
            p_w = base_pos_np + self._rotate_vec_by_yaw(yaw, rel_p_safe)

        # Ensure minimum world Z so it doesn't clip feet on spawn
        min_z_world = 1.10  # meters (tune 1.05~1.25)
        if float(p_w[2]) < min_z_world:
            p_w[2] = min_z_world

        self._box.set_world_pose(position=p_w)
        self._box.set_linear_velocity(v_w)
        try:
            self._box.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        except Exception:
            pass

        if self.debug_mode:
            print(f">>> THROW: rel_p={rel_p}, rel_v={rel_v} -> pos_w={p_w}, vel_w={v_w}")

    # ---------------- Policy I/O inference ----------------
    def _infer_policy_io_dims(self) -> Tuple[Optional[int], Optional[int]]:
        obs_dim = None
        act_dim = None

        try:
            params = list(self.policy.named_parameters())
        except Exception:
            params = []

        # infer obs_dim from first actor layer: [256, obs_dim]
        for name, p in params:
            try:
                if p.ndim == 2:
                    out_f, in_f = int(p.shape[0]), int(p.shape[1])
                    if out_f == 256 and in_f not in (256, 128):
                        obs_dim = in_f
                        break
            except Exception:
                pass

        # infer act_dim from actor output layer: [act_dim, hidden]
        cand = []
        for name, p in params:
            try:
                if p.ndim == 2:
                    out_f, in_f = int(p.shape[0]), int(p.shape[1])
                    if out_f > 1 and out_f < 128 and in_f in (256, 128):
                        cand.append(out_f)
            except Exception:
                pass
        if len(cand) > 0:
            act_dim = min(cand)  # should pick 29 (critic head is 1)

        return obs_dim, act_dim

    def _select_obs_mode(self) -> str:
        # Try to match policy_obs_dim by choosing between (torque / no_torque) layouts.
        if self.policy_obs_dim is None or self.policy_act_dim is None:
            return "torque"

        for mode in ["torque", "no_torque"]:
            try:
                obs = self._build_observation(mode)
                if obs.numel() == int(self.policy_obs_dim):
                    return mode
            except Exception:
                continue

        # fallback
        if self.debug_mode:
            print(
                f"[WARN] Could not match policy_obs_dim={self.policy_obs_dim} with known layouts. "
                f"Will use torque layout with pad/trunc fallback."
            )
        return "torque"

    # ---------------- Observation ----------------
    def _try_get_joint_torque_like_training(self, full_joint_pos: torch.Tensor) -> torch.Tensor:
        """Return joint torque observation for the controlled 29 joints (policy order).

        Training uses robot.data.applied_torque (or joint_effort) scaled by 1/80 and clamped.
        In Isaac Sim deployment, the most reliable way to match that distribution is:
          1) use measured joint efforts if available, else
          2) use the implicit-PD torque we compute for control (self._last_tau_full), else
          3) fall back to a PD estimate from the last target pose.
        """
        act_dim = int(self.policy_act_dim or len(POLICY_JOINT_ORDER))

        # Try Isaac Sim API first.
        jt_full = torch.zeros_like(full_joint_pos)
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
                if cand_t.shape == jt_full.shape:
                    jt_full = cand_t
                    if self.debug_mode and (not self._printed_torque_hint):
                        self._printed_torque_hint = True
                        print("[DEBUG] torque obs: using IsaacSim measured efforts")
            except Exception:
                pass

        # If API is unavailable or wrong shape, use our PD torques (best match to IsaacLab).
        if (jt_full.abs().sum().item() == 0.0) and (self._last_tau_full is not None):
            jt_full = self._last_tau_full
            if self.debug_mode and (not self._printed_torque_hint):
                self._printed_torque_hint = True
                print("[DEBUG] torque obs: using implicit-PD efforts (deployment-computed)")

        # Last resort: compute from last target pose.
        if (jt_full.abs().sum().item() == 0.0) and (self._last_target_pos_full is not None) and (self._kp_full_t is not None) and (self._kd_full_t is not None):
            try:
                vel = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
                jt_full = self._kp_full_t * (self._last_target_pos_full - full_joint_pos) - self._kd_full_t * vel
                if self._tau_limit_full_t is not None:
                    jt_full = torch.clamp(jt_full, -self._tau_limit_full_t, self._tau_limit_full_t)
                if self.debug_mode and (not self._printed_torque_hint):
                    self._printed_torque_hint = True
                    print("[DEBUG] torque obs: using PD estimate from last target pose")
            except Exception:
                pass

        # gather only policy joints (policy order)
        jt = jt_full[self.policy_to_robot_indices]
        jt = torch.clamp(jt * (1.0 / 80.0), -1.0, 1.0)
        if jt.numel() != act_dim:
            # safety fallback
            jt = jt[:act_dim] if jt.numel() > act_dim else torch.cat([jt, torch.zeros(act_dim - jt.numel(), device=self.device)], dim=0)
        return jt


    def _read_contact_11(self, raw: bool = False) -> torch.Tensor:
        """Read 11 contact magnitudes in the exact order used in env_cfg.py.

        Returns:
          - raw=True: unscaled Newton magnitudes (used for hold-latch logic)
          - raw=False: scaled (1/300) and gated by toss_signal (used as policy obs)
        """
        forces = []
        for name in CONTACT_SENSOR_LINKS:
            sensor = self.contact_sensors.get(name, None)
            val = 0.0
            if sensor and self._is_running:
                try:
                    # ContactSensor API differs by Isaac Sim version; we try a few options.
                    if hasattr(sensor, "get_net_contact_forces"):
                        f = sensor.get_net_contact_forces()
                        val = float(np.linalg.norm(np.asarray(f)))
                    else:
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

        raw_forces = torch.tensor(forces, device=self.device, dtype=torch.float32)
        if raw:
            return raw_forces

        contact = raw_forces * (1.0 / 300.0)
        if not self._toss_active:
            contact *= 0.0
        return contact

    def _get_torso_pos_w(self, base_pos_np: np.ndarray) -> torch.Tensor:
        """Best-effort torso_link world position.

        Training uses torso_link for several rewards and for the hold-latch logic.
        If we cannot query the torso link pose, fall back to root pose.
        """
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(f"{ROBOT_PRIM_PATH}/torso_link")
            if prim and prim.IsValid():
                xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                t = xf.ExtractTranslation()
                return torch.tensor([float(t[0]), float(t[1]), float(t[2])], device=self.device, dtype=torch.float32)
        except Exception:
            pass
        return torch.tensor(base_pos_np, device=self.device, dtype=torch.float32)

    def _obj_params_vec(self) -> torch.Tensor:
        """Match mdp.object_params() exactly: [size_n(3), mass_n(1), fric_n(1), rest_n(1)]."""
        size = torch.tensor(self._box_size, device=self.device, dtype=torch.float32).view(3)
        mass = torch.tensor([self._box_mass], device=self.device, dtype=torch.float32)
        fric = torch.tensor([self._box_friction], device=self.device, dtype=torch.float32)
        rest = torch.tensor([self._box_restitution], device=self.device, dtype=torch.float32)

        size_n = torch.stack(
            [
                (size[0] - 0.32) / 0.06,
                (size[1] - 0.24) / 0.05,
                (size[2] - 0.24) / 0.05,
            ],
            dim=-1,
        )
        mass_n = (mass - 3.25) / 1.75
        fric_n = (fric - 0.70) / 0.20
        rest_n = (rest - 0.03) / 0.03
        return torch.cat([size_n, mass_n, fric_n, rest_n], dim=-1)

    def _update_hold_latch_like_training(
        self,
        torso_pos_w: torch.Tensor,
        base_pos_np: np.ndarray,
        base_lin_w: torch.Tensor,
        raw_contact_11: torch.Tensor,
        obj_pos_w: torch.Tensor,
        obj_lin_w: torch.Tensor,
    ) -> None:
        """Port of mdp.rewards._update_hold_latch() (single-env version)."""
        if not self._toss_active:
            self._hold_latched = False
            self._hold_steps = 0
            return

        dist = torch.norm(obj_pos_w - torso_pos_w).item()
        rel_speed = torch.norm(obj_lin_w - base_lin_w).item()
        z_ok = float(obj_pos_w[2].item()) > 0.30

        # contact gate (same sensors/thresholds as training)
        tf = float(raw_contact_11[0].item())
        lf = float(torch.maximum(raw_contact_11[4], raw_contact_11[5]).item())
        rf = float(torch.maximum(raw_contact_11[9], raw_contact_11[10]).item())

        bilateral = (lf > 5.0) and (rf > 5.0)
        torso_plus = (tf > 10.0) and ((lf > 3.0) or (rf > 3.0))
        contact_gate = bilateral or torso_plus

        stable = z_ok and (dist < 0.55) and (rel_speed < 1.55) and contact_gate

        if stable and (not self._hold_latched):
            self._hold_latched = True
            self._hold_steps = 0
            self._hold_anchor_xy[:] = np.array([base_pos_np[0], base_pos_np[1]], dtype=np.float32)

        if self._hold_latched:
            self._hold_steps += 1

            # unlatch if clearly dropped/missed after latching (same as training)
            if (float(obj_pos_w[2].item()) < 0.18) or (dist > 1.35):
                self._hold_latched = False
                self._hold_steps = 0

    # ---------------- Observation ----------------
    def _build_observation(self, mode: str) -> torch.Tensor:
        # phase signals: toss(1), hold(1), hold_anchor_err(2)
        toss_signal = torch.tensor([1.0 if self._toss_active else 0.0], device=self.device, dtype=torch.float32)
        hold_signal = torch.tensor([1.0 if self._hold_latched else 0.0], device=self.device, dtype=torch.float32)

        base_pos_np, base_quat_raw = self._robot.get_world_pose()
        base_quat = self._to_wxyz(base_quat_raw)
        q = torch.tensor(base_quat, device=self.device, dtype=torch.float32)

        base_lin_w = torch.tensor(self._robot.get_linear_velocity(), device=self.device, dtype=torch.float32)
        base_ang_w = torch.tensor(self._robot.get_angular_velocity(), device=self.device, dtype=torch.float32)

        # gravity/base vel in base frame
        g_world = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        g_b = self._quat_rotate_inverse(q, g_world)
        lin_b = self._quat_rotate_inverse(q, base_lin_w)
        ang_b = self._quat_rotate_inverse(q, base_ang_w)

        # full DOF states
        j_pos_full = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
        j_vel_full = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)

        # gather only controlled joints in policy order
        j_pos = j_pos_full[self.policy_to_robot_indices]
        j_vel = j_vel_full[self.policy_to_robot_indices]

        if mode == "torque":
            j_torque = self._try_get_joint_torque_like_training(j_pos_full)
            proprio = torch.cat([g_b, lin_b, ang_b, j_pos, j_vel, j_torque], dim=-1)
        elif mode == "no_torque":
            proprio = torch.cat([g_b, lin_b, ang_b, j_pos, j_vel], dim=-1)
        else:
            raise ValueError(f"Unknown obs mode: {mode}")

        prev_act = self.prev_action_policy_order

        # object_rel (15)
        obj_pos_np, obj_quat_raw = self._box.get_world_pose()
        obj_quat = self._to_wxyz(obj_quat_raw)

        obj_pos_w = torch.tensor(obj_pos_np, device=self.device, dtype=torch.float32)
        obj_lin_w = torch.tensor(self._box.get_linear_velocity(), device=self.device, dtype=torch.float32)
        try:
            obj_ang_w = torch.tensor(self._box.get_angular_velocity(), device=self.device, dtype=torch.float32)
        except Exception:
            obj_ang_w = torch.zeros(3, device=self.device, dtype=torch.float32)

        r_pos = torch.tensor(base_pos_np, device=self.device, dtype=torch.float32)
        rel_p_b = self._quat_rotate_inverse(q, obj_pos_w - r_pos)
        rel_v_b = self._quat_rotate_inverse(q, obj_lin_w - base_lin_w)
        rel_w_b = self._quat_rotate_inverse(q, obj_ang_w - base_ang_w)

        b_q = torch.tensor(obj_quat, device=self.device, dtype=torch.float32)
        rel_q = self._quat_mul(self._quat_conj(q), b_q)
        rel_r6 = self._quat_to_rot6d(rel_q)

        obj_rel = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b], dim=-1)

        # contact (11) + hold latch update (needs raw)
        raw_contact = self._read_contact_11(raw=True)
        torso_pos_w = self._get_torso_pos_w(np.asarray(base_pos_np, dtype=np.float32))
        self._update_hold_latch_like_training(
            torso_pos_w=torso_pos_w,
            base_pos_np=np.asarray(base_pos_np, dtype=np.float32),
            base_lin_w=base_lin_w,
            raw_contact_11=raw_contact,
            obj_pos_w=obj_pos_w,
            obj_lin_w=obj_lin_w,
        )

        # refresh hold signal after update
        hold_signal = torch.tensor([1.0 if self._hold_latched else 0.0], device=self.device, dtype=torch.float32)
        if self._hold_latched:
            err_xy = torch.tensor(
                [float(base_pos_np[0]) - float(self._hold_anchor_xy[0]), float(base_pos_np[1]) - float(self._hold_anchor_xy[1])],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            err_xy = torch.zeros(2, device=self.device, dtype=torch.float32)

        obj_params = self._obj_params_vec()

        contact = raw_contact * (1.0 / 300.0)
        if not self._toss_active:
            contact *= 0.0

        # FULL policy obs order (env_cfg.py):
        # [toss(1), hold(1), hold_anchor_err(2), proprio, prev_actions, obj_rel, obj_params(6), contact(11)]
        obs = torch.cat([toss_signal, hold_signal, err_xy, proprio, prev_act, obj_rel, obj_params, contact], dim=-1)

        if not torch.isfinite(obs).all():
            raise RuntimeError("non-finite value in observation")
        return obs


    def _get_observation(self) -> torch.Tensor:
        obs = self._build_observation(self.obs_mode)

        # Ensure shape matches policy (optional fallback for experimentation)
        if self.policy_obs_dim is not None and obs.numel() != int(self.policy_obs_dim):
            want = int(self.policy_obs_dim)
            have = int(obs.numel())
            if self.debug_mode:
                print(f"[WARN] obs_dim mismatch: have={have} want={want} (pad/trunc fallback)")
            if have < want:
                pad = torch.zeros(want - have, device=self.device, dtype=torch.float32)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                obs = obs[:want]

        return obs

    # ---------------- Control loop ----------------
    def _robot_control(self, step_size: float):
        dt = float(step_size)
        if dt <= 0.0:
            return

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
                self._is_running = True

        if (self._robot is None) or (not bool(getattr(self._robot, "handles_initialized", False))):
            return

        self._physics_step_count += 1

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
                    f"toss={self._toss_active} latched={self._hold_latched} park={self._box_hold_mode} "
                    f"callback={'ON' if self._callback_registered else 'OFF'}"
                )
                print(f"        base_pos={np.array(p).round(3)} base_quat_wxyz={np.array(q).round(3)}")
            except Exception:
                pass

        if not self._is_running:
            return

        self._sim_time_since_reset += dt

        act_dim = int(self.policy_act_dim or len(POLICY_JOINT_ORDER))
        zero_act = torch.zeros(act_dim, device=self.device, dtype=torch.float32)

        if self._sim_time_since_reset < self.reset_settle_sec:
            self._apply_policy_action(zero_act)
            return

        if not self._use_policy:
            self._apply_policy_action(zero_act)
            return

        self._accum_dt += dt

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

        # update prev action
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
        """Apply the policy action.

        IMPORTANT: Isaac Lab training uses ImplicitActuator (PD -> torque) even though the action is a position target.
        If we only send joint_positions targets in Isaac Sim, it depends on USD drive maxForce and drive setup.
        For Unitree G1 USD, those defaults are often too weak -> robot collapses (ragdoll).
        Hence we reproduce IsaacLab: compute PD torques and apply joint_efforts.
        """
        if self._robot is None:
            return
        if not bool(getattr(self._robot, "handles_initialized", False)):
            return
        if self.default_pos is None or self.action_scale is None or self.policy_to_robot_indices is None:
            return

        # action vector in robot DOF order
        action_robot_order = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        action_robot_order[self.policy_to_robot_indices] = action_policy_order

        # target_pos = default_pos + scale * action
        target_pos = self.default_pos + (self.action_scale * action_robot_order)
        self._last_target_pos_full = target_pos.detach()

        if self.use_effort_pd and (self._kp_full_t is not None) and (self._kd_full_t is not None):
            try:
                cur_pos = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
                cur_vel = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)

                tau = self._kp_full_t * (target_pos - cur_pos) - self._kd_full_t * cur_vel
                if self._tau_limit_full_t is not None:
                    tau = torch.clamp(tau, -self._tau_limit_full_t, self._tau_limit_full_t)

                self._last_tau_full = tau.detach()

                tau_np = tau.detach().cpu().numpy().astype(np.float32)
                self._robot.apply_action(ArticulationAction(joint_efforts=tau_np))
                return
            except Exception:
                if self.debug_mode:
                    print("[WARN] effort-PD apply failed, falling back to joint_positions:")
                    traceback.print_exc()

        # Fallback: position targets (if effort mode is not available)
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

        legs_kp, legs_kd = KP_KD["legs_and_waist"]
        load_kp, load_kd = KP_KD["arms_load"]
        pose_kp, pose_kd = KP_KD["arms_pose"]
        fin_kp, fin_kd = KP_KD["fingers_lock"]

        for i, name in enumerate(self.robot_dof_names):
            if name in LEGS_AND_WAIST_JOINTS:
                kp[i] = legs_kp
                kd[i] = legs_kd
            elif name in ARMS_LOAD_JOINTS:
                kp[i] = load_kp
                kd[i] = load_kd
            elif name in ARMS_POSE_JOINTS:
                kp[i] = pose_kp
                kd[i] = pose_kd
            elif name in FINGERS_LOCK_JOINTS:
                kp[i] = fin_kp
                kd[i] = fin_kd
            else:
                # default fallback (should rarely happen)
                kp[i] = pose_kp
                kd[i] = pose_kd
        # Cache gains for IsaacLab-like implicit PD and for torque observations.
        self._kp_full_t = torch.tensor(kp, device=self.device, dtype=torch.float32)
        self._kd_full_t = torch.tensor(kd, device=self.device, dtype=torch.float32)

        # Conservative torque limits (used only when effort-PD is enabled).
        tau_lim = np.full(self.num_dof, 250.0, dtype=np.float32)
        for i, name in enumerate(self.robot_dof_names):
            if name in LEGS_AND_WAIST_JOINTS:
                tau_lim[i] = 450.0
            elif (name in ARMS_LOAD_JOINTS) or (name in ARMS_POSE_JOINTS):
                tau_lim[i] = 300.0
            elif name in FINGERS_LOCK_JOINTS:
                tau_lim[i] = 80.0
        self._tau_limit_full_t = torch.tensor(tau_lim, device=self.device, dtype=torch.float32)

        # If the controller supports max-efforts, raise them to avoid 'ragdoll' due to low defaults.
        try:
            if hasattr(ctrl, 'set_max_efforts'):
                ctrl.set_max_efforts(tau_lim)
        except Exception:
            pass

        # Try to switch controller to effort mode if available (safe no-op otherwise).
        try:
            if hasattr(ctrl, 'switch_control_mode'):
                ctrl.switch_control_mode('effort')
            elif hasattr(ctrl, 'set_control_mode'):
                ctrl.set_control_mode('effort')
        except Exception:
            pass


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
            print(f"[DEBUG] set_gains ok={ok} (legs={legs_kp}/{legs_kd}, load={load_kp}/{load_kd}, pose={pose_kp}/{pose_kd}, fingers={fin_kp}/{fin_kd})")

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