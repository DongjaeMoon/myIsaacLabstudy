# deploy/catch/g1_catch_policy_v23.py
# UROP_v23 Isaac Sim deploy policy wrapper.

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import omni.usd
    from pxr import Usd, UsdGeom
except Exception:  # static checks outside Isaac Sim
    omni = None
    Usd = None
    UsdGeom = None

from isaacsim.core.utils.types import ArticulationAction

from catch.controllers.policy_controller import PolicyController


POLICY_OBS_DIM = 100
GRAVITY_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)
# v23 actor uses the real torso-mounted AprilTag camera extrinsic.
# The returned camera quaternion is already optical/OpenCV: x right, y down, z forward.
CAMERA_PARENT_BODY_NAMES = ("torso_link", "torso")
CAMERA_TRANSLATION_T = np.array([0.05762, 0.01753, 0.42987], dtype=np.float32)
CAMERA_QUAT_T = np.array([0.91496, 0.0, 0.40355, 0.0], dtype=np.float32)  # wxyz, torso -> camera optical
TORSO_TRANSLATION_ROOT_B = np.array([-0.003963499795645475, 0.0, 0.04399999976158142], dtype=np.float32)
DEFAULT_CAMERA_OFFSET_B = TORSO_TRANSLATION_ROOT_B + CAMERA_TRANSLATION_T
HOLD_ANCHOR_B = np.array([0.34, 0.0, 0.23], dtype=np.float32)


# -----------------------------------------------------------------------------
# Quaternion helpers. Internally this file uses Isaac Lab convention: w, x, y, z.
# -----------------------------------------------------------------------------
def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float32)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float32)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return quat_mul(quat_mul(q, np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)), quat_conj(q))[1:4]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q), v)


def body_to_opencv(v_body: np.ndarray) -> np.ndarray:
    # v23 _camera_pose_vel_w() already returns the camera optical/OpenCV frame.
    # Do NOT apply the old v21 body/head -> OpenCV axis remapping here.
    return np.asarray(v_body, dtype=np.float32).reshape(3)


def _as_np3(x, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if x is None:
        return np.asarray(default, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return np.asarray(default, dtype=np.float32)
    return arr[:3]


# -----------------------------------------------------------------------------
# Path resolution. Keep everything relative to myIsaacLabstudy/.
# -----------------------------------------------------------------------------
def find_project_root(start_file: str | Path) -> Path:
    p = Path(start_file).resolve()
    for candidate in [p.parent, *p.parents]:
        if (candidate / "UROP").exists() and (candidate / "deploy").exists():
            return candidate
    # g1_catch_policy.py lives in myIsaacLabstudy/deploy/catch/ in the intended layout.
    return p.parents[2]


def resolve_policy_and_env(policy_file: str | Path, env_yaml: str | Path | None = None) -> tuple[str, str]:
    policy_file = Path(os.path.expanduser(os.path.expandvars(str(policy_file)))).resolve()
    if policy_file.is_dir():
        policy_file = policy_file / "policy.pt"
    if not policy_file.exists():
        raise FileNotFoundError(f"policy.pt not found: {policy_file}")

    if env_yaml is not None:
        env_path = Path(os.path.expanduser(os.path.expandvars(str(env_yaml)))).resolve()
        if not env_path.exists():
            raise FileNotFoundError(f"env.yaml not found: {env_path}")
        return str(policy_file), str(env_path)

    # Common Isaac Lab RSL-RL layouts around .../<run>/exported/policy.pt.
    run_dir = policy_file.parent.parent if policy_file.parent.name == "exported" else policy_file.parent
    candidates = [
        policy_file.parent / "env.yaml",
        run_dir / "env.yaml",
        run_dir / "params" / "env.yaml",
        run_dir / "config" / "env.yaml",
    ]
    for c in candidates:
        if c.exists():
            return str(policy_file), str(c.resolve())

    raise FileNotFoundError(
        "Could not find env.yaml for policy. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )


class G1CatchPolicyV23(PolicyController):
    """UROP_v23 actor-policy wrapper for Isaac Sim interactive deploy.

    Actor observation order, exactly:
      projected_gravity(3), base_ang_vel(3), joint_pos_rel(29), joint_vel*0.05(29),
      prev_actions(29), object_rel_pos_opencv(3), object_rel_lin_vel_opencv(3), tag_visible(1).
    """

    def __init__(
        self,
        prim_path: str,
        policy_file_path: str,
        usd_path: str,
        env_yaml_path: str | None = None,
        name: str = "G1",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(name, prim_path, None, usd_path, position, orientation)
        policy_file, env_yaml = resolve_policy_and_env(policy_file_path, env_yaml_path)
        self.load_policy(policy_file, env_yaml)

        self.force_tag_visible = False
        self.debug_enabled = True
        self.debug_print_first_n = 5
        self.debug_print_every = 100
        self.debug_action_gain = 1.0
        self.debug_force_zero_action = False

        self._quat_raw_is_xyzw: Optional[bool] = None
        self._policy_counter = 0
        self._debug_step = 0

        self._previous_action = np.zeros(29, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self._last_obs: np.ndarray | None = None
        self._last_raw_action: np.ndarray | None = None
        self._last_target_positions: np.ndarray | None = None
        self._camera_source = "uninitialized"

        init_state = self.policy_env_params.get("scene", {}).get("robot", {}).get("init_state", {}) or {}
        self.default_root_pos = np.asarray(init_state.get("pos", [0.0, 0.0, 0.78]), dtype=np.float32)
        self.default_root_rot = np.asarray(init_state.get("rot", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
        self.default_root_lin_vel = np.asarray(init_state.get("lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.default_root_ang_vel = np.asarray(init_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)

    # ------------------------------------------------------------------
    # Initialization and static contract checks.
    # ------------------------------------------------------------------
    def initialize(self):
        super().initialize(set_articulation_props=True)
        usd_dof_names = list(self.robot.dof_names)

        action_cfg = self.policy_env_params.get("actions", {}).get("policy", {}) or {}
        self.controlled_joint_names = list(action_cfg.get("joint_names", []))
        if len(self.controlled_joint_names) != 29:
            raise RuntimeError(f"UROP_v23 deploy expects 29 action joints, got {len(self.controlled_joint_names)}")

        scale_cfg = action_cfg.get("scale", {}) or {}
        offset_cfg = action_cfg.get("offset", {}) or {}
        self.action_scales = np.asarray([float(scale_cfg[name]) for name in self.controlled_joint_names], dtype=np.float32)
        self.action_offsets = np.asarray([float(offset_cfg[name]) for name in self.controlled_joint_names], dtype=np.float32)
        self.controlled_to_usd_idx = [usd_dof_names.index(name) for name in self.controlled_joint_names]

        self._full_default_pos = self.default_pos.copy()
        self._full_default_vel = np.zeros_like(self.default_pos, dtype=np.float32)
        for i, usd_idx in enumerate(self.controlled_to_usd_idx):
            self._full_default_pos[usd_idx] = self.action_offsets[i]

        # Lock quaternion convention after the articulation is initialized.
        _, q_raw = self.robot.get_world_pose()
        self._detect_and_lock_quat_order(q_raw)

        self.reset_policy_state()
        self._debug_print_static_mapping()
        self.reset_robot_to_ready()

    def reset_policy_state(self):
        self._policy_counter = 0
        self._debug_step = 0
        self._previous_action[:] = 0.0
        self.action[:] = 0.0
        self._last_obs = None
        self._last_raw_action = None
        self._last_target_positions = None

    def reset_robot_to_ready(self):
        """Best-effort reset to the same catch-ready state used by v23 training."""
        self.reset_policy_state()
        try:
            self.robot.set_world_pose(position=self.default_root_pos, orientation=self.default_root_rot)
        except Exception:
            pass
        for method_name, value in (
            ("set_linear_velocity", self.default_root_lin_vel),
            ("set_angular_velocity", self.default_root_ang_vel),
            ("set_joint_positions", self._full_default_pos),
            ("set_joint_velocities", self._full_default_vel),
        ):
            try:
                getattr(self.robot, method_name)(value)
            except Exception:
                pass
        try:
            self.robot.apply_action(ArticulationAction(joint_positions=self._full_default_pos.copy()))
        except Exception:
            pass


    def apply_ready_action(self):
        """Hold the robot at the v23 catch-ready pose without querying the policy.

        This is a debug/safety mode only.  The normal interactive demo keeps the
        learned policy running continuously while the box is held and released.
        """
        self.action[:] = 0.0
        self._previous_action[:] = 0.0
        target_positions = self._full_default_pos.copy()
        self._last_target_positions = target_positions.copy()
        try:
            self.robot.apply_action(ArticulationAction(joint_positions=target_positions))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Quaternion and pose helpers.
    # ------------------------------------------------------------------
    def _detect_and_lock_quat_order(self, q_raw):
        if self._quat_raw_is_xyzw is not None:
            return
        q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
        if q.size != 4:
            self._quat_raw_is_xyzw = False
            return
        # Upright identity usually has scalar near 1. Isaac Sim versions differ.
        if abs(q[3]) > 0.90 and abs(q[0]) < 0.50:
            self._quat_raw_is_xyzw = True
        elif abs(q[0]) > 0.90 and abs(q[3]) < 0.50:
            self._quat_raw_is_xyzw = False
        else:
            # Use existing Isaac Lab/yaml convention as fallback.
            self._quat_raw_is_xyzw = False
        print(f"[G1CatchPolicyV23] quaternion raw order -> {'xyzw' if self._quat_raw_is_xyzw else 'wxyz'} | raw={q}")

    def _to_wxyz(self, q_raw) -> np.ndarray:
        q = np.asarray(q_raw, dtype=np.float32).reshape(-1)
        if q.size != 4:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._detect_and_lock_quat_order(q)
        if self._quat_raw_is_xyzw:
            return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        return q.astype(np.float32)

    def _from_wxyz_for_core(self, q_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
        if self._quat_raw_is_xyzw:
            return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
        return q

    def _root_pose_vel_w(self):
        root_pos, q_raw = self.robot.get_world_pose()
        root_pos = _as_np3(root_pos)
        root_quat = self._to_wxyz(q_raw)
        lin_vel = _as_np3(getattr(self.robot, "get_linear_velocity", lambda: None)())
        ang_vel = _as_np3(getattr(self.robot, "get_angular_velocity", lambda: None)())
        return root_pos, root_quat, lin_vel, ang_vel

    def _stage_body_pose_w(self, body_name: str):
        """Return (pos_w, quat_wxyz, prim_path) for a robot body prim if it exists.

        Isaac Lab's Articulation body names are not always direct children of the
        articulation prim in the flattened USD.  The first v23 deploy patch only
        checked /World/G1/<body_name>, so it could silently fall back to the root
        camera offset even when a head/torso link existed deeper in the stage.
        """
        if omni is None or UsdGeom is None or Usd is None:
            return None
        try:
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(getattr(self, "prim_path", "/World/G1"))
            if not root or not root.IsValid():
                return None

            candidates = []
            direct = stage.GetPrimAtPath(f"{root.GetPath().pathString}/{body_name}")
            if direct and direct.IsValid():
                candidates.append(direct)

            for prim in Usd.PrimRange(root):
                try:
                    if prim.GetName() == body_name:
                        candidates.append(prim)
                except Exception:
                    pass

            for prim in candidates:
                try:
                    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    t = mat.ExtractTranslation()
                    q = mat.ExtractRotationQuat()
                    imag = q.GetImaginary()
                    pos = np.array([t[0], t[1], t[2]], dtype=np.float32)
                    quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
                    return pos, quat, prim.GetPath().pathString
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def _camera_pose_vel_w(self):
        root_pos, root_quat, root_lin_vel, root_ang_vel = self._root_pose_vel_w()

        # Match UROP_v23 mdp.observations._camera_pose_vel_w(): torso_link -> camera optical.
        for body_name in CAMERA_PARENT_BODY_NAMES:
            pose = self._stage_body_pose_w(body_name)
            if pose is not None:
                parent_pos, parent_quat, path = pose
                cam_offset_w = quat_apply(parent_quat, CAMERA_TRANSLATION_T)
                cam_pos = parent_pos + cam_offset_w
                cam_quat = quat_mul(parent_quat, CAMERA_QUAT_T)
                # Isaac Sim SingleArticulation does not expose every link velocity through this wrapper.
                # Use root velocity plus angular offset compensation as a stable deploy approximation.
                cam_lin = root_lin_vel + np.cross(root_ang_vel, cam_offset_w).astype(np.float32)
                self._camera_source = f"stage:{body_name}+calib ({path})"
                return cam_pos.astype(np.float32), cam_quat.astype(np.float32), cam_lin.astype(np.float32)

        self._camera_source = "fallback_root_offset+calib"
        cam_offset_w = quat_apply(root_quat, DEFAULT_CAMERA_OFFSET_B)
        cam_pos = root_pos + cam_offset_w
        cam_quat = quat_mul(root_quat, CAMERA_QUAT_T)
        cam_lin = root_lin_vel + np.cross(root_ang_vel, cam_offset_w).astype(np.float32)
        return cam_pos.astype(np.float32), cam_quat.astype(np.float32), cam_lin.astype(np.float32)

    # ------------------------------------------------------------------
    # Observation: exact v23 actor contract, no old privileged terms.
    # ------------------------------------------------------------------
    def _compute_tag_visible(self, cam_pos, cam_quat, obj_pos) -> np.float32:
        if self.force_tag_visible:
            return np.float32(1.0)

        # v23 camera quaternion is optical/OpenCV already: x right, y down, z forward.
        rel_cam = quat_rotate_inverse(cam_quat, obj_pos - cam_pos)
        right, down, forward = float(rel_cam[0]), float(rel_cam[1]), float(rel_cam[2])
        dist = float(np.linalg.norm(rel_cam) + 1e-6)
        h_angle = math.atan2(abs(right), max(forward, 1e-4))
        v_angle = math.atan2(abs(down), max(forward, 1e-4))
        visible = (
            forward > 0.05
            and dist < 2.20
            and h_angle < 1.45 * 0.5
            and v_angle < 1.15 * 0.5
        )
        return np.float32(1.0 if visible else 0.0)

    def _compute_observation(self, obj_pos_w, obj_lin_vel_w, tag_visible_override: float | None = None) -> np.ndarray:
        root_pos, root_quat, root_lin_vel_w, root_ang_vel_w = self._root_pose_vel_w()
        cam_pos, cam_quat, cam_lin_vel_w = self._camera_pose_vel_w()

        gravity_b = quat_rotate_inverse(root_quat, GRAVITY_W)
        base_ang_vel_b = quat_rotate_inverse(root_quat, root_ang_vel_w)

        q_all = np.asarray(self.robot.get_joint_positions(), dtype=np.float32)
        qd_all = np.asarray(self.robot.get_joint_velocities(), dtype=np.float32)
        q = q_all[self.controlled_to_usd_idx]
        qd = qd_all[self.controlled_to_usd_idx]
        joint_pos_rel = np.clip(q - self.action_offsets, -3.5, 3.5)
        joint_vel = np.clip(qd * 0.05, -8.0, 8.0)

        obj_pos_w = np.asarray(obj_pos_w, dtype=np.float32).reshape(3)
        obj_lin_vel_w = np.asarray(obj_lin_vel_w, dtype=np.float32).reshape(3)
        tag = self._compute_tag_visible(cam_pos, cam_quat, obj_pos_w)
        if tag_visible_override is not None:
            tag = np.float32(float(tag_visible_override) > 0.5)

        rel_pos_body = quat_rotate_inverse(cam_quat, obj_pos_w - cam_pos)
        rel_vel_body = quat_rotate_inverse(cam_quat, obj_lin_vel_w - cam_lin_vel_w)
        object_rel_pos = np.clip(body_to_opencv(rel_pos_body), -4.0, 4.0)
        object_rel_lin_vel = np.clip(body_to_opencv(rel_vel_body), -8.0, 8.0)

        if tag < 0.5:
            object_rel_pos[:] = 0.0
            object_rel_lin_vel[:] = 0.0

        obs = np.concatenate(
            [
                np.clip(gravity_b, -1.5, 1.5),
                np.clip(base_ang_vel_b, -12.0, 12.0),
                joint_pos_rel,
                joint_vel,
                self._previous_action,
                object_rel_pos,
                object_rel_lin_vel,
                np.array([tag], dtype=np.float32),
            ]
        ).astype(np.float32)

        if obs.shape[0] != POLICY_OBS_DIM:
            raise RuntimeError(f"Bad UROP_v23 actor obs dim: {obs.shape[0]} != {POLICY_OBS_DIM}")
        return obs

    def probe_policy_outputs(self, obj_pos_w, obj_lin_vel_w=None):
        """Print policy outputs for controlled diagnostic observations.

        Interpretation:
          - tag0_object_zeroed should be almost idle. If it is already saturated,
            the exported policy/normalizer is probably mismatched.
          - tag1_stationary_object being highly saturated means the policy reacts
            strongly to a visible held object; use this to diagnose whether the
            policy learned delayed/no-toss behavior robustly.
        """
        obj_pos_w = np.asarray(obj_pos_w, dtype=np.float32).reshape(3)
        if obj_lin_vel_w is None:
            obj_lin_vel_w = np.zeros(3, dtype=np.float32)
        obj_lin_vel_w = np.asarray(obj_lin_vel_w, dtype=np.float32).reshape(3)

        saved_prev = self._previous_action.copy()
        saved_action = self.action.copy()
        cases = [
            ("tag0_object_zeroed", 0.0, np.zeros(3, dtype=np.float32)),
            ("tag1_stationary_object", 1.0, np.zeros(3, dtype=np.float32)),
            ("tag1_current_velocity", 1.0, obj_lin_vel_w),
        ]
        print("\n" + "=" * 100)
        print("[G1CatchPolicyV23] POLICY OUTPUT PROBE")
        print(f"  camera_source={self._camera_source}")
        print(f"  obj_pos_w={self._fmt(obj_pos_w)} obj_vel_w={self._fmt(obj_lin_vel_w)}")
        for name, tag, vel in cases:
            self._previous_action[:] = 0.0
            obs = self._compute_observation(obj_pos_w, vel, tag_visible_override=tag)
            raw = self._compute_action(obs).astype(np.float32)
            clipped = np.clip(raw, -1.0, 1.0)
            sat = int((np.abs(clipped) >= 0.98).sum())
            print(
                f"  {name:24s} tag={tag:.0f} "
                f"obs[min,max,mean]=({obs.min():+.3f},{obs.max():+.3f},{obs.mean():+.3f}) "
                f"raw[min,max,mean]=({raw.min():+.3f},{raw.max():+.3f},{raw.mean():+.3f}) sat={sat:02d}/29"
            )
        self._previous_action = saved_prev
        self.action = saved_action
        print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # Policy/action application.
    # ------------------------------------------------------------------
    def forward(self, dt: float, obj_pos_w, obj_lin_vel_w, tag_visible_override: float | None = None):
        if self._policy_counter % int(self._decimation) == 0:
            obs = self._compute_observation(obj_pos_w, obj_lin_vel_w, tag_visible_override)
            raw_action = self._compute_action(obs)
            if raw_action.shape[0] != 29:
                raise RuntimeError(f"Bad UROP_v23 action dim: {raw_action.shape[0]} != 29")
            clipped = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
            if self.debug_force_zero_action:
                clipped[:] = 0.0
            self.action = clipped
            self._previous_action = clipped.copy()
            self._last_obs = obs.copy()
            self._last_raw_action = raw_action.copy()

        target_positions = self._full_default_pos.copy()
        controlled_targets = self.action_offsets + self.action * self.action_scales * float(self.debug_action_gain)
        for i, usd_idx in enumerate(self.controlled_to_usd_idx):
            target_positions[usd_idx] = controlled_targets[i]

        self._last_target_positions = target_positions.copy()
        self.robot.apply_action(ArticulationAction(joint_positions=target_positions))

        if self._debug_should_print():
            self._debug_print_step(obj_pos_w, obj_lin_vel_w)

        self._policy_counter += 1
        self._debug_step += 1

    def _debug_should_print(self) -> bool:
        return bool(
            self.debug_enabled
            and (
                self._debug_step < int(self.debug_print_first_n)
                or (int(self.debug_print_every) > 0 and self._debug_step % int(self.debug_print_every) == 0)
            )
        )

    def _fmt(self, x) -> str:
        return np.array2string(np.asarray(x), precision=3, suppress_small=True)

    def _debug_print_static_mapping(self):
        print("\n" + "=" * 110)
        print("[G1CatchPolicyV23] STATIC CONTRACT")
        print(f"  actor obs dim: {POLICY_OBS_DIM}")
        print(f"  action dim: {len(self.controlled_joint_names)}")
        print(f"  sim dt={self._dt:.4f}, decimation={self._decimation}, policy dt={self._dt * self._decimation:.4f}")
        print("  ACTION/OBS JOINT ORDER -> USD DOF INDEX / SCALE / OFFSET")
        usd_names = list(self.robot.dof_names)
        for i, (name, usd_idx, scale, offset) in enumerate(
            zip(self.controlled_joint_names, self.controlled_to_usd_idx, self.action_scales, self.action_offsets)
        ):
            print(f"    [{i:02d}] {name:28s} -> usd[{usd_idx:02d}] {usd_names[usd_idx]:28s} scale={scale:4.2f} offset={offset:+6.3f}")
        print("=" * 110 + "\n")

    def _debug_print_step(self, obj_pos_w, obj_lin_vel_w):
        if self._last_obs is None or self._last_raw_action is None:
            return
        obs = self._last_obs
        raw = self._last_raw_action
        root_pos, root_quat, root_lin, root_ang = self._root_pose_vel_w()
        cam_pos, cam_quat, _cam_vel = self._camera_pose_vel_w()
        tag = obs[-1]
        rel_opencv = obs[93:96]
        rel_vel_opencv = obs[96:99]
        sat = int((np.abs(self.action) >= 0.98).sum())
        print("\n" + "-" * 110)
        print(f"[G1CatchPolicyV23][step {self._debug_step:05d}] tag={tag:.0f} sat={sat} action_gain={self.debug_action_gain:.2f}")
        print(f"  root_pos={self._fmt(root_pos)} root_ang_b={self._fmt(obs[3:6])} gravity_b={self._fmt(obs[0:3])}")
        print(f"  camera_source={self._camera_source}")
        print(f"  cam_pos={self._fmt(cam_pos)} obj_pos={self._fmt(obj_pos_w)} obj_vel={self._fmt(obj_lin_vel_w)}")
        print(f"  object_rel_pos_opencv={self._fmt(rel_opencv)} object_rel_vel_opencv={self._fmt(rel_vel_opencv)}")
        print(
            f"  obs[min,max,mean]=({obs.min():+.3f}, {obs.max():+.3f}, {obs.mean():+.3f}) "
            f"raw_action[min,max]=({raw.min():+.3f}, {raw.max():+.3f}) "
            f"joint_rel_max={np.max(np.abs(obs[6:35])):.3f} "
            f"joint_vel_scaled_max={np.max(np.abs(obs[35:64])):.3f} "
            f"prev_action_max={np.max(np.abs(obs[64:93])):.3f}"
        )
        order = np.argsort(-np.abs(self.action))[:10]
        for k in order:
            print(
                f"    act[{k:02d}] {self.controlled_joint_names[k]:28s} "
                f"raw={raw[k]:+7.3f} clip={self.action[k]:+7.3f} "
                f"scale={self.action_scales[k]:4.2f} target={self.action_offsets[k] + self.action[k] * self.action_scales[k]:+7.3f}"
            )
        print("-" * 110 + "\n")


# Backward-compatible local alias for code that imports G1CatchPolicy from this module.
G1CatchPolicy = G1CatchPolicyV23
