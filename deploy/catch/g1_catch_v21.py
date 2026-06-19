# deploy/catch/g1_catch_v21.py
# Interactive Isaac Sim demo for UROP_v21 bulky-object catching policy.
#
# v4 behavior:
#   - The learned policy runs continuously by default, including while the box is
#     held/positioned by the user.
#   - The box is kinematic while "held" so it behaves like an object in a human
#     thrower's hand instead of falling under gravity or being advanced by the old
#     autonomous toss event.
#   - Keyboard mode continuously moves a commanded held pose.  Mouse/gizmo mode
#     lets the Isaac Sim viewport transform manipulator move /World/CatchBox while
#     the policy keeps reading the live pose.
#   - Releasing the box only switches the object from kinematic to dynamic and
#     applies a release velocity; it does not reset/disarm the policy.

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path

import carb
import numpy as np

try:
    from isaacsim.examples.interactive.base_sample import BaseSample
except Exception:
    from isaacsim.examples.interactive.base_sample import BaseSample

try:
    from isaacsim.core.api.objects import DynamicCuboid
except Exception:
    try:
        from isaacsim.core.objects import DynamicCuboid
    except Exception:
        from omni.isaac.core.objects import DynamicCuboid

try:
    from isaacsim.core.utils.prims import get_prim_at_path
except Exception:
    try:
        from omni.isaac.core.utils.prims import get_prim_at_path
    except Exception:
        get_prim_at_path = None

try:
    from omni.physx import get_physx_simulation_interface
except Exception:
    get_physx_simulation_interface = None

try:
    from pxr import UsdPhysics, PhysxSchema
except Exception:
    UsdPhysics = None
    PhysxSchema = None

try:
    import omni.appwindow
except Exception:
    omni = None

from catch.g1_catch_policy import G1CatchPolicy, HOLD_ANCHOR_B, quat_apply, quat_rotate_inverse


OBJECT_SIZE = np.array([0.30, 0.23, 0.21], dtype=np.float32)
OBJECT_MASS = 2.0
GRAVITY_W = np.array([0.0, 0.0, -9.81], dtype=np.float32)


MOVE_KEYS = {"W", "S", "A", "D", "Q", "E"}


def project_root_from_this_file() -> Path:
    p = Path(__file__).resolve()
    for candidate in [p.parent, *p.parents]:
        if (candidate / "UROP").exists() and (candidate / "deploy").exists():
            return candidate
    return p.parents[2]


def default_policy_path(project_root: Path) -> Path:
    env_override = os.environ.get("G1_CATCH_POLICY") or os.environ.get("G1_CATCH_POLICY_FILE")
    if env_override:
        return Path(env_override).expanduser()
    return project_root / "logs" / "rsl_rl" / "UROP_v21" / "2026-06-18_00-33-55" / "exported" / "policy.pt"


def default_env_yaml_path() -> Path | None:
    env_override = os.environ.get("G1_CATCH_ENV_YAML")
    return Path(env_override).expanduser() if env_override else None


def key_name(event) -> str:
    raw = getattr(event, "input", "")
    name = getattr(raw, "name", str(raw))
    return str(name).upper().replace("KEY_", "").split(".")[-1]


def _as_np3(x, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if x is None:
        return np.asarray(default, dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return np.asarray(default, dtype=np.float32)
    return arr[:3]


class G1CatchV21Deploy(BaseSample):
    """Single-env interactive deploy for the real demo scenario.

    Default mode is the real experiment analogue:
      robot policy: always running
      box: held by a kinematic user hand until release

    Controls:
      W/S: move held box forward/back in the initial robot/sender frame
      A/D: move held box left/right
      Q/E: move held box up/down
      LEFT/RIGHT bracket: decrease/increase assisted throw arrival time
      SPACE: manual release using the measured hand/box velocity
      T: assisted ballistic throw from current held pose to the learned hold anchor
      H: grab/hold the box again at its current pose; policy stays running
      M: toggle keyboard-controlled hold vs Isaac Sim mouse/gizmo hold
      P: toggle policy enabled for debugging only; default ON
      G: action gain 1.0/0.5 for safety/debug
      R: reset robot and box; policy ON, box HELD
      V: toggle force-tag-visible override
      Y: print policy output probe
      Z: zero-action debug toggle
    """

    def __init__(self) -> None:
        super().__init__()
        self.project_root = project_root_from_this_file()
        self.policy_file = default_policy_path(self.project_root)
        self.env_yaml = default_env_yaml_path()
        self.usd_path = self.project_root / "UROP" / "UROP_v21" / "usd" / "g1_29dof_full_collider_flattened.usd"

        self.robot_policy: G1CatchPolicy | None = None
        self.box = None
        self._demo_world = None

        # Manual hand/object state.  held_rel_b is expressed in the sender frame
        # captured at reset (initial robot/root frame), not in the live falling base.
        self.held_rel_b = np.array([1.45, 0.00, 0.32], dtype=np.float32)
        self.held_mode = True
        self.script_controls_box = True  # False means Isaac Sim mouse/gizmo controls the kinematic box.
        self.policy_enabled = True

        self.arrival_time = 0.65
        self.launch_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.manual_release_velocity_gain = 1.15
        self.manual_release_max_speed = 5.5
        self.min_manual_throw_speed = 0.25
        self.force_tag_visible_override: float | None = None
        self._max_action_gain = 1.0

        self.sender_origin_pos = np.array([0.0, 0.0, 0.78], dtype=np.float32)
        self.sender_origin_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._keyboard = None
        self._input = None
        self._keyboard_sub = None
        self._keys_down: set[str] = set()
        self._step_count = 0

        self._held_pose_w_prev: np.ndarray | None = None
        self._box_pose_w_prev: np.ndarray | None = None
        self._box_obs_vel_w = np.zeros(3, dtype=np.float32)
        self._box_obs_vel_alpha = 0.45
        self._box_kinematic_state: bool | None = None
        self._last_mode_print_step = -100000

        # Speeds in sender frame: x forward/back, y left/right, z up/down.
        self.hand_speed_b = np.array([0.75, 0.65, 0.45], dtype=np.float32)
        self.hand_fast_multiplier = 2.2

    # ------------------------------------------------------------------
    # BaseSample hooks.
    # ------------------------------------------------------------------
    def setup_scene(self):
        world = self.get_world()
        self._demo_world = world
        world.scene.add_default_ground_plane()

        self.robot_policy = G1CatchPolicy(
            prim_path="/World/G1",
            policy_file_path=str(self.policy_file),
            env_yaml_path=str(self.env_yaml) if self.env_yaml else None,
            usd_path=str(self.usd_path),
            name="G1_v21",
            position=np.array([0.0, 0.0, 0.78], dtype=np.float32),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        world.scene.add(self.robot_policy.robot)

        start_pos = np.array([1.45, 0.0, 1.10], dtype=np.float32)
        self.box = world.scene.add(
            DynamicCuboid(
                prim_path="/World/CatchBox",
                name="CatchBox",
                position=start_pos,
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                scale=OBJECT_SIZE,
                size=1.0,
                mass=OBJECT_MASS,
                color=np.array([0.22, 0.66, 0.26], dtype=np.float32),
            )
        )

    async def setup_post_load(self):
        self._demo_world = self.get_world()
        if self.robot_policy is None:
            raise RuntimeError("robot_policy was not created in setup_scene().")
        self.robot_policy.initialize()
        self._install_keyboard()
        await self.reset_demo_async()
        self._demo_world.add_physics_callback("g1_catch_v21_policy_step", self.physics_step)
        self._print_help()

    async def setup_pre_reset(self):
        if self._demo_world is not None:
            try:
                self._demo_world.remove_physics_callback("g1_catch_v21_policy_step")
            except Exception:
                pass

    async def setup_post_reset(self):
        if self.robot_policy is not None:
            self.robot_policy.reset_robot_to_ready()
        await self.reset_demo_async()
        if self._demo_world is not None:
            self._demo_world.add_physics_callback("g1_catch_v21_policy_step", self.physics_step)

    def world_cleanup(self):
        if self._demo_world is not None:
            try:
                self._demo_world.remove_physics_callback("g1_catch_v21_policy_step")
            except Exception:
                pass

        if self._input is not None and self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass

        self._keyboard_sub = None
        self._keyboard = None
        self._input = None
        self.robot_policy = None
        self.box = None
        self._demo_world = None

    # ------------------------------------------------------------------
    # Demo state and physics.
    # ------------------------------------------------------------------
    async def reset_demo_async(self):
        self.held_mode = True
        self.script_controls_box = True
        self.policy_enabled = True
        self._step_count = 0
        self._keys_down.clear()
        self._held_pose_w_prev = None
        self._box_pose_w_prev = None
        self._box_obs_vel_w[:] = 0.0
        self._box_kinematic_state = None

        if self.robot_policy is not None:
            self.robot_policy.reset_robot_to_ready()
            self.robot_policy.reset_policy_state()
            self.robot_policy.debug_action_gain = self._max_action_gain
            self._capture_sender_frame()

        self._set_box_held_physics(True)
        self._place_box_at_held_pose(step_size=1.0 / 100.0, zero_velocity=True)
        await asyncio.sleep(0)
        print("[G1CatchV21] reset: policy ON, box HELD/keyboard-controlled")

    def physics_step(self, step_size: float):
        if self.robot_policy is None or self.box is None:
            return
        dt = max(float(step_size), 1e-6)

        if self.held_mode:
            self._set_box_held_physics(True)
            if self.script_controls_box:
                self._update_keyboard_hand_pose(dt)
                obj_pos, obj_vel = self._place_box_at_held_pose(step_size=dt, zero_velocity=False)
            else:
                # Mouse/gizmo mode: do not overwrite the viewport transform.  Keep
                # the object kinematic and estimate AprilTag velocity from pose differences.
                obj_pos, _quat = self.box.get_world_pose()
                obj_pos = _as_np3(obj_pos)
                obj_vel = self._estimate_box_velocity_from_pose(obj_pos, dt)
                self._box_obs_vel_w = obj_vel.astype(np.float32)
                self._sync_held_rel_from_world(obj_pos)
        else:
            self._set_box_held_physics(False)
            obj_pos, _quat = self.box.get_world_pose()
            obj_pos = _as_np3(obj_pos)
            actual_vel = self._get_box_linear_velocity()
            if np.linalg.norm(actual_vel) < 1e-5:
                actual_vel = self._estimate_box_velocity_from_pose(obj_pos, dt)
            obj_vel = actual_vel.astype(np.float32)
            self._box_obs_vel_w = obj_vel.copy()

        if self.policy_enabled:
            self.robot_policy.debug_action_gain = float(self._max_action_gain)
            self.robot_policy.forward(
                dt=dt,
                obj_pos_w=np.asarray(obj_pos, dtype=np.float32),
                obj_lin_vel_w=np.asarray(obj_vel, dtype=np.float32),
                tag_visible_override=self.force_tag_visible_override,
            )
        else:
            self.robot_policy.apply_ready_action()

        self._maybe_print_mode(dt, obj_pos, obj_vel)
        self._step_count += 1

    def _maybe_print_mode(self, dt: float, obj_pos: np.ndarray, obj_vel: np.ndarray):
        if self._step_count - self._last_mode_print_step < 100:
            return
        self._last_mode_print_step = self._step_count
        mode = "HELD-keyboard" if self.held_mode and self.script_controls_box else "HELD-mouse" if self.held_mode else "RELEASED-dynamic"
        print(
            f"[G1CatchV21][mode] step={self._step_count:05d} policy={'ON' if self.policy_enabled else 'OFF'} "
            f"box={mode} pos={np.round(obj_pos, 3)} obs_vel={np.round(obj_vel, 3)} gain={self._max_action_gain:.2f}"
        )

    # ------------------------------------------------------------------
    # Robot/sender-frame and box physics helpers.
    # ------------------------------------------------------------------
    def _root_pose(self):
        if self.robot_policy is None:
            return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        root_pos, root_quat, _lin, _ang = self.robot_policy._root_pose_vel_w()
        return root_pos.astype(np.float32), root_quat.astype(np.float32)

    def _capture_sender_frame(self):
        root_pos, root_quat = self._root_pose()
        self.sender_origin_pos = np.asarray(root_pos, dtype=np.float32).copy()
        self.sender_origin_quat = np.asarray(root_quat, dtype=np.float32).copy()

    def _held_world_pose(self):
        pos = self.sender_origin_pos + quat_apply(self.sender_origin_quat, self.held_rel_b)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return pos.astype(np.float32), quat

    def _sync_held_rel_from_world(self, pos_w: np.ndarray):
        rel = quat_rotate_inverse(self.sender_origin_quat, np.asarray(pos_w, dtype=np.float32) - self.sender_origin_pos)
        self.held_rel_b = np.array(
            [
                np.clip(rel[0], 0.75, 2.30),
                np.clip(rel[1], -0.85, 0.85),
                np.clip(rel[2], 0.00, 0.85),
            ],
            dtype=np.float32,
        )

    def _target_world_pos(self):
        root_pos, root_quat = self._root_pose()
        return (root_pos + quat_apply(root_quat, HOLD_ANCHOR_B)).astype(np.float32)

    def _update_keyboard_hand_pose(self, dt: float):
        direction = np.zeros(3, dtype=np.float32)
        if "W" in self._keys_down:
            direction[0] += 1.0
        if "S" in self._keys_down:
            direction[0] -= 1.0
        if "A" in self._keys_down:
            direction[1] += 1.0
        if "D" in self._keys_down:
            direction[1] -= 1.0
        if "Q" in self._keys_down:
            direction[2] += 1.0
        if "E" in self._keys_down:
            direction[2] -= 1.0

        if np.linalg.norm(direction) > 1.0:
            direction = direction / np.linalg.norm(direction)
        speed = self.hand_speed_b.copy()
        if "LEFT_SHIFT" in self._keys_down or "RIGHT_SHIFT" in self._keys_down or "SHIFT" in self._keys_down:
            speed *= self.hand_fast_multiplier
        self.held_rel_b += direction * speed * float(dt)
        self.held_rel_b[0] = float(np.clip(self.held_rel_b[0], 0.75, 2.30))
        self.held_rel_b[1] = float(np.clip(self.held_rel_b[1], -0.85, 0.85))
        self.held_rel_b[2] = float(np.clip(self.held_rel_b[2], 0.00, 0.85))

    def _place_box_at_held_pose(self, step_size: float, zero_velocity: bool = False):
        if self.box is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        pos, quat = self._held_world_pose()
        if zero_velocity or self._held_pose_w_prev is None:
            vel = np.zeros(3, dtype=np.float32)
        else:
            raw_vel = (pos - self._held_pose_w_prev) / max(float(step_size), 1e-6)
            raw_vel = np.clip(raw_vel, -self.manual_release_max_speed, self.manual_release_max_speed).astype(np.float32)
            vel = (1.0 - self._box_obs_vel_alpha) * self._box_obs_vel_w + self._box_obs_vel_alpha * raw_vel
        self._held_pose_w_prev = pos.copy()
        self._box_pose_w_prev = pos.copy()
        self._box_obs_vel_w = vel.astype(np.float32)

        try:
            self.box.set_world_pose(position=pos, orientation=quat)
        except Exception:
            self.box.set_world_pose(pos, quat)
        # Keep velocity only as the policy-observation estimate while the box is held.
        # Do NOT call set_linear_velocity/set_angular_velocity on a kinematic
        # PxRigidDynamic; PhysX logs "Body must be non-kinematic!".
        return pos, self._box_obs_vel_w.copy()

    def _estimate_box_velocity_from_pose(self, pos_w: np.ndarray, step_size: float) -> np.ndarray:
        pos_w = np.asarray(pos_w, dtype=np.float32).reshape(3)
        if self._box_pose_w_prev is None:
            vel = np.zeros(3, dtype=np.float32)
        else:
            vel = (pos_w - self._box_pose_w_prev) / max(float(step_size), 1e-6)
            vel = np.clip(vel, -self.manual_release_max_speed, self.manual_release_max_speed).astype(np.float32)
        self._box_pose_w_prev = pos_w.copy()
        return vel

    def _get_box_linear_velocity(self) -> np.ndarray:
        for method_name in ("get_linear_velocity", "get_world_linear_velocity"):
            method = getattr(self.box, method_name, None)
            if method is not None:
                try:
                    return np.asarray(method(), dtype=np.float32).reshape(-1)[:3]
                except Exception:
                    pass
        return np.zeros(3, dtype=np.float32)

    def _set_box_velocity(self, lin_vel: np.ndarray, ang_vel: np.ndarray):
        for method_name, value in (("set_linear_velocity", lin_vel), ("set_angular_velocity", ang_vel)):
            method = getattr(self.box, method_name, None)
            if method is not None:
                try:
                    method(np.asarray(value, dtype=np.float32))
                except Exception:
                    pass

    def _box_prim(self):
        if get_prim_at_path is None:
            return None
        try:
            return get_prim_at_path("/World/CatchBox")
        except Exception:
            return None

    def _set_box_held_physics(self, held: bool):
        """Best-effort switch between human-held kinematic box and released dynamic box."""
        held = bool(held)
        previous_kinematic_state = self._box_kinematic_state
        if previous_kinematic_state is held:
            return

        # Only zero PhysX velocity when we know the body is currently dynamic.
        # Calling setLinearVelocity on a kinematic body produces red PhysX errors.
        if held and previous_kinematic_state is False:
            self._set_box_velocity(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

        self._box_kinematic_state = held

        # Object wrappers differ across Isaac Sim versions, so use several
        # best-effort routes.  The USD attributes are the important fallback.
        for method_name in ("set_kinematic_enabled", "set_rigid_body_kinematic_enabled"):
            method = getattr(self.box, method_name, None)
            if method is not None:
                try:
                    method(held)
                except Exception:
                    pass
        for method_name in ("set_gravity_enabled", "set_disable_gravity"):
            method = getattr(self.box, method_name, None)
            if method is not None:
                try:
                    if method_name == "set_gravity_enabled":
                        method(not held)
                    else:
                        method(held)
                except Exception:
                    pass

        prim = self._box_prim()
        if prim is not None and getattr(prim, "IsValid", lambda: False)():
            if UsdPhysics is not None:
                try:
                    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
                    rb_api.CreateKinematicEnabledAttr().Set(held)
                except Exception:
                    pass
            if PhysxSchema is not None:
                try:
                    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    physx_api.CreateDisableGravityAttr().Set(held)
                except Exception:
                    pass

        if get_physx_simulation_interface is not None:
            try:
                get_physx_simulation_interface().flush_changes()
            except Exception:
                pass
        print(f"[G1CatchV21] box physics -> {'HELD kinematic/no-gravity' if held else 'RELEASED dynamic/gravity'}")

    # ------------------------------------------------------------------
    # Release modes.
    # ------------------------------------------------------------------
    def release_box_manual(self):
        """Release the box with the velocity produced by the user's hand motion."""
        if self.box is None:
            return
        pos, quat = self._held_world_pose() if self.script_controls_box else self.box.get_world_pose()
        pos = _as_np3(pos)
        quat = np.asarray(quat, dtype=np.float32).reshape(-1)[:4]
        v0 = self._box_obs_vel_w.astype(np.float32) * float(self.manual_release_velocity_gain)
        speed = float(np.linalg.norm(v0))
        if speed < self.min_manual_throw_speed:
            # If the user simply taps SPACE while the box is stationary, give a
            # tiny forward release instead of a pure drop.  Use T for a full
            # repeatable ballistic throw.
            target = self._target_world_pos()
            direction = target - pos
            n = float(np.linalg.norm(direction))
            if n > 1e-6:
                v0 = direction / n * self.min_manual_throw_speed
        speed = float(np.linalg.norm(v0))
        if speed > self.manual_release_max_speed:
            v0 *= self.manual_release_max_speed / max(speed, 1e-6)
        self._release_with_velocity(pos, quat, v0.astype(np.float32), self.launch_ang_vel, label="MANUAL_RELEASE")

    def launch_box_assisted(self):
        """Assisted ballistic throw to the learned hold anchor; policy remains running."""
        if self.box is None:
            return
        start_pos, start_quat = self._held_world_pose() if self.script_controls_box else self.box.get_world_pose()
        start_pos = _as_np3(start_pos)
        start_quat = np.asarray(start_quat, dtype=np.float32).reshape(-1)[:4]
        target_pos = self._target_world_pos()
        T = float(np.clip(self.arrival_time, 0.35, 1.20))
        v0 = (target_pos - start_pos - 0.5 * GRAVITY_W * T * T) / T
        v0 = np.clip(v0, -5.5, 5.5).astype(np.float32)
        self._release_with_velocity(start_pos, start_quat, v0, self.launch_ang_vel, label=f"ASSISTED_THROW T={T:.2f}s")

    def _release_with_velocity(self, pos, quat, lin_vel, ang_vel, label: str):
        self.held_mode = False
        self._keys_down.clear()
        try:
            self.box.set_world_pose(position=pos, orientation=quat)
        except Exception:
            self.box.set_world_pose(pos, quat)
        self._set_box_held_physics(False)
        self._set_box_velocity(lin_vel, ang_vel)
        self._box_obs_vel_w = np.asarray(lin_vel, dtype=np.float32).copy()
        self._box_pose_w_prev = np.asarray(pos, dtype=np.float32).copy()
        self._held_pose_w_prev = None
        print(f"[G1CatchV21] {label}: policy still {'ON' if self.policy_enabled else 'OFF'} pos={np.round(pos,3)} v0={np.round(lin_vel,3)}")

    def hold_box_again(self, capture_current_pose: bool = True):
        if self.box is None:
            return
        if capture_current_pose:
            pos, _quat = self.box.get_world_pose()
            self._sync_held_rel_from_world(_as_np3(pos))
        self.held_mode = True
        self._held_pose_w_prev = None
        self._box_pose_w_prev = None
        self._box_obs_vel_w[:] = 0.0
        self._set_box_held_physics(True)
        self._place_box_at_held_pose(step_size=1.0 / 100.0, zero_velocity=True)
        print(f"[G1CatchV21] HOLD again: policy still {'ON' if self.policy_enabled else 'OFF'} held_rel_b={np.round(self.held_rel_b,3)}")

    # ------------------------------------------------------------------
    # Keyboard.
    # ------------------------------------------------------------------
    def _install_keyboard(self):
        if omni is None:
            print("[G1CatchV21] omni.appwindow not available; keyboard disabled.")
            return
        try:
            appwindow = omni.appwindow.get_default_app_window()
            self._keyboard = appwindow.get_keyboard()
            self._input = carb.input.acquire_input_interface()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        except Exception as exc:
            print(f"[G1CatchV21] keyboard setup failed: {exc}")

    def _on_keyboard_event(self, event) -> bool:
        key = key_name(event)
        try:
            event_type = event.type
            is_press = event_type == carb.input.KeyboardEventType.KEY_PRESS
            is_repeat = event_type == carb.input.KeyboardEventType.KEY_REPEAT
            is_release = event_type == carb.input.KeyboardEventType.KEY_RELEASE
        except Exception:
            return True

        if is_release:
            self._keys_down.discard(key)
            return True

        if key in MOVE_KEYS or key in ("LEFT_SHIFT", "RIGHT_SHIFT", "SHIFT"):
            if is_press or is_repeat:
                self._keys_down.add(key)
            return True

        # Avoid firing toggles multiple times from KEY_REPEAT.
        if not is_press:
            return True

        handled = True
        if key in ("LEFT_BRACKET", "BRACKETLEFT", "["):
            self.arrival_time = max(0.35, self.arrival_time - 0.05)
            print(f"[G1CatchV21] assisted arrival_time={self.arrival_time:.2f}s")
        elif key in ("RIGHT_BRACKET", "BRACKETRIGHT", "]"):
            self.arrival_time = min(1.20, self.arrival_time + 0.05)
            print(f"[G1CatchV21] assisted arrival_time={self.arrival_time:.2f}s")
        elif key in ("SPACE",):
            self.release_box_manual()
        elif key in ("T",):
            self.launch_box_assisted()
        elif key in ("H",):
            self.hold_box_again(capture_current_pose=True)
        elif key in ("M",):
            self.script_controls_box = not self.script_controls_box
            if self.held_mode:
                if self.script_controls_box:
                    pos, _quat = self.box.get_world_pose()
                    self._sync_held_rel_from_world(_as_np3(pos))
                    self._held_pose_w_prev = None
                    self._place_box_at_held_pose(step_size=1.0 / 100.0, zero_velocity=True)
                self._set_box_held_physics(True)
            print(f"[G1CatchV21] held control mode -> {'KEYBOARD' if self.script_controls_box else 'MOUSE/GIZMO'}; policy remains {'ON' if self.policy_enabled else 'OFF'}")
        elif key in ("P",):
            self.policy_enabled = not self.policy_enabled
            print(f"[G1CatchV21] policy_enabled={self.policy_enabled} (debug only)")
        elif key in ("Y",):
            if self.robot_policy is not None and self.box is not None:
                pos, _quat = self.box.get_world_pose()
                self.robot_policy.probe_policy_outputs(_as_np3(pos), self._box_obs_vel_w.copy())
        elif key in ("G",):
            self._max_action_gain = 0.5 if self._max_action_gain > 0.75 else 1.0
            if self.robot_policy is not None:
                self.robot_policy.debug_action_gain = self._max_action_gain
            print(f"[G1CatchV21] max_action_gain={self._max_action_gain:.2f}")
        elif key in ("R",):
            asyncio.ensure_future(self.reset_demo_async())
            print("[G1CatchV21] reset requested")
        elif key in ("V",):
            if self.force_tag_visible_override is None:
                self.force_tag_visible_override = 1.0
                if self.robot_policy is not None:
                    self.robot_policy.force_tag_visible = True
                print("[G1CatchV21] force tag visible: ON")
            else:
                self.force_tag_visible_override = None
                if self.robot_policy is not None:
                    self.robot_policy.force_tag_visible = False
                print("[G1CatchV21] force tag visible: geometric")
        elif key in ("Z",):
            if self.robot_policy is not None:
                self.robot_policy.debug_force_zero_action = not self.robot_policy.debug_force_zero_action
                print(f"[G1CatchV21] zero action debug={self.robot_policy.debug_force_zero_action}")
        else:
            handled = False
        return True if handled else True

    def _print_help(self):
        print("\n" + "=" * 96)
        print("G1 Catch V21 interactive deploy loaded")
        print(f"  project_root : {self.project_root}")
        print(f"  policy       : {self.policy_file}")
        print(f"  env_yaml     : {self.env_yaml if self.env_yaml else 'auto-resolve near policy'}")
        print(f"  usd          : {self.usd_path}")
        print("  default      : policy ON continuously; box HELD kinematic until release")
        print("  keyboard box : hold W/S/A/D/Q/E; hold Shift for fast movement")
        print("  mouse box    : press M, select /World/CatchBox, move with Isaac Sim transform gizmo")
        print("  release      : SPACE manual release from current hand velocity; T assisted ballistic throw")
        print("  other keys   : H hold again, R reset, [/] assisted T time, P policy debug toggle, G gain, V tag, Y probe, Z zero-action")
        print("=" * 96 + "\n")
