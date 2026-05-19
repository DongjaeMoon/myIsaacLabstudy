from __future__ import annotations

import argparse
import time

import numpy as np

from unitree_sdk2py.utils.thread import RecurrentThread

from .config_schema import CatchRealConfig
from .dds_io import DDSInterface
from .math_utils import interpolate
from .modes import ControllerMode
from .motor_command import MotorCommandWriter
from .obs_builder import ObservationBuilder
from .object_observation import ObjectObservation, ObjectObservationProvider
from .policy_runner import PolicyRunner
from .safety import apply_joint_target_safety, is_lowstate_fresh
from .state_reader import StateReader


class G1CatchRealController:
    FAKE_OBJECT_TOGGLE_KEY = "o"

    def __init__(self, cfg: CatchRealConfig, args: argparse.Namespace):
        self.cfg = cfg
        self.args = args

        self.control_dt = cfg.runtime.control_dt
        self.policy_dt = cfg.runtime.policy_dt
        self.device = args.device or cfg.runtime.device
        self.print_interval = (
            1.0 / float(args.print_rate)
            if args.print_rate is not None and args.print_rate > 0.0
            else float(cfg.runtime.print_every)
        )
        self.no_policy = bool(args.no_policy)
        self.move_duration = (
            float(args.move_duration)
            if args.move_duration is not None
            else float(cfg.modes.default_pose_transition_duration_s)
        )

        self.dds = DDSInterface(cfg.communication)
        max_motor_index = max(self.cfg.robot.motor_indices)
        if max_motor_index >= self.dds.command_slots:
            raise RuntimeError(
                f"Configured motor index {max_motor_index} exceeds LowCmd size {self.dds.command_slots}. "
                "Check robot.motor_indices in the YAML."
            )

        self.policy_runner = PolicyRunner(self.cfg, no_policy=self.no_policy, device=self.device)
        self.state_reader = StateReader(self.cfg)
        self.obs_builder = ObservationBuilder(self.cfg)
        self.object_provider = ObjectObservationProvider(self.cfg)
        self.command_writer: MotorCommandWriter | None = None
        self.policy_mode = ControllerMode(self.cfg.policy_runtime.policy_mode_name)

        self.current_mode = ControllerMode.DAMPING
        self.reference_pose_name = args.start_pose
        self.reference_pose_q = self.cfg.poses[self.reference_pose_name].copy()
        self.current_target_q: np.ndarray | None = None
        self.last_sent_q_des: np.ndarray | None = None
        self.transition_active = False
        self.transition_start_time = 0.0
        self.transition_duration = self.move_duration
        self.transition_start_q: np.ndarray | None = None
        self.transition_goal_q: np.ndarray | None = None

        self.started = False
        self.exit_requested = False
        self.control_thread: RecurrentThread | None = None
        self.last_policy_time = -1e9
        self.last_status_time = 0.0
        self.lowstate_warned_stale = False
        self.pending_auto_start_policy = False
        self.last_object_observation = self.object_provider.get()

    def init(self) -> None:
        self._print_startup_summary()
        self.policy_runner.load_if_enabled()
        self.dds.init_motion_switcher()
        self.dds.init_channels()
        self.command_writer = MotorCommandWriter(
            self.cfg,
            low_cmd=self.dds.low_cmd,
            publisher=self.dds.lowcmd_publisher,
            crc=self.dds.crc,
            command_slots=self.dds.command_slots,
        )

    def start(self) -> None:
        self.control_thread = RecurrentThread(interval=self.control_dt, target=self.control_step, name="g1_catch_real")
        while not self.dds.mode_machine_ready:
            time.sleep(0.1)
        self.control_thread.Start()
        print("[G1] DDS control loop started.")
        print(
            "[G1] Keys: "
            f"S=safe_stand, P/C=catch_ready, {self.cfg.policy_runtime.autonomous_key.upper()}=policy(auto), "
            f"{self.cfg.policy_runtime.manual_debug_key.upper()}=policy(debug), "
            f"{self.FAKE_OBJECT_TOGGLE_KEY.upper()}=fake_object, H=hold, D=damping, Q/ESC=exit"
        )

    def stop(self) -> None:
        if self.control_thread is not None:
            try:
                self.control_thread.Wait(timeout=1.0)
            except Exception:
                pass

    def request_mode(self, mode: ControllerMode, duration: float | None = None, announce: bool = True) -> None:
        if mode == ControllerMode.DAMPING:
            self.current_mode = ControllerMode.DAMPING
            self.transition_active = False
            self.pending_auto_start_policy = False
            self.policy_runner.zero_action()
            if announce:
                print("[G1] Mode -> DAMPING")
            return

        pose_name = self._pose_name_for_mode(mode)
        target_q = self.cfg.poses[pose_name].copy()
        now = time.monotonic()

        if self.current_target_q is None:
            start_q = target_q.copy()
        elif self.current_mode == ControllerMode.DAMPING and self.dds.low_state is not None:
            start_q, _ = self.state_reader.read_motor_state(self.dds.low_state)
        else:
            start_q = self.current_target_q.copy()

        self.current_mode = mode
        self.reference_pose_name = pose_name
        self.reference_pose_q = target_q.copy()
        self.transition_active = True
        self.transition_start_time = now
        self.transition_duration = float(duration if duration is not None else self.move_duration)
        self.transition_start_q = start_q
        self.transition_goal_q = target_q
        self.policy_runner.zero_action()
        self.policy_runner.reset_target(target_q)
        self.last_policy_time = -1e9
        self.pending_auto_start_policy = (
            mode == ControllerMode.CATCH_READY
            and self.cfg.policy_runtime.auto_start_after_ready
            and self._policy_is_available()
        )

        if announce:
            print(f"[G1] Mode -> {mode.value} (transition {self.transition_duration:.2f}s)")

    def handle_key(self, key: str) -> None:
        if key is None:
            return

        if key == self.cfg.modes.keyboard.get("safe_stand", "s"):
            self.request_mode(ControllerMode.SAFE_STAND)
        elif key in {
            self.cfg.modes.keyboard.get("catch_ready_primary", "p"),
            self.cfg.modes.keyboard.get("catch_ready_secondary", "c"),
        }:
            self.request_mode(ControllerMode.CATCH_READY)
        elif key == self.cfg.modes.keyboard.get("hold", "h"):
            self.request_mode(ControllerMode.HOLD)
        elif key == self.cfg.modes.keyboard.get("damping", "d"):
            self.request_mode(ControllerMode.DAMPING)
        elif key == self.cfg.policy_runtime.autonomous_key:
            self._request_policy_mode(manual_debug=False)
        elif key == self.cfg.policy_runtime.manual_debug_key:
            self._request_policy_mode(manual_debug=True)
        elif key == self.FAKE_OBJECT_TOGGLE_KEY:
            fake_enabled = self.object_provider.toggle_fake()
            print(f"[G1] Fake object observation -> {'ON' if fake_enabled else 'OFF'}")
        elif key in {
            self.cfg.modes.keyboard.get("exit_primary", "q"),
            self.cfg.modes.keyboard.get("exit_secondary", "esc"),
            "esc",
        }:
            print("[G1] Exit requested; switching to damping.")
            self.request_mode(ControllerMode.DAMPING, announce=False)
            self.exit_requested = True

    def control_step(self) -> None:
        now = time.monotonic()
        if self.dds.low_state is None:
            return

        self._initialize_targets_from_current_state()

        lowstate_fresh = is_lowstate_fresh(
            self.dds.low_state,
            self.dds.low_state_time,
            now,
            self.cfg.safety.lowstate_timeout_s,
        )
        if not lowstate_fresh:
            if not self.lowstate_warned_stale:
                print("[G1] LowState is stale; switching to damping until DDS state recovers.")
                self.lowstate_warned_stale = True
                self.current_mode = ControllerMode.DAMPING
                self.transition_active = False
                self.pending_auto_start_policy = False
                self.policy_runner.zero_action()
            self.command_writer.send_damping_command(self.dds.mode_machine)
            self._maybe_print_status(
                now,
                q=None,
                dq=None,
                gravity=None,
                object_observation=None,
                lowstate_fresh=False,
            )
            return

        self.lowstate_warned_stale = False
        q, dq = self.state_reader.read_motor_state(self.dds.low_state)
        _, _, projected_gravity, base_ang_vel = self.state_reader.read_base_state(self.dds.low_state)
        object_observation = self.object_provider.get()
        self.last_object_observation = object_observation
        obs = self.obs_builder.build(
            q=q,
            dq=dq,
            projected_gravity=projected_gravity,
            base_ang_vel=base_ang_vel,
            reference_pose_q=self.reference_pose_q,
            prev_action=self.policy_runner.prev_action,
            object_rel_pos=object_observation.rel_pos,
            object_rel_lin_vel=object_observation.rel_lin_vel,
            tag_visible=object_observation.tag_visible,
            current_mode_value=self.current_mode.value,
        )

        if self.current_mode == ControllerMode.DAMPING:
            self.policy_runner.zero_action()
            self.command_writer.send_damping_command(self.dds.mode_machine)
            self._maybe_print_status(
                now,
                q=q,
                dq=dq,
                gravity=projected_gravity,
                object_observation=object_observation,
                lowstate_fresh=True,
            )
            return

        nominal_target_q = self._compute_nominal_target(now, obs)
        q_des = apply_joint_target_safety(self.cfg, nominal_target_q, self.last_sent_q_des)
        self.command_writer.set_position_command(self.dds.mode_machine, q_des)
        self.current_target_q = q_des.copy()
        self.last_sent_q_des = q_des.copy()
        self._maybe_print_status(
            now,
            q=q,
            dq=dq,
            gravity=projected_gravity,
            object_observation=object_observation,
            lowstate_fresh=True,
        )

    def safe_shutdown(self) -> None:
        if self.cfg.safety.damping_on_exit and self.command_writer is not None:
            try:
                self.command_writer.send_damping_command(self.dds.mode_machine)
                time.sleep(0.05)
                self.command_writer.send_damping_command(self.dds.mode_machine)
            except Exception:
                pass

    def _print_startup_summary(self) -> None:
        kp_min = float(np.min(self.cfg.control.kp))
        kp_max = float(np.max(self.cfg.control.kp))
        kd_min = float(np.min(self.cfg.control.kd))
        kd_max = float(np.max(self.cfg.control.kd))
        policy_path_str = str(self.cfg.policy.path) if self.cfg.policy.path is not None else "None"
        print(f"[G1] loaded config path : {self.cfg.config_path}")
        print(f"[G1] net iface          : {self.args.net_iface or self.cfg.communication.net_iface or 'default DDS'}")
        print(f"[G1] controlled joints  : {self.cfg.robot.controlled_joint_names}")
        print(f"[G1] motor indices      : {self.cfg.robot.motor_indices}")
        print(
            f"[G1] control_dt/policy_dt: {self.control_dt:.4f}s / {self.policy_dt:.4f}s "
            f"({1.0 / self.control_dt:.1f} Hz / {1.0 / self.policy_dt:.1f} Hz)"
        )
        print(f"[G1] pose names         : {sorted([name for name in self.cfg.poses.keys() if name != 'catch'])}")
        print(f"[G1] start pose         : {self.reference_pose_name}")
        print(f"[G1] move duration      : {self.move_duration:.2f}s")
        print(f"[G1] policy disabled    : {self.no_policy}")
        print(f"[G1] policy enabled     : {not self.no_policy}")
        print(f"[G1] policy path        : {policy_path_str}")
        print(f"[G1] target lowpass     : alpha={self.cfg.runtime.target_lowpass_alpha:.3f}")
        print(f"[G1] kp summary         : min={kp_min:.2f}, max={kp_max:.2f}")
        print(f"[G1] kd summary         : min={kd_min:.2f}, max={kd_max:.2f}")
        print(f"[G1] obs dim/action dim : {self.cfg.observation.num_obs} / {self.cfg.policy.num_actions}")
        print(
            f"[G1] policy mode keys   : "
            f"auto={self.cfg.policy_runtime.autonomous_key.upper()}, "
            f"debug={self.cfg.policy_runtime.manual_debug_key.upper()}"
        )
        print(f"[G1] object source      : {self.object_provider.status_label()}")

    def _pose_name_for_mode(self, mode: ControllerMode) -> str:
        if mode == ControllerMode.SAFE_STAND:
            return "safe_stand"
        if mode == ControllerMode.CATCH_READY:
            return "catch_ready"
        if mode == self.policy_mode:
            return self.cfg.policy_runtime.default_policy_reference_pose
        if mode == ControllerMode.HOLD:
            return "hold"
        return self.reference_pose_name

    def _mode_from_pose_name(self, pose_name: str) -> ControllerMode:
        if pose_name == "safe_stand":
            return ControllerMode.SAFE_STAND
        if pose_name == "catch_ready":
            return ControllerMode.CATCH_READY
        if pose_name == "hold":
            return ControllerMode.HOLD
        raise ValueError(f"Unsupported start pose '{pose_name}'")

    def _initialize_targets_from_current_state(self) -> None:
        if self.started:
            return
        q_now, _ = self.state_reader.read_motor_state(self.dds.low_state)
        self.current_target_q = q_now.copy()
        self.last_sent_q_des = q_now.copy()
        self.policy_runner.reset_target(self.cfg.poses[self.cfg.policy_runtime.default_policy_reference_pose].copy())
        self.started = True
        initial_mode = self._mode_from_pose_name(self.reference_pose_name)
        self.request_mode(initial_mode, duration=self.move_duration, announce=False)
        print(f"[G1] Initial transition -> {initial_mode.value} over {self.move_duration:.2f}s")

    def _compute_nominal_target(self, now: float, obs: np.ndarray) -> np.ndarray:
        if self.current_mode == self.policy_mode and self._policy_is_available():
            if (now - self.last_policy_time) >= self.policy_dt:
                self.policy_runner.last_target_q = self.policy_runner.compute_target(obs, self.reference_pose_q)
                self.last_policy_time = now
            return self.policy_runner.last_target_q.copy()

        if self.transition_active:
            assert self.transition_start_q is not None
            assert self.transition_goal_q is not None
            if self.transition_duration <= 1e-6:
                alpha = 1.0
            else:
                alpha = float(np.clip((now - self.transition_start_time) / self.transition_duration, 0.0, 1.0))
            target = interpolate(self.transition_start_q, self.transition_goal_q, alpha)
            if alpha >= 1.0:
                self.transition_active = False
                if self.pending_auto_start_policy:
                    self.pending_auto_start_policy = False
                    self._request_policy_mode(manual_debug=False)
            return target

        return self.reference_pose_q.copy()

    def _maybe_print_status(
        self,
        now: float,
        q: np.ndarray | None,
        dq: np.ndarray | None,
        gravity: np.ndarray | None,
        object_observation: ObjectObservation | None,
        lowstate_fresh: bool,
    ) -> None:
        if self.print_interval <= 0.0:
            return
        if (now - self.last_status_time) < self.print_interval:
            return
        self.last_status_time = now
        policy_running = self.current_mode == self.policy_mode and self._policy_is_available()

        if q is None or dq is None or gravity is None or object_observation is None:
            print(
                f"[G1] mode={self.current_mode.value:<12s} "
                f"max|q-q_ref|=n/a max|dq|=n/a gravity=n/a "
                f"tag_visible=n/a object_rel_pos=n/a policy_running={policy_running} "
                f"lowstate_fresh={lowstate_fresh}"
            )
            return

        q_ref_err = float(np.max(np.abs(q - self.reference_pose_q)))
        max_dq = float(np.max(np.abs(dq)))
        gravity_str = np.array2string(np.round(gravity, 3), precision=3, separator=",")
        object_pos_str = np.array2string(np.round(object_observation.rel_pos, 3), precision=3, separator=",")
        tag_visible = int(float(object_observation.tag_visible[0]) > 0.5)
        print(
            f"[G1] mode={self.current_mode.value:<12s} "
            f"max|q-q_ref|={q_ref_err:.3f} max|dq|={max_dq:.3f} "
            f"gravity={gravity_str} tag_visible={tag_visible} "
            f"object_rel_pos={object_pos_str} policy_running={policy_running} "
            f"lowstate_fresh={lowstate_fresh}"
        )

    def _policy_is_available(self) -> bool:
        return (not self.no_policy) and self.policy_runner.is_loaded

    def _request_policy_mode(self, manual_debug: bool) -> None:
        if not self._policy_is_available():
            print(
                "[G1] Autonomous policy mode requested, but policy is not loaded or policy execution is disabled. "
                "Staying in the current scripted mode."
            )
            return

        self.request_mode(self.policy_mode, announce=False)
        if manual_debug:
            print("[G1] Mode -> manual/debug policy mode (using internal catch mode)")
        else:
            print("[G1] Mode -> autonomous policy running (using internal catch mode)")
