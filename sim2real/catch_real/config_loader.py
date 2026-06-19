from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config_schema import (
    CameraConfig,
    CameraEstimatorConfig,
    CatchRealConfig,
    CommunicationConfig,
    ControlConfig,
    ImuConfig,
    MetadataConfig,
    ModesConfig,
    ObservationConfig,
    ObservationTermConfig,
    PolicyActionEntry,
    PolicyConfig,
    PolicyRuntimeConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    VirtualObjectConfig,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{name}' to be a mapping, got {type(value).__name__}")
    return value


def _ensure_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Expected '{name}' to be a list, got {type(value).__name__}")
    return value


def _resolve_existing_path(
    raw_path: str | None,
    config_path: Path,
    *,
    require_exists: bool = True,
) -> Path | None:
    if raw_path in (None, ""):
        return None

    candidate = Path(raw_path)
    if candidate.is_absolute():
        if require_exists and not candidate.exists():
            raise FileNotFoundError(f"Configured path does not exist: {candidate}")
        return candidate.resolve()

    repo_root = _repo_root()
    candidates = [
        (config_path.parent / candidate).resolve(),
        (repo_root / candidate).resolve(),
    ]
    for resolved in candidates:
        if resolved.exists():
            return resolved

    if not require_exists:
        return candidates[-1]

    raise FileNotFoundError(
        f"Could not resolve path '{raw_path}' relative to config '{config_path.parent}' "
        f"or repository root '{repo_root}'."
    )


def _validate_unique_strings(values: list[str], name: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"'{name}' contains duplicates: {duplicates}")


def _validate_unique_ints(values: list[int], name: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"'{name}' contains duplicates: {duplicates}")


def _optional_float_array(raw: Any, name: str, expected_len: int) -> np.ndarray | None:
    if raw in (None, ""):
        return None
    values = np.array(raw, dtype=np.float64).reshape(-1)
    if values.size != expected_len:
        raise ValueError(f"{name} must contain {expected_len} floats, got {values.size}")
    return values


def _pose_dict_to_array(
    pose_name: str,
    pose_raw: dict[str, Any],
    controlled_joint_names: list[str],
) -> np.ndarray:
    pose_raw = _ensure_mapping(pose_raw, f"poses.{pose_name}")
    missing = [joint for joint in controlled_joint_names if joint not in pose_raw]
    extras = [joint for joint in pose_raw.keys() if joint not in controlled_joint_names]
    if missing or extras:
        raise ValueError(
            f"Pose '{pose_name}' must define exactly the controlled joints. "
            f"Missing={missing}, Extra={extras}"
        )
    return np.array([float(pose_raw[joint]) for joint in controlled_joint_names], dtype=np.float64)


def _build_pd_arrays(control_raw: dict[str, Any], controlled_joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    default_pd = _ensure_mapping(control_raw.get("default_pd", {}), "control.default_pd")
    kp_default = float(default_pd["kp"])
    kd_default = float(default_pd["kd"])
    overrides = _ensure_mapping(control_raw.get("pd_overrides", {}), "control.pd_overrides")

    kp = np.full(len(controlled_joint_names), kp_default, dtype=np.float64)
    kd = np.full(len(controlled_joint_names), kd_default, dtype=np.float64)
    joint_name_to_slot = {name: index for index, name in enumerate(controlled_joint_names)}

    for joint_name, override_raw in overrides.items():
        if joint_name not in joint_name_to_slot:
            raise ValueError(f"PD override references unknown joint '{joint_name}'")
        override = _ensure_mapping(override_raw, f"control.pd_overrides.{joint_name}")
        slot = joint_name_to_slot[joint_name]
        kp[slot] = float(override.get("kp", kp[slot]))
        kd[slot] = float(override.get("kd", kd[slot]))

    return kp, kd


def _build_joint_limits(
    robot_raw: dict[str, Any],
    controlled_joint_names: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    joint_limits_raw = robot_raw.get("joint_limits")
    if joint_limits_raw in (None, {}):
        return None, None

    joint_limits = _ensure_mapping(joint_limits_raw, "robot.joint_limits")
    lower = np.full(len(controlled_joint_names), -np.inf, dtype=np.float64)
    upper = np.full(len(controlled_joint_names), np.inf, dtype=np.float64)

    for index, joint_name in enumerate(controlled_joint_names):
        if joint_name not in joint_limits:
            continue
        limit_raw = _ensure_mapping(joint_limits[joint_name], f"robot.joint_limits.{joint_name}")
        if "min" not in limit_raw or "max" not in limit_raw:
            raise ValueError(f"Joint limit for '{joint_name}' must contain 'min' and 'max'")
        lower[index] = float(limit_raw["min"])
        upper[index] = float(limit_raw["max"])
        if lower[index] > upper[index]:
            raise ValueError(f"Joint limit min/max invalid for '{joint_name}'")

    return lower, upper


def load_catch_real_config(config_path: str | Path, policy_override: str | None = None) -> CatchRealConfig:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML in '{config_path}' must be a mapping")

    known_top_level = {
        "name",
        "version",
        "runtime_version",
        "metadata",
        "runtime",
        "communication",
        "robot",
        "control",
        "poses",
        "policy",
        "policy_runtime",
        "observation",
        "modes",
        "camera",
        "virtual_object",
        "safety",
    }
    unknown_top_level = sorted(set(raw) - known_top_level)
    if unknown_top_level:
        print(f"[G1][config] WARNING unknown top-level config keys ignored: {unknown_top_level}")

    metadata_raw = _ensure_mapping(raw.get("metadata", {}), "metadata")
    runtime_raw = _ensure_mapping(raw.get("runtime", {}), "runtime")
    communication_raw = _ensure_mapping(raw.get("communication", {}), "communication")
    robot_raw = _ensure_mapping(raw.get("robot", {}), "robot")
    control_raw = _ensure_mapping(raw.get("control", {}), "control")
    poses_raw = _ensure_mapping(raw.get("poses", {}), "poses")
    policy_raw = _ensure_mapping(raw.get("policy", {}), "policy")
    policy_runtime_raw = _ensure_mapping(raw.get("policy_runtime", {}), "policy_runtime")
    observation_raw = _ensure_mapping(raw.get("observation", {}), "observation")
    modes_raw = _ensure_mapping(raw.get("modes", {}), "modes")
    camera_raw = _ensure_mapping(raw.get("camera", {}), "camera")
    camera_estimator_raw = _ensure_mapping(camera_raw.get("estimator", {}), "camera.estimator")
    virtual_object_raw = _ensure_mapping(raw.get("virtual_object", {}), "virtual_object")
    safety_raw = _ensure_mapping(raw.get("safety", {}), "safety")

    controlled_joint_names = [str(name) for name in _ensure_list(robot_raw.get("controlled_joint_names"), "robot.controlled_joint_names")]
    motor_indices = [int(index) for index in _ensure_list(robot_raw.get("motor_indices"), "robot.motor_indices")]
    num_motors = int(robot_raw["num_motors"])

    if len(controlled_joint_names) != num_motors:
        raise ValueError(
            f"robot.controlled_joint_names length mismatch: expected {num_motors}, got {len(controlled_joint_names)}"
        )
    if len(motor_indices) != num_motors:
        raise ValueError(f"robot.motor_indices length mismatch: expected {num_motors}, got {len(motor_indices)}")
    _validate_unique_strings(controlled_joint_names, "robot.controlled_joint_names")
    _validate_unique_ints(motor_indices, "robot.motor_indices")

    if min(motor_indices) < 0:
        raise ValueError(f"robot.motor_indices must be non-negative, got {motor_indices}")

    safe_stand = _pose_dict_to_array("safe_stand", poses_raw.get("safe_stand"), controlled_joint_names)
    catch_ready = _pose_dict_to_array("catch_ready", poses_raw.get("catch_ready"), controlled_joint_names)
    hold = _pose_dict_to_array("hold", poses_raw.get("hold"), controlled_joint_names)
    poses = {
        "safe_stand": safe_stand,
        "catch_ready": catch_ready,
        "catch": catch_ready.copy(),
        "hold": hold,
    }

    kp, kd = _build_pd_arrays(control_raw, controlled_joint_names)
    if len(kp) != num_motors:
        raise ValueError(f"kp length mismatch: expected {num_motors}, got {len(kp)}")
    if len(kd) != num_motors:
        raise ValueError(f"kd length mismatch: expected {num_motors}, got {len(kd)}")

    action_order_raw = _ensure_list(policy_raw.get("action_order"), "policy.action_order")
    action_order: list[PolicyActionEntry] = []
    action_joint_names: list[str] = []
    action_scales: list[float] = []
    joint_name_to_slot = {name: index for index, name in enumerate(controlled_joint_names)}
    for index, entry_raw in enumerate(action_order_raw):
        entry = _ensure_mapping(entry_raw, f"policy.action_order[{index}]")
        joint_name = str(entry["name"])
        if joint_name not in joint_name_to_slot:
            raise ValueError(f"policy.action_order[{index}] references unknown joint '{joint_name}'")
        action_order.append(
            PolicyActionEntry(
                name=joint_name,
                scale=float(entry["scale"]),
                group=str(entry.get("group", "")),
            )
        )
        action_joint_names.append(joint_name)
        action_scales.append(float(entry["scale"]))

    if len(action_order) != int(policy_raw["num_actions"]):
        raise ValueError(
            f"policy.action_order length mismatch: expected {policy_raw['num_actions']}, got {len(action_order)}"
        )
    _validate_unique_strings(action_joint_names, "policy.action_order names")
    action_scale_array = np.array(action_scales, dtype=np.float64)
    if len(action_scale_array) != int(policy_raw["num_actions"]):
        raise ValueError(
            f"action_scale length mismatch: expected {policy_raw['num_actions']}, got {len(action_scale_array)}"
        )
    action_slot_indices = np.array([joint_name_to_slot[name] for name in action_joint_names], dtype=np.int64)

    term_raw_list = _ensure_list(observation_raw.get("terms"), "observation.terms")
    terms: list[ObservationTermConfig] = []
    for index, entry_raw in enumerate(term_raw_list):
        entry = _ensure_mapping(entry_raw, f"observation.terms[{index}]")
        terms.append(
            ObservationTermConfig(
                name=str(entry["name"]),
                dim=int(entry["dim"]),
                source=str(entry["source"]),
                scale=float(entry.get("scale", 1.0)),
            )
        )
    obs_dim = sum(term.dim for term in terms)
    expected_obs_dim = int(observation_raw["num_obs"])
    if obs_dim != expected_obs_dim:
        raise ValueError(f"Observation dimension mismatch: YAML terms sum to {obs_dim}, but num_obs={expected_obs_dim}")

    term_dims = {term.name: term.dim for term in terms}
    if term_dims.get("joint_pos_rel") != num_motors:
        raise ValueError(f"joint_pos_rel dim must equal {num_motors}")
    if term_dims.get("joint_vel") != num_motors:
        raise ValueError(f"joint_vel dim must equal {num_motors}")

    # UROP-v23 calls this actor channel "prev_actions".  Older deploy YAMLs
    # used "previous_action".  Accept either spelling, but require the action
    # dimension when either term is present.
    prev_action_dim = term_dims.get("prev_actions", term_dims.get("previous_action"))
    if prev_action_dim is not None and prev_action_dim != int(policy_raw["num_actions"]):
        raise ValueError(
            f"prev_actions/previous_action dim must equal {policy_raw['num_actions']}, got {prev_action_dim}"
        )

    mode_names = [str(name) for name in _ensure_list(modes_raw.get("names"), "modes.names")]
    if "mode_one_hot" in term_dims and term_dims["mode_one_hot"] != len(mode_names):
        raise ValueError(
            f"mode_one_hot dim must equal len(modes.names); got {term_dims['mode_one_hot']} vs {len(mode_names)}"
        )
    if str(modes_raw["initial"]) not in poses:
        raise ValueError(f"modes.initial='{modes_raw['initial']}' must be one of {sorted(poses.keys())}")

    keyboard_raw = _ensure_mapping(modes_raw.get("keyboard", {}), "modes.keyboard")
    transitions_raw = _ensure_mapping(modes_raw.get("transitions", {}), "modes.transitions")

    policy_mode_name = str(policy_runtime_raw.get("policy_mode_name", "catch")).lower()
    if policy_mode_name not in mode_names:
        raise ValueError(
            f"policy_runtime.policy_mode_name='{policy_mode_name}' must be one of modes.names={mode_names}"
        )
    if policy_mode_name not in {"safe_stand", "catch_ready", "catch", "hold"}:
        raise ValueError(
            "policy_runtime.policy_mode_name must map to an existing controller mode value: "
            "safe_stand, catch_ready, catch, or hold"
        )

    default_policy_reference_pose = str(
        policy_runtime_raw.get("default_policy_reference_pose", "catch_ready")
    )
    if default_policy_reference_pose not in poses:
        raise ValueError(
            "policy_runtime.default_policy_reference_pose="
            f"'{default_policy_reference_pose}' must be one of {sorted(poses.keys())}"
        )

    policy_path_raw = policy_override if policy_override is not None else policy_raw.get("path")
    policy_path = _resolve_existing_path(policy_path_raw, config_path) if policy_path_raw not in (None, "") else None
    contract_path = _resolve_existing_path(
        metadata_raw.get("contract_path"),
        config_path,
        require_exists=False,
    )

    joint_limit_lower, joint_limit_upper = _build_joint_limits(robot_raw, controlled_joint_names)

    runtime = RuntimeConfig(
        control_dt=float(runtime_raw["control_dt"]),
        policy_dt=float(runtime_raw["policy_dt"]),
        device=str(runtime_raw.get("device", "auto")),
        print_every=float(runtime_raw.get("print_every", 1.0)),
        require_enter_before_start=bool(runtime_raw.get("require_enter_before_start", True)),
        default_move_duration=float(runtime_raw.get("default_move_duration", 3.0)),
        target_lowpass_alpha=float(runtime_raw.get("target_lowpass_alpha", 1.0)),
    )
    communication = CommunicationConfig(
        backend=str(communication_raw.get("backend", "unitree_dds")),
        net_iface=None if communication_raw.get("net_iface") in (None, "") else str(communication_raw["net_iface"]),
        msg_type=str(communication_raw.get("msg_type", "hg")),
        lowcmd_topic=str(communication_raw.get("lowcmd_topic", "rt/lowcmd")),
        lowstate_topic=str(communication_raw.get("lowstate_topic", "rt/lowstate")),
        release_high_level_mode=bool(communication_raw.get("release_high_level_mode", True)),
    )
    imu = ImuConfig(
        quaternion_order=str(_ensure_mapping(robot_raw.get("imu", {}), "robot.imu").get("quaternion_order", "auto")),
        policy_body_frame=str(_ensure_mapping(robot_raw.get("imu", {}), "robot.imu").get("policy_body_frame", "isaaclab")),
        use_base_linear_velocity=bool(_ensure_mapping(robot_raw.get("imu", {}), "robot.imu").get("use_base_linear_velocity", False)),
    )
    robot = RobotConfig(
        name=str(robot_raw.get("name", "unitree_g1")),
        num_motors=num_motors,
        controlled_joint_names=controlled_joint_names,
        motor_indices=motor_indices,
        imu=imu,
        joint_limit_lower=joint_limit_lower,
        joint_limit_upper=joint_limit_upper,
    )
    control = ControlConfig(
        command_type=str(control_raw.get("command_type", "joint_position_pd")),
        q_des_source=str(control_raw.get("q_des_source", "q_ref_plus_residual_action")),
        dq_des=float(control_raw.get("dq_des", 0.0)),
        tau_ff=float(control_raw.get("tau_ff", 0.0)),
        kp=kp,
        kd=kd,
    )
    policy = PolicyConfig(
        path=policy_path,
        num_actions=int(policy_raw["num_actions"]),
        action_clip=float(policy_raw.get("action_clip", 1.0)),
        action_order=action_order,
        action_joint_names=action_joint_names,
        action_scales=action_scale_array,
        action_slot_indices=action_slot_indices,
    )
    policy_runtime = PolicyRuntimeConfig(
        autonomous_key=str(policy_runtime_raw.get("autonomous_key", "a")).lower(),
        manual_debug_key=str(
            policy_runtime_raw.get("manual_debug_key", keyboard_raw.get("catch", "k"))
        ).lower(),
        policy_mode_name=policy_mode_name,
        default_policy_reference_pose=default_policy_reference_pose,
        auto_start_after_ready=bool(policy_runtime_raw.get("auto_start_after_ready", False)),
        object_source=str(policy_runtime_raw.get("object_source", "zeros")).lower(),
        object_observation_frame=str(
            policy_runtime_raw.get(
                "object_observation_frame",
                observation_raw.get("object_frame", observation_raw.get("frame", "policy_body")),
            )
        ).lower(),
        fake_object_debug=bool(policy_runtime_raw.get("fake_object_debug", False)),
        gate_policy_until_object_visible=bool(
            policy_runtime_raw.get("gate_policy_until_object_visible", False)
        ),
        object_visible_blend_duration_s=float(
            policy_runtime_raw.get("object_visible_blend_duration_s", 0.4)
        ),
        gantry_upper_body_only=bool(policy_runtime_raw.get("gantry_upper_body_only", False)),
        gantry_lower_body_action_scale=float(
            policy_runtime_raw.get("gantry_lower_body_action_scale", 1.0)
        ),
        gantry_upper_body_action_scale=float(
            policy_runtime_raw.get("gantry_upper_body_action_scale", 1.0)
        ),
    )
    observation = ObservationConfig(
        contract_name=str(observation_raw.get("contract_name", "real_catch_v1")),
        num_obs=expected_obs_dim,
        frame=str(observation_raw.get("frame", "policy_body")),
        use_torque_obs=bool(observation_raw.get("use_torque_obs", False)),
        use_base_linear_velocity_obs=bool(observation_raw.get("use_base_linear_velocity_obs", False)),
        q_reference=str(observation_raw.get("q_reference", "current_mode_reference_pose")),
        terms=terms,
    )
    modes = ModesConfig(
        names=mode_names,
        initial=str(modes_raw["initial"]),
        keyboard={key: str(value).lower() for key, value in keyboard_raw.items()},
        default_pose_transition_duration_s=float(
            transitions_raw.get("default_pose_transition_duration_s", runtime.default_move_duration)
        ),
    )
    camera = CameraConfig(
        enabled=bool(camera_raw.get("enabled", False)),
        server_address=str(camera_raw.get("server_address", "127.0.0.1")),
        port=int(camera_raw.get("port", 5555)),
        image_show=bool(camera_raw.get("image_show", False)),
        intrinsics_yaml=_resolve_existing_path(
            camera_raw.get("intrinsics_yaml"),
            config_path,
            require_exists=False,
        ),
        extrinsics_yaml=_resolve_existing_path(
            camera_raw.get("extrinsics_yaml"),
            config_path,
            require_exists=False,
        ),
        tag_yaml=_resolve_existing_path(
            camera_raw.get("tag_yaml"),
            config_path,
            require_exists=False,
        ),
        extrinsics_parent_frame=(
            None if camera_raw.get("extrinsics_parent_frame") in (None, "")
            else str(camera_raw.get("extrinsics_parent_frame"))
        ),
        dynamic_body_to_camera=bool(camera_raw.get("dynamic_body_to_camera", False)),
        body_to_torso_urdf=_resolve_existing_path(
            camera_raw.get("body_to_torso_urdf"),
            config_path,
            require_exists=False,
        ),
        body_link_name=(
            None if camera_raw.get("body_link_name") in (None, "")
            else str(camera_raw.get("body_link_name"))
        ),
        torso_link_name=str(camera_raw.get("torso_link_name", "torso_link")),
        waist_joint_names=[
            str(name)
            for name in _ensure_list(
                camera_raw.get(
                    "waist_joint_names",
                    ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                ),
                "camera.waist_joint_names",
            )
        ],
        training_camera_translation=_optional_float_array(
            camera_raw.get("training_camera_translation"),
            "camera.training_camera_translation",
            3,
        ),
        training_camera_quat_wxyz=_optional_float_array(
            camera_raw.get("training_camera_quat_wxyz"),
            "camera.training_camera_quat_wxyz",
            4,
        ),
        training_camera_convention=str(
            camera_raw.get("training_camera_convention", "")
        ).lower(),
        estimator=CameraEstimatorConfig(
            lost_timeout_s=float(camera_estimator_raw.get("lost_timeout_s", 0.2)),
            min_valid_detections=max(int(camera_estimator_raw.get("min_valid_detections", 1)), 1),
            position_filter_alpha=float(camera_estimator_raw.get("position_filter_alpha", 0.35)),
            velocity_filter_alpha=float(camera_estimator_raw.get("velocity_filter_alpha", 0.25)),
            angular_velocity_filter_alpha=float(
                camera_estimator_raw.get(
                    "angular_velocity_filter_alpha",
                    camera_estimator_raw.get("velocity_filter_alpha", 0.25),
                )
            ),
            status_print_interval_s=float(camera_estimator_raw.get("status_print_interval_s", 1.0)),
        ),
    )
    _validate_unique_strings(camera.waist_joint_names, "camera.waist_joint_names")
    for joint_name in camera.waist_joint_names:
        if joint_name not in controlled_joint_names:
            raise ValueError(
                f"camera.waist_joint_names references unknown controlled joint '{joint_name}'"
            )
    virtual_object = VirtualObjectConfig(
        enabled_when_no_camera=bool(virtual_object_raw.get("enabled_when_no_camera", True)),
        initial_pos_base=np.array(virtual_object_raw.get("initial_pos_base", [0.0, 0.0, 0.0]), dtype=np.float64),
        initial_vel_base=np.array(virtual_object_raw.get("initial_vel_base", [0.0, 0.0, 0.0]), dtype=np.float64),
    )
    safety = SafetyConfig(
        damping_on_exit=bool(safety_raw.get("damping_on_exit", True)),
        damping_kd=float(safety_raw.get("damping_kd", 8.0)),
        clamp_action=bool(safety_raw.get("clamp_action", True)),
        clamp_joint_targets_to_limits=bool(safety_raw.get("clamp_joint_targets_to_limits", False)),
        max_target_delta_per_control_step=(
            None if safety_raw.get("max_target_delta_per_control_step") is None
            else float(safety_raw.get("max_target_delta_per_control_step"))
        ),
        max_target_delta_per_policy_step=(
            None if safety_raw.get("max_target_delta_per_policy_step") is None
            else float(safety_raw.get("max_target_delta_per_policy_step"))
        ),
        lowstate_timeout_s=float(safety_raw.get("lowstate_timeout_s", 0.10)),
        max_abs_joint_velocity_warn=float(safety_raw.get("max_abs_joint_velocity_warn", 20.0)),
        max_abs_action_warn=float(safety_raw.get("max_abs_action_warn", 0.95)),
        emergency_gravity_z_min=float(safety_raw.get("emergency_gravity_z_min", -0.3)),
    )

    if communication.backend != "unitree_dds":
        raise ValueError(f"Unsupported communication.backend '{communication.backend}'; expected 'unitree_dds'")
    if communication.msg_type.lower() != "hg":
        raise ValueError(f"Unsupported communication.msg_type '{communication.msg_type}'; expected 'hg' for G1")
    if control.command_type != "joint_position_pd":
        raise ValueError(f"Unsupported control.command_type '{control.command_type}'")

    return CatchRealConfig(
        config_path=config_path,
        repo_root=_repo_root(),
        name=str(raw.get("name", "g1_catch_real")),
        version=str(raw.get("version", raw.get("runtime_version", "0.0.0"))),
        metadata=MetadataConfig(
            urop_version=str(
                metadata_raw.get(
                    "urop_version",
                    policy_raw.get("version", raw.get("runtime_version", raw.get("version", "unknown"))),
                )
            ),
            contract_path=contract_path,
            run_safety=str(metadata_raw.get("run_safety", "real_policy_allowed")),
            notes=[str(note) for note in metadata_raw.get("notes", [])],
        ),
        runtime=runtime,
        communication=communication,
        robot=robot,
        control=control,
        poses=poses,
        policy=policy,
        policy_runtime=policy_runtime,
        observation=observation,
        modes=modes,
        camera=camera,
        virtual_object=virtual_object,
        safety=safety,
    )
