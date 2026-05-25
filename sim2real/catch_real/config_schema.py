from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RuntimeConfig:
    control_dt: float
    policy_dt: float
    device: str
    print_every: float
    require_enter_before_start: bool
    default_move_duration: float
    target_lowpass_alpha: float


@dataclass(frozen=True)
class CommunicationConfig:
    backend: str
    net_iface: str | None
    msg_type: str
    lowcmd_topic: str
    lowstate_topic: str
    release_high_level_mode: bool


@dataclass(frozen=True)
class ImuConfig:
    quaternion_order: str
    policy_body_frame: str
    use_base_linear_velocity: bool


@dataclass(frozen=True)
class RobotConfig:
    name: str
    num_motors: int
    controlled_joint_names: list[str]
    motor_indices: list[int]
    imu: ImuConfig
    joint_limit_lower: np.ndarray | None
    joint_limit_upper: np.ndarray | None


@dataclass(frozen=True)
class ControlConfig:
    command_type: str
    q_des_source: str
    dq_des: float
    tau_ff: float
    kp: np.ndarray
    kd: np.ndarray


@dataclass(frozen=True)
class PolicyActionEntry:
    name: str
    scale: float
    group: str


@dataclass(frozen=True)
class PolicyConfig:
    path: Path | None
    num_actions: int
    action_clip: float
    action_order: list[PolicyActionEntry]
    action_joint_names: list[str]
    action_scales: np.ndarray
    action_slot_indices: np.ndarray


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    autonomous_key: str
    manual_debug_key: str
    policy_mode_name: str
    default_policy_reference_pose: str
    auto_start_after_ready: bool
    object_source: str
    fake_object_debug: bool
    gate_policy_until_object_visible: bool
    object_visible_blend_duration_s: float
    gantry_upper_body_only: bool
    gantry_lower_body_action_scale: float
    gantry_upper_body_action_scale: float


@dataclass(frozen=True)
class ObservationTermConfig:
    name: str
    dim: int
    source: str
    scale: float


@dataclass(frozen=True)
class ObservationConfig:
    contract_name: str
    num_obs: int
    frame: str
    use_torque_obs: bool
    use_base_linear_velocity_obs: bool
    q_reference: str
    terms: list[ObservationTermConfig]


@dataclass(frozen=True)
class ModesConfig:
    names: list[str]
    initial: str
    keyboard: dict[str, str]
    default_pose_transition_duration_s: float


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool
    server_address: str
    port: int
    image_show: bool
    intrinsics_yaml: Path | None
    extrinsics_yaml: Path | None
    tag_yaml: Path | None


@dataclass(frozen=True)
class VirtualObjectConfig:
    enabled_when_no_camera: bool
    initial_pos_base: np.ndarray
    initial_vel_base: np.ndarray


@dataclass(frozen=True)
class SafetyConfig:
    damping_on_exit: bool
    damping_kd: float
    clamp_action: bool
    clamp_joint_targets_to_limits: bool
    max_target_delta_per_control_step: float | None
    max_target_delta_per_policy_step: float | None
    lowstate_timeout_s: float
    max_abs_joint_velocity_warn: float
    max_abs_action_warn: float
    emergency_gravity_z_min: float


@dataclass(frozen=True)
class CatchRealConfig:
    config_path: Path
    repo_root: Path
    name: str
    version: str
    runtime: RuntimeConfig
    communication: CommunicationConfig
    robot: RobotConfig
    control: ControlConfig
    poses: dict[str, np.ndarray]
    policy: PolicyConfig
    policy_runtime: PolicyRuntimeConfig
    observation: ObservationConfig
    modes: ModesConfig
    camera: CameraConfig
    virtual_object: VirtualObjectConfig
    safety: SafetyConfig
