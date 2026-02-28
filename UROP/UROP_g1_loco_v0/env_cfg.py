import os
import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    CurriculumTermCfg as CurrTerm,
    SceneEntityCfg
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp_isaac
from . import mdp as custom_mdp
from . import scene_objects_cfg


@configclass
class dj_urop_g1_loco_v0_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot = scene_objects_cfg.dj_robot_cfg

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left_ankle_roll_link|right_ankle_roll_link|torso_link)",
        history_length=3,
        track_air_time=True,
    )

@configclass
class CommandsCfg:
    base_velocity = mdp_isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp_isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
        ),
    )

# [핵심 수정 1] 부위별로 Action Scale을 쪼개서 부드러운 하체 제어 유도 (별표 없이 43개 모두 나열)
@configclass
class ActionsCfg:
    # 1. 다리 (보행의 핵심): 스케일을 0.5 -> 0.25로 줄여서 세밀하고 부드러운 움직임 유도
    legs = mdp_isaac.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ],
        scale=0.5,
        use_default_offset=True,
    )
    # 2. 허리 및 팔: 스케일을 0.05로 대폭 줄여서 팔을 크게 휘적거리지 않고 균형만 잡게 함
    arms_and_waist = mdp_isaac.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ],
        scale=0.05,
        use_default_offset=True,
    )
    # 3. 손가락: 보행에 불필요하므로 스케일 0.0 (AI가 제어하지 못하게 완전히 묶음)
    hands = mdp_isaac.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint"
        ],
        scale=0.0,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp_isaac.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp_isaac.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp_isaac.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp_isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp_isaac.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp_isaac.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp_isaac.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp_isaac.is_terminated, weight=-200.0)

    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.5,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_mdp.track_ang_vel_z_world_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )

    feet_air_time = RewTerm(
        func=custom_mdp.feet_air_time_positive_biped, weight=0.75,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]), "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=custom_mdp.feet_slide, weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]), "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    
    hop_penalty = RewTerm(
        func=custom_mdp.both_feet_air_penalty, weight=-3.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    
    # [핵심 수정 2] 높이 페널티 완화: -5.0 -> -1.0 으로 줄여서 로봇이 자연스럽게 무릎 반동을 이용하도록 허용
    base_height_penalty = RewTerm(
        func=custom_mdp.base_height_penalty, weight=-1.0, 
        params={"target_height": 0.78, "asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z_l2 = RewTerm(func=mdp_isaac.lin_vel_z_l2, weight=-2.0)

    dof_pos_limits = RewTerm(
        func=mdp_isaac.joint_pos_limits, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"])},
    )
    
    # [핵심 수정 3] 떨림 방지 페널티 강화: action_rate (이전 행동과 너무 다르게 움직이는 것 방지)
    action_rate_l2 = RewTerm(func=mdp_isaac.action_rate_l2, weight=-0.02) # -0.005 -> -0.02 로 강화
    joint_vel_l2 = RewTerm(func=mdp_isaac.joint_vel_l2, weight=-0.0005)    # 관절이 휙휙 돌아가는 속도 자체에 페널티 추가
    
    flat_orientation_l2 = RewTerm(func=mdp_isaac.flat_orientation_l2, weight=-1.0)
    
    # 에너지 절약 (부드러운 움직임 보조)
    dof_acc_l2 = RewTerm(
        func=mdp_isaac.joint_acc_l2, weight=-1.25e-7, 
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp_isaac.joint_torques_l2, weight=-1.5e-5, # -1.5e-7 -> -1.5e-5 로 상향하여 힘을 과하게 쓰는 것 억제
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"])},
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_isaac.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp_isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]), "threshold": 1.0},
    )

@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp_isaac.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp_isaac.reset_joints_by_scale, mode="reset", params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp_isaac.push_by_setting_velocity,
        mode="interval", interval_range_s=(10.0, 15.0), params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class CurriculumCfg:
    pass

@configclass
class dj_urop_g1_loco_v0_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_g1_loco_v0_SceneCfg = dj_urop_g1_loco_v0_SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2                 
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 120                  
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt