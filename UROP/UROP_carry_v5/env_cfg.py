# [/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v5/env_cfg.py]

from __future__ import annotations

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
)
from isaaclab.scene import InteractiveSceneCfg

import isaaclab.envs.mdp as mdp_isaac

from . import mdp as custom_mdp
from . import scene_objects_cfg


# =============================================================================
# Scene
# =============================================================================

@configclass
class dj_urop_carry_v5_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(150.0, 150.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=500.0,
        ),
    )

    robot = scene_objects_cfg.dj_robot_cfg
    object = scene_objects_cfg.bulky_object_cfg

    contact_forces = scene_objects_cfg.full_body_contact_forces_cfg

    contact_torso = scene_objects_cfg.contact_torso_cfg
    contact_l_shoulder_yaw = scene_objects_cfg.contact_l_shoulder_yaw_cfg
    contact_l_elbow = scene_objects_cfg.contact_l_elbow_cfg
    contact_l_wrist_roll = scene_objects_cfg.contact_l_wrist_roll_cfg
    contact_l_wrist_pitch = scene_objects_cfg.contact_l_wrist_pitch_cfg
    contact_l_wrist_yaw = scene_objects_cfg.contact_l_wrist_yaw_cfg
    contact_l_hand = scene_objects_cfg.contact_l_hand_cfg
    contact_r_shoulder_yaw = scene_objects_cfg.contact_r_shoulder_yaw_cfg
    contact_r_elbow = scene_objects_cfg.contact_r_elbow_cfg
    contact_r_wrist_roll = scene_objects_cfg.contact_r_wrist_roll_cfg
    contact_r_wrist_pitch = scene_objects_cfg.contact_r_wrist_pitch_cfg
    contact_r_wrist_yaw = scene_objects_cfg.contact_r_wrist_yaw_cfg
    contact_r_hand = scene_objects_cfg.contact_r_hand_cfg


# =============================================================================
# Commands
# =============================================================================

@configclass
class CommandsCfg:
    # Slower and cleaner forward target first.
    base_velocity = mdp_isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp_isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.10, 0.22),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


# =============================================================================
# Actions
# =============================================================================

@configclass
class ActionsCfg:
    # Split leg action scales by function to suppress cross-step / twisting gait.

    legs_sagittal = mdp_isaac.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ],
        scale=0.38,
        use_default_offset=True,
        preserve_order=True,
    )

    legs_frontal = mdp_isaac.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_roll_joint",
            "left_ankle_roll_joint",
            "right_hip_roll_joint",
            "right_ankle_roll_joint",
        ],
        scale=0.10,
        use_default_offset=True,
        preserve_order=True,
    )

    legs_yaw = mdp_isaac.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
        ],
        scale=0.05,
        use_default_offset=True,
        preserve_order=True,
    )


# =============================================================================
# Observations
# =============================================================================

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_proprio = ObsTerm(
            func=custom_mdp.robot_proprio,
            params={"include_torque": True, "torque_scale": 1.0 / 80.0},
        )
        carry_command = ObsTerm(func=custom_mdp.carry_command)
        prev_actions = ObsTerm(func=custom_mdp.prev_actions)
        object_rel = ObsTerm(
            func=custom_mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "noise_std": 0.0, "clip": 5.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        robot_proprio = ObsTerm(
            func=custom_mdp.robot_proprio,
            params={"include_torque": True, "torque_scale": 1.0 / 80.0},
        )
        carry_command = ObsTerm(func=custom_mdp.carry_command)
        prev_actions = ObsTerm(func=custom_mdp.prev_actions)
        object_rel = ObsTerm(
            func=custom_mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "noise_std": 0.0, "clip": 5.0},
        )

        object_params = ObsTerm(func=custom_mdp.object_params)
        contact = ObsTerm(func=custom_mdp.contact_features)
        reset_grace = ObsTerm(func=custom_mdp.reset_grace_feature)
        target_rel = ObsTerm(func=custom_mdp.carry_target_rel_pos_feature)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# =============================================================================
# Rewards
# =============================================================================

@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp_isaac.is_terminated, weight=-50.0)
    alive = RewTerm(func=custom_mdp.alive_bonus, weight=0.2)

    # locomotion is the main goal now
    track_lin_vel_xy = RewTerm(
        func=custom_mdp.track_lin_vel_xy_exp,
        weight=3.2,
        params={"std": 0.28},
    )
    track_ang_vel_z = RewTerm(
        func=custom_mdp.track_ang_vel_z_exp,
        weight=0.0,
        params={"std": 0.35},
    )

    lin_vel_z_l2 = RewTerm(func=custom_mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=custom_mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=custom_mdp.flat_orientation_l2, weight=-1.0)

    # replace fixed nominal-height penalty with per-episode loaded-height reference
    root_height_ref = RewTerm(func=custom_mdp.root_height_ref_l2, weight=-1.8)

    joint_vel_l2 = RewTerm(func=custom_mdp.joint_vel_l2, weight=-1.0e-3)
    joint_acc_l2 = RewTerm(func=custom_mdp.joint_acc_l2, weight=-2.0e-7)
    joint_torques_l2 = RewTerm(
        func=custom_mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"torque_scale": 1.0},
    )
    action_rate_l2 = RewTerm(func=custom_mdp.action_rate_l2, weight=-0.01)
    feet_slide = RewTerm(
        func=custom_mdp.feet_slide,
        weight=-0.10,
        params={"contact_force_threshold": 5.0},
    )

    # still useful, but not sufficient alone
    single_support = RewTerm(
        func=custom_mdp.single_support_bonus,
        weight=0.12,
        params={"force_threshold": 5.0},
    )

    # preserve lower-body loaded stance softly, so policy does not instantly twist legs
    leg_ref_deviation = RewTerm(
        func=custom_mdp.leg_ref_deviation_l2,
        weight=-0.10,
        params={"deadzone": 0.10},
    )

    # object terms are support/safety, not main objective
    object_center = RewTerm(
        func=custom_mdp.object_center_reward,
        weight=0.20,
        params={"std": 0.12},
    )
    object_upright = RewTerm(
        func=custom_mdp.object_upright_reward,
        weight=0.18,
        params={"std": 0.35},
    )
    hold_object_vel = RewTerm(
        func=custom_mdp.hold_object_vel_reward,
        weight=0.12,
        params={"lin_std": 0.6, "ang_std": 1.5},
    )
    hug_contact = RewTerm(
        func=custom_mdp.hug_contact_bonus,
        weight=0.02,
        params={"torso_force_scale": 25.0, "limb_force_scale": 20.0},
    )
    bilateral_contact = RewTerm(
        func=custom_mdp.bilateral_contact_bonus,
        weight=0.01,
        params={"force_scale": 20.0},
    )
    not_drop = RewTerm(
        func=custom_mdp.not_drop_bonus,
        weight=0.20,
        params={"min_height": 0.18},
    )

    # upper-body regularizers stay mild because upper body is already servoed
    arm_ref_deviation = RewTerm(
        func=custom_mdp.arm_ref_deviation_l2,
        weight=-0.02,
        params={"deadzone": 0.08},
    )
    wrist_ref_deviation = RewTerm(
        func=custom_mdp.wrist_ref_deviation_l2,
        weight=-0.05,
        params={"deadzone": 0.05},
    )


# =============================================================================
# Terminations
# =============================================================================

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_isaac.time_out, time_out=True)

    fall = DoneTerm(
        func=custom_mdp.robot_fallen,
        params={"min_root_height": 0.45, "max_tilt_deg": 45.0},
    )
    drop = DoneTerm(
        func=custom_mdp.object_dropped,
        params={"min_object_height": 0.12, "max_object_rel_dist": 0.95, "use_grace": True},
    )
    object_tilt = DoneTerm(
        func=custom_mdp.object_tilt_exceeded,
        params={"max_tilt_deg": 60.0, "use_grace": True},
    )
    bad_state = DoneTerm(func=custom_mdp.numerical_instability)


# =============================================================================
# Events
# =============================================================================

@configclass
class EventCfg:
    reset_from_bank = EventTerm(
        func=custom_mdp.reset_from_catch_success_bank,
        mode="reset",
        params={
            #"bank_path": "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/tools/catch_success_bank.pt",
            "bank_path": "/home/idim5080-2/mdj/myIsaacLabstudy/UROP/UROP_carry_v3/tools/catch_success_bank.pt",
            "pos_noise_xy": 0.002,
            "yaw_noise_rad": 0.0,
            "vel_noise_scale": 0.0,
            "grace_steps": 10,
            "randomize_object": False,
        },
    )

    hold_upper_ref = EventTerm(
        func=custom_mdp.hold_upper_body_reference,
        mode="interval",
        interval_range_s=(0.02, 0.02),
    )

    decay_grace = EventTerm(
        func=custom_mdp.decay_reset_grace,
        mode="interval",
        interval_range_s=(0.02, 0.02),
    )


# =============================================================================
# Env cfg
# =============================================================================

@configclass
class dj_urop_carry_v5_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_carry_v5_SceneCfg = dj_urop_carry_v5_SceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0

        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation

        if hasattr(self.sim, "disable_contact_processing"):
            self.sim.disable_contact_processing = False


@configclass
class dj_urop_carry_v5_EnvCfg_PLAY(dj_urop_carry_v5_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.18)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0