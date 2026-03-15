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
)
from isaaclab.scene import InteractiveSceneCfg

from . import scene_objects_cfg
from . import mdp as mdp


@configclass
class dj_urop_carry_v0_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot = scene_objects_cfg.dj_robot_cfg
    object = scene_objects_cfg.bulky_object_cfg

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


@configclass
class CommandsCfg:
    # command manager를 쓰지 않고, carry 전용 command를 이벤트로 직접 샘플링합니다.
    command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """29-DOF joint position action (finger joints excluded)."""

    legs_sagittal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
        ],
        scale=0.32,
    )
    legs_frontal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_roll_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_ankle_roll_joint",
        ],
        scale=0.22,
    )
    legs_yaw = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
        scale=0.12,
    )
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        scale=0.22,
    )
    left_arm_load = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_pitch_joint", "left_elbow_joint"],
        scale=0.45,
    )
    right_arm_load = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=0.45,
    )
    left_arm_pose = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        ],
        scale=0.25,
    )
    right_arm_pose = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        scale=0.25,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        carry_cmd = ObsTerm(func=mdp.carry_command)
        carry_stage = ObsTerm(func=mdp.carry_stage)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "drop_prob": 0.0, "noise_std": 0.01},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        carry_cmd = ObsTerm(func=mdp.carry_command)
        carry_stage = ObsTerm(func=mdp.carry_stage)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "drop_prob": 0.0, "noise_std": 0.0},
        )
        obj_params = ObsTerm(func=mdp.object_params)
        contact = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_names": [
                    "contact_torso",
                    "contact_l_shoulder_yaw", "contact_l_elbow",
                    "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
                    "contact_r_shoulder_yaw", "contact_r_elbow",
                    "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand",
                ],
                "scale": 1.0 / 300.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # survival / stabilization
    alive = RewTerm(func=mdp.alive_bonus, weight=0.25)
    upright = RewTerm(func=mdp.upright_reward, weight=1.00)
    height = RewTerm(func=mdp.root_height_reward, weight=0.60, params={"target_z": 0.78, "sigma": 0.12})

    # carry locomotion task
    track_lin = RewTerm(func=mdp.command_tracking_lin_vel_reward, weight=2.50, params={"sigma": 0.28})
    track_yaw = RewTerm(func=mdp.command_tracking_ang_vel_reward, weight=1.20, params={"sigma": 0.35})

    # keep object hugged and stable
    arm_pose = RewTerm(func=mdp.arm_carry_pose_reward, weight=0.60, params={"sigma": 0.65})
    obj_center = RewTerm(func=mdp.object_centering_reward, weight=2.20, params={"sigma": 0.12})
    obj_upright = RewTerm(func=mdp.object_upright_reward, weight=1.00)
    obj_rel_vel = RewTerm(func=mdp.object_relative_velocity_reward, weight=1.00, params={"torso_body_name": "torso_link", "sigma": 0.45})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.00, params={"min_z": 0.45, "max_dist": 1.10})
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=1.40,
        params={
            "sensor_names_left": [
                "contact_l_shoulder_yaw", "contact_l_elbow",
                "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
            ],
            "sensor_names_right": [
                "contact_r_shoulder_yaw", "contact_r_elbow",
                "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand",
            ],
            "sensor_name_torso": "contact_torso",
            "thr": 2.0,
        },
    )

    # penalties
    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.25, params={"w_lin_z": 1.0, "w_ang_xy": 0.4})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.03)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00005)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.08)
    impact = RewTerm(
        func=mdp.impact_peak_penalty,
        weight=-0.003,
        params={
            "sensor_names": [
                "contact_torso",
                "contact_l_shoulder_yaw", "contact_l_elbow", "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
                "contact_r_shoulder_yaw", "contact_r_elbow", "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand",
            ],
            "force_thr": 280.0,
        },
    )
    obj_spin = RewTerm(func=mdp.object_spin_penalty, weight=-0.01, params={"scale": 1.0})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.45, "max_tilt_deg": 60.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.42, "max_dist": 1.20, "max_rel_x": 0.95})
    tilt = DoneTerm(func=mdp.object_tilted, params={"min_up_z": 0.15})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset object directly into carry state
    reset_object_in_carry_pose = EventTerm(
        func=mdp.reset_object_in_carry_pose,
        mode="reset",
        params={
            "stage0": {
                "hold_x": (0.39, 0.42), "hold_y": (-0.01, 0.01), "hold_z": (0.34, 0.36),
                "box_yaw_deg": (-2.0, 2.0),
                "mass_range": (2.0, 2.8),
                "friction_range": (0.65, 0.90),
                "restitution_range": (0.00, 0.03),
                "size_jitter": (0.96, 1.02),
            },
            "stage1": {
                "hold_x": (0.37, 0.41), "hold_y": (-0.015, 0.015), "hold_z": (0.33, 0.36),
                "box_yaw_deg": (-3.0, 3.0),
                "mass_range": (2.4, 3.6),
                "friction_range": (0.60, 0.92),
                "restitution_range": (0.00, 0.04),
                "size_jitter": (0.95, 1.04),
            },
            "stage2": {
                "hold_x": (0.36, 0.41), "hold_y": (-0.02, 0.02), "hold_z": (0.32, 0.36),
                "box_yaw_deg": (-4.0, 4.0),
                "mass_range": (2.8, 4.4),
                "friction_range": (0.55, 0.95),
                "restitution_range": (0.00, 0.05),
                "size_jitter": (0.94, 1.06),
            },
            "stage3": {
                "hold_x": (0.35, 0.42), "hold_y": (-0.03, 0.03), "hold_z": (0.31, 0.37),
                "box_yaw_deg": (-5.0, 5.0),
                "mass_range": (3.0, 5.0),
                "friction_range": (0.55, 0.95),
                "restitution_range": (0.00, 0.05),
                "size_jitter": (0.93, 1.08),
            },
            "apply_physx": True,
        },
    )

    # sample carry command at reset
    reset_carry_command = EventTerm(
        func=mdp.resample_carry_command,
        mode="reset",
        params={
            "stage0": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "stage1": {"lin_vel_x": (0.10, 0.30), "lin_vel_y": (-0.04, 0.04), "ang_vel_z": (-0.10, 0.10)},
            "stage2": {"lin_vel_x": (0.10, 0.45), "lin_vel_y": (-0.10, 0.10), "ang_vel_z": (-0.20, 0.20)},
            "stage3": {"lin_vel_x": (-0.05, 0.60), "lin_vel_y": (-0.18, 0.18), "ang_vel_z": (-0.45, 0.45)},
        },
    )

    # resample during episode to force genuine carrying, not one fixed motion
    resample_carry_command = EventTerm(
        func=mdp.resample_carry_command,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        params={
            "stage0": {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
            "stage1": {"lin_vel_x": (0.10, 0.30), "lin_vel_y": (-0.04, 0.04), "ang_vel_z": (-0.10, 0.10)},
            "stage2": {"lin_vel_x": (0.10, 0.45), "lin_vel_y": (-0.10, 0.10), "ang_vel_z": (-0.20, 0.20)},
            "stage3": {"lin_vel_x": (-0.05, 0.60), "lin_vel_y": (-0.18, 0.18), "ang_vel_z": (-0.45, 0.45)},
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            #"stage0_iters": 10,
            #"stage1_iters": 10,
            #"stage2_iters": 10,
            "stage0_iters": 500,
            "stage1_iters": 1200,
            "stage2_iters": 1800,
            "num_steps_per_env": 64,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_carry_v0_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_carry_v0_SceneCfg = dj_urop_carry_v0_SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation

        try:
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
                self.sim.physx.enable_external_forces_every_iteration = True
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "num_velocity_iterations"):
                self.sim.physx.num_velocity_iterations = 1
        except Exception:
            pass


@configclass
class dj_urop_carry_v0_EnvCfg_Play(dj_urop_carry_v0_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.stage_schedule.params["eval_stage"] = 3
        self.observations.policy.enable_corruption = False
