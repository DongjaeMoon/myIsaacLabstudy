import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from . import scene_objects_cfg


CONTACT_SENSOR_NAMES = [
    "contact_torso",
    "contact_l_shoulder_yaw",
    "contact_l_elbow",
    "contact_l_wrist_roll",
    "contact_l_wrist_pitch",
    "contact_l_wrist_yaw",
    "contact_l_hand",
    "contact_r_shoulder_yaw",
    "contact_r_elbow",
    "contact_r_wrist_roll",
    "contact_r_wrist_pitch",
    "contact_r_wrist_yaw",
    "contact_r_hand",
]

# Actor policy observation contract: real-robot-available terms only.
POLICY_OBS_DIM = 100


@configclass
class dj_urop_v22_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(150.0, 150.0),
            physics_material=scene_objects_cfg.GROUND_PHYSICS_MATERIAL,
        ),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot = scene_objects_cfg.dj_robot_cfg
    object = scene_objects_cfg.bulky_object_cfg

    # Privileged contact sensors: critic only, never actor observation.
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
    command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    # DO NOT REORDER: this must stay aligned with the 29-DOF G1 action contract.
    policy = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(scene_objects_cfg.CONTROLLED_JOINT_NAMES),
        scale=dict(scene_objects_cfg.ACTION_SCALE_BY_JOINT),
        offset=dict(scene_objects_cfg.READY_POSE),
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 3: IMU-derived gravity direction.
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"noise_std": 0.010})
        # 3: IMU angular velocity.
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity, params={"noise_std": 0.025})
        # 29: deploy catch_ready-relative joint position.
        joint_pos_rel = ObsTerm(func=mdp.controlled_joint_pos_rel, params={"noise_std": 0.006})
        # 29: scaled joint velocity.
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities, params={"scale": 0.05, "noise_std": 0.20})
        # 29: previous controller action.
        prev_actions = ObsTerm(func=mdp.prev_actions)
        # 3+3+1: AprilTag-like object observation in camera optical frame.
        object_rel_pos = ObsTerm(func=mdp.object_rel_pos, params={"camera_frame": "opencv", "noise_std": 0.018})
        object_rel_lin_vel = ObsTerm(func=mdp.object_rel_lin_vel, params={"camera_frame": "opencv", "noise_std": 0.08})
        tag_visible = ObsTerm(func=mdp.tag_visible)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        # Privileged asymmetric critic terms.
        phase = ObsTerm(func=mdp.toss_state)
        hold_signal = ObsTerm(func=mdp.hold_state)
        drop_signal = ObsTerm(func=mdp.drop_state)
        robot_state = ObsTerm(func=mdp.critic_robot_state, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        obj_rel_full = ObsTerm(func=mdp.object_rel_full_state)
        obj_truth = ObsTerm(func=mdp.object_truth_state)
        root_state = ObsTerm(func=mdp.root_state_privileged)
        hold_anchor_err = ObsTerm(func=mdp.hold_anchor_error, params={"scale": 1.0})
        obj_params = ObsTerm(func=mdp.object_params)
        contact = ObsTerm(func=mdp.contact_forces, params={"sensor_names": CONTACT_SENSOR_NAMES, "scale": 1.0 / 300.0})

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # Balance / survival.
    alive = RewTerm(func=mdp.alive_bonus, weight=0.08)
    upright = RewTerm(func=mdp.upright_reward, weight=1.35, params={"sigma": 0.24})
    height = RewTerm(func=mdp.root_height_reward, weight=0.80, params={"target_z": 0.78, "sigma": 0.11})
    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.045)

    # False-positive suppression and reaction timing.
    ready_posture = RewTerm(func=mdp.ready_posture_reward, weight=0.55)
    idle_until_reaction = RewTerm(func=mdp.idle_until_reaction_reward, weight=1.20)
    early_arm_motion = RewTerm(func=mdp.early_arm_motion_penalty, weight=-2.25)
    no_toss_hug = RewTerm(func=mdp.no_toss_contact_like_penalty, weight=-2.00)
    reaction_timing = RewTerm(func=mdp.reaction_timing_reward, weight=1.00)

    # Hug geometry. These are gated by reaction/near-catch/post-catch phase, not by release alone.
    hand_side_proximity = RewTerm(func=mdp.hand_side_proximity_reward, weight=1.65)
    hug_bracket = RewTerm(func=mdp.hug_bracket_reward, weight=1.55)
    hug_symmetry = RewTerm(func=mdp.hug_symmetry_reward, weight=0.95)
    chest_pocket = RewTerm(func=mdp.chest_pocket_reward, weight=2.60)
    hug_depth = RewTerm(func=mdp.hug_depth_reward, weight=1.35)
    elbow_flexion = RewTerm(func=mdp.elbow_flexion_reward, weight=0.55)
    whole_body_absorb = RewTerm(func=mdp.whole_body_absorption_reward, weight=0.42)

    # Object reception and stabilization.
    object_anchor = RewTerm(func=mdp.object_anchor_reward, weight=1.50)
    object_velocity_damping = RewTerm(func=mdp.object_velocity_damping_reward, weight=2.35)
    object_height_safety = RewTerm(func=mdp.object_height_safety_reward, weight=0.50)
    successful_hold = RewTerm(func=mdp.successful_hold_reward, weight=5.00)
    sustained_hold = RewTerm(func=mdp.sustained_hold_reward, weight=2.20, params={"scale_steps": 60.0})
    stable_hug = RewTerm(func=mdp.stable_hug_reward, weight=1.80)
    post_catch_stillness = RewTerm(func=mdp.post_catch_stillness_reward, weight=0.80)

    # Failure / anti-hacking.
    object_drop = RewTerm(func=mdp.object_drop_penalty, weight=-4.50)
    object_escape = RewTerm(func=mdp.object_escape_penalty, weight=-2.50)
    front_shelf = RewTerm(func=mdp.front_shelf_penalty, weight=-1.20)

    # Smoothness / tremor suppression. Post-catch functions are phase-gated internally.
    joint_vel = RewTerm(func=mdp.joint_velocity_penalty, weight=-0.020)
    lower_body_joint_vel = RewTerm(func=mdp.lower_body_joint_velocity_penalty, weight=-0.055)
    action_mag = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.010)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.080)
    action_accel = RewTerm(func=mdp.action_acceleration_penalty, weight=-0.050)
    lower_body_action_rate = RewTerm(func=mdp.lower_body_action_rate_penalty, weight=-0.120)
    post_catch_action_rate = RewTerm(func=mdp.post_catch_action_rate_penalty, weight=-0.240)
    base_ang_vel_post_catch = RewTerm(func=mdp.base_ang_vel_post_catch_penalty, weight=-0.080)
    post_catch_tremor = RewTerm(func=mdp.post_catch_tremor_penalty, weight=-0.200)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 55})
    robot_fell = DoneTerm(func=mdp.robot_fell, params={"min_root_z": 0.46, "max_tilt_xy": 0.86})
    object_dropped = DoneTerm(func=mdp.object_dropped, params={"drop_z": 0.23, "grace_steps_after_release": 12})
    object_escaped = DoneTerm(func=mdp.object_escaped, params={"max_dist": 2.60, "behind_x": -0.62})
    invalid_object = DoneTerm(func=mdp.invalid_object_state)


@configclass
class EventCfg:
    object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": {"x": (0.85, 1.18), "y": (0.88, 1.16), "z": (0.88, 1.14)},
        },
    )
    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.60, 1.10),
            "dynamic_friction_range": (0.50, 1.00),
            "restitution_range": (0.0, 0.10),
            "num_buckets": 96,
            "make_consistent": True,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.1, 3.6),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 0.60,
        },
    )
    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(scene_objects_cfg.CONTROLLED_JOINT_NAMES)),
            "stiffness_distribution_params": (0.90, 1.10),
            "damping_distribution_params": (0.90, 1.16),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True, "joint_pos_noise": 0.014, "joint_vel_noise": 0.018},
    )
    reset_autonomous_episode = EventTerm(
        func=mdp.reset_autonomous_episode,
        mode="reset",
        params={
            "release_time_range_s": (0.42, 1.15),
            "delayed_release_time_range_s": (1.25, 2.65),
            "sender_x_range": (1.10, 1.70),
            "sender_y_range": (-0.28, 0.28),
            "sender_z_rel_range": (0.20, 0.46),
            "arrival_time_range_s": (0.55, 1.05),
            "target_noise_xyz": (0.07, 0.10, 0.08),
            "object_size_scale_range": (0.85, 1.18),
            "object_mass_range": (1.1, 3.6),
            "object_friction_range": (0.60, 1.10),
            "object_restitution_range": (0.0, 0.10),
            "obs_noise_scale_range": (0.65, 1.35),
            "tag_available_prob": 0.96,
        },
    )
    advance_toss = EventTerm(
        func=mdp.advance_autonomous_toss,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"hold_jitter_std": 0.002},
    )
    random_push = EventTerm(
        func=mdp.random_push,
        mode="interval",
        interval_range_s=(0.55, 1.25),
        params={
            "robot_push_prob": 0.22,
            "object_push_prob": 0.22,
            "robot_lin_vel_xy_range": (-0.16, 0.16),
            "robot_ang_vel_z_range": (-0.28, 0.28),
            "object_lin_vel_range": (-0.24, 0.24),
            "object_ang_vel_range": (-0.85, 0.85),
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={"thresholds": (12_000, 35_000, 75_000, 130_000), "force_stage": None},
    )


@configclass
class dj_urop_v22_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v22_SceneCfg = dj_urop_v22_SceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.5
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
        self.scene.replicate_physics = False

        assert scene_objects_cfg.EXPECTED_ACTION_DIM == 29
        assert len(scene_objects_cfg.CONTROLLED_JOINT_NAMES) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(scene_objects_cfg.ACTION_SCALE) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(self.actions.policy.joint_names) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert scene_objects_cfg.EXPECTED_POLICY_OBS_DIM == POLICY_OBS_DIM == 100
        assert abs(self.decimation * self.sim.dt - 0.02) < 1e-9


@configclass
class dj_urop_v22_EnvCfg_Play(dj_urop_v22_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.episode_length_s = 7.5
        self.curriculum.stage_schedule.params["force_stage"] = 2
        self.events.reset_autonomous_episode.params["stage_episode_probabilities"] = (
            (0.90, 0.08, 0.00, 0.02, 0.00),
            (0.90, 0.08, 0.00, 0.02, 0.00),
            (0.90, 0.08, 0.00, 0.02, 0.00),
            (0.90, 0.08, 0.00, 0.02, 0.00),
            (0.90, 0.08, 0.00, 0.02, 0.00),
        )
        self.events.reset_autonomous_episode.params["obs_noise_scale_range"] = (0.0, 0.0)
        self.events.reset_autonomous_episode.params["target_noise_xyz"] = (0.035, 0.050, 0.045)
        self.events.reset_autonomous_episode.params["sender_y_range"] = (-0.12, 0.12)
        self.events.reset_autonomous_episode.params["arrival_time_range_s"] = (0.65, 0.95)
        self.events.random_push = None
        self.events.robot_actuator_gains = None


@configclass
class dj_urop_v22_EnvCfg_Demo(dj_urop_v22_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 7.5
        self.curriculum.stage_schedule.params["force_stage"] = 0
        self.events.reset_autonomous_episode.params["stage_episode_probabilities"] = (
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.events.reset_autonomous_episode.params["release_time_range_s"] = (0.55, 0.70)
        self.events.reset_autonomous_episode.params["sender_x_range"] = (1.18, 1.36)
        self.events.reset_autonomous_episode.params["sender_y_range"] = (-0.045, 0.045)
        self.events.reset_autonomous_episode.params["sender_z_rel_range"] = (0.30, 0.39)
        self.events.reset_autonomous_episode.params["arrival_time_range_s"] = (0.72, 0.88)
        self.events.reset_autonomous_episode.params["target_noise_xyz"] = (0.015, 0.020, 0.020)
        self.events.reset_autonomous_episode.params["obs_noise_scale_range"] = (0.0, 0.0)
        self.events.reset_autonomous_episode.params["tag_available_prob"] = 1.0
        self.events.reset_autonomous_episode.params["demo_mode"] = True
        self.events.reset_autonomous_episode.params["demo_release_time_s"] = 0.62
        self.events.reset_autonomous_episode.params["demo_arrival_time_s"] = 0.76
        self.events.object_scale = None
        self.events.object_material = None
        self.events.object_mass = None
        self.events.robot_actuator_gains = None
        self.events.random_push = None
