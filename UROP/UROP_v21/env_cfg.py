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

# Actor policy observation contract. Contact force, object truth, episode type,
# object physical parameters, and other simulator-only terms are critic-only.
POLICY_OBS_DIM = 100


@configclass
class dj_urop_v21_SceneCfg(InteractiveSceneCfg):
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

    # Contact sensors are privileged. They are exposed to the critic only, not to
    # the actor policy observation.
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
    # DO NOT REORDER: must match sim2real deploy policy action order.
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
        # 3: IMU-derived gravity direction in robot/base frame.
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"noise_std": 0.010})
        # 3: IMU angular velocity in robot/base frame.
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity, params={"noise_std": 0.025})
        # 29: controlled joint positions relative to deploy catch_ready q_ref.
        joint_pos_rel = ObsTerm(func=mdp.controlled_joint_pos_rel, params={"noise_std": 0.006})
        # 29: joint velocities, scaled to match deploy YAML observation scale.
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities, params={"scale": 0.05, "noise_std": 0.20})
        # 29: last action sent by the policy/controller.
        prev_actions = ObsTerm(func=mdp.prev_actions)
        # 3: AprilTag/object translation in camera optical frame: x right, y down, z forward.
        object_rel_pos = ObsTerm(func=mdp.object_rel_pos, params={"camera_frame": "opencv", "noise_std": 0.018})
        # 3: AprilTag/object linear velocity in the same camera optical frame.
        object_rel_lin_vel = ObsTerm(func=mdp.object_rel_lin_vel, params={"camera_frame": "opencv", "noise_std": 0.08})
        # 1: current tag detection bit with dropout; 0 means object pose/velocity terms are zeroed.
        tag_visible = ObsTerm(func=mdp.tag_visible)

        def __post_init__(self):
            self.concatenate_terms = True
            # Noise/dropout is implemented inside each real-observation term so
            # that object pose and tag_visible stay temporally consistent.
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        # Privileged terms for asymmetric actor-critic / teacher-student style training.
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
    alive = RewTerm(func=mdp.alive_bonus, weight=0.10)
    upright = RewTerm(func=mdp.upright_reward, weight=1.40, params={"sigma": 0.24})
    height = RewTerm(func=mdp.root_height_reward, weight=0.85, params={"target_z": 0.78, "sigma": 0.11})
    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.060)

    # False-positive suppression and timing.
    idle_until_reaction = RewTerm(func=mdp.idle_until_reaction_reward, weight=0.70)
    early_arm_motion = RewTerm(func=mdp.early_arm_motion_penalty, weight=-0.90)
    reaction_timing = RewTerm(func=mdp.reaction_timing_reward, weight=0.85)

    # Whole-body catch geometry. These terms are geometric/proprioceptive shaping;
    # they do not expose contact force to the actor.
    hand_side_proximity = RewTerm(func=mdp.hand_side_proximity_reward, weight=2.15)
    hug_symmetry = RewTerm(func=mdp.hug_symmetry_reward, weight=0.90)
    whole_body_absorb = RewTerm(func=mdp.whole_body_absorption_reward, weight=0.35)

    # Object stabilization.
    object_anchor = RewTerm(func=mdp.object_anchor_reward, weight=2.30)
    object_velocity_damping = RewTerm(func=mdp.object_velocity_damping_reward, weight=2.15)
    object_height_safety = RewTerm(func=mdp.object_height_safety_reward, weight=0.55)
    successful_hold = RewTerm(func=mdp.successful_hold_reward, weight=3.20)
    sustained_hold = RewTerm(func=mdp.sustained_hold_reward, weight=1.60, params={"scale_steps": 50.0})

    # Failure / regularization.
    object_drop = RewTerm(func=mdp.object_drop_penalty, weight=-4.00)
    object_escape = RewTerm(func=mdp.object_escape_penalty, weight=-2.50)
    no_toss_hug = RewTerm(func=mdp.no_toss_contact_like_penalty, weight=-1.20)
    joint_vel = RewTerm(func=mdp.joint_velocity_penalty, weight=-0.030)
    action_mag = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.012)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.060)
    action_accel = RewTerm(func=mdp.action_acceleration_penalty, weight=-0.030)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 45})
    robot_fell = DoneTerm(func=mdp.robot_fell, params={"min_root_z": 0.46, "max_tilt_xy": 0.86})
    object_dropped = DoneTerm(func=mdp.object_dropped, params={"drop_z": 0.22, "grace_steps_after_release": 8})
    object_escaped = DoneTerm(func=mdp.object_escaped, params={"max_dist": 2.75, "behind_x": -0.65})
    invalid_object = DoneTerm(func=mdp.invalid_object_state)


@configclass
class EventCfg:
    # Startup/pre-start randomization. Shape scaling is applied before physics starts.
    object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": {"x": (0.80, 1.25), "y": (0.82, 1.22), "z": (0.82, 1.18)},
        },
    )

    # Reset-time physics randomization.
    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.55, 1.15),
            "dynamic_friction_range": (0.45, 1.05),
            "restitution_range": (0.0, 0.12),
            "num_buckets": 96,
            "make_consistent": True,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 4.0),
            "operation": "abs",
            "distribution": "uniform",
            "recompute_inertia": True,
            "min_mass": 0.50,
        },
    )
    robot_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(scene_objects_cfg.CONTROLLED_JOINT_NAMES)),
            "stiffness_distribution_params": (0.86, 1.14),
            "damping_distribution_params": (0.82, 1.18),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Reset state and autonomous toss schedule.
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True, "joint_pos_noise": 0.018, "joint_vel_noise": 0.025},
    )
    reset_autonomous_episode = EventTerm(
        func=mdp.reset_autonomous_episode,
        mode="reset",
        params={
            "release_time_range_s": (0.02, 0.32),
            "delayed_release_time_range_s": (0.65, 2.20),
            "sender_x_range": (1.05, 1.85),
            "sender_y_range": (-0.34, 0.34),
            "sender_z_rel_range": (0.16, 0.52),
            "arrival_time_range_s": (0.42, 0.92),
            "object_size_scale_range": (0.80, 1.25),
            "object_mass_range": (1.0, 4.0),
            "object_friction_range": (0.55, 1.15),
            "object_restitution_range": (0.0, 0.12),
            "obs_noise_scale_range": (0.75, 1.45),
            "tag_available_prob": 0.94,
        },
    )

    # Keep delayed/no-toss objects held until release, then let them fly.
    advance_toss = EventTerm(
        func=mdp.advance_autonomous_toss,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"hold_jitter_std": 0.003},
    )

    # Sparse disturbances. This is deliberately important for robustness.
    random_push = EventTerm(
        func=mdp.random_push,
        mode="interval",
        interval_range_s=(0.35, 0.95),
        params={
            "robot_push_prob": 0.35,
            "object_push_prob": 0.30,
            "robot_lin_vel_xy_range": (-0.20, 0.20),
            "robot_ang_vel_z_range": (-0.35, 0.35),
            "object_lin_vel_range": (-0.30, 0.30),
            "object_ang_vel_range": (-1.10, 1.10),
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={"thresholds": (15_000, 40_000, 80_000, 140_000), "force_stage": None},
    )


@configclass
class dj_urop_v21_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v21_SceneCfg = dj_urop_v21_SceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # 50 Hz policy, 100 Hz physics. This matches the usual deploy/control loop contract.
        self.decimation = 2
        self.episode_length_s = 7.5
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation

        # Per-environment mass/material/scale randomization needs independent physics parsing.
        # If your Isaac Lab version complains, remove object_scale first and keep mass/material.
        self.scene.replicate_physics = False

        assert scene_objects_cfg.EXPECTED_ACTION_DIM == 29
        assert len(scene_objects_cfg.CONTROLLED_JOINT_NAMES) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(scene_objects_cfg.ACTION_SCALE) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(self.actions.policy.joint_names) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert POLICY_OBS_DIM == 100
        assert abs(self.decimation * self.sim.dt - 0.02) < 1e-9


@configclass
class dj_urop_v21_EnvCfg_Play(dj_urop_v21_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.episode_length_s = 7.5
        # Evaluation/demo: final curriculum distribution, deterministic observation, no random pushes.
        self.curriculum.stage_schedule.params["force_stage"] = 4
        self.events.reset_autonomous_episode.params["obs_noise_scale_range"] = (0.0, 0.0)
        self.events.random_push = None
        self.events.robot_actuator_gains = None