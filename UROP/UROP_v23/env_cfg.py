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


# v23 keeps only the most useful privileged contact channels to reduce sensor overhead.
CONTACT_SENSOR_NAMES = ["contact_torso", "contact_l_hand", "contact_r_hand"]
POLICY_OBS_DIM = 100


@configclass
class dj_urop_v23_SceneCfg(InteractiveSceneCfg):
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

    # Critic-only sparse contact sensors. No contact force is exposed to the actor.
    contact_torso = scene_objects_cfg.contact_torso_cfg
    contact_l_hand = scene_objects_cfg.contact_l_hand_cfg
    contact_r_hand = scene_objects_cfg.contact_r_hand_cfg


@configclass
class CommandsCfg:
    command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
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
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"noise_std": 0.010})
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity, params={"noise_std": 0.025})
        joint_pos_rel = ObsTerm(func=mdp.controlled_joint_pos_rel, params={"noise_std": 0.006})
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities, params={"scale": 0.05, "noise_std": 0.18})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        object_rel_pos = ObsTerm(func=mdp.object_rel_pos, params={"camera_frame": "opencv", "noise_std": 0.016})
        object_rel_lin_vel = ObsTerm(func=mdp.object_rel_lin_vel, params={"camera_frame": "opencv", "noise_std": 0.07})
        tag_visible = ObsTerm(func=mdp.tag_visible)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
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
    # Compact reward stack: fewer manager terms than v21/v22, with stronger anti-tremor shaping.
    alive = RewTerm(func=mdp.alive_bonus, weight=0.08)
    stand_balance = RewTerm(func=mdp.stand_balance_reward, weight=2.10)
    idle_ready = RewTerm(func=mdp.idle_ready_stand_reward, weight=1.15)
    early_arm = RewTerm(func=mdp.pre_catch_arm_motion_penalty, weight=-1.25)

    timing = RewTerm(func=mdp.catch_timing_reward, weight=0.95)
    hug_catch = RewTerm(func=mdp.hug_catch_reward, weight=2.25)
    object_stability = RewTerm(func=mdp.object_stability_reward, weight=2.60)
    successful_hold = RewTerm(func=mdp.successful_hold_reward, weight=3.60)
    sustained_hold = RewTerm(func=mdp.sustained_hold_reward, weight=1.35, params={"scale_steps": 45.0})

    drop_escape = RewTerm(func=mdp.drop_escape_penalty, weight=-4.20)
    smooth_action = RewTerm(func=mdp.smooth_action_penalty, weight=-0.20)
    post_catch_stillness = RewTerm(func=mdp.post_catch_stillness_penalty, weight=-0.42)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 40})
    robot_fell = DoneTerm(func=mdp.robot_fell, params={"min_root_z": 0.48, "max_tilt_xy": 0.74})
    object_dropped = DoneTerm(func=mdp.object_dropped, params={"drop_z": 0.22, "grace_steps_after_release": 8})
    object_escaped = DoneTerm(func=mdp.object_escaped, params={"max_dist": 2.55, "behind_x": -0.55})
    invalid_object = DoneTerm(func=mdp.invalid_object_state)


@configclass
class EventCfg:
    # Reset-time physics randomization only. v23 removes scale randomization to avoid extra overhead.
    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.65, 1.05),
            "dynamic_friction_range": (0.55, 0.95),
            "restitution_range": (0.0, 0.06),
            "num_buckets": 48,
            "make_consistent": True,
        },
    )
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.2, 3.2),
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
            "stiffness_distribution_params": (0.92, 1.08),
            "damping_distribution_params": (0.92, 1.12),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True, "joint_pos_noise": 0.012, "joint_vel_noise": 0.016},
    )
    reset_autonomous_episode = EventTerm(
        func=mdp.reset_autonomous_episode,
        mode="reset",
        params={
            "release_time_range_s": (0.18, 0.82),
            "delayed_release_time_range_s": (0.90, 2.30),
            "sender_x_range": (0.95, 1.45),
            "sender_y_range": (-0.18, 0.18),
            "sender_z_rel_range": (0.34, 0.58),
            "arrival_time_range_s": (0.34, 0.62),
            "object_size_scale_range": (0.88, 1.14),
            "object_mass_range": (1.2, 3.2),
            "object_friction_range": (0.65, 1.05),
            "object_restitution_range": (0.0, 0.06),
            "obs_noise_scale_range": (0.70, 1.35),
            "tag_available_prob": 0.90,
            "no_toss_tag_available_prob": 0.55,
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
        interval_range_s=(0.40, 0.90),
        params={
            "robot_push_prob": 0.40,
            "object_push_prob": 0.18,
            "robot_lin_vel_xy_range": (-0.16, 0.16),
            "robot_ang_vel_z_range": (-0.26, 0.26),
            "object_lin_vel_range": (-0.18, 0.18),
            "object_ang_vel_range": (-0.65, 0.65),
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={"thresholds": (12_000, 32_000, 65_000, 110_000), "force_stage": None},
    )


@configclass
class dj_urop_v23_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v23_SceneCfg = dj_urop_v23_SceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.8
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation
        # Per-env mass/material randomization may need independent physics parsing.
        self.scene.replicate_physics = False

        assert scene_objects_cfg.EXPECTED_ACTION_DIM == 29
        assert scene_objects_cfg.EXPECTED_POLICY_OBS_DIM == 100
        assert len(scene_objects_cfg.CONTROLLED_JOINT_NAMES) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(scene_objects_cfg.ACTION_SCALE) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(self.actions.policy.joint_names) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert POLICY_OBS_DIM == 100
        assert abs(self.decimation * self.sim.dt - 0.02) < 1e-9


@configclass
class dj_urop_v23_EnvCfg_Play(dj_urop_v23_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.curriculum.stage_schedule.params["force_stage"] = 2
        self.events.reset_autonomous_episode.params["obs_noise_scale_range"] = (0.0, 0.0)
        self.events.reset_autonomous_episode.params["stage_episode_probabilities"] = (
            (0.78, 0.08, 0.08, 0.04, 0.02),
            (0.78, 0.08, 0.08, 0.04, 0.02),
            (0.78, 0.08, 0.08, 0.04, 0.02),
            (0.78, 0.08, 0.08, 0.04, 0.02),
            (0.78, 0.08, 0.08, 0.04, 0.02),
        )
        self.events.random_push = None
        self.events.robot_actuator_gains = None


@configclass
class dj_urop_v23_EnvCfg_Demo(dj_urop_v23_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.curriculum.stage_schedule.params["force_stage"] = 0
        self.events.object_material = None
        self.events.object_mass = None
        self.events.robot_actuator_gains = None
        self.events.random_push = None
        self.events.reset_autonomous_episode.params.update(
            {
                "stage_episode_probabilities": (
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 0.0),
                ),
                "release_time_range_s": (0.28, 0.42),
                "delayed_release_time_range_s": (0.28, 0.42),
                "sender_x_range": (1.05, 1.25),
                "sender_y_range": (-0.04, 0.04),
                "sender_z_rel_range": (0.40, 0.50),
                "arrival_time_range_s": (0.40, 0.52),
                "object_size_scale_range": (0.98, 1.02),
                "object_mass_range": (1.8, 2.2),
                "object_friction_range": (0.75, 0.90),
                "object_restitution_range": (0.00, 0.03),
                "obs_noise_scale_range": (0.0, 0.0),
                "tag_available_prob": 1.0,
                "no_toss_tag_available_prob": 1.0,
            }
        )
