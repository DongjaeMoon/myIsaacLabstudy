import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
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


@configclass
class dj_urop_v19_SceneCfg(InteractiveSceneCfg):
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
        # Keep the v15 deploy contract exactly: total 104 dims, same order.
        projected_gravity = ObsTerm(func=mdp.projected_gravity)  # 3
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity)  # 3
        joint_pos_rel = ObsTerm(func=mdp.controlled_joint_pos_rel)  # 29
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities, params={"scale": 0.05})  # 29
        prev_actions = ObsTerm(func=mdp.prev_actions)  # 29
        object_rel_pos = ObsTerm(func=mdp.object_rel_pos)  # 3
        object_rel_lin_vel = ObsTerm(func=mdp.object_rel_lin_vel)  # 3
        tag_visible = ObsTerm(func=mdp.tag_visible)  # 1
        # v19 semantic change only: [unknown/no tag, light, medium, heavy] mass prior.
        mode_one_hot = ObsTerm(func=mdp.mode_one_hot)  # 4

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class CriticCfg(ObsGroup):
        phase = ObsTerm(func=mdp.toss_state)
        handover_task = ObsTerm(func=mdp.handover_task_state)
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
    alive = RewTerm(func=mdp.alive_bonus, weight=0.25)
    upright = RewTerm(func=mdp.upright_reward, weight=1.80)
    height = RewTerm(func=mdp.root_height_reward, weight=1.05, params={"target_z": 0.78, "sigma": 0.10})

    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.13, params={"w_lin": 1.0, "w_ang": 0.30})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.025)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00002)
    action_mag = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.014)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.10)
    action_accel = RewTerm(func=mdp.action_acceleration_penalty, weight=-0.055)
    lower_body_action_rate = RewTerm(func=mdp.lower_body_action_rate_penalty, weight=-0.065)
    foot_slip = RewTerm(func=mdp.foot_slip_penalty, weight=-0.10, params={"ground_height_thr": 0.16})

    # Waiting/ignore rewards are intentionally strong: visible tag alone must not trigger hugging.
    wait_ready_pose = RewTerm(func=mdp.ready_pose_when_waiting, weight=3.0, params={"sigma": 0.22})
    wait_joint_still = RewTerm(func=mdp.waiting_joint_stillness_reward, weight=0.9, params={"sigma": 1.3})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-1.7, params={"sigma": 0.14})
    wait_yaw_drift = RewTerm(func=mdp.wait_yaw_drift_penalty, weight=-0.95, params={"sigma": 0.20})
    visible_far_ready = RewTerm(func=mdp.visible_far_ready_reward, weight=2.4, params={"sigma": 0.20, "min_x": 0.48})
    premature_receive = RewTerm(
        func=mdp.premature_receive_penalty,
        weight=-2.2,
        params={"min_x": 0.48, "action_weight": 0.25},
    )
    lower_body_ready = RewTerm(
        func=mdp.lower_body_ready_reward,
        weight=1.0,
        params={"sigma_wait": 0.16, "sigma_active": 0.32},
    )

    # Receive/hold rewards activate only for real handover-release episodes near commit/release.
    catch_region = RewTerm(func=mdp.catch_target_region_reward, weight=1.6, params={"sigma": 0.28})
    upper_body_receive = RewTerm(func=mdp.upper_body_receive_reward, weight=1.25, params={"sigma": 0.27})
    catch_vel_match = RewTerm(
        func=mdp.catch_velocity_match_reward,
        weight=0.9,
        params={"torso_body_name": "torso_link", "sigma": 0.70},
    )
    mass_brace = RewTerm(func=mdp.mass_conditioned_receive_pose_reward, weight=1.7, params={"sigma": 0.36})
    heavy_stability = RewTerm(func=mdp.heavy_object_stability_reward, weight=1.0, params={"lin_sigma": 0.16, "ang_sigma": 0.38})
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=2.2,
        params={
            "sensor_names_left": [
                "contact_l_shoulder_yaw",
                "contact_l_elbow",
                "contact_l_wrist_roll",
                "contact_l_wrist_pitch",
                "contact_l_wrist_yaw",
            ],
            "sensor_names_right": [
                "contact_r_shoulder_yaw",
                "contact_r_elbow",
                "contact_r_wrist_roll",
                "contact_r_wrist_pitch",
                "contact_r_wrist_yaw",
            ],
            "sensor_name_torso": "contact_torso",
            "thr": 1.5,
        },
    )
    impact = RewTerm(func=mdp.impact_peak_penalty, weight=-0.0045, params={"sensor_names": CONTACT_SENSOR_NAMES, "force_thr": 230.0})

    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=1.8, params={"torso_body_name": "torso_link", "sigma": 0.45})
    hold_pose = RewTerm(func=mdp.hold_pose_reward, weight=2.3, params={"sigma": 0.18})
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=1.0)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=2.8, params={"min_steps": 22})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.25, params={"min_z": 0.42, "max_dist": 1.8})

    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=1.55, params={"lin_sigma": 0.10, "yaw_sigma": 0.30})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-1.45, params={"sigma": 0.10})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 44})
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.50, "max_tilt_deg": 45.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.30, "max_dist": 2.0})
    runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.30})
    unsafe_lower_body = DoneTerm(func=mdp.unsafe_lower_body_deviation, params={"max_abs_dev": 0.85})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_autonomous_episode = EventTerm(
        func=mdp.reset_handover_episode,
        mode="reset",
        params={
            "park": {"pos_x": (1.55, 1.85), "pos_y": (-0.15, 0.15), "pos_z": (-0.62, -0.52)},
            # Deliberately keep many visible-but-not-receive episodes throughout training.
            "task_probability_by_stage": {
                "stage0": {"hidden": 0.35, "visible_static": 0.65, "approach_no_release": 0.00, "handover_release": 0.00},
                "stage1": {"hidden": 0.18, "visible_static": 0.36, "approach_no_release": 0.22, "handover_release": 0.24},
                "stage2": {"hidden": 0.12, "visible_static": 0.28, "approach_no_release": 0.22, "handover_release": 0.38},
                "stage3": {"hidden": 0.08, "visible_static": 0.22, "approach_no_release": 0.20, "handover_release": 0.50},
            },
            "joint_noise": {
                "lower_body_pos": (-0.030, 0.030),
                "waist_pos": (-0.025, 0.025),
                "arm_pos": (-0.060, 0.060),
                "wrist_pos": (-0.040, 0.040),
                "velocity": (-0.10, 0.10),
            },
            "root_xy_range": (-0.03, 0.03),
            "root_yaw_range": (-0.08, 0.08),
            "object_randomization": {
                "mass_class_ranges": {
                    "light": (1.0, 2.4),
                    "medium": (2.4, 4.5),
                    "heavy": (4.5, 8.0),
                },
                "mass_class_prob_by_stage": {
                    "stage0": {"light": 0.45, "medium": 0.40, "heavy": 0.15},
                    "stage1": {"light": 0.44, "medium": 0.42, "heavy": 0.14},
                    "stage2": {"light": 0.34, "medium": 0.42, "heavy": 0.24},
                    "stage3": {"light": 0.28, "medium": 0.40, "heavy": 0.32},
                },
                "friction_range": (0.25, 1.60),
                "restitution_range": (0.00, 0.16),
                "size_scale_range": (0.75, 1.35),
                "prior_unknown_prob": 0.03,
                "prior_mismatch_prob": 0.04,
                "apply_physx": True,
            },
            "robot_material_randomization": {
                "friction_range": (0.40, 1.35),
                "restitution_range": (0.00, 0.04),
                "apply_physx": True,
            },
            "floor_material_randomization": {
                "friction_range": (0.35, 1.45),
            },
            "observation_randomization": {
                "projected_gravity_noise_std_range": (0.005, 0.030),
                "base_ang_vel_noise_std_range": (0.01, 0.08),
                "joint_pos_noise_std_range": (0.002, 0.020),
                "joint_vel_noise_std_range": (0.02, 0.20),
                "obj_pos_noise_range": (0.005, 0.070),
                "obj_vel_noise_range": (0.03, 0.40),
                "drop_prob_range": (0.00, 0.30),
                "hold_prob_range": (0.00, 0.18),
                "alpha_range": (0.18, 0.92),
                "depth_scale_range": (0.88, 1.12),
                "lateral_scale_range": (0.86, 1.14),
                "height_scale_range": (0.90, 1.10),
                "pos_bias_range": (-0.040, 0.040),
                "vel_scale_range": (0.70, 1.30),
            },
            "handover_trajectory_by_stage": {
                "stage1": {
                    "start_x": (0.72, 1.05), "start_y": (-0.18, 0.18), "start_z": (0.16, 0.36),
                    "goal_x": (0.16, 0.28), "goal_y": (-0.07, 0.07), "goal_z": (0.08, 0.24),
                    "stop_x": (0.52, 0.78), "stop_y": (-0.15, 0.15), "stop_z": (0.10, 0.32),
                    "static_x": (0.70, 1.20), "static_y": (-0.30, 0.30), "static_z": (0.08, 0.40),
                    "start_wait_s": {"stage1": (0.40, 1.40)},
                    "move_duration_s": (1.20, 2.60),
                    "pre_release_pause_s": (0.15, 0.55),
                    "commit_lead_s": (0.25, 0.50),
                    "release_vel_x": (-0.08, 0.03), "release_vel_y": (-0.04, 0.04), "release_vel_z": (-0.08, 0.06),
                },
                "stage2": {
                    "start_x": (0.72, 1.25), "start_y": (-0.28, 0.28), "start_z": (0.10, 0.44),
                    "goal_x": (0.12, 0.30), "goal_y": (-0.11, 0.11), "goal_z": (0.04, 0.30),
                    "stop_x": (0.48, 0.85), "stop_y": (-0.23, 0.23), "stop_z": (0.06, 0.36),
                    "static_x": (0.62, 1.35), "static_y": (-0.40, 0.40), "static_z": (0.04, 0.46),
                    "start_wait_s": {"stage2": (0.30, 1.70)},
                    "move_duration_s": (1.00, 2.80),
                    "pre_release_pause_s": (0.08, 0.60),
                    "commit_lead_s": (0.25, 0.58),
                    "release_vel_x": (-0.12, 0.04), "release_vel_y": (-0.06, 0.06), "release_vel_z": (-0.12, 0.08),
                },
                "stage3": {
                    "start_x": (0.65, 1.45), "start_y": (-0.40, 0.40), "start_z": (0.02, 0.52),
                    "goal_x": (0.08, 0.32), "goal_y": (-0.16, 0.16), "goal_z": (0.00, 0.34),
                    "stop_x": (0.48, 0.95), "stop_y": (-0.30, 0.30), "stop_z": (0.02, 0.42),
                    "static_x": (0.58, 1.55), "static_y": (-0.48, 0.48), "static_z": (0.00, 0.55),
                    "start_wait_s": {"stage3": (0.20, 2.00)},
                    "move_duration_s": (0.85, 3.20),
                    "pre_release_pause_s": (0.05, 0.70),
                    "commit_lead_s": (0.22, 0.65),
                    "release_vel_x": (-0.16, 0.06), "release_vel_y": (-0.10, 0.10), "release_vel_z": (-0.16, 0.10),
                },
            },
        },
    )

    random_push = EventTerm(
        func=mdp.push_robot_root_velocity,
        mode="interval",
        interval_range_s=(0.7, 2.0),
        params={
            "stage0_xy_range": (-0.12, 0.12),
            "stage1_xy_range": (-0.20, 0.20),
            "stage2_xy_range": (-0.32, 0.32),
            "stage3_xy_range": (-0.42, 0.42),
            "stage0_yaw_range": (-0.08, 0.08),
            "stage1_yaw_range": (-0.15, 0.15),
            "stage2_yaw_range": (-0.24, 0.24),
            "stage3_yaw_range": (-0.32, 0.32),
            "z_velocity_range": (-0.02, 0.02),
            "hold_xy_scale": 0.65,
            "hold_yaw_scale": 0.60,
            "max_xy_speed": 1.30,
            "max_yaw_speed": 0.95,
        },
    )

    handover = EventTerm(
        func=mdp.handover_object_curriculum,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "commit_x": 0.42,
            "debug_print": False,
            "debug_print_rate_s": 0.75,
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 500,
            "stage1_iters": 1000,
            "stage2_iters": 1500,
            "num_steps_per_env": 64,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v19_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v19_SceneCfg = dj_urop_v19_SceneCfg(num_envs=128, env_spacing=3.0)
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

        assert scene_objects_cfg.EXPECTED_ACTION_DIM == 29
        assert len(scene_objects_cfg.CONTROLLED_JOINT_NAMES) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(scene_objects_cfg.ACTION_SCALE) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert len(self.actions.policy.joint_names) == scene_objects_cfg.EXPECTED_ACTION_DIM
        assert scene_objects_cfg.EXPECTED_POLICY_OBS_DIM == 104
        assert abs(self.decimation * self.sim.dt - 0.02) < 1e-9

        try:
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
                self.sim.physx.enable_external_forces_every_iteration = True
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "num_velocity_iterations"):
                self.sim.physx.num_velocity_iterations = 1
        except Exception:
            pass


@configclass
class dj_urop_v19_EnvCfg_Play(dj_urop_v19_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 6
        self.scene.env_spacing = 3.2
        self.episode_length_s = 8.0
        self.curriculum.stage_schedule.params["eval_stage"] = 3
        self.events.random_push.interval_range_s = (9999.0, 9999.0)
        self._set_mild_play_randomization()

    def _set_mild_play_randomization(self):
        self.events.reset_autonomous_episode.params["task_probability_by_stage"] = {
            "stage3": {"hidden": 0.10, "visible_static": 0.30, "approach_no_release": 0.25, "handover_release": 0.35},
        }
        self.events.reset_autonomous_episode.params["object_randomization"] = {
            "mass_class_ranges": {"light": (1.2, 2.2), "medium": (2.6, 4.0), "heavy": (4.8, 6.8)},
            "mass_class_prob_by_stage": {"stage3": {"light": 0.30, "medium": 0.40, "heavy": 0.30}},
            "friction_range": (0.55, 1.15),
            "restitution_range": (0.00, 0.06),
            "size_scale_range": (0.90, 1.15),
            "prior_unknown_prob": 0.00,
            "prior_mismatch_prob": 0.00,
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["robot_material_randomization"] = {
            "friction_range": (0.70, 1.10),
            "restitution_range": (0.00, 0.02),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["floor_material_randomization"] = {"friction_range": (0.75, 1.15)}
        self.events.reset_autonomous_episode.params["observation_randomization"] = {
            "projected_gravity_noise_std_range": (0.0, 0.006),
            "base_ang_vel_noise_std_range": (0.0, 0.012),
            "joint_pos_noise_std_range": (0.0, 0.004),
            "joint_vel_noise_std_range": (0.0, 0.035),
            "obj_pos_noise_range": (0.0, 0.018),
            "obj_vel_noise_range": (0.0, 0.080),
            "drop_prob_range": (0.0, 0.04),
            "hold_prob_range": (0.0, 0.04),
            "alpha_range": (0.65, 1.00),
            "depth_scale_range": (0.97, 1.03),
            "lateral_scale_range": (0.96, 1.04),
            "height_scale_range": (0.97, 1.03),
            "pos_bias_range": (-0.010, 0.010),
            "vel_scale_range": (0.92, 1.08),
        }
        self.events.handover.params["debug_print"] = False


@configclass
class dj_urop_v19_EnvCfg_Debug(dj_urop_v19_EnvCfg_Play):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.events.handover.params["debug_print"] = True
        self.events.handover.params["debug_print_rate_s"] = 0.50
        # Make the GUI visibly include all event types, not just successful handovers.
        self.events.reset_autonomous_episode.params["task_probability_by_stage"] = {
            "stage3": {"hidden": 0.125, "visible_static": 0.375, "approach_no_release": 0.250, "handover_release": 0.250},
        }
