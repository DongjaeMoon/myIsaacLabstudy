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
class dj_urop_v18_SceneCfg(InteractiveSceneCfg):
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
        # 3
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # 3
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity)
        # 29
        joint_pos_rel = ObsTerm(func=mdp.controlled_joint_pos_rel)
        # 29, scaled to match deploy YAML observation scale.
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities, params={"scale": 0.05})
        # 29
        prev_actions = ObsTerm(func=mdp.prev_actions)
        # 3
        object_rel_pos = ObsTerm(func=mdp.object_rel_pos)
        # 3
        object_rel_lin_vel = ObsTerm(func=mdp.object_rel_lin_vel)
        # 1
        tag_visible = ObsTerm(func=mdp.tag_visible)
        # 4. Intentionally kept clean because it is a semantic phase signal.
        mode_one_hot = ObsTerm(func=mdp.mode_one_hot)

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
    alive = RewTerm(func=mdp.alive_bonus, weight=0.25)
    upright = RewTerm(func=mdp.upright_reward, weight=1.75)
    height = RewTerm(func=mdp.root_height_reward, weight=1.10, params={"target_z": 0.78, "sigma": 0.10})

    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.055, params={"w_lin": 1.0, "w_ang": 0.25})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.025)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00002)
    action_mag = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.015)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.095)
    action_accel = RewTerm(func=mdp.action_acceleration_penalty, weight=-0.050)
    lower_body_action_rate = RewTerm(func=mdp.lower_body_action_rate_penalty, weight=-0.055)
    foot_slip = RewTerm(func=mdp.foot_slip_penalty, weight=-0.10, params={"ground_height_thr": 0.16})

    wait_ready_pose = RewTerm(func=mdp.ready_pose_when_waiting, weight=2.4, params={"sigma": 0.24})
    wait_joint_still = RewTerm(func=mdp.waiting_joint_stillness_reward, weight=0.9, params={"sigma": 1.5})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-0.85, params={"sigma": 0.22})
    wait_yaw_drift = RewTerm(func=mdp.wait_yaw_drift_penalty, weight=-0.55, params={"sigma": 0.28})
    lower_body_ready = RewTerm(
        func=mdp.lower_body_ready_reward,
        weight=1.8,
        params={"sigma_wait": 0.18, "sigma_active": 0.30},
    )

    # v18 anti-overhug shaping: seeing a tag is not enough to hug.
    visual_wait_patience = RewTerm(func=mdp.visual_wait_patience_reward, weight=1.15, params={"action_sigma": 0.55})
    premature_hug = RewTerm(func=mdp.premature_hug_penalty, weight=-0.50, params={"action_w": 1.0, "pose_w": 0.25})
    lateral_intercept = RewTerm(func=mdp.lateral_intercept_reward, weight=0.45, params={"deadband": 0.12, "speed_sigma": 0.42})
    incoming_receive_pose = RewTerm(func=mdp.incoming_receive_pose_reward, weight=2.6, params={"sigma": 0.52, "hold_blend": 0.70})
    head_region = RewTerm(func=mdp.head_region_penalty, weight=-0.65)

    catch_region = RewTerm(func=mdp.catch_target_region_reward, weight=2.8, params={"sigma": 0.44})
    upper_body_receive = RewTerm(func=mdp.upper_body_receive_reward, weight=2.6, params={"sigma": 0.40})
    catch_vel_match = RewTerm(
        func=mdp.catch_velocity_match_reward,
        weight=1.7,
        params={"torso_body_name": "torso_link", "sigma": 1.05},
    )
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=3.6,
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
    impact = RewTerm(func=mdp.impact_peak_penalty, weight=-0.005, params={"sensor_names": CONTACT_SENSOR_NAMES, "force_thr": 235.0})

    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=2.4, params={"torso_body_name": "torso_link", "sigma": 0.64})
    hold_pose = RewTerm(func=mdp.hold_pose_reward, weight=2.8, params={"sigma": 0.30})
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=1.0)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=3.8, params={"min_steps": 16})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.4, params={"min_z": 0.34, "max_dist": 2.1})

    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=1.5, params={"lin_sigma": 0.10, "yaw_sigma": 0.30})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-0.55, params={"sigma": 0.18})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 40})
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.50, "max_tilt_deg": 45.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.18, "max_dist": 2.8, "min_active_steps": 10})
    runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.45})
    unsafe_lower_body = DoneTerm(func=mdp.unsafe_lower_body_deviation, params={"max_abs_dev": 0.98})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_autonomous_episode = EventTerm(
        func=mdp.reset_autonomous_episode,
        mode="reset",
        params={
            "park": {"pos_x": (1.70, 2.15), "pos_y": (-0.80, 0.80), "pos_z": (-0.72, -0.55)},
            "wait_time_ranges": {
                "stage1": (0.80, 2.40),
                "stage2": (0.45, 3.20),
                "stage3": (0.25, 4.00),
            },
            "toss_probability_by_stage": {
                "stage0": 0.0,
                "stage1": 0.72,
                "stage2": 0.78,
                "stage3": 0.82,
            },
            "joint_noise": {
                "lower_body_pos": (-0.045, 0.045),
                "waist_pos": (-0.040, 0.040),
                "arm_pos": (-0.090, 0.090),
                "wrist_pos": (-0.070, 0.070),
                "velocity": (-0.18, 0.18),
            },
            "root_xy_range": (-0.03, 0.03),
            "root_yaw_range": (-0.08, 0.08),
            "object_randomization": {
                "mass_range": (1.0, 6.5),
                "friction_range": (0.35, 1.55),
                "restitution_range": (0.00, 0.16),
                "size_scale_range": (0.75, 1.30),
                "apply_physx": True,
            },
            "robot_material_randomization": {
                "friction_range": (0.40, 1.30),
                "restitution_range": (0.00, 0.04),
                "apply_physx": True,
            },
            "floor_material_randomization": {
                "friction_range": (0.35, 1.40),
            },
            "observation_randomization": {
                "projected_gravity_noise_std_range": (0.006, 0.035),
                "base_ang_vel_noise_std_range": (0.015, 0.095),
                "joint_pos_noise_std_range": (0.003, 0.024),
                "joint_vel_noise_std_range": (0.030, 0.220),
                "prev_action_noise_std_range": (0.002, 0.016),
                "mode_noise_std_range": (0.003, 0.024),
                "obj_pos_noise_range": (0.006, 0.065),
                "obj_vel_noise_range": (0.035, 0.360),
                "obj_pos_bias_range": (-0.028, 0.028),
                "obj_vel_bias_range": (-0.075, 0.075),
                "obj_pos_scale_range": (0.91, 1.09),
                "obj_vel_scale_range": (0.82, 1.18),
                "drop_prob_range": (0.01, 0.24),
                "false_positive_prob_range": (0.0, 0.010),
                "tag_visible_noise_std_range": (0.006, 0.040),
                "alpha_range": (0.25, 0.90),
                "latency_steps_range": (0, 4),
                "noise_spike_prob_range": (0.00, 0.045),
                "noise_spike_scale_range": (2.0, 5.5),
            },
            "visibility_randomization": {
                "pre_toss_visible_probability_by_stage": {
                    "stage0": 0.0,
                    "stage1": 0.70,
                    "stage2": 0.84,
                    "stage3": 0.92,
                },
                "idle_visible_probability_by_stage": {
                    "stage0": 0.92,
                    "stage1": 0.42,
                    "stage2": 0.34,
                    "stage3": 0.28,
                },
                "visibility_start_s": (0.0, 0.55),
                "idle_visible_pose": {
                    "pos_x": (0.75, 1.85),
                    "pos_y": (-0.65, 0.65),
                    "pos_z": (-0.26, 0.38),
                    "vel_x": (-0.035, 0.070),
                    "vel_y": (-0.070, 0.070),
                    "vel_z": (-0.030, 0.030),
                },
            },
            "delivery_randomization": {
                # Hidden generator mixture. Actor never observes this; it only sees rel_pos/rel_vel/tag_visible.
                "toss_family_prob_by_stage": {
                    "stage1": {"ballistic": 0.46, "carried_release": 0.28, "push_release": 0.18, "late_visible": 0.08},
                    "stage2": {"ballistic": 0.34, "carried_release": 0.30, "push_release": 0.20, "lob": 0.08, "late_visible": 0.08},
                    "stage3": {"ballistic": 0.26, "carried_release": 0.30, "push_release": 0.22, "lob": 0.12, "late_visible": 0.10},
                },
                "no_toss_family_prob_by_stage": {
                    "stage0": {"idle": 0.55, "pass_by": 0.25, "approach_abort": 0.20},
                    "stage1": {"idle": 0.55, "pass_by": 0.25, "approach_abort": 0.20},
                    "stage2": {"idle": 0.42, "pass_by": 0.34, "approach_abort": 0.24},
                    "stage3": {"idle": 0.34, "pass_by": 0.38, "approach_abort": 0.28},
                },
                "pre_release_start_probability_by_stage": {
                    "stage0": 0.0,
                    "stage1": 0.42,
                    "stage2": 0.64,
                    "stage3": 0.78,
                },
                "pre_release_start_pose": {
                    "pos_x": (0.85, 2.10),
                    "pos_y": (-0.75, 0.75),
                    "pos_z": (-0.30, 0.48),
                },
                "abort_visible_pose": {
                    "pos_x": (0.70, 1.75),
                    "pos_y": (-0.65, 0.65),
                    "pos_z": (-0.25, 0.40),
                    "vel_x": (-0.06, 0.09),
                    "vel_y": (-0.08, 0.08),
                    "vel_z": (-0.03, 0.03),
                },
                "pass_by_visible_pose": {
                    "pos_x": (0.55, 1.55),
                    "pos_y": (-0.95, 0.95),
                    "pos_z": (-0.22, 0.42),
                    "vel_x": (-0.06, 0.08),
                    "vel_y": (-0.65, 0.65),
                    "vel_z": (-0.04, 0.04),
                },
            },
        },
    )

    hold_visible_object = EventTerm(
        func=mdp.update_visible_object_before_toss,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"max_hold_speed": 0.85, "jitter_amp": 0.010},
    )

    random_push = EventTerm(
        func=mdp.push_robot_root_velocity,
        mode="interval",
        interval_range_s=(1.4, 3.5),
        params={
            "stage0_xy_range": (-0.05, 0.05),
            "stage1_xy_range": (-0.10, 0.10),
            "stage2_xy_range": (-0.18, 0.18),
            "stage3_xy_range": (-0.28, 0.28),
            "stage0_yaw_range": (-0.04, 0.04),
            "stage1_yaw_range": (-0.08, 0.08),
            "stage2_yaw_range": (-0.13, 0.13),
            "stage3_yaw_range": (-0.22, 0.22),
            "z_velocity_range": (-0.02, 0.02),
            "hold_xy_scale": 0.70,
            "hold_yaw_scale": 0.65,
            "max_xy_speed": 0.95,
            "max_yaw_speed": 0.75,
        },
    )

    toss = EventTerm(
        func=mdp.toss_object_relative_curriculum,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "max_throws_per_episode": 1,
            "throw_prob_stage1": 1.0,
            "throw_prob_stage2": 1.0,
            "throw_prob_stage3": 1.0,
            "stage1": {
                # Base family: easy but not deterministic ballistic toss.
                "sampler": "target_ballistic",
                "spawn_x": (0.38, 0.68),
                "spawn_y": (-0.14, 0.14),
                "spawn_z": (0.10, 0.38),
                "target_x": (0.08, 0.26),
                "target_y": (-0.11, 0.11),
                "target_z": (0.06, 0.30),
                "flight_time": (0.34, 0.62),
                "max_speed": 1.55,
                "max_vy_abs": 0.42,
                "max_vz_abs": 1.80,
                "roll": (-0.03, 0.03), "pitch": (-0.04, 0.04), "yaw": (-0.10, 0.10),
                "ang_vel_x": (-0.08, 0.08), "ang_vel_y": (-0.08, 0.08), "ang_vel_z": (-0.16, 0.16),
                "carried_release": {
                    "sampler": "independent",
                    "spawn_x": (0.18, 0.52), "spawn_y": (-0.18, 0.18), "spawn_z": (0.06, 0.32),
                    "vel_x": (-0.65, -0.12), "vel_y": (-0.18, 0.18), "vel_z": (-0.16, 0.18),
                    "max_speed": 1.10,
                    "roll": (-0.03, 0.03), "pitch": (-0.04, 0.04), "yaw": (-0.10, 0.10),
                    "ang_vel_x": (-0.06, 0.06), "ang_vel_y": (-0.06, 0.06), "ang_vel_z": (-0.12, 0.12),
                },
                "push_release": {
                    "sampler": "independent",
                    "spawn_x": (0.22, 0.62), "spawn_y": (-0.22, 0.22), "spawn_z": (0.02, 0.34),
                    "vel_x": (-1.05, -0.28), "vel_y": (-0.32, 0.32), "vel_z": (-0.20, 0.22),
                    "max_speed": 1.35,
                },
                "late_visible_toss": {
                    "sampler": "target_ballistic",
                    "spawn_x": (0.36, 0.70), "spawn_y": (-0.16, 0.16), "spawn_z": (0.08, 0.38),
                    "target_x": (0.08, 0.26), "target_y": (-0.12, 0.12), "target_z": (0.06, 0.30),
                    "flight_time": (0.32, 0.58), "max_speed": 1.65, "max_vy_abs": 0.45, "max_vz_abs": 1.85,
                },
            },
            "stage2": {
                "sampler": "target_ballistic",
                "spawn_x": (0.34, 0.92),
                "spawn_y": (-0.38, 0.38),
                "spawn_z": (0.02, 0.50),
                "target_x": (0.02, 0.34),
                "target_y": (-0.26, 0.26),
                "target_z": (0.00, 0.40),
                "flight_time": (0.30, 0.72),
                "max_speed": 2.20,
                "max_vy_abs": 0.95,
                "max_vz_abs": 2.25,
                "roll": (-0.04, 0.04), "pitch": (-0.05, 0.05), "yaw": (-0.24, 0.24),
                "ang_vel_x": (-0.16, 0.16), "ang_vel_y": (-0.16, 0.16), "ang_vel_z": (-0.34, 0.34),
                "carried_release": {
                    "sampler": "independent",
                    "spawn_x": (0.16, 0.62), "spawn_y": (-0.34, 0.34), "spawn_z": (-0.04, 0.42),
                    "vel_x": (-0.85, -0.08), "vel_y": (-0.32, 0.32), "vel_z": (-0.24, 0.28),
                    "max_speed": 1.35,
                },
                "push_release": {
                    "sampler": "independent",
                    "spawn_x": (0.18, 0.70), "spawn_y": (-0.40, 0.40), "spawn_z": (-0.08, 0.44),
                    "vel_x": (-1.35, -0.20), "vel_y": (-0.55, 0.55), "vel_z": (-0.30, 0.34),
                    "max_speed": 1.85,
                },
                "lob": {
                    "sampler": "target_ballistic",
                    "spawn_x": (0.42, 1.05), "spawn_y": (-0.34, 0.34), "spawn_z": (0.22, 0.64),
                    "target_x": (0.02, 0.34), "target_y": (-0.24, 0.24), "target_z": (0.04, 0.42),
                    "flight_time": (0.42, 0.86), "max_speed": 2.20, "max_vy_abs": 0.90, "max_vz_abs": 2.55,
                },
                "late_visible_toss": {
                    "sampler": "target_ballistic",
                    "spawn_x": (0.32, 0.85), "spawn_y": (-0.36, 0.36), "spawn_z": (0.00, 0.50),
                    "target_x": (0.02, 0.34), "target_y": (-0.26, 0.26), "target_z": (0.00, 0.40),
                    "flight_time": (0.28, 0.68), "max_speed": 2.25, "max_vy_abs": 0.95, "max_vz_abs": 2.30,
                },
            },
            "stage3": {
                "sampler": "target_ballistic",
                "spawn_x": (0.28, 1.35),
                "spawn_y": (-0.72, 0.72),
                "spawn_z": (-0.18, 0.68),
                "target_x": (-0.10, 0.46),
                "target_y": (-0.52, 0.52),
                "target_z": (-0.14, 0.54),
                "flight_time": (0.24, 0.95),
                "max_speed": 2.95,
                "max_vy_abs": 1.50,
                "max_vz_abs": 2.90,
                "roll": (-0.06, 0.06), "pitch": (-0.07, 0.07), "yaw": (-0.65, 0.65),
                "ang_vel_x": (-0.34, 0.34), "ang_vel_y": (-0.34, 0.34), "ang_vel_z": (-0.80, 0.80),
                "carried_release": {
                    "sampler": "independent",
                    "spawn_x": (0.12, 0.78), "spawn_y": (-0.55, 0.55), "spawn_z": (-0.14, 0.54),
                    "vel_x": (-1.05, -0.03), "vel_y": (-0.52, 0.52), "vel_z": (-0.35, 0.38),
                    "max_speed": 1.70,
                },
                "push_release": {
                    "sampler": "independent",
                    "spawn_x": (0.14, 0.82), "spawn_y": (-0.62, 0.62), "spawn_z": (-0.16, 0.58),
                    "vel_x": (-1.75, -0.16), "vel_y": (-0.85, 0.85), "vel_z": (-0.45, 0.48),
                    "max_speed": 2.35,
                },
                "lob": {
                    "sampler": "target_ballistic",
                    "spawn_x": (0.42, 1.45), "spawn_y": (-0.65, 0.65), "spawn_z": (0.18, 0.78),
                    "target_x": (-0.08, 0.44), "target_y": (-0.48, 0.48), "target_z": (-0.08, 0.56),
                    "flight_time": (0.42, 1.10), "max_speed": 3.00, "max_vy_abs": 1.40, "max_vz_abs": 3.10,
                },
                "late_visible_toss": {
                    "sampler": "target_ballistic",
                    "spawn_x": (0.28, 1.20), "spawn_y": (-0.68, 0.68), "spawn_z": (-0.16, 0.66),
                    "target_x": (-0.10, 0.44), "target_y": (-0.50, 0.50), "target_z": (-0.12, 0.52),
                    "flight_time": (0.22, 0.86), "max_speed": 3.00, "max_vy_abs": 1.45, "max_vz_abs": 2.95,
                },
            },
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 300,
            "stage1_iters": 1400,
            "stage2_iters": 2200,
            "num_steps_per_env": 64,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v18_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v18_SceneCfg = dj_urop_v18_SceneCfg(num_envs=128, env_spacing=3.0)
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
class dj_urop_v18_EnvCfg_Play(dj_urop_v18_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 3.2
        self.episode_length_s = 7.0

        # Play uses fixed stage-3 evaluation.
        self.curriculum.stage_schedule.params["eval_stage"] = 3

        # ============================================================
        # [PLAY-EASY OVERRIDE]
        # Mild visualization/evaluation setting.
        # Training remains hard; only Play is made easier and cleaner.
        # ============================================================

        # 1) Disable random external push during Play.
        # Keep the term configured, but make it never trigger within a 7 s episode.
        self.events.random_push.interval_range_s = (9999.0, 9999.0)

        # 2) Use reasonable random timing.
        self.events.reset_autonomous_episode.params["wait_time_ranges"] = {
            "stage1": (1.20, 2.00),
            "stage2": (1.00, 2.50),
            "stage3": (0.80, 2.00),
        }

        # 3) For visualization, make toss happen every episode.
        # This is only for Play. Training still keeps toss/no-toss mixture.
        self.events.reset_autonomous_episode.params["toss_probability_by_stage"] = {
            "stage0": 0.0,
            "stage1": 1.0,
            "stage2": 1.0,
            "stage3": 1.0,
        }

        # 4) Use mild object/material randomization for Play.
        self.events.reset_autonomous_episode.params["object_randomization"] = {
            "mass_range": (2.5, 4.0),
            "friction_range": (0.60, 1.10),
            "restitution_range": (0.00, 0.06),
            "size_scale_range": (1.00, 1.00),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["robot_material_randomization"] = {
            "friction_range": (0.70, 1.10),
            "restitution_range": (0.00, 0.02),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["floor_material_randomization"] = {
            "friction_range": (0.75, 1.15),
        }

        # 5) Use mild observation noise/dropout for Play.
        self.events.reset_autonomous_episode.params["observation_randomization"] = {
            "projected_gravity_noise_std_range": (0.002, 0.008),
            "base_ang_vel_noise_std_range": (0.005, 0.020),
            "joint_pos_noise_std_range": (0.001, 0.006),
            "joint_vel_noise_std_range": (0.010, 0.050),
            "prev_action_noise_std_range": (0.001, 0.006),
            "mode_noise_std_range": (0.001, 0.008),
            "obj_pos_noise_range": (0.004, 0.020),
            "obj_vel_noise_range": (0.015, 0.100),
            "obj_pos_bias_range": (-0.010, 0.010),
            "obj_vel_bias_range": (-0.025, 0.025),
            "obj_pos_scale_range": (0.98, 1.02),
            "obj_vel_scale_range": (0.95, 1.05),
            "drop_prob_range": (0.005, 0.030),
            "false_positive_prob_range": (0.0, 0.002),
            "tag_visible_noise_std_range": (0.002, 0.010),
            "alpha_range": (0.70, 0.98),
            "latency_steps_range": (0, 2),
            "noise_spike_prob_range": (0.0, 0.01),
            "noise_spike_scale_range": (2.0, 3.5),
        }

        self.events.reset_autonomous_episode.params["visibility_randomization"] = {
            "pre_toss_visible_probability_by_stage": {"stage0": 0.0, "stage1": 1.0, "stage2": 1.0, "stage3": 1.0},
            "idle_visible_probability_by_stage": {"stage0": 1.0, "stage1": 0.35, "stage2": 0.35, "stage3": 0.35},
            "visibility_start_s": (0.0, 0.15),
            "idle_visible_pose": {
                "pos_x": (0.85, 1.35),
                "pos_y": (-0.25, 0.25),
                "pos_z": (-0.10, 0.25),
                "vel_x": (-0.02, 0.03),
                "vel_y": (-0.03, 0.03),
                "vel_z": (-0.01, 0.01),
            },
        }

        self.events.reset_autonomous_episode.params["delivery_randomization"] = {
            "toss_family_prob_by_stage": {
                "stage3": {"ballistic": 0.45, "carried_release": 0.25, "push_release": 0.18, "late_visible": 0.12},
            },
            "no_toss_family_prob_by_stage": {
                "stage3": {"idle": 0.45, "pass_by": 0.35, "approach_abort": 0.20},
            },
            "pre_release_start_probability_by_stage": {"stage3": 0.65},
            "pre_release_start_pose": {"pos_x": (0.85, 1.45), "pos_y": (-0.35, 0.35), "pos_z": (-0.12, 0.34)},
            "pass_by_visible_pose": {
                "pos_x": (0.65, 1.30), "pos_y": (-0.55, 0.55), "pos_z": (-0.12, 0.32),
                "vel_x": (-0.04, 0.06), "vel_y": (-0.35, 0.35), "vel_z": (-0.02, 0.02),
            },
        }

        # 6) One reasonable toss per episode.
        self.events.toss.params["max_throws_per_episode"] = 1
        self.events.toss.params["throw_prob_stage1"] = 1.0
        self.events.toss.params["throw_prob_stage2"] = 1.0
        self.events.toss.params["throw_prob_stage3"] = 1.0

        # Since eval_stage=3, only stage3 actually matters in Play.
        self.events.toss.params["stage3"] = {
            "sampler": "target_ballistic",

            # Spawn moderately in front of the robot.
            "spawn_x": (0.38, 0.58),
            "spawn_y": (-0.12, 0.12),
            "spawn_z": (0.20, 0.38),

            # Target near the chest/upper torso receiving region.
            "target_x": (0.08, 0.22),
            "target_y": (-0.07, 0.07),
            "target_z": (0.10, 0.25),

            # Not too fast, not too slow.
            "flight_time": (0.30, 0.44),
            "max_speed": 1.75,
            "max_vy_abs": 0.40,
            "max_vz_abs": 1.80,

            # Mild object orientation/angular velocity.
            "roll": (-0.02, 0.02),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.08, 0.08),
            "ang_vel_x": (-0.05, 0.05),
            "ang_vel_y": (-0.05, 0.05),
            "ang_vel_z": (-0.12, 0.12),
            "carried_release": {
                "sampler": "independent",
                "spawn_x": (0.18, 0.50), "spawn_y": (-0.16, 0.16), "spawn_z": (0.08, 0.30),
                "vel_x": (-0.55, -0.10), "vel_y": (-0.12, 0.12), "vel_z": (-0.10, 0.12),
                "max_speed": 0.95,
            },
            "push_release": {
                "sampler": "independent",
                "spawn_x": (0.22, 0.55), "spawn_y": (-0.18, 0.18), "spawn_z": (0.06, 0.32),
                "vel_x": (-0.95, -0.25), "vel_y": (-0.20, 0.20), "vel_z": (-0.16, 0.18),
                "max_speed": 1.25,
            },
            "late_visible_toss": {
                "sampler": "target_ballistic",
                "spawn_x": (0.38, 0.62), "spawn_y": (-0.14, 0.14), "spawn_z": (0.18, 0.40),
                "target_x": (0.08, 0.24), "target_y": (-0.08, 0.08), "target_z": (0.10, 0.26),
                "flight_time": (0.30, 0.46), "max_speed": 1.75, "max_vy_abs": 0.42, "max_vz_abs": 1.85,
            },
        }
