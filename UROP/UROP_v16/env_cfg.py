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

_V16_CONTRACT_PRINTED = False


def _print_v16_contract_once(cfg: "dj_urop_v16_EnvCfg") -> None:
    global _V16_CONTRACT_PRINTED
    if _V16_CONTRACT_PRINTED:
        return

    joint_names = list(scene_objects_cfg.CONTROLLED_JOINT_NAMES)
    finger_tokens = ("thumb", "index", "middle", "hand")
    has_finger_action = any(token in name for name in joint_names for token in finger_tokens)

    print(
        "[UROP_v16] contract: "
        f"policy_obs_dim={scene_objects_cfg.EXPECTED_POLICY_OBS_DIM} "
        f"action_dim={scene_objects_cfg.EXPECTED_ACTION_DIM} "
        f"fingers_excluded={not has_finger_action}"
    )
    print(f"[UROP_v16] controlled_joint_order={joint_names}")
    print(f"[UROP_v16] policy_obs_components={scene_objects_cfg.POLICY_OBS_COMPONENT_DIMS}")

    reset_cfg = cfg.events.reset_autonomous_episode.params
    joint_rand = reset_cfg["joint_randomization"]
    base_rand = reset_cfg["base_randomization"]
    obs_rand = reset_cfg["observation_randomization"]
    print(
        "[UROP_v16] reset curriculum: "
        f"joint_pos_scale_by_stage={joint_rand['joint_pos_scale_by_stage']} "
        f"joint_vel_scale_by_stage={joint_rand['joint_vel_scale_by_stage']}"
    )
    print(
        "[UROP_v16] base curriculum: "
        f"orientation_scale_by_stage={base_rand['base_orientation_scale_by_stage']} "
        f"velocity_scale_by_stage={base_rand['base_velocity_scale_by_stage']}"
    )
    print(
        "[UROP_v16] wait/no-toss curriculum: "
        f"wait_time_ranges={reset_cfg['wait_time_ranges']} "
        f"no_toss_probability_by_stage={reset_cfg['no_toss_probability_by_stage']}"
    )
    print(
        "[UROP_v16] object obs curriculum: "
        f"noise_scale_by_stage={obs_rand['noise_scale_by_stage']} "
        f"dropout_scale_by_stage={obs_rand['dropout_scale_by_stage']} "
        f"latency_steps_by_stage={obs_rand['latency_steps_by_stage']}"
    )

    _V16_CONTRACT_PRINTED = True


@configclass
class dj_urop_v16_SceneCfg(InteractiveSceneCfg):
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
    # [v16] DO NOT REORDER: must match sim2real deploy policy action order exactly.
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
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity)
        joint_pos_rel = ObsTerm(func=mdp.controlled_joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities, params={"scale": 0.05})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        object_rel_pos = ObsTerm(func=mdp.object_rel_pos)
        object_rel_lin_vel = ObsTerm(func=mdp.object_rel_lin_vel)
        tag_visible = ObsTerm(func=mdp.tag_visible)
        # [v16] Kept to preserve the 104-D actor contract used by deployment tooling.
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
    upright = RewTerm(func=mdp.upright_reward, weight=2.00)
    height = RewTerm(func=mdp.root_height_reward, weight=1.15, params={"target_z": 0.78, "sigma": 0.10})

    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.14, params={"w_lin": 1.0, "w_ang": 0.35})
    wait_base_ang_vel = RewTerm(
        func=mdp.wait_base_angular_velocity_penalty,
        weight=-0.22,
        params={"roll_pitch_weight": 1.0, "yaw_weight": 0.55},
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.03)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00002)
    action_mag = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.012)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.10)
    action_accel = RewTerm(func=mdp.action_acceleration_penalty, weight=-0.05)
    lower_body_action_rate = RewTerm(func=mdp.lower_body_action_rate_penalty, weight=-0.05)
    foot_slip = RewTerm(func=mdp.foot_slip_penalty, weight=-0.10, params={"ground_height_thr": 0.16})

    # [v16] Pre-toss robustness and no-toss stability.
    wait_survival = RewTerm(func=mdp.wait_survival_bonus, weight=0.45)
    wait_ready_pose = RewTerm(func=mdp.ready_pose_when_waiting, weight=3.20, params={"sigma": 0.24})
    wait_joint_still = RewTerm(func=mdp.waiting_joint_stillness_reward, weight=0.95, params={"sigma": 1.25})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-1.85, params={"sigma": 0.14})
    wait_yaw_drift = RewTerm(func=mdp.wait_yaw_drift_penalty, weight=-1.00, params={"sigma": 0.20})
    lower_body_ready = RewTerm(
        func=mdp.lower_body_ready_reward,
        weight=1.80,
        params={"sigma_wait": 0.18, "sigma_active": 0.30},
    )
    lower_body_reference = RewTerm(func=mdp.lower_body_reference_penalty, weight=-0.10, params={"sigma": 0.32})
    no_toss_quiet_upper_body = RewTerm(
        func=mdp.no_toss_upper_body_motion_penalty,
        weight=-0.08,
        params={"waist_action_start": 12, "upper_action_start": 15, "joint_vel_weight": 0.35},
    )

    catch_region = RewTerm(func=mdp.catch_target_region_reward, weight=1.70, params={"sigma": 0.28})
    upper_body_receive = RewTerm(func=mdp.upper_body_receive_reward, weight=1.40, params={"sigma": 0.26})
    catch_vel_match = RewTerm(
        func=mdp.catch_velocity_match_reward,
        weight=1.00,
        params={"torso_body_name": "torso_link", "sigma": 0.75},
    )
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=2.40,
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
    impact = RewTerm(
        func=mdp.impact_peak_penalty,
        weight=-0.004,
        params={"sensor_names": CONTACT_SENSOR_NAMES, "force_thr": 220.0},
    )

    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=1.80, params={"torso_body_name": "torso_link", "sigma": 0.45})
    hold_pose = RewTerm(func=mdp.hold_pose_reward, weight=2.30, params={"sigma": 0.18})
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=1.00)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=2.60, params={"min_steps": 20})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.20, params={"min_z": 0.42, "max_dist": 1.8})

    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=1.50, params={"lin_sigma": 0.10, "yaw_sigma": 0.30})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-1.40, params={"sigma": 0.10})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 40})
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.50, "max_tilt_deg": 45.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.30, "max_dist": 2.0})
    runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.28})
    unsafe_lower_body = DoneTerm(func=mdp.unsafe_lower_body_deviation, params={"max_abs_dev": 0.95})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_autonomous_episode = EventTerm(
        func=mdp.reset_autonomous_episode,
        mode="reset",
        params={
            "park": {"pos_x": (1.55, 1.85), "pos_y": (-0.15, 0.15), "pos_z": (-0.62, -0.52)},
            # [v16] Randomized pre-toss stabilization window.
            "wait_time_ranges": {
                "stage1": (0.50, 1.50),
                "stage2": (0.35, 2.40),
                "stage3": (0.20, 3.50),
            },
            # [v16] Explicit no-toss curriculum instead of only implicit toss probability.
            "no_toss_probability_by_stage": {
                "stage0": 1.00,
                "stage1": 0.30,
                "stage2": 0.20,
                "stage3": 0.15,
            },
            "joint_randomization": {
                "joint_pos_ranges": {
                    "hip_pitch": (-0.12, 0.12),
                    "hip_roll": (-0.10, 0.10),
                    "hip_yaw": (-0.10, 0.10),
                    "knee": (-0.16, 0.16),
                    "ankle_pitch": (-0.09, 0.09),
                    "ankle_roll": (-0.08, 0.08),
                    "waist_yaw": (-0.12, 0.12),
                    "waist_roll": (-0.10, 0.10),
                    "waist_pitch": (-0.10, 0.10),
                    "shoulder_pitch": (-0.26, 0.26),
                    "shoulder_roll": (-0.24, 0.24),
                    "shoulder_yaw": (-0.22, 0.22),
                    "elbow": (-0.24, 0.24),
                    "wrist_roll": (-0.18, 0.18),
                    "wrist_pitch": (-0.18, 0.18),
                    "wrist_yaw": (-0.18, 0.18),
                },
                "joint_pos_scale_by_stage": {
                    "stage0": 0.45,
                    "stage1": 0.65,
                    "stage2": 0.85,
                    "stage3": 1.00,
                },
                "joint_vel_range": (-0.50, 0.50),
                "joint_vel_scale_by_stage": {
                    "stage0": 0.40,
                    "stage1": 0.55,
                    "stage2": 0.75,
                    "stage3": 1.00,
                },
            },
            "base_randomization": {
                "root_xy_range": (-0.05, 0.05),
                "root_xy_scale_by_stage": {
                    "stage0": 0.50,
                    "stage1": 0.75,
                    "stage2": 1.00,
                    "stage3": 1.00,
                },
                # [v16] Keep default to non-penetrating height reset; only upward offsets are applied if enabled.
                "base_height_range": (0.00, 0.00),
                "base_roll_deg_range": (-5.0, 5.0),
                "base_pitch_deg_range": (-5.0, 5.0),
                "base_yaw_deg_range": (-10.0, 10.0),
                "base_orientation_scale_by_stage": {
                    "stage0": 0.40,
                    "stage1": 0.60,
                    "stage2": 0.80,
                    "stage3": 1.00,
                },
                "base_lin_vel_xy_range": (-0.15, 0.15),
                "base_lin_vel_z_range": (-0.03, 0.03),
                "base_ang_vel_rp_range": (-0.30, 0.30),
                "base_ang_vel_yaw_range": (-0.30, 0.30),
                "base_velocity_scale_by_stage": {
                    "stage0": 0.35,
                    "stage1": 0.55,
                    "stage2": 0.80,
                    "stage3": 1.00,
                },
            },
            "object_randomization": {
                "mass_range": (1.0, 6.5),
                "friction_range": (0.25, 1.50),
                "restitution_range": (0.00, 0.18),
                "size_scale_range": (0.90, 1.15),
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
                "projected_gravity_noise_std_range": (0.005, 0.030),
                "base_ang_vel_noise_std_range": (0.010, 0.100),
                "joint_pos_noise_std_range": (0.002, 0.020),
                "joint_vel_noise_std_range": (0.020, 0.150),
                "obj_pos_noise_range": (0.005, 0.030),
                "obj_vel_noise_range": (0.020, 0.150),
                "drop_prob_range": (0.00, 0.18),
                "alpha_range": (0.35, 0.85),
                "noise_scale_by_stage": {
                    "stage0": 0.50,
                    "stage1": 0.70,
                    "stage2": 0.85,
                    "stage3": 1.00,
                },
                "dropout_scale_by_stage": {
                    "stage0": 0.25,
                    "stage1": 0.50,
                    "stage2": 0.75,
                    "stage3": 1.00,
                },
                "latency_steps_by_stage": {
                    "stage0": (0, 1),
                    "stage1": (1, 1),
                    "stage2": (1, 2),
                    "stage3": (1, 3),
                },
            },
            "debug_print": False,
            "debug_env_local_index": 0,
        },
    )

    random_push = EventTerm(
        func=mdp.push_robot_root_velocity,
        mode="interval",
        interval_range_s=(0.8, 2.2),
        params={
            "stage0_xy_range": (-0.12, 0.12),
            "stage1_xy_range": (-0.20, 0.20),
            "stage2_xy_range": (-0.30, 0.30),
            "stage3_xy_range": (-0.40, 0.40),
            "stage0_yaw_range": (-0.08, 0.08),
            "stage1_yaw_range": (-0.15, 0.15),
            "stage2_yaw_range": (-0.22, 0.22),
            "stage3_yaw_range": (-0.30, 0.30),
            "z_velocity_range": (-0.02, 0.02),
            "hold_xy_scale": 0.70,
            "hold_yaw_scale": 0.65,
            "max_xy_speed": 1.40,
            "max_yaw_speed": 1.00,
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
                "variants": [
                    {
                        "name": "handover",
                        "weight": 0.65,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.24, 0.40),
                        "spawn_y": (-0.10, 0.10),
                        "spawn_z": (0.16, 0.34),
                        "target_x": (0.08, 0.20),
                        "target_y": (-0.08, 0.08),
                        "target_z": (0.10, 0.24),
                        "flight_time": (0.34, 0.52),
                        "max_speed": 1.05,
                        "max_vy_abs": 0.20,
                        "max_vz_abs": 1.10,
                        "roll": (-0.03, 0.03),
                        "pitch": (-0.04, 0.04),
                        "yaw": (-0.10, 0.10),
                        "ang_vel_x": (-0.05, 0.05),
                        "ang_vel_y": (-0.05, 0.05),
                        "ang_vel_z": (-0.08, 0.08),
                    },
                    {
                        "name": "weak_toss",
                        "weight": 0.35,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.30, 0.50),
                        "spawn_y": (-0.12, 0.12),
                        "spawn_z": (0.18, 0.36),
                        "target_x": (0.04, 0.20),
                        "target_y": (-0.10, 0.10),
                        "target_z": (0.08, 0.24),
                        "flight_time": (0.24, 0.40),
                        "max_speed": 1.45,
                        "max_vy_abs": 0.35,
                        "max_vz_abs": 1.70,
                        "roll": (-0.04, 0.04),
                        "pitch": (-0.05, 0.05),
                        "yaw": (-0.12, 0.12),
                        "ang_vel_x": (-0.08, 0.08),
                        "ang_vel_y": (-0.08, 0.08),
                        "ang_vel_z": (-0.14, 0.14),
                    },
                ]
            },
            "stage2": {
                "variants": [
                    {
                        "name": "handover",
                        "weight": 0.25,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.26, 0.46),
                        "spawn_y": (-0.18, 0.18),
                        "spawn_z": (0.16, 0.38),
                        "target_x": (0.04, 0.22),
                        "target_y": (-0.14, 0.14),
                        "target_z": (0.06, 0.28),
                        "flight_time": (0.30, 0.48),
                        "max_speed": 1.25,
                        "max_vy_abs": 0.30,
                        "max_vz_abs": 1.40,
                        "roll": (-0.03, 0.03),
                        "pitch": (-0.04, 0.04),
                        "yaw": (-0.10, 0.10),
                        "ang_vel_x": (-0.06, 0.06),
                        "ang_vel_y": (-0.06, 0.06),
                        "ang_vel_z": (-0.10, 0.10),
                    },
                    {
                        "name": "weak_toss",
                        "weight": 0.55,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.34, 0.62),
                        "spawn_y": (-0.24, 0.24),
                        "spawn_z": (0.12, 0.44),
                        "target_x": (0.00, 0.24),
                        "target_y": (-0.18, 0.18),
                        "target_z": (0.04, 0.30),
                        "flight_time": (0.22, 0.44),
                        "max_speed": 1.95,
                        "max_vy_abs": 0.80,
                        "max_vz_abs": 2.10,
                        "roll": (-0.04, 0.04),
                        "pitch": (-0.05, 0.05),
                        "yaw": (-0.14, 0.14),
                        "ang_vel_x": (-0.10, 0.10),
                        "ang_vel_y": (-0.10, 0.10),
                        "ang_vel_z": (-0.18, 0.18),
                    },
                    {
                        "name": "strong_toss",
                        "weight": 0.20,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.38, 0.76),
                        "spawn_y": (-0.32, 0.32),
                        "spawn_z": (0.10, 0.50),
                        "target_x": (-0.02, 0.28),
                        "target_y": (-0.22, 0.22),
                        "target_z": (0.02, 0.34),
                        "flight_time": (0.18, 0.42),
                        "max_speed": 2.35,
                        "max_vy_abs": 1.00,
                        "max_vz_abs": 2.35,
                        "roll": (-0.05, 0.05),
                        "pitch": (-0.06, 0.06),
                        "yaw": (-0.24, 0.24),
                        "ang_vel_x": (-0.14, 0.14),
                        "ang_vel_y": (-0.14, 0.14),
                        "ang_vel_z": (-0.28, 0.28),
                    },
                ]
            },
            "stage3": {
                "variants": [
                    {
                        "name": "handover",
                        "weight": 0.20,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.25, 0.48),
                        "spawn_y": (-0.22, 0.22),
                        "spawn_z": (0.16, 0.42),
                        "target_x": (0.02, 0.24),
                        "target_y": (-0.16, 0.16),
                        "target_z": (0.04, 0.30),
                        "flight_time": (0.28, 0.54),
                        "max_speed": 1.30,
                        "max_vy_abs": 0.35,
                        "max_vz_abs": 1.50,
                        "roll": (-0.03, 0.03),
                        "pitch": (-0.04, 0.04),
                        "yaw": (-0.12, 0.12),
                        "ang_vel_x": (-0.06, 0.06),
                        "ang_vel_y": (-0.06, 0.06),
                        "ang_vel_z": (-0.10, 0.10),
                    },
                    {
                        "name": "weak_toss",
                        "weight": 0.45,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.32, 0.70),
                        "spawn_y": (-0.35, 0.35),
                        "spawn_z": (0.10, 0.50),
                        "target_x": (-0.02, 0.28),
                        "target_y": (-0.22, 0.22),
                        "target_z": (0.02, 0.34),
                        "flight_time": (0.20, 0.48),
                        "max_speed": 2.10,
                        "max_vy_abs": 0.95,
                        "max_vz_abs": 2.20,
                        "roll": (-0.04, 0.04),
                        "pitch": (-0.05, 0.05),
                        "yaw": (-0.18, 0.18),
                        "ang_vel_x": (-0.10, 0.10),
                        "ang_vel_y": (-0.10, 0.10),
                        "ang_vel_z": (-0.20, 0.20),
                    },
                    {
                        "name": "strong_toss",
                        "weight": 0.35,
                        "sampler": "target_ballistic",
                        "spawn_x": (0.34, 0.88),
                        "spawn_y": (-0.45, 0.45),
                        "spawn_z": (0.05, 0.56),
                        "target_x": (-0.04, 0.32),
                        "target_y": (-0.24, 0.24),
                        "target_z": (0.00, 0.38),
                        "flight_time": (0.18, 0.55),
                        "max_speed": 2.60,
                        "max_vy_abs": 1.20,
                        "max_vz_abs": 2.40,
                        "roll": (-0.05, 0.05),
                        "pitch": (-0.06, 0.06),
                        "yaw": (-0.35, 0.35),
                        "ang_vel_x": (-0.18, 0.18),
                        "ang_vel_y": (-0.18, 0.18),
                        "ang_vel_z": (-0.42, 0.42),
                    },
                ]
            },
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 300,
            "stage1_iters": 900,
            "stage2_iters": 1300,
            "num_steps_per_env": 64,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v16_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v16_SceneCfg = dj_urop_v16_SceneCfg(num_envs=128, env_spacing=3.0)
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
        assert all(
            token not in name
            for name in self.actions.policy.joint_names
            for token in ("thumb", "index", "middle", "hand")
        )

        try:
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
                self.sim.physx.enable_external_forces_every_iteration = True
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "num_velocity_iterations"):
                self.sim.physx.num_velocity_iterations = 1
        except Exception:
            pass

        _print_v16_contract_once(self)


@configclass
class dj_urop_v16_EnvCfg_Play(dj_urop_v16_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 3.2
        self.episode_length_s = 7.5

        # [v16] Default play mode is still robustness-oriented stage-3 evaluation.
        self.curriculum.stage_schedule.params["eval_stage"] = 3

        # Keep focus on reset robustness rather than random pushes during visual evaluation.
        self.events.random_push.interval_range_s = (9999.0, 9999.0)

        # [v16] Play keeps randomized wait/no-toss behavior by default for realism.
        self.events.reset_autonomous_episode.params["wait_time_ranges"] = {
            "stage1": (0.50, 1.50),
            "stage2": (0.35, 2.40),
            "stage3": (0.20, 3.50),
        }
        self.events.reset_autonomous_episode.params["no_toss_probability_by_stage"] = {
            "stage0": 1.00,
            "stage1": 0.20,
            "stage2": 0.15,
            "stage3": 0.15,
        }
        self.events.reset_autonomous_episode.params["observation_randomization"] = {
            "projected_gravity_noise_std_range": (0.002, 0.015),
            "base_ang_vel_noise_std_range": (0.005, 0.040),
            "joint_pos_noise_std_range": (0.001, 0.008),
            "joint_vel_noise_std_range": (0.010, 0.060),
            "obj_pos_noise_range": (0.002, 0.015),
            "obj_vel_noise_range": (0.010, 0.080),
            "drop_prob_range": (0.00, 0.10),
            "alpha_range": (0.45, 0.90),
            "noise_scale_by_stage": {
                "stage0": 0.60,
                "stage1": 0.80,
                "stage2": 0.90,
                "stage3": 1.00,
            },
            "dropout_scale_by_stage": {
                "stage0": 0.20,
                "stage1": 0.45,
                "stage2": 0.70,
                "stage3": 1.00,
            },
            "latency_steps_by_stage": {
                "stage0": (0, 0),
                "stage1": (0, 1),
                "stage2": (1, 1),
                "stage3": (1, 2),
            },
        }
        self.events.reset_autonomous_episode.params["debug_print"] = True
        self.events.reset_autonomous_episode.params["debug_env_local_index"] = 0

        # Slightly narrower object/material spread in Play, but still not visualization-only easy mode.
        self.events.reset_autonomous_episode.params["object_randomization"] = {
            "mass_range": (2.0, 5.0),
            "friction_range": (0.45, 1.20),
            "restitution_range": (0.00, 0.08),
            "size_scale_range": (0.95, 1.08),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["robot_material_randomization"] = {
            "friction_range": (0.55, 1.15),
            "restitution_range": (0.00, 0.03),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["floor_material_randomization"] = {
            "friction_range": (0.55, 1.20),
        }
