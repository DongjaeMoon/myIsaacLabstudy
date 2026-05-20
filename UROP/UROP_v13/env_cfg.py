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
class dj_urop_v13_SceneCfg(InteractiveSceneCfg):
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
        # 4
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

    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.12, params={"w_lin": 1.0, "w_ang": 0.30})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.025)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00002)
    action_mag = RewTerm(func=mdp.action_magnitude_penalty, weight=-0.015)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.06)
    foot_slip = RewTerm(func=mdp.foot_slip_penalty, weight=-0.10, params={"ground_height_thr": 0.16})

    wait_ready_pose = RewTerm(func=mdp.ready_pose_when_waiting, weight=2.8, params={"sigma": 0.22})
    wait_joint_still = RewTerm(func=mdp.waiting_joint_stillness_reward, weight=0.8, params={"sigma": 1.4})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-1.6, params={"sigma": 0.14})
    lower_body_ready = RewTerm(
        func=mdp.lower_body_ready_reward,
        weight=1.6,
        params={"sigma_wait": 0.16, "sigma_active": 0.26},
    )

    catch_region = RewTerm(func=mdp.catch_target_region_reward, weight=1.7, params={"sigma": 0.28})
    upper_body_receive = RewTerm(func=mdp.upper_body_receive_reward, weight=1.4, params={"sigma": 0.26})
    catch_vel_match = RewTerm(
        func=mdp.catch_velocity_match_reward,
        weight=1.0,
        params={"torso_body_name": "torso_link", "sigma": 0.75},
    )
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=2.4,
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
    impact = RewTerm(func=mdp.impact_peak_penalty, weight=-0.004, params={"sensor_names": CONTACT_SENSOR_NAMES, "force_thr": 220.0})

    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=1.8, params={"torso_body_name": "torso_link", "sigma": 0.45})
    hold_pose = RewTerm(func=mdp.hold_pose_reward, weight=2.3, params={"sigma": 0.18})
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=1.0)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=2.6, params={"min_steps": 20})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.2, params={"min_z": 0.42, "max_dist": 1.8})

    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=1.5, params={"lin_sigma": 0.10, "yaw_sigma": 0.30})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-1.4, params={"sigma": 0.10})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 40})
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.50, "max_tilt_deg": 45.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.30, "max_dist": 2.0})
    runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.28})
    unsafe_lower_body = DoneTerm(func=mdp.unsafe_lower_body_deviation, params={"max_abs_dev": 0.85})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

    reset_autonomous_episode = EventTerm(
        func=mdp.reset_autonomous_episode,
        mode="reset",
        params={
            "park": {"pos_x": (1.55, 1.85), "pos_y": (-0.15, 0.15), "pos_z": (-0.62, -0.52)},
            "wait_time_ranges": {
                "stage1": (1.20, 2.00),
                "stage2": (1.00, 2.80),
                "stage3": (0.80, 3.50),
            },
            "joint_noise": {
                "lower_body_pos": (-0.025, 0.025),
                "waist_pos": (-0.020, 0.020),
                "arm_pos": (-0.050, 0.050),
                "wrist_pos": (-0.030, 0.030),
                "velocity": (-0.08, 0.08),
            },
            "root_xy_range": (-0.02, 0.02),
            "root_yaw_range": (-0.05, 0.05),
            "object_randomization": {
                "mass_range": (2.0, 4.5),
                "friction_range": (0.55, 1.00),
                "restitution_range": (0.00, 0.06),
                "size_scale_range": (0.95, 1.05),
                "apply_physx": True,
            },
            "robot_material_randomization": {
                "friction_range": (0.70, 1.00),
                "restitution_range": (0.00, 0.02),
                "apply_physx": True,
            },
            "floor_material_randomization": {
                "friction_range": (0.75, 1.05),
            },
            "observation_randomization": {
                "pos_noise_range": (0.004, 0.020),
                "vel_noise_range": (0.02, 0.12),
                "drop_prob_range": (0.00, 0.08),
                "alpha_range": (0.35, 0.85),
            },
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
                # Very easy close handover-like toss.
                # z values are RELATIVE to robot root/pelvis, not world height.
                "sampler": "target_ballistic",
                "spawn_x": (0.32, 0.44),
                "spawn_y": (-0.04, 0.04),
                "spawn_z": (0.18, 0.32),
                "target_x": (0.08, 0.18),
                "target_y": (-0.04, 0.04),
                "target_z": (0.08, 0.22),
                "flight_time": (0.22, 0.34),
                "max_speed": 1.45,
                "max_vy_abs": 0.25,
                "max_vz_abs": 1.80,
                "roll": (-0.02, 0.02),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.05, 0.05),
                "ang_vel_x": (-0.05, 0.05),
                "ang_vel_y": (-0.05, 0.05),
                "ang_vel_z": (-0.08, 0.08),
            },
            "stage2": {
                # Gentle short toss.
                "sampler": "target_ballistic",
                "spawn_x": (0.34, 0.55),
                "spawn_y": (-0.08, 0.08),
                "spawn_z": (0.16, 0.38),
                "target_x": (0.06, 0.22),
                "target_y": (-0.06, 0.06),
                "target_z": (0.06, 0.26),
                "flight_time": (0.22, 0.38),
                "max_speed": 1.75,
                "max_vy_abs": 0.35,
                "max_vz_abs": 2.00,
                "roll": (-0.03, 0.03),
                "pitch": (-0.04, 0.04),
                "yaw": (-0.08, 0.08),
                "ang_vel_x": (-0.08, 0.08),
                "ang_vel_y": (-0.08, 0.08),
                "ang_vel_z": (-0.12, 0.12),
            },
            "stage3": {
                # Realistic but still admissible close toss.
                "sampler": "target_ballistic",
                "spawn_x": (0.36, 0.65),
                "spawn_y": (-0.14, 0.14),
                "spawn_z": (0.12, 0.44),
                "target_x": (0.04, 0.26),
                "target_y": (-0.10, 0.10),
                "target_z": (0.04, 0.30),
                "flight_time": (0.24, 0.42),
                "max_speed": 2.05,
                "max_vy_abs": 0.50,
                "max_vz_abs": 2.20,
                "roll": (-0.04, 0.04),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.12, 0.12),
                "ang_vel_x": (-0.12, 0.12),
                "ang_vel_y": (-0.12, 0.12),
                "ang_vel_z": (-0.18, 0.18),
            },
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 300,
            "stage1_iters": 1200,
            "stage2_iters": 2200,
            "num_steps_per_env": 64,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v13_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v13_SceneCfg = dj_urop_v13_SceneCfg(num_envs=128, env_spacing=3.0)
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

        assert len(scene_objects_cfg.CONTROLLED_JOINT_NAMES) == scene_objects_cfg.EXPECTED_ACTION_DIM
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
class dj_urop_v13_EnvCfg_Play(dj_urop_v13_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 3.2
        self.episode_length_s = 7.0

        # Play defaults to gentle stage-1 close toss.
        self.curriculum.stage_schedule.params["eval_stage"] = 1

        self.events.reset_autonomous_episode.params["wait_time_ranges"] = {
            "stage1": (1.50, 1.50),
            "stage2": (1.50, 1.50),
            "stage3": (1.50, 1.50),
        }
        self.events.reset_autonomous_episode.params["object_randomization"] = {
            "mass_range": (2.8, 3.6),
            "friction_range": (0.75, 0.95),
            "restitution_range": (0.00, 0.03),
            "size_scale_range": (0.98, 1.02),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["robot_material_randomization"] = {
            "friction_range": (0.80, 0.95),
            "restitution_range": (0.00, 0.01),
            "apply_physx": True,
        }
        self.events.reset_autonomous_episode.params["floor_material_randomization"] = {
            "friction_range": (0.85, 0.95),
        }
        self.events.reset_autonomous_episode.params["observation_randomization"] = {
            "pos_noise_range": (0.003, 0.006),
            "vel_noise_range": (0.02, 0.05),
            "drop_prob_range": (0.00, 0.02),
            "alpha_range": (0.75, 0.95),
        }

        self.events.toss.params["max_throws_per_episode"] = 1
        self.events.toss.params["throw_prob_stage1"] = 1.0
        self.events.toss.params["throw_prob_stage2"] = 1.0
        self.events.toss.params["throw_prob_stage3"] = 1.0

        self.events.toss.params["stage1"] = {
            "sampler": "target_ballistic",
            "spawn_x": (0.32, 0.44),
            "spawn_y": (-0.04, 0.04),
            "spawn_z": (0.18, 0.32),
            "target_x": (0.08, 0.18),
            "target_y": (-0.04, 0.04),
            "target_z": (0.08, 0.22),
            "flight_time": (0.22, 0.34),
            "max_speed": 1.45,
            "max_vy_abs": 0.25,
            "max_vz_abs": 1.80,
            "roll": (-0.02, 0.02),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.05, 0.05),
            "ang_vel_x": (-0.05, 0.05),
            "ang_vel_y": (-0.05, 0.05),
            "ang_vel_z": (-0.08, 0.08),
        }

        self.events.toss.params["stage2"] = {
            "sampler": "target_ballistic",
            "spawn_x": (0.34, 0.55),
            "spawn_y": (-0.08, 0.08),
            "spawn_z": (0.16, 0.38),
            "target_x": (0.06, 0.22),
            "target_y": (-0.06, 0.06),
            "target_z": (0.06, 0.26),
            "flight_time": (0.22, 0.38),
            "max_speed": 1.75,
            "max_vy_abs": 0.35,
            "max_vz_abs": 2.00,
            "roll": (-0.03, 0.03),
            "pitch": (-0.04, 0.04),
            "yaw": (-0.08, 0.08),
            "ang_vel_x": (-0.08, 0.08),
            "ang_vel_y": (-0.08, 0.08),
            "ang_vel_z": (-0.12, 0.12),
        }

        self.events.toss.params["stage3"] = {
            "sampler": "target_ballistic",
            "spawn_x": (0.36, 0.65),
            "spawn_y": (-0.14, 0.14),
            "spawn_z": (0.12, 0.44),
            "target_x": (0.04, 0.26),
            "target_y": (-0.10, 0.10),
            "target_z": (0.04, 0.30),
            "flight_time": (0.24, 0.42),
            "max_speed": 2.05,
            "max_vy_abs": 0.50,
            "max_vz_abs": 2.20,
            "roll": (-0.04, 0.04),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.12, 0.12),
            "ang_vel_x": (-0.12, 0.12),
            "ang_vel_y": (-0.12, 0.12),
            "ang_vel_z": (-0.18, 0.18),
        }
