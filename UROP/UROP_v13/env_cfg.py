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
    command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    legs_sagittal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        ],
        scale=0.25,
    )
    legs_frontal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_roll_joint",
            "left_ankle_roll_joint",
            "right_hip_roll_joint",
            "right_ankle_roll_joint",
        ],
        scale=0.16,
    )
    legs_yaw = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
        scale=0.08,
    )
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        scale=0.15,
    )
    left_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_pitch_joint", "left_elbow_joint"],
        scale=0.42,
    )
    right_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=0.42,
    )
    left_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        scale=0.26,
    )
    right_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=0.26,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_angular_velocity)
        joint_pos = ObsTerm(func=mdp.controlled_joint_positions)
        joint_vel = ObsTerm(func=mdp.controlled_joint_velocities)
        prev_actions = ObsTerm(func=mdp.prev_actions)
        obj_rel = ObsTerm(func=mdp.object_rel_pos_vel, params={"pos_scale": 1.0, "vel_scale": 1.0, "apply_noise": True})
        phase = ObsTerm(func=mdp.toss_state)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

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
    height = RewTerm(func=mdp.root_height_reward, weight=1.0, params={"target_z": 0.78, "sigma": 0.10})

    base_motion = RewTerm(func=mdp.base_motion_penalty, weight=-0.18, params={"w_lin": 1.0, "w_ang": 0.30})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.04)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00004)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.08)
    foot_slip = RewTerm(func=mdp.foot_slip_penalty, weight=-0.12, params={"ground_height_thr": 0.16})

    wait_ready_pose = RewTerm(func=mdp.ready_pose_when_waiting, weight=2.5, params={"sigma": 0.28})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-2.0, params={"sigma": 0.16})
    lower_body_ready = RewTerm(
        func=mdp.lower_body_ready_reward,
        weight=1.6,
        params={"sigma_wait": 0.18, "sigma_active": 0.28},
    )

    chest_receive = RewTerm(func=mdp.chest_receive_region_reward, weight=1.2, params={"sigma": 0.35})
    upper_body_receive = RewTerm(func=mdp.upper_body_receive_reward, weight=1.0, params={"sigma": 0.30})
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=2.0,
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
    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=1.6, params={"torso_body_name": "torso_link", "sigma": 0.55})
    hold_pose = RewTerm(func=mdp.hold_pose_reward, weight=2.0, params={"sigma": 0.22})
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=1.2)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=2.8, params={"min_steps": 24})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=0.8, params={"min_z": 0.42, "max_dist": 2.0})
    impact = RewTerm(func=mdp.impact_peak_penalty, weight=-0.004, params={"sensor_names": CONTACT_SENSOR_NAMES, "force_thr": 220.0})

    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=1.6, params={"lin_sigma": 0.12, "yaw_sigma": 0.35})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-1.5, params={"sigma": 0.10})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.successful_hold_complete, params={"min_steps": 40})
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.47, "max_tilt_deg": 55.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.30, "max_dist": 2.1})
    runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.32})
    unsafe_lower_body = DoneTerm(func=mdp.unsafe_lower_body_deviation, params={"max_abs_dev": 1.0})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_parked = EventTerm(
        func=mdp.reset_object_parked,
        mode="reset",
        params={
            "park": {"pos_x": (1.55, 1.75), "pos_y": (-0.10, 0.10), "pos_z": (-0.62, -0.54)},
            "object_randomization": {
                "mass_range": (2.4, 4.6),
                "friction_range": (0.60, 0.95),
                "restitution_range": (0.00, 0.05),
                "size_scale_range": (0.94, 1.06),
                "apply_physx": True,
            },
            "observation_randomization": {
                "pos_noise_range": (0.005, 0.018),
                "vel_noise_range": (0.03, 0.10),
                "drop_prob_range": (0.00, 0.08),
                "alpha_range": (0.65, 1.00),
            },
        },
    )

    toss = EventTerm(
        func=mdp.toss_object_relative_curriculum,
        mode="interval",
        interval_range_s=(1.40, 1.80),
        params={
            "max_throws_per_episode": 1,
            "throw_prob_stage1": 1.00,
            "throw_prob_stage2": 0.95,
            "throw_prob_stage3": 0.90,
            "stage1": {
                "pos_x": (0.42, 0.50),
                "pos_y": (-0.04, 0.04),
                "pos_z": (0.26, 0.36),
                "vel_x": (-0.18, -0.05),
                "vel_y": (-0.04, 0.04),
                "vel_z": (-0.02, 0.05),
                "roll": (-0.03, 0.03),
                "pitch": (-0.04, 0.04),
                "yaw": (-0.08, 0.08),
                "ang_vel_x": (-0.08, 0.08),
                "ang_vel_y": (-0.08, 0.08),
                "ang_vel_z": (-0.12, 0.12),
            },
            "stage2": {
                "pos_x": (0.46, 0.56),
                "pos_y": (-0.05, 0.05),
                "pos_z": (0.24, 0.38),
                "vel_x": (-0.70, -0.40),
                "vel_y": (-0.06, 0.06),
                "vel_z": (-0.04, 0.08),
                "roll": (-0.04, 0.04),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.10, 0.10),
                "ang_vel_x": (-0.12, 0.12),
                "ang_vel_y": (-0.12, 0.12),
                "ang_vel_z": (-0.18, 0.18),
            },
            "stage3": {
                "pos_x": (0.50, 0.60),
                "pos_y": (-0.08, 0.08),
                "pos_z": (0.24, 0.40),
                "vel_x": (-1.00, -0.65),
                "vel_y": (-0.10, 0.10),
                "vel_z": (-0.06, 0.10),
                "roll": (-0.05, 0.05),
                "pitch": (-0.06, 0.06),
                "yaw": (-0.14, 0.14),
                "ang_vel_x": (-0.18, 0.18),
                "ang_vel_y": (-0.18, 0.18),
                "ang_vel_z": (-0.24, 0.24),
            },
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 400,
            "stage1_iters": 1200,
            "stage2_iters": 2000,
            "num_steps_per_env": 64,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v13_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v13_SceneCfg = dj_urop_v13_SceneCfg(num_envs=64, env_spacing=2.6)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.0
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
class dj_urop_v13_EnvCfg_Play(dj_urop_v13_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 3
        self.scene.env_spacing = 3.0
        self.episode_length_s = 7.0

        self.curriculum.stage_schedule.params["eval_stage"] = 0

        self.events.reset_object_parked.params["object_randomization"] = {
            "mass_range": (3.0, 3.4),
            "friction_range": (0.75, 0.85),
            "restitution_range": (0.00, 0.02),
            "size_scale_range": (0.98, 1.02),
            "apply_physx": True,
        }
        self.events.reset_object_parked.params["observation_randomization"] = {
            "pos_noise_range": (0.003, 0.006),
            "vel_noise_range": (0.02, 0.04),
            "drop_prob_range": (0.00, 0.02),
            "alpha_range": (0.90, 1.00),
        }

        self.events.toss.interval_range_s = (1.55, 1.55)
        self.events.toss.params["throw_prob_stage1"] = 1.0
        self.events.toss.params["stage1"] = {
            "pos_x": (0.46, 0.48),
            "pos_y": (-0.02, 0.02),
            "pos_z": (0.30, 0.32),
            "vel_x": (-0.10, -0.08),
            "vel_y": (0.00, 0.00),
            "vel_z": (0.00, 0.02),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
            "ang_vel_x": (0.0, 0.0),
            "ang_vel_y": (0.0, 0.0),
            "ang_vel_z": (0.0, 0.0),
        }
