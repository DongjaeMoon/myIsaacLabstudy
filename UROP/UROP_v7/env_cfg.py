#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v7/env_cfg.py]
import os
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
class dj_urop_v7_SceneCfg(InteractiveSceneCfg):
    """Catch-only scene (G1 + a bulky object)."""

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

    # Contact sensors (palms often have NO contact reporter in official g1.usd -> use wrists/elbows/torso)
    contact_torso = scene_objects_cfg.contact_torso_cfg
    contact_l_shoulder_pitch = scene_objects_cfg.contact_l_shoulder_pitch_cfg
    contact_l_shoulder_roll = scene_objects_cfg.contact_l_shoulder_roll_cfg
    contact_l_shoulder_yaw = scene_objects_cfg.contact_l_shoulder_yaw_cfg
    contact_l_elbow = scene_objects_cfg.contact_l_elbow_cfg
    contact_l_hand = scene_objects_cfg.contact_l_hand_cfg
    contact_r_shoulder_pitch = scene_objects_cfg.contact_r_shoulder_pitch_cfg
    contact_r_shoulder_roll = scene_objects_cfg.contact_r_shoulder_roll_cfg
    contact_r_shoulder_yaw = scene_objects_cfg.contact_r_shoulder_yaw_cfg
    contact_r_elbow = scene_objects_cfg.contact_r_elbow_cfg
    contact_r_hand = scene_objects_cfg.contact_r_hand_cfg


@configclass
class CommandsCfg:
    # Receive-only environment: no locomotion command.
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
        scale=0.16,
    )
    legs_yaw = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
        scale=0.07,
    )
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        scale=0.14,
    )
    left_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_pitch_joint", "left_elbow_joint"],
        scale=0.60,
    )
    right_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=0.60,
    )
    left_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        ],
        scale=0.35,
    )
    right_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        scale=0.35,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # phase signals
        toss_signal = ObsTerm(func=mdp.toss_state)
        hold_signal = ObsTerm(func=mdp.hold_state)
        hold_anchor_err = ObsTerm(func=mdp.hold_anchor_error, params={"scale": 1.0})

        # robot proprio (gravity dir + base vel + controlled joints pos/vel/torque)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)

        # object relative pose/vel in base frame
        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "drop_prob": 0.0, "noise_std": 0.0},
        )
        # domain randomization parameters (normalized)
        obj_params = ObsTerm(func=mdp.object_params)

        # contact magnitudes from multiple sensors
        contact = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_names": [
                    "contact_torso",
                    "contact_l_shoulder_pitch",
                    "contact_l_shoulder_roll",
                    "contact_l_shoulder_yaw",
                    "contact_l_elbow",
                    "contact_l_hand",
                    "contact_r_shoulder_pitch",
                    "contact_r_shoulder_roll",
                    "contact_r_shoulder_yaw",
                    "contact_r_elbow",
                    "contact_r_hand",
                ],
                "scale": 1.0 / 300.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # base stabilization
    alive = RewTerm(func=mdp.alive_bonus, weight=0.50)
    upright = RewTerm(func=mdp.upright_reward, weight=1.00)
    height = RewTerm(func=mdp.root_height_reward, weight=1.00, params={"target_z": 0.78, "sigma": 0.12})

    base_vel = RewTerm(func=mdp.base_velocity_penalty, weight=-0.15, params={"w_lin": 1.0, "w_ang": 0.35})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.0012)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00002)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.012)

    # before toss: stand still + keep receive-ready posture
    ready_pose_wait = RewTerm(func=mdp.ready_pose_when_waiting, weight=3.0, params={"sigma": 0.40})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-3.0, params={"sigma": 0.18})

    # catching / receiving
    reach = RewTerm(func=mdp.torso_reach_object_reward, weight=0.7, params={"sigma": 0.75})
    hands_reach = RewTerm(func=mdp.hands_reach_object_reward, weight=1.0, params={"sigma": 0.35})
    support_under = RewTerm(func=mdp.hands_support_under_box_reward, weight=1.0, params={"sigma": 0.16})

    hold_pose = RewTerm(
        func=mdp.hold_pose_reward,
        weight=2.5,
        params={"torso_body_name": "torso_link", "sigma": 0.22, "target_offset": (0.05, 0.0, 0.05)},
    )
    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=1.0, params={"torso_body_name": "torso_link", "sigma": 0.60})
    contact_symmetric = RewTerm(
        func=mdp.contact_hold_bonus_symmetric,
        weight=2.0,
        params={
            "sensor_names_left": ["contact_l_elbow", "contact_l_hand"],
            "sensor_names_right": ["contact_r_elbow", "contact_r_hand"],
            "sensor_names_torso": ["contact_torso"],
            "thr": 2.0,
        },
    )
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.0, params={"min_z": 0.45, "max_dist": 2.2})

    # impact penalty: discourage violent collision spikes
    impact = RewTerm(
        func=mdp.impact_peak_penalty,
        weight=-0.003,
        params={
            "sensor_names": [
                "contact_torso",
                "contact_l_shoulder_pitch",
                "contact_l_shoulder_roll",
                "contact_l_shoulder_yaw",
                "contact_l_elbow",
                "contact_l_hand",
                "contact_r_shoulder_pitch",
                "contact_r_shoulder_roll",
                "contact_r_shoulder_yaw",
                "contact_r_elbow",
                "contact_r_hand",
            ],
            "force_thr": 320.0,
        },
    )

    # latch success: once we believe the object is truly caught
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=3.0)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=3.0, params={"min_steps": 30})

    # 핵심: once caught, stop shuffling / stop walking away
    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=3.0, params={"lin_sigma": 0.14, "yaw_sigma": 0.45})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-3.0, params={"sigma": 0.12})
    post_hold_leg_motion = RewTerm(func=mdp.post_hold_leg_motion_penalty, weight=-0.06)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.45, "max_tilt_deg": 60.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.20, "max_dist": 3.0})
    post_hold_runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.40})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # park object away (to remove early-contact exploitation) + randomize physics
    reset_object_parked = EventTerm(
        func=mdp.reset_object_parked,
        mode="reset",
        params={
            "park": {"pos_x": (-0.05, 0.10), "pos_y": (1.10, 1.35), "pos_z": (-0.60, -0.55)},
        },
    )

    # toss/handover once per episode
    toss = EventTerm(
        func=mdp.toss_object_relative_curriculum,
        mode="interval",
        interval_range_s=(0.9, 2.2),
        params={
            "max_throws_per_episode": 1,
            "throw_prob_stage1": 1.0,
            "throw_prob_stage2": 0.90,

            # stage0: super-easy handover (almost no impact)
            "stage0": {
                "pos_x": (0.46, 0.56), "pos_y": (-0.03, 0.03), "pos_z": (0.34, 0.42),
                "vel_x": (-0.35, -0.15), "vel_y": (-0.05, 0.05), "vel_z": (-0.03, 0.05),
            },
            # stage1: gentle throw
            "stage1": {
                "pos_x": (0.50, 0.64), "pos_y": (-0.06, 0.06), "pos_z": (0.30, 0.44),
                "vel_x": (-0.95, -0.70), "vel_y": (-0.10, 0.10), "vel_z": (-0.02, 0.12),
            },
            # stage2: harder throw (more speed + slightly wider lateral)
            "stage2": {
                "pos_x": (0.52, 0.72), "pos_y": (-0.10, 0.10), "pos_z": (0.30, 0.46),
                "vel_x": (-1.45, -0.95), "vel_y": (-0.14, 0.14), "vel_z": (-0.04, 0.14),
            },
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 600,
            "stage1_iters": 2000,
            "num_steps_per_env": 96,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v7_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v7_SceneCfg = dj_urop_v7_SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # sim dt / decimation
        self.decimation = 2
        self.episode_length_s = 7.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # PhysX stability (optional; avoids noisy velocity updates warning)
        try:
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
                self.sim.physx.enable_external_forces_every_iteration = True
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "num_velocity_iterations"):
                self.sim.physx.num_velocity_iterations = 1
        except Exception:
            pass

        # allow env-vars to override curriculum quickly
        p = self.curriculum.stage_schedule.params
        p["stage0_iters"] = int(os.environ.get("UROP_STAGE0_ITERS", str(p["stage0_iters"])))
        p["stage1_iters"] = int(os.environ.get("UROP_STAGE1_ITERS", str(p["stage1_iters"])))
        p["num_steps_per_env"] = int(os.environ.get("UROP_NUM_STEPS_PER_ENV", str(p["num_steps_per_env"])))
        p["eval_stage"] = int(os.environ.get("UROP_EVAL_STAGE", str(p.get("eval_stage", -1))))