from dataclasses import MISSING
import math
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, ContactSensor, SensorBaseCfg

from isaaclab.assets import RigidObject, RigidObjectCfg

from isaaclab.controllers import DifferentialIKControllerCfg

import torch
import math
import os
from . import scene_objects_cfg
from . import mdp as mdp
import isaaclab.envs.mdp as isaac_mdp

import isaaclab.sim as sim_utils

##
# Pre-defined configs
##


##
# Scene definition
##

@configclass
class dj_urop_SceneCfg(InteractiveSceneCfg):
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos = (0.0, 0.0, 0.0),
            rot = (1.0, 0.0, 0.0, 0.0),
        ),
    )

    # #sky
    sky = AssetBaseCfg(
        prim_path="/World/sky",
        spawn=sim_utils.UsdFileCfg(usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Environments/Hospital/Props/SM_SkySphere.usd")
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

    robot = scene_objects_cfg.dj_robot_cfg
    object = scene_objects_cfg.bulky_object_cfg

    # UROP/UROP_v0/env_cfg.py (dj_urop_SceneCfg 안)

    contact_torso = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    contact_lhand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_roll_rubber_hand",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    contact_rhand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_roll_rubber_hand",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    command=mdp.NullCommandCfg()

@configclass
class ActionsCfg:
    # 1) 다리 HOLD (Go2 때처럼 scale=0.0로 “정지” 느낌)
    #    ⚠️ 단점: action 차원은 그대로 커짐(학습이 약간 느려질 수 있음)
    legs_hold = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ],
        scale=0.85,
    )

    # 2) 허리: 조금만
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint"],
        scale=0.4,
    )

    # 3) 팔: policy가 제어 (shoulder 크게, elbow 중간, wrist는 roll만 있음)
    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
        ],
        scale=1.5,
    )

    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint",
        ],
        scale=1.5,
    )




@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.robot_proprio)
        contact = ObsTerm(
        func=mdp.contact_forces,
        params={"sensor_names": ["contact_torso", "contact_lhand", "contact_rhand"], "scale": 1.0/300.0},
        )
        obj_rel = ObsTerm(
        func=mdp.object_rel_state,
        params={"drop_prob": 0.0, "noise_std": 0.0, "pos_scale": 1.0, "vel_scale": 1.0},
        )
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # stage-scaled inside reward fn (env_cfg weight는 부호만 담당)
    alive = RewTerm(func=mdp.alive_bonus_curriculum, weight=1.0, params={"w0": 0.2, "w1": 0.05, "w2": 0.02})
    height = RewTerm(
        func=mdp.root_height_reward_curriculum,
        weight=1.0,
        params={"target_z": 0.78, "sigma": 0.08, "w0": 1.0, "w1": 0.2, "w2": 0.1},
    )
    base_vel = RewTerm(
        func=mdp.base_velocity_penalty_curriculum,
        weight=-0.5,
        params={"w_lin": 1.0, "w_ang": 0.2, "w0": 0.2, "w1": 0.05, "w2": 0.03},
    )

    hold = RewTerm(
        func=mdp.hold_object_close_curriculum,
        weight=1.0,
        params={"sigma": 0.7, "w0": 0.0, "w1": 2.0, "w2": 2.0},
    )
    not_drop = RewTerm(
        func=mdp.object_not_dropped_bonus_curriculum,
        weight=3.0,
        params={"min_z": 0.25, "w0": 0.0, "w1": 0.5, "w2": 0.5},
    )

    impact = RewTerm(
        func=mdp.impact_peak_penalty_curriculum,
        weight=-0.00,
        params={
            "sensor_names": ["contact_torso", "contact_lhand", "contact_rhand"],
            "force_thr_stage1": 400.0,
            "force_thr_stage2": 300.0,
            "w0": 0.0,
            "w1": 0.03,
            "w2": 0.05,
        },
    )

    action_rate = RewTerm(
        func=mdp.action_rate_penalty_curriculum,
        weight=-0.05,
        params={"w0": 0.05, "w1": 0.02, "w2": 0.01},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen, params={"min_root_z": 0.55})
    drop = DoneTerm(func=mdp.object_dropped_curriculum, params={"min_z": 0.55})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base_vel = EventTerm(
        func=mdp.reset_robot_base_velocity_curriculum,
        mode="reset",
        params={
            "asset_name": "robot",
            "stage0": {"lin_x": (-1.0, 0.6), "lin_y": (-0.4, 0.4), "yaw_rate": (-1.5, 1.5)},
            "stage1": {"lin_x": (-0.2, 0.2), "lin_y": (-0.2, 0.2), "yaw_rate": (-0.8, 0.8)},
            "stage2": {"lin_x": (-0.15, 0.15), "lin_y": (-0.1, 0.1), "yaw_rate": (-0.4, 0.4)},
        },
    )

    toss = EventTerm(
        func=mdp.reset_and_toss_object_curriculum,
        mode="reset",
        params={
            "asset_name": "object",
            "stage0": {  # toss off
                "pos_x": (2.0, 2.4), "pos_y": (-0.2, 0.2), "pos_z": (0.25, 0.35),
                "vel_x": (0.0, 0.0), "vel_y": (0.0, 0.0), "vel_z": (0.0, 0.0),
            },
            "stage1": {  # gentle
                "pos_x": (0.30, 0.35), "pos_y": (-0.15, 0.15), "pos_z": (0.9, 1.1),
                "vel_x": (-0.8, -0.5), "vel_y": (-0.2, 0.2), "vel_z": (-0.1, 0.1),
            },
            "stage2": {  # harder
                "pos_x": (0.3, 0.5), "pos_y": (-0.15, 0.15), "pos_z": (0.9, 1.2),
                "vel_x": (-2.0, -0.8), "vel_y": (-0.3, 0.3), "vel_z": (-0.2, 0.2),
            },
        },
    )

    push_recovery = EventTerm(
        func=mdp.push_robot_velocity_impulse,
        mode="interval",
        interval_range_s=(0.8, 2.0),
        params={
            "asset_name": "robot",
            "lin_vel_xy": (0.25, 0.75),
            "lin_vel_z": (0.00, 0.10),
            "yaw_rate": (-0.8, 0.8),
            "stage0_scale": 1.0,
            "stage1_scale": 0.6,
            "stage2_scale": 0.5,
            "body_names": ["torso_link"],  # (A) 외력 버전에서 쓰려고 남겨둠
        },
    )


@configclass
class CurriculumCfg:
    """CurriculumManager가 파싱하는 항목들은 전부 CurriculumTermCfg여야 함."""

    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,   # 아래에서 mdp/curriculum.py에 만들 함수
        params={
            "stage0_iters": 2000,
            "stage1_iters": 4000,
            "num_steps_per_env": 96,   # runner의 rollout length와 반드시 동일
            "eval_stage": -1,          # -1이면 자동, 0/1/2면 강제 고정(play용)
        },
    )


##
# Environment configuration
##

@configclass
class dj_urop_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: dj_urop_SceneCfg = dj_urop_SceneCfg(num_envs=64, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2 #몇 번의 dt마다 강화학습 step이 진행될지 ex) decimation=2 dt=1/120이면, physics는 120fps로 진행되지만 reward 계산 등 학습의 step은 60fps
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (1920, 1080)
        # simulation settings
        self.sim.dt = 1 / 120 #physics가 계산되는 dt
        self.sim.render_interval = self.decimation

        # ---- teacher/student observation gating (dims 유지) ----
        mode = os.environ.get("UROP_MODE", "teacher").lower()
        if mode == "teacher":
            self.observations.policy.obj_rel.params["drop_prob"] = 0.0
        else:
            self.observations.policy.obj_rel.params["drop_prob"] = 1.0

                # ---- curriculum schedule params from env vars ----
        p = self.curriculum.stage_schedule.params
        p["stage0_iters"] = int(os.environ.get("UROP_STAGE0_ITERS", str(p["stage0_iters"])))
        p["stage1_iters"] = int(os.environ.get("UROP_STAGE1_ITERS", str(p["stage1_iters"])))
        p["num_steps_per_env"] = int(os.environ.get("UROP_NUM_STEPS_PER_ENV", str(p["num_steps_per_env"])))
        p["eval_stage"] = int(os.environ.get("UROP_EVAL_STAGE", str(p.get("eval_stage", -1))))
