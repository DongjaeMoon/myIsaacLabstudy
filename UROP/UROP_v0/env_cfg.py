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
        scale=0.0,
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
    # Stage0(기본 0으로 두고 stage에서 켬)
    alive = RewTerm(func=mdp.alive_bonus, weight=0.0)
    height = RewTerm(func=mdp.root_height_reward, weight=0.0, params={"target_z": 0.78, "sigma": 0.08})
    base_vel = RewTerm(func=mdp.base_velocity_penalty, weight=0.0, params={"w_lin": 1.0, "w_ang": 0.2})

    # Catch task
    hold = RewTerm(func=mdp.hold_object_close, weight=2.0, params={"sigma": 0.7})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=0.5, params={"min_z": 0.25})

    impact = RewTerm(
        func=mdp.impact_peak_penalty,
        weight=-0.05,
        params={"sensor_names": ["contact_torso", "contact_lhand", "contact_rhand"], "force_thr": 400.0},
    )

    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.02)



@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen, params={"min_root_z": 0.55})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.2})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base_vel = EventTerm(
    func=mdp.reset_robot_base_velocity,
    mode="reset",
    params={"lin_x": (-0.6, 0.6), "lin_y": (-0.4, 0.4), "yaw_rate": (-1.5, 1.5)},
    )

    toss = EventTerm(
        func=mdp.reset_and_toss_object,
        mode="reset",
        params={
            "asset_name": "object",
            "pos_x": (0.3, 0.5),
            "pos_y": (-0.15, 0.15),
            "pos_z": (0.9, 1.2),
            "vel_x": (-2.0, -0.8),
            "vel_y": (-0.3, 0.3),
            "vel_z": (-0.2, 0.2),
        },
    )
    



@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


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

        stage = int(os.environ.get("UROP_STAGE", "1"))       # 0/1/2
        mode = os.environ.get("UROP_MODE", "teacher").lower()  # teacher/student

        # teacher/student: GT(obj_rel) 가림 (차원 유지)
        if mode == "teacher":
            self.observations.policy.obj_rel.params["drop_prob"] = 0.0
        else:
            self.observations.policy.obj_rel.params["drop_prob"] = 1.0
   
        # Stage 스케줄
        if stage == 0:
            # ---- balance/recovery ----
            self.actions.legs_hold.scale = 0.25
            self.actions.waist.scale = 0.15
            self.actions.left_arm.scale = 0.0
            self.actions.right_arm.scale = 0.0

            # toss: 사실상 끔(멀리, 속도 0) — 하지만 object는 존재(obs 차원 유지)
            self.events.toss.params.update({
                "pos_x": (2.0, 2.4), "pos_y": (-0.2, 0.2), "pos_z": (0.25, 0.35),
                "vel_x": (0.0, 0.0), "vel_y": (0.0, 0.0), "vel_z": (0.0, 0.0),
            })

            # Stage0 reward ON, catch reward OFF
            self.rewards.alive.weight = 0.2
            self.rewards.height.weight = 1.0
            self.rewards.base_vel.weight = -0.2
            self.rewards.hold.weight = 0.0
            self.rewards.not_drop.weight = 0.0
            self.rewards.impact.weight = 0.0
            self.rewards.action_rate.weight = -0.05
            # Stage0에서는 object가 떨어져도 학습(밸런스/리커버리) 끊기지 않게 drop 종료 비활성화
            self.terminations.drop.params["min_z"] = -10.0


        elif stage == 1:
            # ---- gentle toss catch ----
            self.actions.legs_hold.scale = 0.20
            self.actions.waist.scale = 0.30
            self.actions.left_arm.scale = 1.0
            self.actions.right_arm.scale = 1.0

            self.events.toss.params.update({
                "pos_x": (0.35, 0.55), "pos_y": (-0.15, 0.15), "pos_z": (0.9, 1.1),
                "vel_x": (-0.8, -0.3), "vel_y": (-0.2, 0.2), "vel_z": (-0.1, 0.1),
            })
            self.events.reset_base_vel.params.update({
                "lin_x": (-0.3, 0.3), "lin_y": (-0.2, 0.2), "yaw_rate": (-0.8, 0.8),
            })


            self.rewards.alive.weight = 0.05
            self.rewards.height.weight = 0.2
            self.rewards.base_vel.weight = -0.05
            self.rewards.hold.weight = 2.0
            self.rewards.not_drop.weight = 0.5
            self.rewards.impact.weight = -0.05
            self.rewards.impact.params["force_thr"] = 400.0
            self.rewards.action_rate.weight = -0.02

        else:
            # ---- full toss + shock mitigation ----
            self.actions.legs_hold.scale = 0.40
            self.actions.waist.scale = 0.40
            self.actions.left_arm.scale = 1.5
            self.actions.right_arm.scale = 1.5

            self.events.toss.params.update({
                "pos_x": (0.3, 0.5), "pos_y": (-0.15, 0.15), "pos_z": (0.9, 1.2),
                "vel_x": (-2.0, -0.8), "vel_y": (-0.3, 0.3), "vel_z": (-0.2, 0.2),
            })

            self.events.reset_base_vel.params.update({
                "lin_x": (-0.15, 0.15), "lin_y": (-0.1, 0.1), "yaw_rate": (-0.4, 0.4),
            })

            self.rewards.alive.weight = 0.02
            self.rewards.height.weight = 0.1
            self.rewards.base_vel.weight = -0.03
            self.rewards.hold.weight = 2.0
            self.rewards.not_drop.weight = 0.5
            self.rewards.impact.weight = -0.10
            self.rewards.impact.params["force_thr"] = 300.0
            self.rewards.action_rate.weight = -0.01




