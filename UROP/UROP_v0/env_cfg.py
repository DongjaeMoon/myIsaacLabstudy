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

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        # object만 필터 (가능하면 켜두는 걸 추천)
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
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
        contact = ObsTerm(func=mdp.contact_forces)
        obj_rel = ObsTerm(func=mdp.object_rel_state)   # 초기엔 넣고, 나중에 student에선 빼도 됨

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    hold = RewTerm(func=mdp.hold_object_close, weight=2.0, params={"sigma": 0.7})
    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=0.5, params={"min_z": 0.25})
    impact = RewTerm(func=mdp.impact_peak_penalty, weight=-1.0, params={"force_thr": 250.0})
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen, params={"min_root_z": 0.55})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.2})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    toss = EventTerm(func=mdp.reset_and_toss_object, mode="reset")
    
    



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
    scene: dj_urop_SceneCfg = dj_urop_SceneCfg(num_envs=64, env_spacing=4.0)
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



