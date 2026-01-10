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
            pos = (0.0, 0.0, -0.76),
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
    # =================================================================
    # [수정된 부분] Contact Sensor 경로 설정
    # =================================================================
    #contact_forces = ContactSensorCfg(
        # Robot 밑에 'go2'나 'dj_arm' 같은 중간 경로가 껴있어도 찾을 수 있게 수정
        # (1) 방법 A: 모든 하위 경로를 다 뒤지도록 (가장 확실함)
        #prim_path="{ENV_REGEX_NS}/Robot/.*", 
        # 만약 위 코드로도 안 된다면 아래 주석을 풀고 써보세요 (경로 깊이 강제 지정)
    #    prim_path="{ENV_REGEX_NS}/Robot/.*/.*_calf",
        
    #    history_length=3, 
    #    track_air_time=True
    #)

    # articulation
    robot: ArticulationCfg = scene_objects_cfg.dj_robot_cfg
    target_ball: RigidObjectCfg = scene_objects_cfg.ball_cfg

    # sensors
    #ft_sensor_example: FrameTransformerCfg = scene_objects_cfg.ft_sensor_example_cfg
    #contact_sensor_example: ContactSensorCfg = scene_objects_cfg.contact_sensor_example_cfg

    # props
    #prop_example = scene_objects_cfg.prop_example_cfg
    #box = scene_objects_cfg.box_cfg

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    command=mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # robot
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=[".*"],
                )
            },
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=[".*"],
                )
            },
            scale=1.0,
        )

        # [추가] 공의 위치 정보
        target_pos = ObsTerm(
            func=mdp.object_pos_rel,
            params={"asset_name": "target_ball"}, # scene_objects_cfg의 이름과 일치해야 함
            scale=1.0,
        )

        # last actions
        actions = ObsTerm(func=mdp.last_action)


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    alive_reward = RewTerm(
        func=mdp.is_alive,
        weight=1.0
    )
    # 2. [추가] 공 근처로 가기 (Base)
    approach_ball = RewTerm(
        func=mdp.distance_to_target,
        params={"asset_name": "target_ball"},
        weight=-1.0, # 거리가 줄어들수록(0에 가까울수록) 페널티가 줄어듦 -> 즉 보상
    )
    
    # 3. [추가] 팔 뻗어서 터치하기 (End Effector)
    # *주의: "hand_link" 부분은 실제 로봇 arm의 끝부분 링크 이름으로 바꿔주세요! (USD 파일 확인 필요)
    reach_ball = RewTerm(
        func=mdp.ee_distance_to_target,
        params={"asset_name": "target_ball", "ee_body_name": "arm_link2"}, 
        weight=-2.0, # 접근보다 더 큰 가중치
    )

    # ----------------------------------------------------
    # [핵심] 걷기 안정화 (Penalty) - 걷는 게 문제라면 이 부분 가중치를 높이세요
    # ----------------------------------------------------
    
    # (1) 로봇이 마구 회전하지 않도록 (Z축 회전 속도 규제)
    ang_vel_xy_penalty = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )
    
    # (2) 로봇이 너무 휘청거리지 않도록 (몸통 방향 규제)
    flat_orientation_penalty = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
    )
    
    # (3) 발을 너무 세게 구르거나 액션이 튀지 않도록
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    # (4) 불필요한 관절 토크 줄이기 (에너지 효율 + 부드러운 움직임)
    torques_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0001,
    )
    #simple_reward = RewTerm(
    #    func=mdp.reward_example,
    #    params={"ft_name": "ft_sensor_example"},
    #    weight=1.0
    #)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # 1. 시간 초과 (기존)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # 2. [추가] 넘어짐 감지 (몸통이 땅에 닿으면 리셋)
    # 로봇의 몸통(base)이 바닥에 닿으면 종료
    #base_contact = DoneTerm(
    #    func=mdp.illegal_contact,
    #    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*base"), "threshold": 1.0},
    #)
    
    # 3. [추가] 높이 기반 넘어짐 감지 (대안)
    # 로봇의 몸통(Root) 높이가 0.25m 밑으로 내려가면 "넘어졌다"고 판단하고 리셋
    bad_height = DoneTerm(
        func=isaac_mdp.root_height_below,
        params={"minimum_height": 0.25}, 
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
    scene: dj_urop_SceneCfg = dj_urop_SceneCfg(num_envs=64, env_spacing=6.0)
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
        self.episode_length_s = 15
        # viewer settings
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (1920, 1080)
        # simulation settings
        self.sim.dt = 1 / 120 #physics가 계산되는 dt
        self.sim.render_interval = self.decimation



