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
# import isaaclab.envs.mdp as mdp

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
    #simple_reward = RewTerm(
    #    func=mdp.reward_example,
    #    params={"ft_name": "ft_sensor_example"},
    #    weight=1.0
    #)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)



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



