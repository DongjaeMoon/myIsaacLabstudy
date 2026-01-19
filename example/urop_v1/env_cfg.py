from __future__ import annotations

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg

from .scene_objects_cfg import DJ_ROBOT_CFG, ball_cfg, arm_tip_contact_sensor_cfg, goal_post_cfg
import isaaclab.envs.mdp as isaac_mdp
from . import mdp as mdp


# =========================
# Task constants
# =========================
EE_BODY_NAME = "arm_link2"
EE_TIP_OFFSET = (-0.16, 0.0, 0.0)

@configclass
class dj_urop_SceneCfg(InteractiveSceneCfg):
    # 1. Ground Plane (v0 복원)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 2. Sky (v0 복원)
    sky = AssetBaseCfg(
        prim_path="/World/sky",
        spawn=sim_utils.UsdFileCfg(
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Environments/Hospital/Props/SM_SkySphere.usd"
        )
    )

    # 3. Lights (v0 복원)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

    # 4. Robot & Objects
    robot: ArticulationCfg = DJ_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    goal_post: RigidObjectCfg = goal_post_cfg
    target_ball: RigidObjectCfg = ball_cfg
    arm_tip_contact = arm_tip_contact_sensor_cfg


@configclass
class ActionsCfg:
    # 팔 가동범위 확대 (Scale 2.5)
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_joint", "arm_joint1", "arm_joint2"],
        scale=2.5, 
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 몸 상태
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=["shoulder_joint", "arm_joint1", "arm_joint2"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="robot", joint_names=["shoulder_joint", "arm_joint1", "arm_joint2"])},
        )

        # 공 상태 (Yaw 회전만 고려)
        ball_pos = ObsTerm(func=mdp.object_pos_rel, params={"asset_name": "target_ball", "yaw_only": True})
        ball_vel = ObsTerm(func=mdp.object_lin_vel_rel, params={"asset_name": "target_ball", "yaw_only": True})

        # 핵심: 공-손끝 상대 벡터
        ball_tip_rel = ObsTerm(
            func=mdp.ee_tip_pos_rel,
            params={"asset_name": "target_ball", "ee_body_name": EE_BODY_NAME, "tip_offset": EE_TIP_OFFSET, "yaw_only": True},
        )

        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # Reset 시 공 발사
    shoot_ball = EventTerm(
        func=mdp.shoot_ball_towards_body,
        mode="reset",
        params=dict(
            asset_name="target_ball",
            robot_name="robot",
            target_body_name="shoulder_link",
            x_offset=2.5,
            y_range=(-0.4, 0.4),
            z_range=(0.6, 1.2),
            speed_range=(1.0, 1.5),
            aim_noise_y=0.2,
            aim_noise_z=0.2,
        ),
    )


@configclass
class RewardsCfg:
    # 닿으면 100점
    save_ball = RewTerm(
        func=mdp.save_ball_reward,
        weight=100.0,
        params=dict(sensor_name="arm_tip_contact", min_force=0.1),
    )

    # 추적 보상
    track_ball = RewTerm(
        func=mdp.track_ball_tip_kernel,
        weight=2.0,
        params=dict(
            asset_name="target_ball",
            ee_body_name=EE_BODY_NAME,
            tip_offset=EE_TIP_OFFSET,
            sigma=0.3, 
        ),
    )

    action_rate_penalty = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    # 성공 시 종료 (닿으면 끝)
    save_success = DoneTerm(
        func=mdp.ball_saved_simple,
        time_out=False,
        params=dict(sensor_name="arm_tip_contact", min_force=0.1),
    )

    # 실패 시 종료 (골 먹힘)
    goal_conceded = DoneTerm(
        func=mdp.ball_past_robot,
        params={"asset_name": "target_ball", "robot_name": "robot", "goal_x_offset": -0.2},
    )

    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={"asset_name": "target_ball"},
    )


@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class dj_urop_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_SceneCfg = dj_urop_SceneCfg(num_envs=64, env_spacing=4.0)
    
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 4.0
        
        self.viewer.eye = (-2.5, 2.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        
        self.sim = SimulationCfg(dt=1 / 120, render_interval=self.decimation)