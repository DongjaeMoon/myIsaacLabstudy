from __future__ import annotations

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
from isaaclab.sim import SimulationCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .scene_objects_cfg import GO2_CFG, ball_cfg, arm_tip_contact_sensor_cfg, goal_post_cfg

import isaaclab.envs.mdp as isaac_mdp
from . import mdp as mdp


# =========================
# Task constants
# =========================
EE_BODY_NAME = "arm_link2"

# NOTE: 기존에 쓰던 값 유지. 만약 tip 방향이 반대면 (+0.16,0,0)로 부호만 바꿔봐.
EE_TIP_OFFSET = (-0.16, 0.0, 0.0)

TIP_RADIUS = 0.08   # tip 근처로 판정할 반경 (너무 작으면 성공이 안 잡힘)
MIN_CONTACT_FORCE = 0.1


@configclass
class dj_urop_SceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=None)

    # robot
    robot: ArticulationCfg = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # goal post (visual)
    goal_post: RigidObjectCfg = goal_post_cfg

    # ball
    target_ball: RigidObjectCfg = ball_cfg

    # contact sensor on arm tip link (regex path)
    arm_tip_contact = arm_tip_contact_sensor_cfg


@configclass
class ActionsCfg:
    """Only control the 3 arm joints (CRITICAL: remove Go2 leg joints from action vector)."""
    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_joint", "arm_joint1", "arm_joint2"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # arm state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        # ball state in robot yaw-frame (MARKOV-ize)
        ball_pos = ObsTerm(func=mdp.object_pos_rel, params={"asset_name": "target_ball", "yaw_only": True})
        ball_vel = ObsTerm(func=mdp.object_lin_vel_rel, params={"asset_name": "target_ball", "yaw_only": True})

        # relative vectors: (ball(t+dt) - tip(t)) in robot yaw-frame
        ball_tip_now = ObsTerm(
            func=mdp.ee_tip_pos_rel,
            params={"asset_name": "target_ball", "ee_body_name": EE_BODY_NAME, "tip_offset": EE_TIP_OFFSET, "yaw_only": True},
        )
        ball_tip_future_0p15 = ObsTerm(
            func=mdp.ball_future_pos_rel_to_tip,
            params={
                "asset_name": "target_ball",
                "ee_body_name": EE_BODY_NAME,
                "tip_offset": EE_TIP_OFFSET,
                "time_horizon_s": 0.15,
                "yaw_only": True,
            },
        )
        ball_tip_future_0p30 = ObsTerm(
            func=mdp.ball_future_pos_rel_to_tip,
            params={
                "asset_name": "target_ball",
                "ee_body_name": EE_BODY_NAME,
                "tip_offset": EE_TIP_OFFSET,
                "time_horizon_s": 0.30,
                "yaw_only": True,
            },
        )

        # last action helps smoothing
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # Reset: shoot the ball toward the arm base (reduces unreachable cases)
    shoot_ball = EventTerm(
        func=mdp.shoot_ball_towards_body,
        mode="reset",
        params=dict(
            asset_name="target_ball",
            robot_name="robot",
            target_body_name="shoulder_link",   # 네 URDF 링크 이름
            x_offset=2.2,
            y_range=(-0.25, 0.25),
            z_range=(0.55, 1.05),               # <-- 프레임 보니까 z 낮으면 팔이 못 막음. 올림.
            speed_range=(2.0, 4.0),
            aim_noise_y=0.15,
            aim_noise_z=0.15,
        ),
    )


@configclass
class RewardsCfg:
    # 성공 보상: "tip 근처 접촉"일 때만 1.0 (그리고 아래 Termination으로 바로 끝내서 학습 효율↑)
    save_ball = RewTerm(
        func=mdp.save_ball_reward,
        weight=20.0,
        params=dict(
            sensor_name="arm_tip_contact",
            asset_name="target_ball",
            robot_name="robot",
            ee_body_name=EE_BODY_NAME,
            tip_offset=EE_TIP_OFFSET,
            tip_radius=TIP_RADIUS,
            min_force=MIN_CONTACT_FORCE,
            goal_x_offset=-0.2,
        ),
    )

    # 미래 예측 shaping: arm_joint2가 쭉 뻗는 게 이득이 되게 만드는 핵심
    track_future = RewTerm(
        func=mdp.track_future_ball_tip_kernel,
        weight=4.0,
        params=dict(
            asset_name="target_ball",
            ee_body_name=EE_BODY_NAME,
            tip_offset=EE_TIP_OFFSET,
            time_horizon_s=0.25,
            sigma=0.25,
        ),
    )

    # 현재 위치 shaping (보조)
    track_now = RewTerm(
        func=mdp.track_ball_tip_kernel,
        weight=1.0,
        params=dict(
            asset_name="target_ball",
            ee_body_name=EE_BODY_NAME,
            tip_offset=EE_TIP_OFFSET,
            sigma=0.20,
        ),
    )

    action_rate_penalty = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.01)
    dof_acc_l2 = RewTerm(func=isaac_mdp.joint_acc_l2, weight=-1e-6)


@configclass
class TerminationsCfg:
    # time limit
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)

    # Success: tip-save면 즉시 종료 (로그가 "진짜 성공률"이 됨)
    save_success = DoneTerm(
        func=mdp.ball_saved_tip,
        time_out=False,
        params=dict(
            sensor_name="arm_tip_contact",
            asset_name="target_ball",
            robot_name="robot",
            ee_body_name=EE_BODY_NAME,
            tip_offset=EE_TIP_OFFSET,
            tip_radius=TIP_RADIUS,
            min_force=MIN_CONTACT_FORCE,
            goal_x_offset=-0.2,
        ),
    )

    # Failure if goal conceded
    goal_conceded = DoneTerm(
        func=mdp.ball_past_robot,
        params={"asset_name": "target_ball", "robot_name": "robot", "goal_x_offset": -0.2},
    )

    # Avoid long timeouts when ball flies away / gets stuck
    ball_out_of_bounds = DoneTerm(
        func=mdp.ball_out_of_bounds,
        params={"asset_name": "target_ball", "x_bounds": (-2.0, 4.0), "y_abs_max": 3.0, "z_bounds": (0.0, 3.0)},
    )

    # Safety: robot fell
    bad_height = DoneTerm(func=mdp.root_height_below, params={"minimum_height": 0.25})


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
        self.episode_length_s = 5.0

        self.viewer.eye = (-3.0, 3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        self.sim = SimulationCfg(dt=1 / 120, render_interval=self.decimation)
        self.sim.physx.bounce_threshold_velocity = 0.2
