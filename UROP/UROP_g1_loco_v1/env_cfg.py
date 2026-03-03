import math
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
    SceneEntityCfg
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp_isaac
from . import mdp as custom_mdp
from . import scene_objects_cfg

@configclass
class dj_urop_g1_loco_v1_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot = scene_objects_cfg.dj_robot_cfg

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left_ankle_roll_link|right_ankle_roll_link|torso_link)",
        history_length=3,
        track_air_time=True,
    )

@configclass
class CommandsCfg:
    base_velocity = mdp_isaac.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp_isaac.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0), 
            lin_vel_y=(-0.5, 0.5), # 기본 예제와 동일하게 옆걸음 허용
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    # 기본 예제와 동일하게 모든 관절을 제어하고 스케일을 0.5로 통일하여 상/하체 밸런스를 맞춥니다.
    joint_pos = mdp_isaac.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp_isaac.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp_isaac.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp_isaac.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp_isaac.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp_isaac.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp_isaac.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp_isaac.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    termination_penalty = RewTerm(func=mdp_isaac.is_terminated, weight=-200.0)

    # 1. 목표 추종 (예제 기본값)
    track_lin_vel_xy_exp = RewTerm(
        func=custom_mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    track_ang_vel_z_exp = RewTerm(
        func=custom_mdp.track_ang_vel_z_world_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )

    # 2. 보행 리듬 (예제 기본값)
    feet_air_time = RewTerm(
        func=custom_mdp.feet_air_time_positive_biped, weight=0.75,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]), "threshold": 0.4},
    )

    # 3. 페널티 동기화 (발목 토크 제외가 가장 중요!)
    lin_vel_z_l2 = RewTerm(func=mdp_isaac.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp_isaac.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp_isaac.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp_isaac.action_rate_l2, weight=-0.005)
    
    dof_acc_l2 = RewTerm(
        func=mdp_isaac.joint_acc_l2, weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    dof_torques_l2 = RewTerm(
        func=mdp_isaac.joint_torques_l2, weight=-2.0e-6, 
        # [핵심] 발목 관절을 페널티 대상에서 완전히 제외하여 땅을 힘차게 찰 수 있게 합니다.
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_hip_.*", "right_hip_.*", ".*_knee_joint"])},
    )

    # 4. 팔/허리 고정 페널티 (안 쓰면 팔을 허우적거림)
    joint_deviation_arms = RewTerm(
        func=mdp_isaac.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp_isaac.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])},
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_isaac.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp_isaac.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]), "threshold": 1.0},
    )

@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp_isaac.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp_isaac.reset_joints_by_scale, mode="reset", params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

@configclass
class dj_urop_g1_loco_v1_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_g1_loco_v1_SceneCfg = dj_urop_g1_loco_v1_SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4                 
        self.episode_length_s = 20.0
        self.sim.dt = 1 / 200                  
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt