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


STATE_BANK_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v2/carry_state_bank.pt"


@configclass
class dj_urop_carry_v2_SceneCfg(InteractiveSceneCfg):
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
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
        ],
        scale=0.3,
    )
    legs_frontal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_roll_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_ankle_roll_joint",
        ],
        scale=0.2,
    )
    legs_yaw = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
        scale=0.1,
    )
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        scale=0.2,
    )
    left_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_pitch_joint", "left_elbow_joint"],
        scale=0.45,
    )
    right_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=0.45,
    )
    left_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        ],
        scale=0.25,
    )
    right_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        scale=0.25,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        carry_cmd = ObsTerm(func=mdp.carry_command)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        obj_rel = ObsTerm(func=mdp.object_rel_state, params={"pos_scale": 1.0, "vel_scale": 1.0, "noise_std": 0.0})

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        carry_cmd = ObsTerm(func=mdp.carry_command)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)
        obj_rel = ObsTerm(func=mdp.object_rel_state, params={"pos_scale": 1.0, "vel_scale": 1.0, "noise_std": 0.0})
        obj_params = ObsTerm(func=mdp.object_params)
        contact = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_names": [
                    "contact_torso",
                    "contact_l_shoulder_yaw", "contact_l_elbow",
                    "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
                    "contact_r_shoulder_yaw", "contact_r_elbow",
                    "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand",
                ],
                "scale": 1.0 / 300.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.alive_bonus, weight=0.5)
    upright = RewTerm(func=mdp.upright_reward, weight=1.5)
    height = RewTerm(func=mdp.root_height_reward, weight=1.0, params={"target_z": 0.78, "sigma": 0.10})

    track_lin = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"sigma": 0.25})
    track_yaw = RewTerm(func=mdp.track_ang_vel_z_exp, weight=1.2, params={"sigma": 0.35})

    obj_center = RewTerm(func=mdp.object_center_reward, weight=2.5, params={"sigma_xyz": (0.14, 0.12, 0.16)})
    obj_upright = RewTerm(func=mdp.object_upright_reward, weight=1.4, params={"max_tilt_deg": 35.0})
    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=1.6, params={"torso_body_name": "torso_link", "sigma": 0.35})
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=2.0,
        params={
            "sensor_names_left": [
                "contact_l_shoulder_yaw", "contact_l_elbow",
                "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
            ],
            "sensor_names_right": [
                "contact_r_shoulder_yaw", "contact_r_elbow",
                "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand",
            ],
            "sensor_name_torso": "contact_torso",
            "thr": 2.0,
        },
    )

    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.03)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00005)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.08)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.45, "max_tilt_deg": 60.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.25, "max_dist": 1.20})
    object_tilt = DoneTerm(func=mdp.object_tilt_exceeded, params={"max_tilt_deg": 60.0})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_from_state_bank = EventTerm(
        func=mdp.reset_from_state_bank,
        mode="reset",
        params={
            "state_bank_path": STATE_BANK_PATH,
            "stage0_cmd": {"zero": True},
            "stage1_cmd": {"vx": (0.00, 0.10), "vy": (0.00, 0.00), "wz": (0.00, 0.00)},
            "stage2_cmd": {"vx": (0.05, 0.25), "vy": (-0.05, 0.05), "wz": (-0.20, 0.20)},
            "stage3_cmd": {"vx": (0.10, 0.40), "vy": (-0.12, 0.12), "wz": (-0.35, 0.35)},
            "xy_noise": 0.01,
            "yaw_noise_deg": 0.0,
            "joint_noise": 0.01,
            "object_linvel_noise": 0.03,
            "object_angvel_noise": 0.08,
        },
    )

    resample_carry_cmd = EventTerm(
        func=mdp.resample_carry_command,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={
            "stage0": {"zero": True},
            "stage1": {"vx": (0.00, 0.10), "vy": (0.00, 0.00), "wz": (0.00, 0.00)},
            "stage2": {"vx": (0.05, 0.25), "vy": (-0.05, 0.05), "wz": (-0.20, 0.20)},
            "stage3": {"vx": (0.10, 0.40), "vy": (-0.12, 0.12), "wz": (-0.35, 0.35)},
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 300,
            "stage1_iters": 700,
            "stage2_iters": 1200,
            "num_steps_per_env": 32,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_carry_v2_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_carry_v2_SceneCfg = dj_urop_carry_v2_SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
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
class dj_urop_carry_v2_EnvCfg_Play(dj_urop_carry_v2_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.stage_schedule.params["eval_stage"] = 3
        self.events.resample_carry_cmd.params["stage3"] = {"vx": (0.20, 0.20), "vy": (0.00, 0.00), "wz": (0.00, 0.00)}
