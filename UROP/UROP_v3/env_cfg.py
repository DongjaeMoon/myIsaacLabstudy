#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/env_cfg.py]
import os
from dataclasses import MISSING
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
from isaaclab.sensors import ContactSensorCfg

from . import scene_objects_cfg
from . import mdp as mdp


@configclass
class dj_urop_v3_SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        #prim_path="{ENV_REGEX_NS}/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(150.0, 150.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot = scene_objects_cfg.dj_robot_cfg
    object = scene_objects_cfg.bulky_object_cfg
    # -------------------------
    # Contact sensors
    # -------------------------
    #contact_object_ground = scene_objects_cfg.contact_object_ground_cfg
    contact_torso = scene_objects_cfg.contact_torso_cfg
    contact_l_shoulder_pitch = scene_objects_cfg.contact_l_shoulder_pitch_cfg
    contact_l_shoulder_roll  = scene_objects_cfg.contact_l_shoulder_roll_cfg
    contact_l_shoulder_yaw   = scene_objects_cfg.contact_l_shoulder_yaw_cfg
    contact_l_elbow          = scene_objects_cfg.contact_l_elbow_cfg
    contact_l_hand           = scene_objects_cfg.contact_l_hand_cfg
    contact_r_shoulder_pitch = scene_objects_cfg.contact_r_shoulder_pitch_cfg
    contact_r_shoulder_roll  = scene_objects_cfg.contact_r_shoulder_roll_cfg
    contact_r_shoulder_yaw   = scene_objects_cfg.contact_r_shoulder_yaw_cfg
    contact_r_elbow          = scene_objects_cfg.contact_r_elbow_cfg
    contact_r_hand           = scene_objects_cfg.contact_r_hand_cfg

@configclass
class CommandsCfg:
    command = mdp.NullCommandCfg()

@configclass
class ActionsCfg:
    # -------------------------
    # LEGS: role-based split
    # -------------------------
    legs_sagittal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
        ],
        # 앞뒤 균형/충격 흡수(무릎/발목)용: 너무 작으면 여전히 넘어짐, 너무 크면 펌핑/쪼그려 치팅
        scale=0.35,   # 시작 추천: 0.25~0.35 (fine-tune면 0.25부터)
    )

    legs_frontal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_roll_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_ankle_roll_joint",
        ],
        # 좌우 균형은 민감함: 크게 주면 바로 와장창
        scale=0.22,   # 시작 추천: 0.15~0.25
    )

    legs_yaw = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
        ],
        # 제자리 캐칭에서는 비틀림이 독이 되기 쉬움 -> 매우 작게
        scale=0.10,   # 시작 추천: 0.05~0.12
    )

    # -------------------------
    # WAIST
    # -------------------------
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint"],
        # 상체 방향 조절은 필요하지만, 과하면 비틀림으로 넘어짐
        scale=0.25,   # 기존 0.3보다 살짝 보수적으로 추천
    )

    # -------------------------
    # ARMS: capture vs wrap split
    # -------------------------
    # (1) "가져오기/포획" 축: shoulder_pitch + elbow (크게 움직여야 박스를 받아냄)
    left_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_pitch_joint", "left_elbow_joint"],
        scale=1.2,    # elbow를 크게 쓰되 너무 난폭하면 충돌/튐
    )
    right_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=1.2,
    )

    # (2) "감싸기/정렬" 축: shoulder_roll/yaw + wrist_roll (미세 조정)
    left_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint"],
        scale=0.7,    # 여기 너무 크면 어깨 쪽으로 치팅/이상한 잡기 나오기 쉬움
    )
    right_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint"],
        scale=0.7,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0/80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)

        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "drop_prob": 0.0, "noise_std": 0.0},
        )

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
                "scale": 1.0/300.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.alive_bonus_curriculum, weight=1.0, params={"w0": 0.2, "w1": 0.05, "w2": 0.02})
    upright = RewTerm(func=mdp.upright_reward_curriculum, weight=0.8, params={"w0": 1.0, "w1": 1.0, "w2": 1.0})
    height = RewTerm(func=mdp.root_height_reward_curriculum, weight=1.0, params={"target_z": 0.78, "sigma": 0.10, "w0": 1.0, "w1": 0.5, "w2": 0.3})

    base_vel = RewTerm(func=mdp.base_velocity_penalty_curriculum, weight=-0.1, params={"w0": 0.2, "w1": 0.08, "w2": 0.06})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty_curriculum, weight=-0.0001)
    torque = RewTerm(func=mdp.torque_l2_penalty_curriculum, weight=-0.00001)

    # catch/hold
    reach = RewTerm(func=mdp.torso_reach_object_reward_curriculum, weight=0.6, params={"sigma": 0.9, "w0": 0.0, "w1": 1.0, "w2": 0.8})
    #hold_pose = RewTerm(func=mdp.hold_pose_reward_curriculum, weight=1.0, params={"target_offset": (0.50, 0.0, 1.00), "sigma": 0.30})
    hold_pose = RewTerm(
    func=mdp.hold_pose_reward_curriculum,
    weight=3.0,
    params={
        "torso_body_name": "torso_link",  # 네 body_names에 맞게 필요시 수정
        "sigma": 0.35,
        # w0,w1,w2는 안 넣어도 기본값 사용됨. 넣고 싶으면 아래처럼:
        # "w0": 0.0, "w1": 2.0, "w2": 2.5,
    },
    )
    #hold_vel = RewTerm(func=mdp.hold_object_vel_reward_curriculum, weight=0.8, params={"sigma": 0.8})
    hold_vel = RewTerm(
    func=mdp.hold_object_vel_reward_curriculum,
    weight=1.0,
    params={
        "torso_body_name": "torso_link",
        "sigma": 0.8,
    },
    )

    contact_hold = RewTerm(
        func=mdp.contact_hold_bonus_curriculum,
        weight=1.5,
        params={
            "sensor_names": [
            "contact_torso",
            #"contact_l_shoulder_pitch",
            #"contact_l_shoulder_roll",
            #"contact_l_shoulder_yaw",
            "contact_l_elbow",
            "contact_l_hand",
            #"contact_r_shoulder_pitch",
            #"contact_r_shoulder_roll",
            #"contact_r_shoulder_yaw",
            "contact_r_elbow",
            "contact_r_hand",
            ],
            "thr": 1.0,
        },
    )

    not_drop = RewTerm(func=mdp.object_not_dropped_bonus_curriculum, weight=1.0, params={"min_z": 0.70})
    impact = RewTerm(
        func=mdp.impact_peak_penalty_curriculum,
        weight=-0.0005,
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
            "force_thr_stage1": 400.0,
            "force_thr_stage2": 300.0,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_penalty_curriculum, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #fall = DoneTerm(func=mdp.robot_fallen, params={"min_root_z": 0.55, "min_upright": 0.4})
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.30, "max_tilt_deg": 66.4})
    drop = DoneTerm(func=mdp.object_dropped_curriculum, params={"min_z": 0.30, "max_dist": 3.0})
    # env_cfg.py : TerminationsCfg 내부 drop 교체
    #drop = DoneTerm(
    #func=mdp.object_ground_contact_curriculum,
    #params={"sensor_name": "contact_object_ground", "force_thr": 1.0},
    #)



@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base_vel = EventTerm(
        func=mdp.reset_robot_base_velocity_curriculum,
        mode="reset",
        params={
            #"asset_name": "robot",
            "stage0": {"lin_x": (0.0, 0.0), "lin_y": (0.0, 0.0), "yaw_rate": (0.0, 0.0)},
            "stage1": {"lin_x": (0.0, 0.0), "lin_y": (0.0, 0.0), "yaw_rate": (0.0, 0.0)},
            "stage2": {"lin_x": (0.0, 0.0), "lin_y": (0.0, 0.0), "yaw_rate": (0.0, 0.0)},
        },
    )

    # v1 toss ranges를 그대로 유지(단, robot-relative로 rotate)
    toss = EventTerm(
        func=mdp.reset_and_toss_object_relative_curriculum,
        mode="reset",
        params={
            #"asset_name": "object",
            "stage0": {  # off (멀리)
                "pos_x": (1.5, 1.6), "pos_y": (0.0, 0.1), "pos_z": (0.25, 0.27),
                "vel_x": (0.0, 0.0), "vel_y": (0.0, 0.0), "vel_z": (0.0, 0.0),
            },
            "stage1": {  # gentle (v1 감 그대로)  :contentReference[oaicite:2]{index=2}
                "pos_x": (0.30, 0.35), "pos_y": (-0.05, 0.05), "pos_z": (0.3, 0.4),
                "vel_x": (-0.8, -0.5), "vel_y": (-0.1, 0.1), "vel_z": (-0.1, 0.1),
            },
            "stage2": {  # harder (v1 감 그대로)  :contentReference[oaicite:3]{index=3}
                "pos_x": (0.3, 0.5), "pos_y": (-0.1, 0.1), "pos_z": (0.3, 0.4),
                "vel_x": (-2.0, -0.8), "vel_y": (-0.1, 0.1), "vel_z": (-0.2, 0.2),
            },
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 2000,
            "stage1_iters": 4000,
            "num_steps_per_env": 96,
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_v3_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v3_SceneCfg = dj_urop_v3_SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # curriculum env vars (v1/v2에서 쓰던 방식 그대로)  :contentReference[oaicite:4]{index=4}
        p = self.curriculum.stage_schedule.params
        p["stage0_iters"] = int(os.environ.get("UROP_STAGE0_ITERS", str(p["stage0_iters"])))
        p["stage1_iters"] = int(os.environ.get("UROP_STAGE1_ITERS", str(p["stage1_iters"])))
        p["num_steps_per_env"] = int(os.environ.get("UROP_NUM_STEPS_PER_ENV", str(p["num_steps_per_env"])))
        p["eval_stage"] = int(os.environ.get("UROP_EVAL_STAGE", str(p.get("eval_stage", -1))))
