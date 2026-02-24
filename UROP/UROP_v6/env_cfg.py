import os
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


@configclass
class dj_urop_v6_SceneCfg(InteractiveSceneCfg):
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

    # Contact sensors (object와의 접촉만 필터링)
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
    # LEGS
    legs_sagittal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
        ],
        scale=0.5,
    )
    legs_frontal = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_roll_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_ankle_roll_joint",
        ],
        scale=0.22,
    )
    legs_yaw = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_yaw_joint", "right_hip_yaw_joint"],
        scale=0.10,
    )

    # WAIST
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint"],
        scale=0.25,
    )

    # ARMS (capture vs wrap)
    left_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_pitch_joint", "left_elbow_joint"],
        scale=1.2,
    )
    right_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=1.2,
    )
    left_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint"],
        scale=0.7,
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
        toss_signal = ObsTerm(func=mdp.toss_state)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
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
                "scale": 1.0 / 300.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # ---------- stability ----------
    alive = RewTerm(func=mdp.alive_bonus_curriculum, weight=1.0, params={"w0": 0.2, "w1": 0.05, "w2": 0.02})
    upright = RewTerm(func=mdp.upright_reward_curriculum, weight=0.8, params={"w0": 1.0, "w1": 1.0, "w2": 1.0})
    height = RewTerm(func=mdp.root_height_reward_curriculum, weight=1.0, params={"target_z": 0.78, "sigma": 0.20, "w0": 1.0, "w1": 0.5, "w2": 0.3})

    # 너무 강하면 "얼어붙는 정책"이 나오기 쉬움 -> 과도하게 크게 두지 말기
    base_vel = RewTerm(func=mdp.base_velocity_penalty_curriculum, weight=-0.05, params={"w0": 0.2, "w1": 0.08, "w2": 0.06})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty_curriculum, weight=-0.0001)
    torque = RewTerm(func=mdp.torque_l2_penalty_curriculum, weight=-0.00001)
    action_rate = RewTerm(func=mdp.action_rate_penalty_curriculum, weight=-0.02)

    # ---------- WAIT(throw 전) 치팅 차단 ----------
    ready_pose_wait = RewTerm(func=mdp.ready_pose_when_waiting, weight=3.0, params={"sigma": 0.5})
    # [핵심] 뒷걸음/베이스 이동으로 시간 벌기 금지
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-2.0, params={"sigma": 0.25})

    # ---------- catch / hold ----------
    reach = RewTerm(func=mdp.torso_reach_object_reward_curriculum, weight=0.6, params={"sigma": 0.9, "w0": 0.0, "w1": 1.0, "w2": 0.8})
    hands_reach = RewTerm(func=mdp.hands_reach_object_reward_curriculum, weight=1.0, params={"sigma": 0.5})

    # [강추] 밑받침 유도 (사람처럼 받기)
    support_under = RewTerm(
        func=mdp.hands_support_under_box_reward_curriculum,
        weight=1.5,
        params={"box_size": (0.4, 0.3, 0.3), "sigma": 0.18},
    )

    hold_pose = RewTerm(func=mdp.hold_pose_reward_curriculum, weight=2.0, params={"torso_body_name": "torso_link", "sigma": 0.35})
    hold_vel = RewTerm(func=mdp.hold_object_vel_reward_curriculum, weight=1.0, params={"torso_body_name": "torso_link", "sigma": 0.8})

    contact_symmetric = RewTerm(
        func=mdp.contact_hold_bonus_symmetric,
        weight=2.0,
        params={
            "sensor_names_left": ["contact_l_elbow", "contact_l_hand"],
            "sensor_names_right": ["contact_r_elbow", "contact_r_hand"],
            "sensor_names_torso": ["contact_torso"],
            "thr": 1.0,
        },
    )

    not_drop = RewTerm(func=mdp.object_not_dropped_bonus_curriculum, weight=1.0, params={"min_z": 0.70})

    # 충격 완화(슬램 방지)
    impact = RewTerm(
        func=mdp.impact_peak_penalty_curriculum,
        weight=-0.002,
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


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 너무 느슨하면 넘어져도 계속 끌고 가며 이상한 정책이 나옴
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.45, "max_tilt_deg": 60.0})
    drop = DoneTerm(func=mdp.object_dropped_curriculum, params={"min_z": 0.25, "max_dist": 3.0})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_parked = EventTerm(
        func=mdp.reset_object_parked,
        mode="reset",
        params={
            # 로봇 옆(좌측)으로 충분히 멀리 주차 -> throw 전 치팅/충돌 감소
            "park": {"pos_x": (-0.1, 0.1), "pos_y": (1.20, 1.40), "pos_z": (-0.62, -0.58)},
        },
    )

    toss = EventTerm(
        func=mdp.toss_object_relative_curriculum,
        mode="interval",
        interval_range_s=(1.0, 2.5),
        params={
            "max_throws_per_episode": 1,
            "throw_prob_stage1": 1.0,
            "throw_prob_stage2": 0.85,

            "stage0": {  # 내부에서 stage0은 toss 무시됨
                "pos_x": (1.5, 1.6), "pos_y": (0.0, 0.1), "pos_z": (0.25, 0.27),
                "vel_x": (0.0, 0.0), "vel_y": (0.0, 0.0), "vel_z": (0.0, 0.0),
            },
            # throw 위치를 너무 가깝게 두면 "초반 뒷걸음으로 시간벌기"가 이득이 됨 -> 조금 멀리
            "stage1": {
                "pos_x": (0.4, 0.5), "pos_y": (-0.05, 0.05), "pos_z": (0.30, 0.40),
                "vel_x": (-1.1, -0.9), "vel_y": (-0.1, 0.1), "vel_z": (0.05, 0.1),
            },
            "stage2": {
                "pos_x": (0.4, 0.6), "pos_y": (-0.05, 0.05), "pos_z": (0.30, 0.42),
                "vel_x": (-1.5, -1.0), "vel_y": (-0.1, 0.1), "vel_z": (0.05, 0.1),
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
class dj_urop_v6_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v6_SceneCfg = dj_urop_v6_SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 7.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # curriculum env vars
        p = self.curriculum.stage_schedule.params
        p["stage0_iters"] = int(os.environ.get("UROP_STAGE0_ITERS", str(p["stage0_iters"])))
        p["stage1_iters"] = int(os.environ.get("UROP_STAGE1_ITERS", str(p["stage1_iters"])))
        p["num_steps_per_env"] = int(os.environ.get("UROP_NUM_STEPS_PER_ENV", str(p["num_steps_per_env"])))
        p["eval_stage"] = int(os.environ.get("UROP_EVAL_STAGE", str(p.get("eval_stage", -1))))
