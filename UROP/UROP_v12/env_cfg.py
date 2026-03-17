#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v12/env_cfg.py]
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
class dj_urop_v12_SceneCfg(InteractiveSceneCfg):
    """Catch-only scene (G1 + a bulky object)."""

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

    # Contact sensors (palms often have NO contact reporter in official g1.usd -> use wrists/elbows/torso)
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
    # Receive-only environment: no locomotion command.
    command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """29-DOF joint position action (finger joints excluded)."""

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
        scale=0.50,
    )
    right_arm_capture = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder_pitch_joint", "right_elbow_joint"],
        scale=0.50,
    )
    left_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        ],
        scale=0.3,
    )
    right_arm_wrap = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        scale=0.3,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # [Actor가 볼 수 있는 현실적인 정보]
        
        # phase signals (카메라 기반으로 박스가 날아오는 것을 인지했다고 가정)
        toss_signal = ObsTerm(func=mdp.toss_state)

        # robot proprio (자신의 관절 각도, 속도, 중력 방향 등)
        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)

        # object relative pose/vel (카메라 비전 트래킹으로 얻을 수 있는 박스의 상대 위치/속도)
        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "drop_prob": 0.0, "noise_std": 0.0},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            # 나중에 Sim-to-Real 할 때 여기에 센서 노이즈를 추가하기 위해 True 유지
            self.enable_corruption = True 

    @configclass
    class CriticCfg(ObsGroup):
        # [Critic만 볼 수 있는 시뮬레이터 내부의 특권 정보 (Privileged Info)]
        
        toss_signal = ObsTerm(func=mdp.toss_state)
        hold_signal = ObsTerm(func=mdp.hold_state)
        hold_anchor_err = ObsTerm(func=mdp.hold_anchor_error, params={"scale": 1.0})

        proprio = ObsTerm(func=mdp.robot_proprio, params={"torque_scale": 1.0 / 80.0})
        prev_actions = ObsTerm(func=mdp.prev_actions)

        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"pos_scale": 1.0, "vel_scale": 1.0, "drop_prob": 0.0, "noise_std": 0.0},
        )
        
        # 도메인 랜덤화 파라미터 (박스 질량, 마찰력 등 - 현실 로봇은 모름)
        obj_params = ObsTerm(func=mdp.object_params)

        # 접촉 센서 (현실 로봇 피부에는 이런 정밀한 센서가 없음)
        contact = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_names": [
                    "contact_torso",
                    "contact_l_shoulder_yaw", "contact_l_elbow",
                    "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
                    "contact_r_shoulder_yaw", "contact_r_elbow",
                    "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand"
                ],
                "scale": 1.0 / 300.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True
            # Critic은 완벽한 정답을 바탕으로 Value를 평가해야 하므로 노이즈 없이(False) 학습
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# env_cfg.py

@configclass
class RewardsCfg:
    # --- [기본 생존 및 제어 안정화] ---
    alive = RewTerm(func=mdp.alive_bonus, weight=0.50)
    upright = RewTerm(func=mdp.upright_reward, weight=1.00)
    height = RewTerm(func=mdp.root_height_reward, weight=1.00, params={"target_z": 0.78, "sigma": 0.12})

    base_vel = RewTerm(func=mdp.base_velocity_penalty, weight=-0.15, params={"w_lin": 1.0, "w_ang": 0.35})
    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty, weight=-0.05)
    torque = RewTerm(func=mdp.torque_l2_penalty, weight=-0.00005)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.1)

    # --- [Stage 0: 박스가 날아오기 전 대기] ---
    ready_pose_wait = RewTerm(func=mdp.ready_pose_when_waiting, weight=3.0, params={"sigma": 0.18})
    wait_base_drift = RewTerm(func=mdp.wait_base_drift_penalty, weight=-3.0, params={"sigma": 0.18})

    # --- [박스가 날아올 때: 몸쪽으로 당기기] ---
    # sigma를 0.35로 좁혀서 몸통에 확 끌어당겨야 보상을 받도록 유도
    reach = RewTerm(func=mdp.torso_reach_object_reward, weight=1.0, params={"sigma": 0.35})
    hands_reach = RewTerm(func=mdp.hands_reach_object_reward, weight=1.0, params={"sigma": 0.35})
    # support_under는 자세를 강제할 수 있으니 삭제하거나 weight를 0으로 꺼두셔도 됩니다 (온몸 껴안기가 알아서 해결함)

    # --- [핵심 1: 온몸 껴안기 뻥튀기 보상] ---
    contact_hug = RewTerm(
        func=mdp.hug_contact_bonus,
        weight=3.0, # 매우 큰 가중치로 껴안기를 강력 유도
        params={
            "sensor_names_left": [
                "contact_l_shoulder_yaw", "contact_l_elbow",
                "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand"
            ],
            "sensor_names_right": [
                "contact_r_shoulder_yaw", "contact_r_elbow",
                "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand"
            ],
            "sensor_name_torso": "contact_torso",
            "thr": 2.0,
        },
    )

    # --- [핵심 2: 저글링(튕기기) 완벽 차단 방어막] ---
    # 방어막 1: 박스와 몸통의 상대 속도를 0으로 만들어 박스를 품 안에서 정지시킴
    hold_vel = RewTerm(func=mdp.hold_object_vel_reward, weight=2.0, params={"torso_body_name": "torso_link", "sigma": 0.35})
    
    # 방어막 2: 손에서 박스가 떨어지면(튕기면) Latch가 풀림. 연속으로 20스텝 이상 잡고 있어야 큰 점수 획득
    hold_latched = RewTerm(func=mdp.hold_latched_bonus, weight=1.0)
    hold_sustain = RewTerm(func=mdp.hold_sustain_bonus, weight=2.0, params={"min_steps": 10})
    
    # 방어막 3: 배구 펀치처럼 때릴 경우 강한 페널티 (임계치를 250.0으로 빡빡하게 낮춤)
    impact = RewTerm(
        func=mdp.impact_peak_penalty,
        weight=-0.005, 
        params={
            "sensor_names": [
                "contact_torso",
                "contact_l_shoulder_yaw", "contact_l_elbow", "contact_l_wrist_roll", "contact_l_wrist_pitch", "contact_l_wrist_yaw", "contact_l_hand",
                "contact_r_shoulder_yaw", "contact_r_elbow", "contact_r_wrist_roll", "contact_r_wrist_pitch", "contact_r_wrist_yaw", "contact_r_hand"
            ],
            "force_thr": 250.0,
        },
    )

    not_drop = RewTerm(func=mdp.object_not_dropped_bonus, weight=1.0, params={"min_z": 0.45, "max_dist": 2.2})

    # --- [받은 후 안정화: 도망가지 말고 제자리에 서있기] ---
    post_hold_still = RewTerm(func=mdp.post_hold_still_reward, weight=2.0, params={"lin_sigma": 0.14, "yaw_sigma": 0.45})
    post_hold_anchor = RewTerm(func=mdp.post_hold_anchor_penalty, weight=-2.0, params={"sigma": 0.12})
    #post_hold_leg_motion = RewTerm(func=mdp.post_hold_leg_motion_penalty, weight=-0.06)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen_degree, params={"min_root_z": 0.45, "max_tilt_deg": 60.0})
    drop = DoneTerm(func=mdp.object_dropped, params={"min_z": 0.35, "max_dist": 3.0})
    post_hold_runaway = DoneTerm(func=mdp.post_hold_runaway, params={"max_anchor_drift": 0.40})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # park object away (to remove early-contact exploitation) + randomize physics
    reset_object_parked = EventTerm(
        func=mdp.reset_object_parked,
        mode="reset",
        params={
            "park": {"pos_x": (1.5, 1.5), "pos_y": (0.0, 0.0), "pos_z": (-0.60, -0.55)},
        },
    )

    toss = EventTerm(
        func=mdp.toss_object_relative_curriculum,
        mode="interval",
        interval_range_s=(1.0, 2.5),
        params={
            "max_throws_per_episode": 1,
            # Stage 0은 함수 내부에서 아예 던지지 않도록(Stand-only) 하드코딩할 예정입니다.
            "throw_prob_stage1": 1.0,  
            "throw_prob_stage2": 0.90, 
            "throw_prob_stage3": 0.90, 

            # stage1: super-easy handover (기존 stage0)
            "stage1": {
                "pos_x": (0.46, 0.56), "pos_y": (-0.03, 0.03), "pos_z": (0.34, 0.42),
                "vel_x": (-0.35, -0.15), "vel_y": (-0.05, 0.05), "vel_z": (-0.03, 0.05),
            },
            # stage2: gentle throw (기존 stage1)
            "stage2": {
                "pos_x": (0.50, 0.64), "pos_y": (-0.06, 0.06), "pos_z": (0.30, 0.44),
                "vel_x": (-0.95, -0.70), "vel_y": (-0.10, 0.10), "vel_z": (-0.02, 0.12),
            },
            # stage3: harder throw (기존 stage2)
            "stage3": {
                "pos_x": (0.52, 0.65), "pos_y": (-0.10, 0.10), "pos_z": (0.30, 0.46),
                "vel_x": (-1.45, -0.95), "vel_y": (-0.14, 0.14), "vel_z": (-0.04, 0.14),
            },
        },
    )
    export_bank = EventTerm(
        func=mdp.export_catch_success_bank,
        mode="interval",
        interval_range_s=(0.10, 0.10),
        params={
            "bank_path": "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/tools/catch_success_bank.pt",
            "min_hold_steps": 12,
            "min_gap_steps": 20,
            "max_total_states": 30000,
            "flush_every": 256,
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule, # (이 함수가 mdp/curriculum.py에 있다면 유지, 아니라면 events.py 등에서 읽는 용도의 dummy여도 됨)
        params={
            #"stage0_iters": 5, 
            #"stage1_iters": 5, 
            #"stage2_iters": 5, 
            "stage0_iters": 500,
            "stage1_iters": 1000, 
            "stage2_iters": 1500,
            "num_steps_per_env": 64,
            "eval_stage": -1, # -1이면 현재 스텝에 따라 자동 난이도 조절
        },
    )


@configclass
class dj_urop_v12_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_v12_SceneCfg = dj_urop_v12_SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # sim dt / decimation
        self.decimation =2
        self.episode_length_s = 6.0
        self.sim.dt = 1 / 100
        self.sim.render_interval = self.decimation

        # PhysX stability (optional; avoids noisy velocity updates warning)
        try:
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "enable_external_forces_every_iteration"):
                self.sim.physx.enable_external_forces_every_iteration = True
            if hasattr(self.sim, "physx") and hasattr(self.sim.physx, "num_velocity_iterations"):
                self.sim.physx.num_velocity_iterations = 1
        except Exception:
            pass

# 🔥 평가(Play) 모드를 위한 환경 클래스를 별도로 상속받아 만듭니다. 🔥
@configclass
class dj_urop_v12_EnvCfg_Play(dj_urop_v12_EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Play 모드에서는 던지는 난이도를 강제 고정 (예: 제일 어려운 stage 3)
        self.curriculum.stage_schedule.params["eval_stage"] = 3
        
        # Play 모드에서는 평가를 위해 무조건 100% 확률로 박스를 던지게 설정
        self.events.toss.params["throw_prob_stage3"] = 1.0 
        
        # (선택) 에피소드 길이를 평가 때는 좀 더 길게 볼 수도 있습니다.
        # self.episode_length_s = 10.0
