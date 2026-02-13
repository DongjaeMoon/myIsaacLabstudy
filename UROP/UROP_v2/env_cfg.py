from dataclasses import MISSING
import os
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

import isaaclab.sim as sim_utils

from . import scene_objects_cfg
from . import mdp


@configclass
class dj_urop_SceneCfg(InteractiveSceneCfg):
    # ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=800.0),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    )

    robot = scene_objects_cfg.dj_robot_cfg
    object = scene_objects_cfg.bulky_object_cfg

    # contact sensors
    contact_torso = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    contact_lhand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_roll_rubber_hand",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )
    contact_rhand = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_roll_rubber_hand",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


@configclass
class CommandsCfg:
    # CommandManager를 쓰지 않고 env.urop_cmd를 events에서 생성/저장해서 obs/reward에서 사용
    command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    legs = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ],
        scale=0.6,
    )
    waist = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist_yaw_joint"],
        scale=0.35,
    )
    left_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint",
        ],
        scale=1.2,
    )
    right_arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint",
        ],
        scale=1.2,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.robot_proprio)
        prev_actions = ObsTerm(func=mdp.previous_actions)
        cmd = ObsTerm(func=mdp.velocity_command)
        obj_rel = ObsTerm(
            func=mdp.object_rel_state,
            params={"drop_prob": 0.0, "noise_std": 0.01, "pos_scale": 1.0, "vel_scale": 1.0, "ang_vel_scale": 1.0},
        )
        hand_vec = ObsTerm(func=mdp.hand_object_vectors)
        contact = ObsTerm(
            func=mdp.contact_forces,
            params={"sensor_names": ["contact_torso", "contact_lhand", "contact_rhand"], "scale": 1.0 / 300.0},
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # locomotion
    alive = RewTerm(func=mdp.alive_bonus_curriculum, weight=1.0, params={"w0": 0.2, "w1": 0.05, "w2": 0.02})
    upright = RewTerm(func=mdp.upright_reward_curriculum, weight=0.1, params={"w0": 1.0, "w1": 1.0, "w2": 1.0})
    height = RewTerm(func=mdp.root_height_reward_curriculum, weight=0.5, params={"target_z": 0.78, "sigma": 0.10, "w0": 1.0, "w1": 0.5, "w2": 0.3})
    track_lin = RewTerm(func=mdp.track_cmd_lin_vel_xy_curriculum, weight=1.2, params={"sigma": 0.35, "w0": 1.0, "w1": 0.8, "w2": 0.8})
    track_yaw = RewTerm(func=mdp.track_cmd_yaw_rate_curriculum, weight=0.4, params={"sigma": 0.6, "w0": 0.5, "w1": 0.5, "w2": 0.5})

    joint_vel = RewTerm(func=mdp.joint_vel_l2_penalty_curriculum, weight=-1e-4, params={"w0": 0.01, "w1": 0.01, "w2": 0.015})
    action_rate = RewTerm(func=mdp.action_rate_penalty_curriculum, weight=-0.01, params={"w0": 0.05, "w1": 0.02, "w2": 0.01})

    # catch & carry
    reach = RewTerm(func=mdp.hand_reach_object_reward_curriculum, weight=0.6, params={"sigma": 0.6, "w0": 0.0, "w1": 1.2, "w2": 1.0})
    #hold_pose = RewTerm(func=mdp.hold_pose_reward_curriculum, weight=1.0, params={"target_offset": (0.55, 0.0, 1.05), "sigma": 0.35, "w0": 0.0, "w1": 2.0, "w2": 2.5})
    hold_pose = RewTerm(
    func=mdp.hold_pose_reward_curriculum,
    weight=1.0,
    params={
        "asset_name": "object",
        "target_offset": (0.45, 0.0, 0.28),
        "xy_sigma": 0.40,
        "z_sigma": 0.25,
        "gate_sigma": 0.90,
    },
    )
    hold_vel = RewTerm(func=mdp.hold_object_vel_reward_curriculum, weight=0.8, params={"sigma": 0.8, "w0": 0.0, "w1": 0.8, "w2": 1.2})
    contact_hold = RewTerm(func=mdp.contact_hold_bonus_curriculum, weight=0.6, params={"sensor_names": ["contact_torso", "contact_lhand", "contact_rhand"], "thr": 1.0, "w0": 0.0, "w1": 0.4, "w2": 0.6})

    not_drop = RewTerm(func=mdp.object_not_dropped_bonus_curriculum, weight=1.0, params={"min_z": 0.55, "max_dist": 2.5, "w0": 0.0, "w1": 0.3, "w2": 0.4})
    impact = RewTerm(func=mdp.impact_peak_penalty_curriculum, weight=-0.001, params={"sensor_names": ["contact_torso", "contact_lhand", "contact_rhand"], "force_thr_stage1": 350.0, "force_thr_stage2": 300.0, "w0": 0.0, "w1": 0.08, "w2": 0.12})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall = DoneTerm(func=mdp.robot_fallen, params={"min_root_z": 0.55, "min_upright": 0.6})
    drop = DoneTerm(func=mdp.object_dropped_curriculum, params={"min_z": 0.55, "max_dist": 2.5})


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cmd = EventTerm(
        func=mdp.reset_velocity_command_curriculum,
        mode="reset",
        params={
            "stage0": {"vx": (0.0, 0.6), "vy": (-0.2, 0.2), "yaw": (-0.6, 0.6), "stand_prob": 0.15},
            "stage1": {"vx": (0.0, 0.35), "vy": (-0.15, 0.15), "yaw": (-0.4, 0.4), "stand_prob": 0.25},
            "stage2": {"vx": (0.2, 0.8), "vy": (-0.2, 0.2), "yaw": (-0.6, 0.6), "stand_prob": 0.05},
        },
    )

    reset_throw = EventTerm(
    func=mdp.reset_and_throw_object_ballistic_curriculum,
    mode="reset",
    params={
        "asset_name": "object",
        "intercept_with_base_vel": True,
        "stage0": {"park_pos": (3.0, 0.0, 0.25)},
        "stage1": {
            "spawn_dist": (1.8, 2.6),
            "spawn_y": (-0.6, 0.6),
            "spawn_z": (0.95, 1.15),
            "flight_t": (0.8, 1.2),
            "target_offset": (0.45, 0.0, 0.28),
            "spin": (-0.0, 0.0),
        },
        "stage2": {
            "spawn_dist": (1.5, 2.4),
            "spawn_y": (-0.7, 0.7),
            "spawn_z": (0.95, 1.25),
            "flight_t": (0.6, 1.0),
            "target_offset": (0.50, 0.0, 0.28),
            "spin": (-0.0, 0.0),
        },
    },
    )

    push_recovery = EventTerm(
        func=mdp.push_robot_velocity_impulse,
        mode="interval",
        interval_range_s=(1.0, 2.5),
        params={
            "asset_name": "robot",
            "lin_vel_xy": (0.15, 0.45),
            "lin_vel_z": (0.0, 0.05),
            "yaw_rate": (-0.5, 0.5),
            "stage0_scale": 1.0,
            "stage1_scale": 0.6,
            "stage2_scale": 0.35,
            "body_names": ["torso_link"],
        },
    )


@configclass
class CurriculumCfg:
    stage_schedule = CurrTerm(
        func=mdp.stage_schedule,
        params={
            "stage0_iters": 4000,
            "stage1_iters": 4000,
            "num_steps_per_env": 96,  # runner cfg와 반드시 동일
            "eval_stage": -1,
        },
    )


@configclass
class dj_urop_EnvCfg(ManagerBasedRLEnvCfg):
    scene: dj_urop_SceneCfg = dj_urop_SceneCfg(num_envs=64, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()

    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 12.0

        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.viewer.resolution = (1920, 1080)

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # teacher/student gating: object state를 “통째로” 끊어버리면 학습이 너무 어려워짐.
        # 일단 teacher로 끝까지 잘 학습시키고, 그 다음에 student로 점진적 이행 추천.
        mode = os.environ.get("UROP_MODE", "teacher").lower()
        self.observations.policy.obj_rel.params["drop_prob"] = 0.0 if mode == "teacher" else 1.0

        # curriculum schedule env vars
        p = self.curriculum.stage_schedule.params
        p["stage0_iters"] = int(os.environ.get("UROP_STAGE0_ITERS", str(p["stage0_iters"])))
        p["stage1_iters"] = int(os.environ.get("UROP_STAGE1_ITERS", str(p["stage1_iters"])))
        p["num_steps_per_env"] = int(os.environ.get("UROP_NUM_STEPS_PER_ENV", str(p["num_steps_per_env"])))
        p["eval_stage"] = int(os.environ.get("UROP_EVAL_STAGE", str(p.get("eval_stage", -1))))
