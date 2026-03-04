#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_g1_loco_v2/scene_objects_cfg.py]

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# --------------------------------------------------------------------------------------
# 29-DOF control set (43 total in your USD - 14 finger joints = 29)
# Keep this list ORDER FIXED. Your policy action ordering will follow this list exactly.
# --------------------------------------------------------------------------------------
G1_29_JOINTS = [
    # Legs (12): (yaw, roll, pitch, knee, ankle_pitch, ankle_roll) x 2
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Arms (14)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_FINGER_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Your official G1 USD (the one you want to train with)
#G1_USD_PATH = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd"
G1_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_g1_loco_v2/g1_29dof.usd"
# --------------------------------------------------------------------------------------
# Key stability settings copied in spirit from IsaacLab’s working locomotion configs:
# - rigid_props: limit depenetration velocity (prevents “explosive” contact resolution)
# - articulation_props: increase solver iterations (helps humanoid stability a lot)
# --------------------------------------------------------------------------------------
dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            retain_accelerations=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base height: keep your 0.78 (works fine if pose + PD are correct)
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            # Legs: slightly crouched & ankle pitched (helps initial stability)
            "left_hip_pitch_joint": -0.20,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.42,
            "left_ankle_pitch_joint": -0.23,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.20,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.42,
            "right_ankle_pitch_joint": -0.23,
            "right_ankle_roll_joint": 0.0,
            # Waist
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # Arms: slightly forward (reduces backward-fall tendency)
            "left_shoulder_pitch_joint": 0.35,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.85,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.85,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # Fingers locked near 0
            **{j: 0.0 for j in G1_FINGER_JOINTS},
        },
        joint_vel={j: 0.0 for j in (G1_29_JOINTS + G1_FINGER_JOINTS)},
    ),
    actuators={
        # Strong legs + waist, but *not* ankles
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=160,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_.*": 120.0,
            },
            damping=5.0,
            armature=0.05,
        ),
        # Critical: ankles need low stiffness + low torque cap
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=50,
            stiffness=20.0,
            damping=2.0,
            armature=0.05,
        ),
        # Arms: moderate stiffness, higher damping, torque cap
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=30,
            stiffness=40.0,
            damping=10.0,
            armature=0.05,
        ),
        # Fingers: lock them (not controlled by policy)
        "hands_lock": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_.*",
                "right_hand_.*",
            ],
            effort_limit_sim=5,
            stiffness=200.0,
            damping=10.0,
            armature=0.001,
        ),
    },
)
