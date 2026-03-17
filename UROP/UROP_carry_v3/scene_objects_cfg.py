# [/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/scene_objects_cfg.py]

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyMaterialCfg


# --------------------------------------------------------------------------------------
# 29-DOF control set
# Keep this ORDER FIXED. Policy action ordering must match this list exactly.
# --------------------------------------------------------------------------------------
G1_29_JOINTS = [
    # Legs (12)
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

G1_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v3/g1_29dof_full_collider_flattened.usd"
#G1_USD_PATH = "/home/idim5080-2/mdj/myIsaacLabstudy/UROP/UROP_carry_v3/g1_29dof_full_collider_flattened.usd"

# --------------------------------------------------------------------------------------
# Robot config
# Base this on loco_v5, not catch/carry_v2.
# Reason: carry must be loaded locomotion, not a brand new task from scratch.
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
        # This init pose is only a fallback/default.
        # Real carry training should reset from filtered catch-success state bank.
        pos=(0.0, 0.0, 0.79),
        joint_pos={
            # Legs
            "left_hip_pitch_joint": -0.10,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.30,
            "left_ankle_pitch_joint": -0.20,
            "left_ankle_roll_joint": 0.0,

            "right_hip_pitch_joint": -0.10,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.30,
            "right_ankle_pitch_joint": -0.20,
            "right_ankle_roll_joint": 0.0,

            # Waist
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            # Arms - neutral standing posture, not catch-ready pose
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.20,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,

            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.20,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,

            # Fingers locked
            **{j: 0.0 for j in G1_FINGER_JOINTS},
        },
        joint_vel={j: 0.0 for j in (G1_29_JOINTS + G1_FINGER_JOINTS)},
    ),
    actuators={
        # Strong legs + waist, but not ankles
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
        # Critical: ankles need low stiffness + limited torque
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
        # Arms used for loaded support + posture
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
            effort_limit_sim=40,
            stiffness={
                ".*_shoulder_pitch_joint": 55.0,
                ".*_shoulder_roll_joint": 45.0,
                ".*_shoulder_yaw_joint": 45.0,
                ".*_elbow_joint": 65.0,
                ".*_wrist_roll_joint": 35.0,
                ".*_wrist_pitch_joint": 35.0,
                ".*_wrist_yaw_joint": 35.0,
            },
            damping=10.0,
            armature=0.05,
        ),
        # Fingers are not policy-controlled
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


# --------------------------------------------------------------------------------------
# Carry object config
# Keep size fixed at first. Randomize only mass/friction/restitution later in events.py.
# --------------------------------------------------------------------------------------
bulky_object_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.32, 0.24, 0.24),
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.85,
            dynamic_friction=0.70,
            restitution=0.01,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=0.02,
            angular_damping=0.02,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=2.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.50, 0.0, 1.00),
    ),
)


# --------------------------------------------------------------------------------------
# Full-body contact sensor for locomotion-style rewards/terminations
# This should be added to SceneCfg in env_cfg.py as "contact_forces".
# --------------------------------------------------------------------------------------
full_body_contact_forces_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    update_period=0.0,
    history_length=3,
    track_air_time=True,
)


# --------------------------------------------------------------------------------------
# Object-specific contact sensors for carry rewards/logic
# --------------------------------------------------------------------------------------
contact_torso_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)

# Left arm
contact_l_shoulder_yaw_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_shoulder_yaw_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_l_elbow_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_elbow_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_l_wrist_roll_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_roll_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_l_wrist_pitch_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_pitch_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_l_wrist_yaw_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_l_hand_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)

# Right arm
contact_r_shoulder_yaw_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_shoulder_yaw_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_r_elbow_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_elbow_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_r_wrist_roll_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_r_wrist_pitch_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_r_wrist_yaw_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_r_hand_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)


# --------------------------------------------------------------------------------------
# Convenient sensor name groups for observations/rewards
# --------------------------------------------------------------------------------------
LEFT_CARRY_CONTACT_SENSOR_NAMES = [
    "contact_l_shoulder_yaw",
    "contact_l_elbow",
    "contact_l_wrist_roll",
    "contact_l_wrist_pitch",
    "contact_l_wrist_yaw",
    "contact_l_hand",
]

RIGHT_CARRY_CONTACT_SENSOR_NAMES = [
    "contact_r_shoulder_yaw",
    "contact_r_elbow",
    "contact_r_wrist_roll",
    "contact_r_wrist_pitch",
    "contact_r_wrist_yaw",
    "contact_r_hand",
]

ALL_CARRY_CONTACT_SENSOR_NAMES = [
    "contact_torso",
    *LEFT_CARRY_CONTACT_SENSOR_NAMES,
    *RIGHT_CARRY_CONTACT_SENSOR_NAMES,
]