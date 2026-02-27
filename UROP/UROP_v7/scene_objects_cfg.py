import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyMaterialCfg

# =============================================================================
# Unitree G1 (official asset)
#   - Policy controls 29 DOF: legs(12) + waist(3) + arms/wrists(14)
#   - Finger joints exist in USD/URDF, but are NOT controlled by the policy.
#     We lock them with a stiff actuator to avoid flapping.
# =============================================================================

dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            # legs
            "left_hip_pitch_joint": -0.15,
            "right_hip_pitch_joint": -0.15,
            "left_knee_joint": 0.30,
            "right_knee_joint": 0.30,
            "left_ankle_pitch_joint": -0.15,
            "right_ankle_pitch_joint": -0.15,
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,

            # waist (29 DOF target includes all three)
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,

            # arms/wrists: receive-ready posture
            "left_shoulder_pitch_joint": 0.20,
            "right_shoulder_pitch_joint": 0.20,
            "left_elbow_joint": 0.55,
            "right_elbow_joint": 0.55,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,

            # fingers locked at zero (not part of 29DOF policy)
            "left_hand_index_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
    ),
    actuators={
        "legs_and_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ],
            stiffness=120.0,
            damping=5.0,
        ),
        "arms_load": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_elbow_joint",
            ],
            stiffness=85.0,
            damping=4.0,
        ),
        "arms_pose": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ],
            stiffness=55.0,
            damping=3.0,
        ),
        "fingers_lock": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_index_0_joint", "left_hand_index_1_joint",
                "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                "right_hand_index_0_joint", "right_hand_index_1_joint",
                "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            ],
            stiffness=220.0,
            damping=10.0,
        ),
    },
)


# =============================================================================
# Object (box) - smaller default size; mass/friction randomized at reset in events.py
# =============================================================================

bulky_object_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.32, 0.24, 0.24),
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.80,
            dynamic_friction=0.60,
            restitution=0.02,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)),
)


# =============================================================================
# Contact sensors
#   NOTE: In the official g1.usd, palm links often do NOT expose contact reporter API.
#         So we attach "hand" sensors to wrist links (reliable rigid bodies).
# =============================================================================

contact_torso_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)

contact_l_shoulder_pitch_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_shoulder_pitch_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_l_shoulder_roll_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_shoulder_roll_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
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
contact_l_hand_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_roll_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)

contact_r_shoulder_pitch_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_shoulder_pitch_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
contact_r_shoulder_roll_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_shoulder_roll_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
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
contact_r_hand_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_roll_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)