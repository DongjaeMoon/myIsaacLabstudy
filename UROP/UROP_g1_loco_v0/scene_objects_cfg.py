import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True, 
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        # 모든 관절 초기 위치 (43개)
        joint_pos={
            "left_hip_pitch_joint": -0.2, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.4, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
            
            "right_hip_pitch_joint": -0.2, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.4, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
            
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            
            "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.0, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
            
            "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0, "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0, "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
            
            "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0, "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0, "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0, "right_hand_index_1_joint": 0.0,
        },
        # 모든 관절 초기 속도 (43개)
        joint_vel={
            "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0, "left_ankle_pitch_joint": 0.0, "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0, "right_ankle_pitch_joint": 0.0, "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.0, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": 0.0, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0, "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0, "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0, "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0, "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0, "right_hand_index_1_joint": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
            ],
            stiffness=120.0,
            damping=5.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
            ],
            stiffness=40.0, 
            damping=2.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                "left_hand_index_0_joint", "left_hand_index_1_joint",
                "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                "right_hand_index_0_joint", "right_hand_index_1_joint"
            ],
            stiffness=2.0,   
            damping=0.1,
        ),
    },
)