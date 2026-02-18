#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/scene_objects_cfg.py]
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import math
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import RigidBodyMaterialCfg
import os

# [1] 로봇 (G1) 정의
dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/usd/G1_23DOF_UROP.usd",
        activate_contact_sensors=True, 
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        # [수정] 정규표현식 대신 명시적 관절 이름 사용
        joint_pos={
            # --- 다리 (Legs) : 약간 굽힌 자세 (Squat) ---
            "left_hip_pitch_joint": -0.2,
            "right_hip_pitch_joint": -0.2,
            "left_knee_joint": 0.4,
            "right_knee_joint": 0.4,
            "left_ankle_pitch_joint": -0.2,
            "right_ankle_pitch_joint": -0.2,
            
            # 나머지 다리 관절은 0.0 (직립)
            "left_hip_roll_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,

            # --- 팔 (Arms) : 받을 준비 자세 ---
            # Shoulder Pitch: 0.2 (살짝 앞으로 들어올림)
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_pitch_joint": 0.2,
            
            # Elbow: 0.5 (굽힘)
            "left_elbow_joint": 0.5,
            "right_elbow_joint": 0.5,

            # 나머지 팔 관절 0.0
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
        },
    ),
    actuators={
        # 1. 하체 & 허리 (강한 지지력 필요)
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint",
            ],
            stiffness=120.0, # 지면 반발력과 박스 무게를 버팀
            damping=5.0,     # 너무 높으면 반응이 느려지므로 6.0 -> 5.0 소폭 조정
        ),
        
        # 2. 상체 - 주요 부하 담당 (어깨 피치, 팔꿈치)
        # 박스 무게를 직접 받는 관절들이라 조금 더 단단해야 처지지 않음
        "arms_load": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint", "left_elbow_joint",
                "right_shoulder_pitch_joint", "right_elbow_joint",
            ],
            stiffness=80.0,  # 기존 60 -> 80 상향 (무게 지탱)
            damping=4.0,     # 진동 방지
        ),

        # 3. 상체 - 위치 제어 담당 (나머지 팔 관절)
        # 유연하게 움직여서 박스를 감싸안거나 충격 흡수
        "arms_pos": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint",
                "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint",
            ],
            stiffness=60.0,  # 유연성 유지
            damping=3.0,
        ),
    },
)

# [2] 던져질 상자 (Bulky Object) 정의
bulky_object_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.4, 0.3, 0.3),  # 택배 상자 크기
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
        # [추가됨] 마찰력 설정: 고무나 거친 종이 박스처럼 마찰을 높임
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5, # 운동 마찰계수 (높게)
            restitution=0.1,      # 반발계수 (튕겨나가지 않게 0.1으로)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,  # True로 하면 중력을 무시하고 그 자리에 고정됨 (또는 코드로 위치 제어 가능)
            disable_gravity=False,    # 이중 안전장치
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True, # 충격 감지를 위해 필수
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 1.0)), # 로봇 앞 1m
)

# -------------------------
# Contact sensor configs
# -------------------------

contact_torso_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)

# Left arm
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
    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_roll_rubber_hand",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)

# Right arm
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
    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_roll_rubber_hand",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    update_period=0.0,
    history_length=1,
)
