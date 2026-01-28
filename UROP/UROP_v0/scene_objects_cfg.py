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

# [1] 로봇 (G1) 정의
dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # ★ 동재님이 만드신 최종 파일 경로
        usd_path="/home/roro_common/mdj/IsaacLab/UROP/UROP_v0/usd/G1_23DOF_UROP.usd",
        activate_contact_sensors=True, # 충격 감지를 위해 필수
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78), # 로봇 키에 맞춰서 살짝 띄움 (골반 높이)
        # 초기 자세 (살짝 무릎 굽힌 자세 추천)
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.2, # 팔을 살짝 앞으로
            ".*_elbow_joint": 0.5,          # 팔 굽힘 (받을 준비)
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*"],
            stiffness=100.0, # 다리는 단단하게
            damping=5.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
            stiffness=40.0,  # 팔은 충격 흡수를 위해 조금 부드럽게
            damping=2.0,
        ),
    },
)

# [2] 던져질 상자 (Bulky Object) 정의
bulky_object_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.4, 0.3, 0.3),  # 택배 상자 크기
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0), # 초기 질량 5kg
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.8, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solve_contact=True,
            max_depenetration_velocity=100.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.0)), # 로봇 앞 1m
)


