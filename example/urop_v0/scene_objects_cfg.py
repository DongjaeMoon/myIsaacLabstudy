

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import math

# =========================================================
# 상호작용할 물체 (빨간 공)
# =========================================================
ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.1,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # True로 하면 중력을 무시하고 그 자리에 고정됨 (또는 코드로 위치 제어 가능)
            disable_gravity=True,    # 이중 안전장치
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.5, 0.0, 1.0), # 로봇 앞 60cm 지점에 배치
    ),
)
# 1. 내 로봇 (Go2 + Arm) 정의
dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # ★여기에 합체한 USD 파일 경로를 넣으세요!
        usd_path="/home/roro_common/mdj/IsaacLab/example/urop_v0/usd/dj_robotarm_on_go2.usd", 
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2), # 로봇이 땅에 파묻히지 않게 살짝 띄움
        # 관절 초기 각도 (필요하면 추가)
        joint_pos={
            # (1) dj_robotarm zero position
            "shoulder_joint": 0.0,
            "arm_joint1": 0.0, 
            "arm_joint2": 0.0,
            
            # (2) Go2 다리 초기 각도 (스크린샷 이름 기반)
            # Unitree 로봇이 서 있을 때의 대략적인 각도입니다.
            ".*_hip_joint": 0.1,    # 엉덩이 살짝 벌리기
            ".*_thigh_joint": 0.8,  # 허벅지 
            ".*_calf_joint": -1.5,  # 종아리
        },
    ),
    actuators={
        # (1) 로봇 팔 관절 (강한 힘)
        "my_arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_joint", "arm_joint1", "arm_joint2"],
            effort_limit_sim=50.0,
            stiffness=50.0, # 아까 설정한 높은 P게인
            damping=5.0,     # D게인
        ),
        # (2) Go2 다리 관절 (모든 나머지 관절)
        # Go2 관절 이름을 정확히 모르면 일단 나머지를 다 잡는 정규표현식 사용
        "go2_legs": ImplicitActuatorCfg(
            joint_names_expr=["FL_.*", "FR_.*", "RL_.*", "RR_.*"], # 혹은 ".*_hip_.*" 등 Go2 관절명 규칙
            effort_limit_sim=25.0,
            stiffness=30.0, # 다리는 좀 부드럽게
            damping=2.0,
        ),
    },
)
'''
box_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/asdf",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sangrul/IsaacLab/0)model/G1/nail.usd",
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos = (0.5, 0.0, 1.0),
        rot = (1.0, 0.0, 0.0, 0.0),
    )
)

robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sangrul/IsaacLab/ksr_RLs/example/urop_v0/usd/G1_hammer.usd",
    ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     joint_pos={
    #         "left_elbow_pitch_joint": 90.0 / 180 * math.pi,
    #     }
    # ),
    actuators={
        "left1": ImplicitActuatorCfg( 
            joint_names_expr=["left_shoulder_pitch_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "left2": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_roll_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "left3": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_yaw_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "left4": ImplicitActuatorCfg(
            joint_names_expr=["left_elbow_pitch_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "left5": ImplicitActuatorCfg(
            joint_names_expr=["left_elbow_roll_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right1": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_pitch_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right2": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_roll_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right3": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_yaw_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right4": ImplicitActuatorCfg(
            joint_names_expr=["right_elbow_pitch_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right5": ImplicitActuatorCfg(
            joint_names_expr=["right_elbow_roll_joint"],
            effort_limit_sim=21.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "left_gripper1": ImplicitActuatorCfg(
            joint_names_expr=["left_plate1_joint"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "left_gripper2": ImplicitActuatorCfg(
            joint_names_expr=["left_plate2_joint"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right_gripper1": ImplicitActuatorCfg(
            joint_names_expr=["right_plate1_joint"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
        "right_gripper2": ImplicitActuatorCfg(
            joint_names_expr=["right_plate2_joint"],
            effort_limit_sim=50.0,
            stiffness=1000.0,
            damping=1000.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

prop_example_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/sangrul/IsaacLab/ksr_RLs/example/urop_v0/usd/table.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos = (1.0, 0.0, 0.0),
        rot = (1.0, 0.0, 0.0, 0.0),
    ),
)

ft_sensor_example_cfg = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_gripper_base",
            name="BaseToLgripper",
            offset=OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(1.0, 0.0, 0, 0),
            )
        ),
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_gripper_base",
            name="BaseToRgripper",
            offset=OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(1.0, 0.0, 0, 0),
            )
        ),
    ],
    update_period=0.0,
    debug_vis=False,
)

contact_sensor_example_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/hammer",
    update_period=0.0,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    track_air_time=True,
)'''
