import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg

# =========================================================
# Interactive objects
# =========================================================
'''ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.05,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,  # True로 하면 중력을 무시하고 그 자리에 고정됨 (또는 코드로 위치 제어 가능)
            disable_gravity=True,    # 이중 안전장치
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            restitution=0.8,        # 통통 튀는 정도 (0.0 ~ 1.0)
            static_friction=0.5,    # (선택) 정지 마찰력
            dynamic_friction=0.5,   # (선택) 운동 마찰력
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(2.0, 0.2, 0.75), # 로봇 앞 60cm 지점에 배치
    ),
)

goal_post_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalPost",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 1.5, 1.0), # 두께 10cm, 폭 1.2m, 높이 0.8m (간이 골대)
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.8), metallic=0.8),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True, # 고정된 물체
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.75, 0.0, 0.4), # 로봇 등 뒤 0.5m 지점
    ),
)'''

# =========================================================
# Robot
# =========================================================
# New robot should be added!!!
'''DJ_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # ★여기에 합체한 USD 파일 경로를 넣으세요!
        #usd_path="/home/roro_common/mdj/IsaacLab/example/urop_v0/usd/dj_robotarm_on_go2.usd", 
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35), # 로봇이 땅에 파묻히지 않게 살짝 띄움
        # 관절 초기 각도 (필요하면 추가)
        joint_pos={
            # (1) dj_robotarm zero position
            "shoulder_joint": 0.0,
            "arm_joint1": -0.5, 
            "arm_joint2": 1.0,
            
            # (2) Go2 다리 초기 각도 (스크린샷 이름 기반)
            # Unitree 로봇이 서 있을 때의 대략적인 각도입니다.
            ".*_hip_joint": 0.1,    # 엉덩이 살짝 벌리기
            ".*_thigh_joint": 0.8,  # 허벅지 
            ".*_calf_joint": -1.5,  # 종아리
        },
    ),
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=50.0,
            damping=2.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_joint", "arm_joint1", "arm_joint2"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=100.0,
            damping=10.0, 
        ),
    },
)'''

# =========================================================
# Contact sensor
# =========================================================
arm_tip_contact_sensor_cfg = ContactSensorCfg(
    #prim_path="{ENV_REGEX_NS}/Robot/dj_robotarm/dj_robotarm/(shoulder_link|arm_link1|arm_link2)",
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
    update_period=0.0,
    history_length=1,
    debug_vis=True,
)