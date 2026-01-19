import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg

# =========================================================
# Interactive objects
# =========================================================
ball_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.09,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.9,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(2.0, 0.0, 0.8),
    ),
)

goal_post_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalPost",
    spawn=sim_utils.CuboidCfg(
        size=(0.1, 2.0, 1.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.0, 0.8)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-0.75, 0.0, 0.75),
    ),
)

# =========================================================
# Robot
# =========================================================
GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # [확인] v1 경로가 맞는지 체크
        usd_path="/home/roro_common/mdj/IsaacLab/example/urop_v1/usd/dj_robotarm_on_go2.usd", 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # [★필수] 이 줄이 있어야 센서가 켜집니다.
        activate_contact_sensors=True, 
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "shoulder_joint": 0.0,
            "arm_joint1": 0.0,
            "arm_joint2": 1.0,
        },
    ),
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=50.0,
            damping=5.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_joint", "arm_joint1", "arm_joint2"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=200.0,
            damping=10.0, 
        ),
    },
)

# =========================================================
# Contact sensor
# =========================================================
arm_tip_contact_sensor_cfg = ContactSensorCfg(
    # 팔이나 어깨 어디든 닿으면 감지 (필터 없음)
    prim_path="{ENV_REGEX_NS}/Robot/dj_robotarm/dj_robotarm/.*_link.*",
    update_period=0.0,
    history_length=1,
    debug_vis=True,
)