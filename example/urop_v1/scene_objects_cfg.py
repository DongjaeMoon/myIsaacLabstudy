import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
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
        rot=(1.0, 0.0, 0.0, 0.0),
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
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

# =========================================================
# Robot
# =========================================================
GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path="/home/roro_common/mjd/assets/unitree_go2_3/usd/go2.usd",
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
        # 팔을 더 “빠릿하게” 움직이게 약간 올림 (너무 튀면 stiffness/damping 낮춰도 됨)
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_joint", "arm_joint1", "arm_joint2"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=200.0,
            damping=20.0,
        ),
    },
)

# =========================================================
# Contact sensor
# =========================================================
arm_tip_contact_sensor_cfg = ContactSensorCfg(
    # NOTE: 네 USD 경로가 Robot/dj_robotarm/dj_robotarm/arm_link2 처럼 중간이 껴도 잡히도록 regex로 완화
    prim_path="{ENV_REGEX_NS}/Robot/.*/arm_link2",
    update_period=0.005,
    history_length=3,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
)
