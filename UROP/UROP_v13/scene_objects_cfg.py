from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import RigidBodyMaterialCfg


G1_USD_PATH = str(Path(__file__).resolve().parent / "usd" / "g1_29dof_full_collider_flattened.usd")

# Contract-fixed control interface shared by scene, observations, actions, and rewards.
CONTROLLED_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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

LOWER_BODY_JOINT_NAMES = CONTROLLED_JOINT_NAMES[:15]

ACTION_SCALE_BY_JOINT = {
    "left_hip_pitch_joint": 0.30,
    "left_hip_roll_joint": 0.20,
    "left_hip_yaw_joint": 0.10,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": 0.30,
    "left_ankle_roll_joint": 0.20,
    "right_hip_pitch_joint": 0.30,
    "right_hip_roll_joint": 0.20,
    "right_hip_yaw_joint": 0.10,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": 0.30,
    "right_ankle_roll_joint": 0.20,
    "waist_yaw_joint": 0.20,
    "waist_roll_joint": 0.20,
    "waist_pitch_joint": 0.20,
    "left_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 0.30,
    "left_shoulder_yaw_joint": 0.30,
    "left_elbow_joint": 0.50,
    "left_wrist_roll_joint": 0.30,
    "left_wrist_pitch_joint": 0.30,
    "left_wrist_yaw_joint": 0.30,
    "right_shoulder_pitch_joint": 0.50,
    "right_shoulder_roll_joint": 0.30,
    "right_shoulder_yaw_joint": 0.30,
    "right_elbow_joint": 0.50,
    "right_wrist_roll_joint": 0.30,
    "right_wrist_pitch_joint": 0.30,
    "right_wrist_yaw_joint": 0.30,
}
ACTION_SCALE = [ACTION_SCALE_BY_JOINT[name] for name in CONTROLLED_JOINT_NAMES]

SAFE_STAND_POSE = {
    "left_hip_pitch_joint": -0.15,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.15,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.55,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.20,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.55,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# q_ref for the deploy policy contract. This must remain aligned with deploy catch_ready.
CATCH_READY_POSE = {
    "left_hip_pitch_joint": -0.15,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.15,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.50,
    "left_shoulder_roll_joint": 0.50,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.50,
    "right_shoulder_roll_joint": -0.50,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
READY_POSE = CATCH_READY_POSE

HOLD_POSE = {
    "left_hip_pitch_joint": -0.15,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.15,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": -0.30,
    "left_shoulder_roll_joint": 0.40,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.80,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.30,
    "right_shoulder_roll_joint": -0.40,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.80,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

LOCKED_FINGER_POSE = {
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
}

POLICY_OBS_COMPONENT_DIMS = {
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "joint_pos_rel": 29,
    "joint_vel": 29,
    "previous_action": 29,
    "object_rel_pos": 3,
    "object_rel_lin_vel": 3,
    "tag_visible": 1,
    "mode_one_hot": 4,
}

EXPECTED_ACTION_DIM = len(CONTROLLED_JOINT_NAMES)
EXPECTED_POLICY_OBS_DIM = sum(POLICY_OBS_COMPONENT_DIMS.values())

OBJECT_BASE_SIZE = (0.34, 0.26, 0.24)
OBJECT_DEFAULT_MASS = 3.2

GROUND_PHYSICS_MATERIAL = RigidBodyMaterialCfg(
    static_friction=0.90,
    dynamic_friction=0.80,
    restitution=0.0,
)

assert EXPECTED_ACTION_DIM == 29
assert EXPECTED_POLICY_OBS_DIM == 104


dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={**CATCH_READY_POSE, **LOCKED_FINGER_POSE},
    ),
    actuators={
        "legs_and_waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            stiffness=120.0,
            damping=10.0,
        ),
        "arms_pitch_elbow": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_pitch_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "right_elbow_joint",
            ],
            stiffness=85.0,
            damping=12.0,
        ),
        "arms_pose": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            stiffness=55.0,
            damping=10.0,
        ),
        "fingers_lock": ImplicitActuatorCfg(
            joint_names_expr=list(LOCKED_FINGER_POSE.keys()),
            stiffness=300.0,
            damping=60.0,
        ),
    },
)

# The nominal object stays close to the real delivery-box geometry. Per-episode mass/material
# randomization is applied in mdp.events; runtime physical resizing is not safely supported by
# IsaacLab while the simulation is playing, so size randomization is tracked as best-effort state.
bulky_object_cfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=OBJECT_BASE_SIZE,
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_DEFAULT_MASS),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.22, 0.66, 0.26)),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.80,
            dynamic_friction=0.72,
            restitution=0.02,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.60, 0.0, -0.55)),
)


contact_torso_cfg = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link",
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
