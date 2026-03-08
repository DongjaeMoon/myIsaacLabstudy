#[/home/idim5080-2/mdj/myIsaacLabstudy/deploy/catch/controllers/config_loader.py]
import fnmatch
import io
import sys
from typing import Dict, List, Tuple
import numpy as np

import carb
import omni
import yaml

def parse_env_config(env_config_path: str = "env.yaml") -> dict:
    class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
        def ignore_unknown(self, node) -> None:
            return None
        def tuple_constructor(loader, node) -> tuple:
            return tuple(loader.construct_sequence(node))

    SafeLoaderIgnoreUnknown.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderIgnoreUnknown.tuple_constructor)
    SafeLoaderIgnoreUnknown.add_constructor(None, SafeLoaderIgnoreUnknown.ignore_unknown)

    file_content = omni.client.read_file(env_config_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    data = yaml.load(file, Loader=SafeLoaderIgnoreUnknown)
    return data

def get_robot_joint_properties(data: dict, joint_names: List[str]):
    actuator_data = data.get("scene").get("robot").get("actuators")

    stiffness_dict, damping_dict, effort_dict, velocity_dict = {}, {}, {}, {}
    armature_dict = {}
    default_pos_dict, default_vel_dict = {}, {}

    for actuator in actuator_data.values():
        joint_names_expr = actuator.get("joint_names_expr", [])

        effort_limit = actuator.get("effort_limit_sim") or actuator.get("effort_limit")
        velocity_limit = actuator.get("velocity_limit_sim") or actuator.get("velocity_limit")
        joint_stiffness = actuator.get("stiffness")
        joint_damping = actuator.get("damping")
        joint_armature = actuator.get("armature", 0.0)

        if isinstance(joint_armature, (float, int)) or joint_armature is None:
            val = 0.0 if joint_armature is None else float(joint_armature)
            for pattern in joint_names_expr:
                armature_dict[pattern] = val
        elif isinstance(joint_armature, dict):
            armature_dict.update(joint_armature)

        if isinstance(effort_limit, (float, int)) or effort_limit is None:
            val = float(sys.maxsize) if effort_limit is None or effort_limit == float("inf") else float(effort_limit)
            for pattern in joint_names_expr:
                effort_dict[pattern] = val
        elif isinstance(effort_limit, dict):
            effort_dict.update(effort_limit)

        if isinstance(velocity_limit, (float, int)) or velocity_limit is None:
            val = float(sys.maxsize) if velocity_limit is None or velocity_limit == float("inf") else float(velocity_limit)
            for pattern in joint_names_expr:
                velocity_dict[pattern] = val
        elif isinstance(velocity_limit, dict):
            velocity_dict.update(velocity_limit)

        if isinstance(joint_stiffness, (float, int)) or joint_stiffness is None:
            val = 0.0 if joint_stiffness is None else float(joint_stiffness)
            for pattern in joint_names_expr:
                stiffness_dict[pattern] = val
        elif isinstance(joint_stiffness, dict):
            stiffness_dict.update(joint_stiffness)

        if isinstance(joint_damping, (float, int)) or joint_damping is None:
            val = 0.0 if joint_damping is None else float(joint_damping)
            for pattern in joint_names_expr:
                damping_dict[pattern] = val
        elif isinstance(joint_damping, dict):
            damping_dict.update(joint_damping)

    init_joint_pos = data.get("scene").get("robot").get("init_state").get("joint_pos")
    if isinstance(init_joint_pos, (float, int)):
        for pattern in stiffness_dict.keys():
            default_pos_dict[pattern] = float(init_joint_pos)
    elif isinstance(init_joint_pos, dict):
        default_pos_dict.update(init_joint_pos)

    init_joint_vel = data.get("scene").get("robot").get("init_state").get("joint_vel")
    if isinstance(init_joint_vel, (float, int)):
        for pattern in stiffness_dict.keys():
            default_vel_dict[pattern] = float(init_joint_vel)
    elif isinstance(init_joint_vel, dict):
        default_vel_dict.update(init_joint_vel)

    stiffness_inorder, damping_inorder = [], []
    effort_limits_inorder, velocity_limits_inorder = [], []
    armature_inorder = []
    default_pos_inorder, default_vel_inorder = [], []

    def find_val(joint, val_dict, default_val):
        for pattern, p_val in val_dict.items():
            if fnmatch.fnmatch(joint, pattern.replace(".", "*") + "*"):
                return float(p_val)
        return default_val

    for joint in joint_names:
        stiffness_inorder.append(find_val(joint, stiffness_dict, 0.0))
        damping_inorder.append(find_val(joint, damping_dict, 0.0))
        effort_limits_inorder.append(find_val(joint, effort_dict, float(sys.maxsize)))
        velocity_limits_inorder.append(find_val(joint, velocity_dict, float(sys.maxsize)))
        armature_inorder.append(find_val(joint, armature_dict, 0.0))
        default_pos_inorder.append(find_val(joint, default_pos_dict, 0.0))
        default_vel_inorder.append(find_val(joint, default_vel_dict, 0.0))

    # [핵심 수정] 무조건 numpy 배열로 강제 리턴
    return (
        np.array(effort_limits_inorder, dtype=np.float32),
        np.array(velocity_limits_inorder, dtype=np.float32),
        np.array(stiffness_inorder, dtype=np.float32),
        np.array(damping_inorder, dtype=np.float32),
        np.array(armature_inorder, dtype=np.float32),
        np.array(default_pos_inorder, dtype=np.float32),
        np.array(default_vel_inorder, dtype=np.float32),
    )

def get_articulation_props(data: dict) -> dict:
    return data.get("scene").get("robot").get("spawn").get("articulation_props")

def get_physics_properties(data: dict) -> dict:
    return data.get("decimation"), data.get("sim").get("dt"), data.get("sim").get("render_interval")
