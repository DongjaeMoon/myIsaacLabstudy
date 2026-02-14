# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example_v3.py]

import torch
import numpy as np
import carb
import omni.appwindow
import omni.timeline

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.physics import ContactSensor

# --------------------------------------------------------------------------
# [설정] 경로
# --------------------------------------------------------------------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v3/2026-02-13_19-46-54/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/usd/G1_23DOF_UROP.usd"
ROBOT_PRIM_PATH = "/World/G1"

# env_cfg.py의 Observation에 정의된 센서 순서
CONTACT_SENSOR_LINKS = [
    "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_rubber_hand",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_rubber_hand"
]

class ExampleV3(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_mode = True
        
        # [중요] env_cfg.py -> ActionsCfg에 정의된 순서 그대로 작성해야 합니다.
        self.policy_joint_order = [
            # legs_sagittal
            "left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint",
            "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint",
            # legs_frontal
            "left_hip_roll_joint", "left_ankle_roll_joint",
            "right_hip_roll_joint", "right_ankle_roll_joint",
            # legs_yaw
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            # waist
            "waist_yaw_joint",
            # left_arm_capture
            "left_shoulder_pitch_joint", "left_elbow_joint",
            # right_arm_capture
            "right_shoulder_pitch_joint", "right_elbow_joint",
            # left_arm_wrap
            "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_wrist_roll_joint",
            # right_arm_wrap
            "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_wrist_roll_joint"
        ]

    # ----------------------------------------------------------------------
    # Math Utils (User Warning Fix: added dim=-1)
    # ----------------------------------------------------------------------
    def _quat_rotate_inverse(self, q, v):
        q_w, q_vec = q[0], q[1:]
        a = v * (2.0 * q_w**2 - 1.0)
        b = torch.linalg.cross(q_vec, v, dim=-1) * q_w * 2.0
        c = q_vec * torch.dot(q_vec, v) * 2.0
        return a - b + c

    def _quat_mul(self, q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        return torch.tensor([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], device=self.device)

    def _quat_conj(self, q):
        return torch.tensor([q[0], -q[1], -q[2], -q[3]], device=self.device)

    def _quat_to_rot6d(self, q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)
        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)
        r20 = 2*(x*z - y*w)
        r21 = 2*(y*z + x*w)
        return torch.tensor([r00, r10, r20, r01, r11, r21], device=self.device)

    # ----------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.78]))

        self.box = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name="box",
                position=np.array([2.0, 0.0, 1.0]),
                scale=np.array([0.4, 0.3, 0.3]),
                mass=5.0,
                color=np.array([0.0, 0.8, 0.0]),
            )
        )

        self.contact_sensors = {}
        for link_name in CONTACT_SENSOR_LINKS:
            sensor_prim_path = f"{ROBOT_PRIM_PATH}/{link_name}/contact_sensor"
            self.contact_sensors[link_name] = world.scene.add(
                ContactSensor(
                    prim_path=sensor_prim_path,
                    name=f"contact_{link_name}",
                    min_threshold=0.0,
                    max_threshold=100000.0,
                    radius=0.05,
                    translation=np.array([0.0, 0.0, 0.0]),
                )
            )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")
        self._box = self._world.scene.get_object("box")

        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )
        self._is_running = False

        self._robot.initialize()
        
        print(f">>> Loading Policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        self.num_dof = self._robot.num_dof
        self.robot_dof_names = self._robot.dof_names
        
        # -----------------------------------------------------------
        # [핵심] Joint Mapping (Policy Order -> Robot Order)
        # -----------------------------------------------------------
        self.policy_to_robot_indices = []
        for policy_joint_name in self.policy_joint_order:
            if policy_joint_name in self.robot_dof_names:
                # 정책의 i번째 관절이 로봇의 몇 번째 인덱스인지 찾음
                idx = self.robot_dof_names.index(policy_joint_name)
                self.policy_to_robot_indices.append(idx)
            else:
                carb.log_error(f"Joint {policy_joint_name} not found in robot USD!")
        
        self.policy_to_robot_indices = torch.tensor(self.policy_to_robot_indices, device=self.device, dtype=torch.long)

        # Buffers (Robot Order 기준)
        self.default_pos = torch.zeros(self.num_dof, device=self.device)
        self.action_scale = torch.zeros(self.num_dof, device=self.device)
        
        # Buffers (Policy Order 기준)
        self.prev_action_policy_order = torch.zeros(23, device=self.device) # Policy expects 23 dims

        # -----------------------------------------------------------
        # Scale & Default Pose Setting
        # -----------------------------------------------------------
        init_pos_dict = {
            "hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2,
            "shoulder_pitch": 0.2, "elbow": 0.5
        }
        
        # ActionsCfg.py 기반 스케일
        scale_dict = {
            "hip_pitch": 0.35, "knee": 0.35, "ankle_pitch": 0.35,
            "hip_roll": 0.22, "ankle_roll": 0.22,
            "hip_yaw": 0.10,
            "waist_yaw": 0.25,
            "shoulder_pitch": 1.2, "elbow": 1.2, # capture
            "shoulder_roll": 0.7, "shoulder_yaw": 0.7, "wrist_roll": 0.7 # wrap
        }

        for i, name in enumerate(self.robot_dof_names):
            # Scale
            found_scale = False
            for key, val in scale_dict.items():
                if key in name:
                    self.action_scale[i] = val
                    found_scale = True
                    break
            if not found_scale: self.action_scale[i] = 1.0

            # Default Pose
            for key, val in init_pos_dict.items():
                if key in name:
                    self.default_pos[i] = val

        self._world.add_physics_callback("physics_step", callback_fn=self._robot_control)
        
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        return

    def _on_timeline_event(self, event):
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._is_running = True
            # Play 누르면 리셋
            if self._robot and self._robot.is_valid():
                self._robot.initialize()
                self._robot.set_joint_positions(self.default_pos.detach().cpu().numpy())
                self._robot.set_joint_velocities(np.zeros(self.num_dof))
                self.prev_action_policy_order[:] = 0.0
                
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._is_running = False

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.K:
            self.throw_box()

    def throw_box(self):
        print(">>> Throwing Box!")
        self.box.set_world_pose(position=np.array([0.5, 0.0, 1.2]))
        self.box.set_linear_velocity(np.array([-2.0, 0.0, 0.5]))

    def _get_observation(self):
        # 1. Base
        base_pos, base_quat = self._robot.get_world_pose()
        base_lin_w = torch.tensor(self._robot.get_linear_velocity(), device=self.device)
        base_ang_w = torch.tensor(self._robot.get_angular_velocity(), device=self.device)
        q = torch.tensor(base_quat, device=self.device)

        g_world = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        g_b = self._quat_rotate_inverse(q, g_world)
        lin_b = self._quat_rotate_inverse(q, base_lin_w)
        ang_b = self._quat_rotate_inverse(q, base_ang_w)

        # 2. Joint (Robot Order)
        j_pos = torch.tensor(self._robot.get_joint_positions(), device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(self._robot.get_joint_velocities(), device=self.device, dtype=torch.float32)
        j_torque = torch.zeros_like(j_pos)

        # 3. Proprioception (Policy Order로 변환 필요? -> NO, obs.py 보면 그냥 data.joint_pos 씀)
        # 하지만 Policy Order랑 Robot Order가 다르다면 여기서 재배열 해주는 게 맞음.
        # 학습 코드 mdp/observations.py: return torch.cat([..., jp, jv, jt])
        # 여기서 jp, jv는 IsaacLab이 주는 순서(Robot USD 순서)임.
        # ★ 결론: Observation의 joint_pos는 Robot Order 그대로 둬도 됨. 
        # (Policy가 내부적으로 어떤 순서의 입력을 기대하는지는 Action Order와 별개일 수 있지만, 보통 같음)
        # 안정성을 위해 여기서도 Policy Order로 맞춰주는 게 좋음.
        
        # [Remapping for Obs] Robot Order -> Policy Order
        # 역매핑을 만들어야 하는데 복잡하므로, 일단 Robot Order 그대로 넣음 (보통 학습 시에도 그렇게 들어감)
        proprio = torch.cat([g_b, lin_b, ang_b, j_pos, j_vel, j_torque])

        # 4. Prev Action (Policy Order)
        prev_act = self.prev_action_policy_order

        # 5. Object Rel
        b_pos, b_quat = self._box.get_world_pose()
        b_lin_w = torch.tensor(self._box.get_linear_velocity(), device=self.device)
        b_ang_w = torch.tensor(self._box.get_angular_velocity(), device=self.device)
        r_pos = torch.tensor(base_pos, device=self.device)

        rel_p_b = self._quat_rotate_inverse(q, torch.tensor(b_pos, device=self.device) - r_pos)
        rel_v_b = self._quat_rotate_inverse(q, b_lin_w - base_lin_w)
        rel_w_b = self._quat_rotate_inverse(q, b_ang_w - base_ang_w)
        b_q = torch.tensor(b_quat, device=self.device)
        rel_q = self._quat_mul(self._quat_conj(q), b_q)
        rel_r6 = self._quat_to_rot6d(rel_q)

        obj_rel = torch.cat([rel_p_b, rel_r6, rel_v_b, rel_w_b])

        # 6. Contact
        forces = []
        for name in CONTACT_SENSOR_LINKS:
            sensor = self.contact_sensors.get(name, None)
            val = 0.0
            if sensor and self._is_running:
                try:
                    reading = sensor.get_current_frame()
                    if "force" in reading:
                        val = np.linalg.norm(reading["force"])
                except: pass
            forces.append(val)
        contact = torch.tensor(forces, device=self.device, dtype=torch.float32) * (1.0/300.0)

        return torch.cat([proprio, prev_act, obj_rel, contact])

    def _robot_control(self, step_size):
        if not self._is_running or not self._robot.handles_initialized:
            return

        try:
            obs = self._get_observation()
        except Exception:
            return 

        with torch.no_grad():
            # Policy는 23개의 값을 Policy Order로 뱉어냄
            action_policy_order = self.policy(obs.unsqueeze(0)).squeeze(0)

        action_policy_order = torch.clamp(action_policy_order, -1.0, 1.0)
        self.prev_action_policy_order = action_policy_order.clone()

        # [핵심] Policy Order Action -> Robot Order Action으로 변환
        # Robot Order크기(23)의 빈 텐서를 만들고, 매핑된 인덱스에 값을 채워넣음
        action_robot_order = torch.zeros(self.num_dof, device=self.device)
        
        # policy_to_robot_indices: [Policy 0번 관절의 로봇 인덱스, Policy 1번의 로봇 인덱스 ...]
        # 예: policy[0](left_hip_pitch) -> robot[12]
        action_robot_order[self.policy_to_robot_indices] = action_policy_order

        # 이제 Robot Order로 정렬됐으니 Scale과 Default Pose 적용 가능
        target_pos = action_robot_order * self.action_scale + self.default_pos
        
        target_np = target_pos.detach().cpu().numpy()
        self.robot.apply_action(ArticulationAction(joint_positions=target_np))
        
        return

    def world_cleanup(self):
        self._timeline_sub = None
        self._input.unsubscribe_to_keyboard_events(self._sub_keyboard)
        return