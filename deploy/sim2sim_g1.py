# 파일 위치: ~/isaaclab/myIsaacLabstudy/deploy/sim2sim_g1.py

import torch
import numpy as np
from collections import deque

# 1. Isaac Sim 앱 실행
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import omni.appwindow
from pxr import UsdLux, Sdf # [추가됨] 조명용 라이브러리

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
import isaacsim.core.utils.prims as prim_utils
from isaacsim.sensors.physics import ContactSensor
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

# --- 설정값 ---
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v1/2026-02-05_23-14-31/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v1/usd/G1_23DOF_UROP.usd"
PHYSICS_DT = 1.0 / 120.0
DECIMATION = 2
POLICY_DT = PHYSICS_DT * DECIMATION

class G1Sim2Sim:
    def __init__(self):
        # 1. World 생성
        self.world = World(stage_units_in_meters=1.0)
        self.world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=POLICY_DT)

        # [추가] 바닥과 조명
        self.world.scene.add_default_ground_plane()
        
        stage = self.world.stage
        light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        light.CreateIntensityAttr(1000.0)

        # 2. G1 로봇 불러오기
        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path="/World/G1")
        
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/G1",
                name="g1"
            )
        )
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.78]))

        # 3. Contact Sensor 추가
        self.contact_sensors = {}
        sensor_paths = [
            "/World/G1/torso_link",
            "/World/G1/left_wrist_roll_rubber_hand",
            "/World/G1/right_wrist_roll_rubber_hand"
        ]
        for path in sensor_paths:
            name = path.split("/")[-1]
            self.contact_sensors[name] = self.world.scene.add(
                ContactSensor(
                    prim_path=f"{path}/contact_sensor",
                    name=f"contact_{name}",
                    min_threshold=0.0,
                    max_threshold=100000.0,
                    radius=0.05,
                    translation=np.array([0, 0, 0]),
                )
            )

        # 4. 박스 (던질 물체) 준비
        self.box = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name="box",
                position=np.array([2.0, 0.0, 1.0]),
                scale=np.array([0.4, 0.3, 0.3]),
                mass=5.0,
                color=np.array([0.0, 0.8, 0.0])
            )
        )

        self.world.reset()

        # 5. Policy 로드
        print(f"Loading Policy: {POLICY_PATH}")
        self.policy = torch.jit.load(POLICY_PATH).to("cuda")
        self.policy.eval()

        # 6. Action Scaling & Joint Order Mapping
        self.joint_names = self.robot.dof_names
        print(f"Robot Joint Names: {self.joint_names}")
        
        self.action_scale = torch.ones(23, device="cuda")
        self.default_pos = torch.zeros(23, device="cuda")

        init_pos_dict = {
            "hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2,
            "shoulder_pitch": 0.2, "elbow": 0.5
        }

        for i, name in enumerate(self.joint_names):
            if "shoulder" in name or "elbow" in name or "wrist" in name:
                self.action_scale[i] = 1.5
            else:
                self.action_scale[i] = 1.0

            for key, val in init_pos_dict.items():
                if key in name:
                    self.default_pos[i] = val
        
        # 컨트롤러 사용법으로 초기 위치 설정
        action = ArticulationAction(joint_positions=self.default_pos.cpu().numpy())
        self.robot.get_articulation_controller().apply_action(action)

        # 7. 키보드 입력
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.T:
            self.throw_box()

    def throw_box(self):
        print("!!! 박스 투척 !!!")
        self.box.set_world_pose(position=np.array([2.2, 0.0, 1.0]))
        self.box.set_linear_velocity(np.array([-5.0, 0.0, 2.0])) 

    def get_observation(self):
        raw_pos = self.robot.get_joint_positions()
        raw_vel = self.robot.get_joint_velocities()
        
        j_pos = torch.tensor(raw_pos, device="cuda", dtype=torch.float32)
        j_vel = torch.tensor(raw_vel, device="cuda", dtype=torch.float32)
        
        base_pos, base_quat = self.robot.get_world_pose()
        base_lin_vel = self.robot.get_linear_velocity()
        base_ang_vel = self.robot.get_angular_velocity()
        
        t_base_lin = torch.tensor(base_lin_vel, device="cuda", dtype=torch.float32)
        t_base_ang = torch.tensor(base_ang_vel, device="cuda", dtype=torch.float32)
        t_base_quat = torch.tensor(base_quat, device="cuda", dtype=torch.float32)

        forces = []
        for name in ["torso_link", "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"]:
            sensor = self.contact_sensors[name]
            reading = sensor.get_current_frame()
            
            f_tensor = torch.zeros(3, device="cuda", dtype=torch.float32)
            if "force" in reading:
                val = reading["force"]
                try:
                    if hasattr(val, "__len__") and len(val) == 3:
                        f_tensor = torch.tensor(val, device="cuda", dtype=torch.float32)
                    elif isinstance(val, np.ndarray) and val.size == 3:
                        f_tensor = torch.tensor(val, device="cuda", dtype=torch.float32)
                except Exception:
                    pass
            forces.append(f_tensor)
            
        t_forces = torch.cat(forces) * (1.0/300.0)

        box_pos, _ = self.box.get_world_pose()
        box_vel = self.box.get_linear_velocity()
        
        rel_pos = torch.tensor(box_pos - base_pos, device="cuda", dtype=torch.float32)
        rel_vel = torch.tensor(box_vel - base_lin_vel, device="cuda", dtype=torch.float32)
        t_obj_rel = torch.cat([rel_pos, rel_vel])

        obs = torch.cat([
            j_pos, j_vel, t_base_lin, t_base_ang, t_base_quat, t_forces, t_obj_rel
        ])
        
        return obs

    def run(self):
        print("Simulation Start...")
        while simulation_app.is_running():
            obs = self.get_observation()
            with torch.no_grad():
                action = self.policy(obs.unsqueeze(0)).squeeze(0)

            target_pos = action * self.action_scale + self.default_pos
            
            for _ in range(DECIMATION):
                # [수정] ArticulationAction 사용
                act = ArticulationAction(joint_positions=target_pos.cpu().numpy())
                self.robot.get_articulation_controller().apply_action(act)
                self.world.step(render=False)
            
            self.world.render()

if __name__ == "__main__":
    sim = G1Sim2Sim()
    sim.run()
    simulation_app.close()