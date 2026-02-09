# ~/isaaclab/myIsaacLabstudy/deploy/sim2sim_g1.py

import torch
import numpy as np
import carb

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.appwindow
import omni.timeline
from pxr import UsdLux, Sdf

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.physics import ContactSensor
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction


# ----------------- 네가 원래 쓰던 상수 경로 유지 -----------------
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v1/2026-02-05_23-14-31/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v1/usd/G1_23DOF_UROP.usd"
ROBOT_PRIM_PATH = "/World/G1"

PHYSICS_DT = 1.0 / 120.0
DECIMATION = 2
POLICY_DT = PHYSICS_DT * DECIMATION

# Play 시작 시 몇 프레임 “물리 view 생성 워밍업”
PHYSICS_WARMUP_STEPS = 3

# 로봇 초기 위치(지면 관통 방지)
ROBOT_SPAWN_POS = np.array([0.0, 0.0, 1.05], dtype=np.float32)


class G1Sim2Sim:
    def __init__(self):
        # World
        self.world = World(stage_units_in_meters=1.0)
        self.world.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=POLICY_DT)

        # Ground + Light
        self.world.scene.add_default_ground_plane()
        stage = self.world.stage
        light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        light.CreateIntensityAttr(1000.0)

        # Robot (USD reference)
        print(f">>> Loading Robot from: {ROBOT_USD_PATH}")
        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = self.world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))

        # Box
        self.box = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/Object",
                name="box",
                position=np.array([2.0, 0.0, 1.0], dtype=np.float32),
                scale=np.array([0.4, 0.3, 0.3], dtype=np.float32),
                mass=5.0,
                color=np.array([0.0, 0.8, 0.0], dtype=np.float32),
            )
        )

        # Contact sensors (링크 이름이 USD에 실제로 존재해야 함)
        self.contact_sensors = {}
        sensor_link_paths = [
            f"{ROBOT_PRIM_PATH}/torso_link",
            f"{ROBOT_PRIM_PATH}/left_wrist_roll_rubber_hand",
            f"{ROBOT_PRIM_PATH}/right_wrist_roll_rubber_hand",
        ]
        for link_path in sensor_link_paths:
            name = link_path.split("/")[-1]
            self.contact_sensors[name] = self.world.scene.add(
                ContactSensor(
                    prim_path=f"{link_path}/contact_sensor",
                    name=f"contact_{name}",
                    min_threshold=0.0,
                    max_threshold=100000.0,
                    radius=0.05,
                    translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                )
            )

        # Policy
        print(f">>> Loading Policy: {POLICY_PATH}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = torch.jit.load(POLICY_PATH).to(self.device)
        self.policy.eval()

        # World reset (stage/scene 등록)
        self.world.reset()

        # DOF & gains & default pose
        self.joint_names = self.robot.dof_names
        self.num_dof = len(self.joint_names)
        print(f"Robot Joint Names ({self.num_dof}): {self.joint_names}")

        self.action_scale = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self.default_pos = torch.zeros(self.num_dof, device=self.device, dtype=torch.float32)
        self.kps = np.zeros(self.num_dof, dtype=np.float32)
        self.kds = np.zeros(self.num_dof, dtype=np.float32)

        init_pos_dict = {
            "hip_pitch": -0.2,
            "knee": 0.4,
            "ankle_pitch": -0.2,
            "shoulder_pitch": 0.2,
            "elbow": 0.5,
        }

        for i, name in enumerate(self.joint_names):
            if ("shoulder" in name) or ("elbow" in name) or ("wrist" in name):
                self.action_scale[i] = 1.5
                self.kps[i] = 40.0
                self.kds[i] = 2.0
            else:
                self.action_scale[i] = 1.0
                self.kps[i] = 300.0
                self.kds[i] = 10.0

            for key, val in init_pos_dict.items():
                if key in name:
                    self.default_pos[i] = float(val)

        # Keyboard
        self._throw_queued = False
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(keyboard, self._on_key)

        # Timeline + flags
        self.timeline = omni.timeline.get_timeline_interface()
        self._was_playing = False
        self._needs_reinit_on_play = True
        self.is_robot_initialized = False

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input == carb.input.KeyboardInput.T:
            # Stop/Pause 중이면 큐에 넣고, Play 중이면 즉시 던짐
            if self.timeline.is_playing():
                self.throw_box()
            else:
                print("[T] queued (will throw when Play starts)")
                self._throw_queued = True

    def throw_box(self):
        print("!!! 박스 투척 !!!")
        self.box.set_world_pose(position=np.array([2.2, 0.0, 1.0], dtype=np.float32))
        self.box.set_linear_velocity(np.array([-5.0, 0.0, 2.0], dtype=np.float32))

    def _apply_gains(self):
        try:
            self.robot.get_articulation_controller().set_gains(kps=self.kps, kds=self.kds)
        except Exception:
            pass

    def _reset_robot_once(self) -> bool:
        """physics view 준비되면 True, 아니면 False."""
        try:
            self.robot.initialize()  # 핵심: articulation view 생성
        except Exception:
            self.is_robot_initialized = False
            return False

        # pose/vel reset (Stop 후에도 지면 관통 방지)
        self.robot.set_world_pose(position=ROBOT_SPAWN_POS)
        self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))

        # joints reset
        self.robot.set_joint_positions(self.default_pos.detach().cpu().numpy())
        self.robot.set_joint_velocities(np.zeros(self.num_dof, dtype=np.float32))

        # gains
        self._apply_gains()

        self.is_robot_initialized = True
        return True

    def _ensure_ready_after_play(self):
        """
        Play 시작 직후: world.reset + 몇 step으로 physics view 생성 시간을 확보하고
        initialize 성공할 때까지 재시도.
        """
        # Stop->Play 때 내부 view가 날아갔을 수 있으니 reset
        self.world.reset()

        # physics view 생성 워밍업
        for _ in range(PHYSICS_WARMUP_STEPS):
            self.world.step(render=False)

        # initialize 재시도 (너무 오래 끌면 안 되니 횟수 제한)
        for _ in range(30):
            if self._reset_robot_once():
                print(">>> Robot re-initialized on Play.")
                return
            self.world.step(render=False)

        # 여기까지 왔다면 아직도 view가 안 생긴 것
        print("[WARN] Robot initialize still not ready. Will retry in next frames.")
        self.is_robot_initialized = False

    def get_observation(self):
        if not self.is_robot_initialized:
            return None

        try:
            raw_pos = self.robot.get_joint_positions()
            raw_vel = self.robot.get_joint_velocities()
        except Exception:
            return None

        if raw_pos is None or raw_vel is None:
            return None

        j_pos = torch.tensor(raw_pos, device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(raw_vel, device=self.device, dtype=torch.float32)

        base_pos, base_quat = self.robot.get_world_pose()
        base_lin = torch.tensor(self.robot.get_linear_velocity(), device=self.device, dtype=torch.float32)
        base_ang = torch.tensor(self.robot.get_angular_velocity(), device=self.device, dtype=torch.float32)
        base_quat = torch.tensor(base_quat, device=self.device, dtype=torch.float32)

        forces = []
        for name in ["torso_link", "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand"]:
            sensor = self.contact_sensors.get(name, None)
            f = torch.zeros(3, device=self.device, dtype=torch.float32)
            if sensor is not None:
                try:
                    reading = sensor.get_current_frame()
                    if isinstance(reading, dict) and ("force" in reading):
                        val = reading["force"]
                        if hasattr(val, "__len__") and len(val) == 3:
                            f = torch.tensor(val, device=self.device, dtype=torch.float32)
                except Exception:
                    pass
            forces.append(f)
        t_forces = torch.cat(forces) * (1.0 / 300.0)

        b_pos, _ = self.box.get_world_pose()
        b_vel = self.box.get_linear_velocity()

        rel_pos = torch.tensor(b_pos - base_pos, device=self.device, dtype=torch.float32)
        rel_vel = torch.tensor(b_vel - self.robot.get_linear_velocity(), device=self.device, dtype=torch.float32)

        obs = torch.cat([j_pos, j_vel, base_lin, base_ang, base_quat, t_forces, rel_pos, rel_vel])
        return obs

    def run(self):
        print("Simulation Ready. Press 'Play' in GUI. (Press 'T' to throw box)")
        while simulation_app.is_running():
            is_playing = self.timeline.is_playing()

            # Play 전환 감지
            if is_playing and (not self._was_playing):
                self._needs_reinit_on_play = True

            if is_playing:
                if self._needs_reinit_on_play:
                    self._ensure_ready_after_play()
                    self._needs_reinit_on_play = False

                    # queued throw 처리
                    if self._throw_queued:
                        self.throw_box()
                        self._throw_queued = False

                # 아직 초기화 안 됐으면 physics만 굴려서 다음 프레임에 준비되게
                if not self.is_robot_initialized:
                    self.world.step(render=False)
                else:
                    # 워밍업: 첫 몇 step은 default pose 고정(활어/폭발 감소)
                    if self.world.current_time_step_index < 3:
                        self._apply_gains()
                        act = ArticulationAction(joint_positions=self.default_pos.detach().cpu().numpy())
                        self.robot.get_articulation_controller().apply_action(act)
                        self.world.step(render=False)
                    else:
                        obs = self.get_observation()
                        if obs is None:
                            # obs None이어도 physics는 진행
                            self.world.step(render=False)
                        else:
                            with torch.no_grad():
                                action = self.policy(obs.unsqueeze(0)).squeeze(0)

                            target = action * self.action_scale + self.default_pos
                            target_np = target.detach().cpu().numpy()

                            # DECIMATION step
                            for _ in range(DECIMATION):
                                act = ArticulationAction(joint_positions=target_np)
                                self.robot.get_articulation_controller().apply_action(act)
                                self.world.step(render=False)

            else:
                # Pause/Stop 상태: view는 유지될 수도 있지만 Stop이면 깨질 수 있으니 다음 Play 때 재초기화
                self.is_robot_initialized = False

            # 렌더링은 항상 1번
            self.world.render()
            self._was_playing = is_playing


if __name__ == "__main__":
    sim = G1Sim2Sim()
    sim.run()
    simulation_app.close()
