# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/example.py]

import torch
import numpy as np
import carb

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.types import ArticulationAction

# --------------------------------------------------------------------------
# [설정] sim2sim_g1.py에 있던 상수들을 가져옵니다.
# --------------------------------------------------------------------------
# ★★★ 경로가 맞는지 꼭 확인하세요! ★★★
POLICY_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/logs/rsl_rl/UROP_v1/2026-02-06_20-31-35/exported/policy.pt"
ROBOT_USD_PATH = "/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v3/usd/G1_23DOF_UROP.usd"
ROBOT_PRIM_PATH = "/World/G1"

class Example(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug_mode = True # 디버깅용 로그 출력 여부

    # ----------------------------------------------------------------------
    # 1. 환경 조성 (sim2sim의 __init__ 부분)
    # ----------------------------------------------------------------------
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # (1) 로봇 추가
        add_reference_to_stage(usd_path=ROBOT_USD_PATH, prim_path=ROBOT_PRIM_PATH)
        self.robot = world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="g1"))
        
        # 로봇 초기 위치 (sim2sim: ROBOT_SPAWN_POS)
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.78]))

        # (2) 박스(Target) 추가
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
        return

    # ----------------------------------------------------------------------
    # 2. 뇌 로드 및 초기화 (sim2sim의 __init__ 뒷부분)
    # ----------------------------------------------------------------------
    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("g1")
        self._box = self._world.scene.get_object("box")

        # 물리 엔진 초기화 대기
        self._robot.initialize()
        
        # (1) Policy 로드
        print(f">>> Loading Policy from: {POLICY_PATH}")
        try:
            self.policy = torch.jit.load(POLICY_PATH).to(self.device)
            self.policy.eval()
            print(">>> Policy Loaded Successfully!")
        except Exception as e:
            carb.log_error(f"Policy Load Failed: {e}")
            return

        # (2) 관절 및 제어 변수 설정
        self.num_dof = self._robot.num_dof
        self.joint_names = self._robot.dof_names
        print(f">>> Robot DOFs: {self.num_dof}")

        # 텐서 버퍼 초기화
        self.default_pos = torch.zeros(self.num_dof, device=self.device)
        self.action_scale = torch.zeros(self.num_dof, device=self.device)

        # (3) Default Pose & Gains 설정 (sim2sim 로직 복사)
        # sim2sim에 있던 init_pos_dict
        init_pos_dict = {
            "hip_pitch": -0.2, "knee": 0.4, "ankle_pitch": -0.2,
            "shoulder_pitch": 0.2, "elbow": 0.5
        }

        # 관절별 설정
        for i, name in enumerate(self.joint_names):
            # Gains & Scale
            if ("shoulder" in name) or ("elbow" in name) or ("wrist" in name):
                self.action_scale[i] = 1.5
                # kps, kds는 여기서 설정 안 해도 USD 설정이나 Controller 설정을 따름
                # 필요하다면 self._robot.get_articulation_controller().set_gains(...) 사용
            else:
                self.action_scale[i] = 1.0

            # Default Pose
            for key, val in init_pos_dict.items():
                if key in name:
                    self.default_pos[i] = float(val)

        # (4) 물리 콜백 등록 (매 프레임 실행될 함수)
        self._world.add_physics_callback("physics_step", callback_fn=self._robot_control)
        return

    # ----------------------------------------------------------------------
    # 3. 신경망 연결 (sim2sim의 run loop + get_observation)
    # ----------------------------------------------------------------------
    def _get_observation(self):
        # (1) Joint Data
        # Isaac Sim에서 numpy로 줌 -> torch로 변환
        j_pos_np = self._robot.get_joint_positions()
        j_vel_np = self._robot.get_joint_velocities()
        
        j_pos = torch.tensor(j_pos_np, device=self.device, dtype=torch.float32)
        j_vel = torch.tensor(j_vel_np, device=self.device, dtype=torch.float32)

        # (2) Base Data
        base_pos, base_quat = self._robot.get_world_pose()
        base_lin = torch.tensor(self._robot.get_linear_velocity(), device=self.device, dtype=torch.float32)
        base_ang = torch.tensor(self._robot.get_angular_velocity(), device=self.device, dtype=torch.float32)
        base_quat = torch.tensor(base_quat, device=self.device, dtype=torch.float32) # (w, x, y, z)

        # (3) Contact Forces (sim2sim에서는 기본적으로 0으로 둠)
        # 센서 설정이 복잡하므로 일단 0으로 채움 (대부분의 학습 코드에서 센서 없으면 0 처리함)
        t_forces = torch.zeros(9, device=self.device, dtype=torch.float32)

        # (4) Relative Object Data (Box)
        b_pos, _ = self._box.get_world_pose()
        b_vel = self._box.get_linear_velocity()
        
        # 로봇 기준 상대 위치/속도 계산
        rel_pos = torch.tensor(b_pos - base_pos, device=self.device, dtype=torch.float32)
        # 주의: b_vel은 numpy array일 수 있음
        rel_vel = torch.tensor(b_vel - self._robot.get_linear_velocity(), device=self.device, dtype=torch.float32)

        # (5) Observation 합치기 (순서 중요!)
        # sim2sim 순서: [j_pos, j_vel, base_lin, base_ang, base_quat, t_forces, rel_pos, rel_vel]
        obs = torch.cat([j_pos, j_vel, base_lin, base_ang, base_quat, t_forces, rel_pos, rel_vel])
        return obs

    def _robot_control(self, step_size):
        # 1. 관측 (Observation)
        try:
            obs = self._get_observation()
        except Exception as e:
            # 초기화 덜 됐을 때 에러 방지
            return 

        # 2. 추론 (Inference)
        with torch.no_grad():
            # 모델은 배치 차원(batch dim)을 기대하므로 .unsqueeze(0) 필요
            # obs shape: (N) -> (1, N)
            action = self.policy(obs.unsqueeze(0)).squeeze(0)

        # 3. 행동 변환 (Action -> Joint Target)
        # Target = Action * Scale + Default_Pose
        target_pos = action * self.action_scale + self.default_pos
        
        # 4. 명령 하달 (Apply)
        # numpy로 변환해서 아이작 심에 전달
        target_np = target_pos.detach().cpu().numpy()
        
        action_cmd = ArticulationAction(joint_positions=target_np)
        self._robot.apply_action(action_cmd)
        
        return

    def world_cleanup(self):
        # 종료 시 정리할 게 있다면 작성
        return