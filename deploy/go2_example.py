# [/home/dongjae/isaaclab/myIsaacLabstudy/deploy/go2_example.py]

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
import numpy as np # [추가] 배열 계산을 위해 numpy 추가

class Go2Example(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # G1 로봇 로드
        # (경로는 동재님 컴퓨터 환경에 맞게 유지했습니다)
        add_reference_to_stage(
            usd_path="/home/dongjae/isaaclab/myIsaacLabstudy/example/urop_v0/usd/dj_robotarm_on_go2.usd", 
            prim_path="/World/Go2"
        )

        robot = world.scene.add(Robot(prim_path="/World/Go2", name="go2"))
        
        # 로봇을 살짝 공중에 띄워서 소환 (바닥에 끼임 방지)
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.5]))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._robot = self._world.scene.get_object("go2")

        # [중요] 로봇이 완전히 준비될 때까지 기다려야 관절 개수를 알 수 있습니다.
        # 물리 엔진이 인식할 때까지 초기화를 확실히 합니다.
        self._robot.initialize() 
        
        # 로봇의 관절 개수(dof)를 가져옵니다. (G1은 23이 나올 겁니다)
        self.num_dof = self._robot.num_dof
        print(f">>> Go2 Loaded: {self.num_dof} DOFs")

        self._world.add_physics_callback("physics_step", callback_fn=self._robot_control)
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return

    def _robot_control(self, step_size):
        # [수정됨] 하드코딩된 리스트 대신, 관절 개수만큼 0으로 채운 배열을 만듭니다.
        # 12개가 아니라 23개(self.num_dof)가 들어갑니다.
        targets = np.zeros(self.num_dof)
        
        # 예시: 특정 관절만 움직여보고 싶다면?
        # targets[3] = 1.0 # 4번째 관절을 1.0으로 움직임

        action = ArticulationAction(joint_positions=targets)
        self._robot.apply_action(action)
        return