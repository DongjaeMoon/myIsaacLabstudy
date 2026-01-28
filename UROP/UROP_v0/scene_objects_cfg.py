import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import math

# =========================================================
# 상호작용할 물체 (빨간 공)
# =========================================================


#ROBOT (G1)
dj_robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # ★여기에 합체한 USD 파일 경로를 넣으세요!
        usd_path="",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3), # 로봇이 땅에 파묻히지 않게 살짝 띄움
        # 관절 초기 각도 (필요하면 추가)
        joint_pos={
            
        },
    ),
    actuators={
        
        
    },
)



