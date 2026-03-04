#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_v11/__init__.py]
import gymnasium as gym

# 학습용 환경 (커리큘럼 자동 진행)
gym.register(
    id="Isaac-Urop-v11",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v11_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV11PPORunnerCfg",
    },
)

# 평가(Play)용 환경 (난이도 고정)
gym.register(
    id="Isaac-Urop-v11-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_v11_EnvCfg_Play", # 🔥 Play 클래스 사용
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropV11PPORunnerCfg",
    },
)