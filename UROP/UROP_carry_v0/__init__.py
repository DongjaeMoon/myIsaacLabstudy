#[/home/dongjae/isaaclab/myIsaacLabstudy/UROP/UROP_carry_v0/__init__.py]
import gymnasium as gym

# 학습용 환경 (커리큘럼 자동 진행)
gym.register(
    id="Isaac-Urop-carry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_carry_v0_EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropCarryV0PPORunnerCfg",
    },
)

# 평가(Play)용 환경 (난이도 고정)
gym.register(
    id="Isaac-Urop-carry-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:dj_urop_carry_v0_EnvCfg_Play", # 🔥 Play 클래스 사용
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UropCarryV0PPORunnerCfg",
    },
)
