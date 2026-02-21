import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##

# NOTE:
# - id는 네가 train/play에서 --task로 넣는 문자열과 동일해야 함.
# - entry_point는 IsaacLab manager-based env 고정.
gym.register(
    id="Isaac-Urop-v5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "UROP_v5.env_cfg:dj_urop_v5_EnvCfg",
        "rsl_rl_cfg_entry_point": "UROP_v5.agents.rsl_rl_ppo_cfg:UropV5PPORunnerCfg",
    },
)
