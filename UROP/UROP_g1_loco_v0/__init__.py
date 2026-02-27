import gymnasium as gym
##
# Register Gym environments.
##

# NOTE:
# - id는 네가 train/play에서 --task로 넣는 문자열과 동일해야 함.
# - entry_point는 IsaacLab manager-based env 고정.
gym.register(
    id="Isaac-Urop-g1-loco-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "UROP_g1_loco_v0.env_cfg:dj_urop_g1_loco_v0_EnvCfg",
        "rsl_rl_cfg_entry_point": "UROP_g1_loco_v0.agents.rsl_rl_ppo_cfg:UropG1LOCOv0PPORunnerCfg",
    },
)
