# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class UropV5PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 250
    experiment_name = "UROP_v5"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        #actor_hidden_dims=[256, 128, 64],
        actor_hidden_dims=[512, 256, 128],
        #critic_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=32,
        # num_mini_batches=8,
        learning_rate=3.0e-4, #5.0e-4->1.0e-4
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0, #1.0->0.5
    )
     # (4) action clip을 여기서 강제로 걸어주기 (train_rsl_rl.py가 agent_cfg.clip_actions를 씀)
    clip_actions = 1.0  # [-1,1]로 자르기(보통 float max abs로 씀)
