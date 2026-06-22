from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class UropV26PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # v26 uses low gains, a smaller object, and a compact reward stack, so we keep PPO modest.
    num_steps_per_env = 64
    max_iterations = 4500
    save_interval = 250
    experiment_name = "UROP_v26"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.22,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0020,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    clip_actions = 1.0
