from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass


@configclass
class UropV22PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 64 steps at 50 Hz = 1.28 s rollout. This covers wait -> react -> contact.
    num_steps_per_env = 64
    max_iterations = 6000
    save_interval = 250
    experiment_name = "UROP_v22"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.28,
        actor_hidden_dims=[256, 256, 128],
        # Critic sees privileged object/contact/domain terms, so it gets a larger first layer.
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0035,
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
