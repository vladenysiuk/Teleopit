"""PPO runner configuration for General-Climbing-G1.

Stage 4 wires the climbing model and observation groups. Full trainable PPO
integration remains gated to Stage 8; ``max_iterations=1`` keeps registry
registration safe without implying a production training run.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from train_mimic.tasks.climbing.config.constants import CLIMBING_EXPERIMENT_NAME

_CLIMBING_MODEL_CLASS = (
    "train_mimic.tasks.climbing.rl.climbing_model:ClimbingModel"
)
_CNN_CFG: dict = {
    "output_channels": (256, 128, 64),
    "kernel_size": 3,
    "activation": "elu",
    "global_pool": "avg",
    "ladder_encoder_cfg": {
        "mode": "relative_rungs",
        "output_dim": 64,
        "hidden_channels": (32, 64),
        "kernel_size": 3,
        "activation": "elu",
    },
}


def make_general_climbing_ppo_runner_cfg(
    experiment_name: str = CLIMBING_EXPERIMENT_NAME,
) -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for General-Climbing-G1."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            class_name=_CLIMBING_MODEL_CLASS,
            hidden_dims=(2048, 1024, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            class_name=_CLIMBING_MODEL_CLASS,
            hidden_dims=(2048, 1024, 512, 256, 128),
            activation="elu",
            obs_normalization=True,
            cnn_cfg=_CNN_CFG,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={
            "actor": ("actor_proprio", "actor_proprio_history", "actor_ladder"),
            "critic": (
                "critic_proprio",
                "critic_proprio_history",
                "critic_ladder",
                "critic_privileged",
            ),
        },
        experiment_name=experiment_name,
        save_interval=2000,
        num_steps_per_env=24,
        max_iterations=1,
        logger="tensorboard",
        upload_model=False,
    )
