"""Medium climbability preset: easy MDP with a full 12-rung ladder."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from train_mimic.tasks.climbing.config.curriculum import (
    CurriculumConfig,
    medium_curriculum_cfg,
)
from train_mimic.tasks.climbing.config.easy import (
    EASY_EPISODE_LENGTH_S,
    EASY_MAX_ITERATIONS,
    EASY_NUM_ENVS,
    EASY_NUM_STEPS_PER_ENV,
    EASY_SAVE_INTERVAL,
    EASY_SEED,
    make_climbing_easy_env_cfg,
    make_climbing_easy_ppo_runner_cfg,
)

# Same training schedule as easy; only the ladder height / rung count changes.
MEDIUM_NUM_ENVS = EASY_NUM_ENVS
MEDIUM_MAX_ITERATIONS = EASY_MAX_ITERATIONS
MEDIUM_SAVE_INTERVAL = EASY_SAVE_INTERVAL
MEDIUM_SEED = EASY_SEED
MEDIUM_EPISODE_LENGTH_S = EASY_EPISODE_LENGTH_S
MEDIUM_NUM_STEPS_PER_ENV = EASY_NUM_STEPS_PER_ENV


def make_climbing_medium_env_cfg(
    *,
    num_envs: int = MEDIUM_NUM_ENVS,
    seed: int = MEDIUM_SEED,
    curriculum: CurriculumConfig | None = None,
    episode_length_s: float = MEDIUM_EPISODE_LENGTH_S,
    env_spacing: float = 4.0,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Easy assisted-hold env with a fixed full-height 12-rung ladder."""
    return make_climbing_easy_env_cfg(
        num_envs=num_envs,
        seed=seed,
        curriculum=curriculum or medium_curriculum_cfg(),
        episode_length_s=episode_length_s,
        env_spacing=env_spacing,
        play=play,
    )


def make_climbing_medium_ppo_runner_cfg(
    *,
    max_iterations: int = MEDIUM_MAX_ITERATIONS,
    save_interval: int = MEDIUM_SAVE_INTERVAL,
    num_steps_per_env: int = MEDIUM_NUM_STEPS_PER_ENV,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner settings for the medium full-ladder experiment."""
    return make_climbing_easy_ppo_runner_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
    )
