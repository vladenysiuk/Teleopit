"""Stage 8 PPO smoke-test configuration for General-Climbing-G1."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from train_mimic.tasks.climbing.config.env import make_general_climbing_env_cfg
from train_mimic.tasks.climbing.config.rl import make_general_climbing_ppo_runner_cfg
from train_mimic.tasks.climbing.debug_mdp import pinned_ladder_cfg

# Owner-facing smoke defaults (Stage 8 plan: 64 envs, ~100 iterations, fixed easy ladder).
SMOKE_NUM_ENVS = 64
SMOKE_MAX_ITERATIONS = 100
SMOKE_SAVE_INTERVAL = 50
SMOKE_SEED = 42
SMOKE_EPISODE_LENGTH_S = 10.0
SMOKE_NUM_STEPS_PER_ENV = 24

# Fast CI defaults — same MDP wiring, fewer envs/iterations.
CI_SMOKE_NUM_ENVS = 4
CI_SMOKE_MAX_ITERATIONS = 3
CI_SMOKE_SAVE_INTERVAL = 2


def make_climbing_smoke_env_cfg(
    *,
    num_envs: int = SMOKE_NUM_ENVS,
    seed: int = SMOKE_SEED,
    episode_length_s: float = SMOKE_EPISODE_LENGTH_S,
    env_spacing: float = 4.0,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Fixed easy ladder, no broad domain randomization, production control rate."""
    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=pinned_ladder_cfg(),
        env_spacing=env_spacing,
        play=play,
    )
    cfg.episode_length_s = episode_length_s
    return cfg


def make_climbing_smoke_ppo_runner_cfg(
    *,
    max_iterations: int = SMOKE_MAX_ITERATIONS,
    save_interval: int = SMOKE_SAVE_INTERVAL,
    num_steps_per_env: int = SMOKE_NUM_STEPS_PER_ENV,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner settings for the Stage 8 infrastructure smoke test."""
    base = make_general_climbing_ppo_runner_cfg()
    return replace(
        base,
        max_iterations=max_iterations,
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
    )
