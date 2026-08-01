"""Stage 9 easy learnability experiment configuration for General-Climbing-G1."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.spec_config import CollisionCfg

from train_mimic.tasks.climbing.config.curriculum import (
    CurriculumConfig,
    curriculum_ladder_cfg,
    curriculum_latch_cfg,
    easy_curriculum_cfg,
)
from train_mimic.tasks.climbing.config.env import (
    _recommended_nconmax,
    _recommended_njmax,
    make_general_climbing_env_cfg,
)
from train_mimic.tasks.climbing.config.rl import make_general_climbing_ppo_runner_cfg
from train_mimic.tasks.climbing.config.robot import (
    ROBOT_BODY_CONAFFINITY,
    ROBOT_BODY_CONTYPE,
)

# Owner-facing easy experiment defaults.
EASY_NUM_ENVS = 256
EASY_MAX_ITERATIONS = 800
EASY_SAVE_INTERVAL = 200
EASY_SEED = 42
EASY_EPISODE_LENGTH_S = 15.0
EASY_NUM_STEPS_PER_ENV = 24

# Fast CI defaults — same MDP wiring, fewer envs/iterations.
CI_EASY_NUM_ENVS = 4
CI_EASY_MAX_ITERATIONS = 12
CI_EASY_SAVE_INTERVAL = 6


def make_climbing_easy_env_cfg(
    *,
    num_envs: int = EASY_NUM_ENVS,
    seed: int = EASY_SEED,
    curriculum: CurriculumConfig | None = None,
    episode_length_s: float = EASY_EPISODE_LENGTH_S,
    env_spacing: float = 4.0,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Easy fixed-ladder env with assisted hold reset and high cylinder friction."""
    curriculum = curriculum or easy_curriculum_cfg()
    ladder_cfg = curriculum_ladder_cfg(curriculum)
    latch_cfg = curriculum_latch_cfg(curriculum)

    cfg = make_general_climbing_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=ladder_cfg,
        latch_cfg=latch_cfg,
        env_spacing=env_spacing,
        play=play,
        curriculum_cfg=curriculum,
    )
    cfg.episode_length_s = episode_length_s

    if curriculum.foot_sliding_friction is not None:
        robot_cfg = deepcopy(cfg.scene.entities["robot"])
        robot_cfg.collisions = (
            CollisionCfg(
                geom_names_expr=(r".*_collision$",),
                contype=ROBOT_BODY_CONTYPE,
                conaffinity=ROBOT_BODY_CONAFFINITY,
                condim=3,
                priority={r"^(left|right)_foot[1-7]_collision$": 1},
                friction={
                    r"^(left|right)_foot[1-7]_collision$": (curriculum.foot_sliding_friction,),
                },
                disable_other_geoms=False,
            ),
        )
        cfg.scene.entities["robot"] = robot_cfg

    cfg.sim.njmax = _recommended_njmax(ladder_cfg, hold_pose=True)
    cfg.sim.nconmax = _recommended_nconmax(ladder_cfg, hold_pose=True)
    return cfg


def make_climbing_easy_ppo_runner_cfg(
    *,
    max_iterations: int = EASY_MAX_ITERATIONS,
    save_interval: int = EASY_SAVE_INTERVAL,
    num_steps_per_env: int = EASY_NUM_STEPS_PER_ENV,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner settings for the Stage 9 easy learnability experiment."""
    base = make_general_climbing_ppo_runner_cfg()
    return replace(
        base,
        max_iterations=max_iterations,
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
    )
