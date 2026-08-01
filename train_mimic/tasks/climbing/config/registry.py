"""Registry wiring for General-Climbing-G1."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.climbing.config.constants import (
    CLIMBING_EXPERIMENT_NAME,
    CLIMBING_TASK_ID,
)
from train_mimic.tasks.climbing.config.env import make_general_climbing_env_cfg
from train_mimic.tasks.climbing.config.rl import make_general_climbing_ppo_runner_cfg

register_mjlab_task(
    task_id=CLIMBING_TASK_ID,
    env_cfg=make_general_climbing_env_cfg(),
    play_env_cfg=make_general_climbing_env_cfg(play=True),
    rl_cfg=make_general_climbing_ppo_runner_cfg(
        experiment_name=CLIMBING_EXPERIMENT_NAME
    ),
    runner_cls=None,
)

__all__ = [
    "CLIMBING_EXPERIMENT_NAME",
    "CLIMBING_TASK_ID",
]
