"""General-Climbing-G1 task package (staged implementation).

Stage 1: ladder spec, sampling, scene debug env.
Stage 2: G1 integration, hand contact identity, task registration.
"""

from train_mimic.tasks.climbing.config.constants import (
    CLIMBING_TASK_ID,
    SUPPORTED_CLIMBING_TASKS,
)
from train_mimic.tasks.climbing.config.ladder import LadderConfig

__all__ = ["CLIMBING_TASK_ID", "LadderConfig", "SUPPORTED_CLIMBING_TASKS"]
