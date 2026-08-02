"""General-Climbing-G1 task package.

Staged implementation (Stages 0–10). See ``train_mimic/tasks/climbing/README.md``
and ``STAGE10_HANDOFF.md`` for documentation and validation commands.
"""

from train_mimic.tasks.climbing.config.constants import (
    CLIMBING_TASK_ID,
    SUPPORTED_CLIMBING_TASKS,
)
from train_mimic.tasks.climbing.config.ladder import LadderConfig

__all__ = ["CLIMBING_TASK_ID", "LadderConfig", "SUPPORTED_CLIMBING_TASKS"]
