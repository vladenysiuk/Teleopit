"""Stage 0 checks for the climbing task package skeleton."""

from __future__ import annotations


def test_climbing_package_imports_and_task_constants() -> None:
    from train_mimic.tasks.climbing import CLIMBING_TASK_ID, SUPPORTED_CLIMBING_TASKS
    from train_mimic.tasks.climbing.config.registry import CLIMBING_EXPERIMENT_NAME

    assert CLIMBING_TASK_ID == "General-Climbing-G1"
    assert CLIMBING_EXPERIMENT_NAME == "g1_general_climbing"
    assert SUPPORTED_CLIMBING_TASKS == (CLIMBING_TASK_ID,)


def test_tracking_registry_unchanged() -> None:
    import pytest

    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.app import DEFAULT_TASK
    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    assert DEFAULT_TASK == GENERAL_TRACKING_TASK
    load_env_cfg(GENERAL_TRACKING_TASK)
