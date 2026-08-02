"""Stage 10 regression suite metadata and guard checks."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_climbing_stage_test_modules_exist() -> None:
    """Every staged gate (0-9) must have a dedicated pytest module."""
    paths = sorted(REPO_ROOT.glob("tests/test_climbing_stage*.py"))
    stages = {int(p.stem.replace("test_climbing_stage", "")) for p in paths}
    assert stages == set(range(10)), f"expected stages 0-9, got {sorted(stages)}"


def test_regression_runner_script_exists() -> None:
    script = REPO_ROOT / "scripts/dev/run_climbing_regression.py"
    assert script.is_file(), "missing scripts/dev/run_climbing_regression.py"


def test_handoff_documentation_exists() -> None:
    handoff = REPO_ROOT / "train_mimic/tasks/climbing/STAGE10_HANDOFF.md"
    readme = REPO_ROOT / "train_mimic/tasks/climbing/README.md"
    assert handoff.is_file()
    assert readme.is_file()
    text = handoff.read_text(encoding="utf-8")
    for section in (
        "## Architecture",
        "## Debug modes",
        "## Final validation matrix",
        "## Known backend limitations",
    ):
        assert section in text, f"missing section {section!r} in STAGE10_HANDOFF.md"


def test_tracking_registry_unchanged() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.app import DEFAULT_TASK
    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    assert DEFAULT_TASK == GENERAL_TRACKING_TASK
    load_env_cfg(GENERAL_TRACKING_TASK)


def test_climbing_task_registered_alongside_tracking() -> None:
    pytest.importorskip("mjlab")
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from train_mimic.app import SUPPORTED_TASKS
    from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
    from train_mimic.tasks.tracking.config.constants import GENERAL_TRACKING_TASK

    assert CLIMBING_TASK_ID in SUPPORTED_TASKS
    load_env_cfg(CLIMBING_TASK_ID)
    load_env_cfg(GENERAL_TRACKING_TASK)
