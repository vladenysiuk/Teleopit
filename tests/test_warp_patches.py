"""Tests for mujoco_warp compatibility patches."""

from __future__ import annotations

from pathlib import Path

import pytest

from train_mimic import warp_patches


def test_frame_axis_patch_is_idempotent_on_installed_package() -> None:
    path = warp_patches._sensor_py_path()
    if path is None:
        pytest.skip("mujoco_warp not installed")
    text = path.read_text(encoding="utf-8")
    if (
        warp_patches._BUGGY_FRAME_AXIS_UNKNOWN not in text
        and warp_patches._FIXED_FRAME_AXIS_UNKNOWN not in text
    ):
        pytest.skip("installed mujoco_warp has neither known buggy nor fixed pattern")

    # First call may patch or no-op; second must no-op.
    warp_patches.apply_mujoco_warp_sensor_patches()
    assert warp_patches.apply_mujoco_warp_sensor_patches() is False
    assert warp_patches._FIXED_FRAME_AXIS_UNKNOWN in path.read_text(encoding="utf-8")
    assert warp_patches._BUGGY_FRAME_AXIS_UNKNOWN not in path.read_text(encoding="utf-8")


def test_frame_axis_patch_rewrites_temp_sensor_file(tmp_path: Path, monkeypatch) -> None:
    sensor = tmp_path / "sensor.py"
    sensor.write_text(
        "def _frame_axis():\n"
        + warp_patches._BUGGY_FRAME_AXIS_UNKNOWN
        + "  return axis\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(warp_patches, "_sensor_py_path", lambda: sensor)
    assert warp_patches.apply_mujoco_warp_sensor_patches() is True
    text = sensor.read_text(encoding="utf-8")
    assert warp_patches._FIXED_FRAME_AXIS_UNKNOWN in text
    assert warp_patches._BUGGY_FRAME_AXIS_UNKNOWN not in text
    assert warp_patches.apply_mujoco_warp_sensor_patches() is False
