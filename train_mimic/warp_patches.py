"""Compatibility patches for mujoco_warp / Warp.

Applied before mjlab imports so Warp codegen sees the fixed source.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# mujoco_warp 3.8.x `_frame_axis` UNKNOWN branch references ``xmat`` without
# defining it. Warp analyzes every branch at codegen time and fails with:
#   WarpCodegenKeyError: Referencing undefined symbol: xmat
# This blocks env construction (CUDA graph capture) on climbing and other
# contact-sensor-heavy scenes, especially under multi-GPU concurrent compile.
_BUGGY_FRAME_AXIS_UNKNOWN = (
    "  else:  # UNKNOWN\n"
    "    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])\n"
)
_FIXED_FRAME_AXIS_UNKNOWN = (
    "  else:  # UNKNOWN\n"
    "    xmat = wp.identity(3, dtype=wp.float32)\n"
    "    axis = wp.vec3(xmat[0, frame_axis], xmat[1, frame_axis], xmat[2, frame_axis])\n"
)


def _sensor_py_path() -> Path | None:
    spec = importlib.util.find_spec("mujoco_warp._src.sensor")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin)
    return path if path.is_file() else None


def apply_mujoco_warp_sensor_patches(*, verbose: bool = False) -> bool:
    """Patch installed ``mujoco_warp`` sensor.py if the ``_frame_axis`` bug is present.

    Idempotent. Must run **before** ``import mujoco_warp`` / ``import mjlab`` so
    Warp compiles the fixed source. Returns True when the file was modified.
    """
    path = _sensor_py_path()
    if path is None:
        if verbose:
            print("[warp_patches] mujoco_warp._src.sensor not found; skip")
        return False

    text = path.read_text(encoding="utf-8")
    if _FIXED_FRAME_AXIS_UNKNOWN in text and _BUGGY_FRAME_AXIS_UNKNOWN not in text:
        if verbose:
            print(f"[warp_patches] already patched: {path}")
        return False
    if _BUGGY_FRAME_AXIS_UNKNOWN not in text:
        if verbose:
            print(f"[warp_patches] no known buggy _frame_axis pattern in {path}")
        return False

    patched = text.replace(_BUGGY_FRAME_AXIS_UNKNOWN, _FIXED_FRAME_AXIS_UNKNOWN, 1)
    if patched == text:
        return False
    path.write_text(patched, encoding="utf-8")
    if verbose:
        print(f"[warp_patches] patched _frame_axis UNKNOWN branch in {path}")
    return True
