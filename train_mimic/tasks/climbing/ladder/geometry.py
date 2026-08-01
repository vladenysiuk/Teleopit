"""Rung geometry factory for the climbing ladder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco

from train_mimic.tasks.climbing.config.ladder import LadderConfig

# MuJoCo default cylinder axis is +Z. Rotate +90 deg about X so the rung spans +Y.
_RUNG_CYLINDER_QUAT_WXYZ = (0.70710678, 0.70710678, 0.0, 0.0)


class RungGeometry(Protocol):
    """Protocol for substitutable rung collision profiles."""

    kind: str

    def add_to_body(
        self,
        body: mujoco.MjsBody,
        *,
        name: str,
        cfg: LadderConfig,
        rgba: tuple[float, float, float, float],
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> mujoco.MjsGeom: ...


@dataclass(frozen=True)
class CylinderRungGeometry:
    """Pure cylindrical rung collision geometry."""

    kind: str = "cylinder"

    def add_to_body(
        self,
        body: mujoco.MjsBody,
        *,
        name: str,
        cfg: LadderConfig,
        rgba: tuple[float, float, float, float],
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> mujoco.MjsGeom:
        geom = body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(cfg.rung_radius, cfg.rung_length * 0.5, 0.0),
            pos=pos,
            quat=_RUNG_CYLINDER_QUAT_WXYZ,
            rgba=rgba,
            contype=1,
            conaffinity=1,
        )
        geom.friction[:] = cfg.rung_friction
        return geom


def create_rung_geometry(kind: str) -> RungGeometry:
    """Return a rung geometry implementation for *kind*."""
    if kind == "cylinder":
        return CylinderRungGeometry()
    raise ValueError(
        f"Unsupported ladder rung geometry_kind={kind!r}. "
        "Only 'cylinder' is implemented."
    )
