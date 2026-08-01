"""Ladder configuration for General-Climbing-G1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LadderConfig:
    """Fixed-topology ladder parameters and sampling ranges."""

    max_rungs: int = 12
    min_active_rungs: int = 4
    max_active_rungs: int = 10

    rung_radius: float = 0.030
    rung_length: float = 0.45
    rung_friction: tuple[float, float, float] = (1.2, 0.005, 0.0001)

    base_height: float = 0.30
    spacing_min: float = 0.22
    spacing_max: float = 0.28

    ladder_distance: float = 0.80
    ladder_distance_range: tuple[float, float] = (0.60, 1.00)
    ladder_yaw_range: tuple[float, float] = (-0.30, 0.30)
    ladder_tilt_range: tuple[float, float] = (-0.05, 0.05)

    rail_radius: float = 0.020
    rail_half_width: float = 0.24

    sites_per_rung: int = 5
    inactive_rung_pose: tuple[float, float, float] = (0.0, 0.0, -50.0)

    # Latch target sites sit on the robot-facing side of each rung so a connect
    # equality aligns the hand-sphere centre with an external resting pose
    # rather than the rung axis (which would fight hand–rung collision).
    latch_site_clearance: float = 0.0015

    geometry_kind: str = "cylinder"

    def __post_init__(self) -> None:
        if self.geometry_kind != "cylinder":
            raise ValueError(
                f"Only geometry_kind='cylinder' is supported now, got {self.geometry_kind!r}."
            )
        if self.max_rungs <= 0:
            raise ValueError("max_rungs must be positive.")
        if not (0 < self.min_active_rungs <= self.max_active_rungs <= self.max_rungs):
            raise ValueError("Require 0 < min_active_rungs <= max_active_rungs <= max_rungs.")
        if self.spacing_min <= 0 or self.spacing_max < self.spacing_min:
            raise ValueError("spacing_min must be positive and spacing_max >= spacing_min.")
        if self.sites_per_rung < 2:
            raise ValueError("sites_per_rung must be at least 2.")
        if self.rung_radius <= 0 or self.rung_length <= 0:
            raise ValueError("rung_radius and rung_length must be positive.")
        if self.latch_site_clearance < 0.0:
            raise ValueError("latch_site_clearance must be non-negative.")

    @property
    def max_ladder_height(self) -> float:
        """Conservative rail span for the compiled fixed topology."""
        return self.base_height + (self.max_rungs - 1) * self.spacing_max
