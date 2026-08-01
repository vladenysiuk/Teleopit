"""Latch configuration for General-Climbing-G1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatchConfig:
    """Attach/detach latch thresholds and capture parameters."""

    attach_threshold: float = 0.5
    """Latch command above this value requests attach."""

    detach_threshold: float = -0.5
    """Latch command below this value requests detach."""

    capture_radius: float = 0.07
    """Maximum hand-to-site distance (m) for attach eligibility.

    With five sites on a 0.45 m rung the max lateral gap to the nearest site is
    about 0.056 m; 0.07 m covers contact plus discretization without large snaps.
    """

    break_force: float | None = None
    """Optional overload detach force (N). ``None`` disables force-based break."""

    solref: tuple[float, float] = (0.02, 1.0)
    """Connect constraint solver reference (timeconst, dampratio)."""

    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    """Connect constraint solver impedance."""

    def __post_init__(self) -> None:
        if self.detach_threshold >= self.attach_threshold:
            raise ValueError(
                "detach_threshold must be strictly less than attach_threshold."
            )
        if self.capture_radius <= 0.0:
            raise ValueError("capture_radius must be positive.")
        if self.break_force is not None and self.break_force <= 0.0:
            raise ValueError("break_force must be positive when enabled.")
