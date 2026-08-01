"""Curriculum configuration for General-Climbing-G1 learnability experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace

from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.latch import LatchConfig


@dataclass(frozen=True)
class CurriculumConfig:
    """Configuration-only curriculum axes for staged climbing experiments.

    Each ``randomize_*`` flag controls whether the corresponding range is
    sampled at reset. Easy presets keep randomization disabled and expose only
    the fixed values needed for the first learnability experiment.
    """

    # --- Axis toggles (expand curriculum one axis at a time) ---
    randomize_rung_count: bool = False
    randomize_spacing: bool = False
    randomize_ladder_pose: bool = False
    randomize_rung_physics: bool = False
    randomize_initial_pose: bool = False
    randomize_latch_capture: bool = False

    # --- Initial latch curriculum ---
    initial_hand_attach_prob: float = 1.0
    """Per-hand probability of force-seeding an attached latch at reset."""

    # --- Ladder layout (fixed values when the matching axis is disabled) ---
    max_rungs: int = 8
    min_active_rungs: int = 6
    max_active_rungs: int = 6
    spacing_min: float = 0.25
    spacing_max: float = 0.25
    ladder_distance_range: tuple[float, float] = (0.85, 0.85)
    ladder_yaw_range: tuple[float, float] = (0.0, 0.0)
    ladder_tilt_range: tuple[float, float] = (0.0, 0.0)

    # --- Rung physics (compile-time radius; sliding friction can vary at reset) ---
    rung_radius: float = 0.035
    rung_radius_range: tuple[float, float] = (0.030, 0.035)
    rung_friction: tuple[float, float, float] = (2.2, 0.005, 0.0001)
    rung_sliding_friction_range: tuple[float, float] = (1.8, 2.4)

    # --- Initial pose (assisted hold IK + optional noise) ---
    use_hold_pose_reset: bool = True
    hold_pose: HoldPoseConfig | None = None
    root_pos_noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    root_yaw_noise_rad: float = 0.0
    joint_pos_noise_std: float = 0.0

    # --- Latch capture / overload curriculum ---
    latch_capture_radius: float = 0.07
    latch_capture_radius_range: tuple[float, float] = (0.05, 0.09)
    latch_break_force: float | None = None

    # --- Optional foot friction bump for cylinder standing (Stage 6 allowed) ---
    foot_sliding_friction: float | None = 2.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.initial_hand_attach_prob <= 1.0):
            raise ValueError("initial_hand_attach_prob must be in [0, 1].")
        if not (0 < self.min_active_rungs <= self.max_active_rungs <= self.max_rungs):
            raise ValueError("Require 0 < min_active_rungs <= max_active_rungs <= max_rungs.")
        if self.spacing_min <= 0 or self.spacing_max < self.spacing_min:
            raise ValueError("spacing_min must be positive and spacing_max >= spacing_min.")
        if self.rung_radius <= 0:
            raise ValueError("rung_radius must be positive.")
        lo, hi = self.rung_radius_range
        if lo <= 0 or hi < lo:
            raise ValueError("rung_radius_range must be positive with max >= min.")
        lo, hi = self.rung_sliding_friction_range
        if lo <= 0 or hi < lo:
            raise ValueError("rung_sliding_friction_range must be positive with max >= min.")
        lo, hi = self.latch_capture_radius_range
        if lo <= 0 or hi < lo:
            raise ValueError("latch_capture_radius_range must be positive with max >= min.")
        if self.latch_break_force is not None and self.latch_break_force <= 0:
            raise ValueError("latch_break_force must be positive when enabled.")
        if self.foot_sliding_friction is not None and self.foot_sliding_friction <= 0:
            raise ValueError("foot_sliding_friction must be positive when set.")

    @property
    def resolved_hold_pose(self) -> HoldPoseConfig:
        return self.hold_pose or HoldPoseConfig()


def easy_curriculum_cfg() -> CurriculumConfig:
    """Stage 9 easy experiment: fixed vertical ladder, assisted hold start, both hands attached."""
    hold = HoldPoseConfig(
        rung_radius_m=0.035,
        rung_friction=(2.2, 0.005, 0.0001),
        foot_friction=2.0,
    )
    return CurriculumConfig(
        randomize_rung_count=False,
        randomize_spacing=False,
        randomize_ladder_pose=False,
        randomize_rung_physics=False,
        randomize_initial_pose=False,
        randomize_latch_capture=False,
        initial_hand_attach_prob=1.0,
        max_rungs=8,
        min_active_rungs=6,
        max_active_rungs=6,
        spacing_min=0.25,
        spacing_max=0.25,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
        rung_radius=0.035,
        rung_radius_range=(0.035, 0.035),
        rung_friction=(2.2, 0.005, 0.0001),
        rung_sliding_friction_range=(2.2, 2.2),
        use_hold_pose_reset=True,
        hold_pose=hold,
        root_pos_noise_std=(0.0, 0.0, 0.0),
        root_yaw_noise_rad=0.0,
        joint_pos_noise_std=0.0,
        latch_capture_radius=0.07,
        latch_capture_radius_range=(0.07, 0.07),
        foot_sliding_friction=2.0,
    )


def curriculum_ladder_cfg(curriculum: CurriculumConfig) -> LadderConfig:
    """Build a ``LadderConfig`` from curriculum fixed values and enabled ranges."""
    spacing_min = curriculum.spacing_min
    spacing_max = curriculum.spacing_max
    if not curriculum.randomize_spacing:
        spacing_max = spacing_min

    min_rungs = curriculum.min_active_rungs
    max_rungs = curriculum.max_active_rungs
    if not curriculum.randomize_rung_count:
        max_rungs = min_rungs

    distance = curriculum.ladder_distance_range
    yaw = curriculum.ladder_yaw_range
    tilt = curriculum.ladder_tilt_range
    if not curriculum.randomize_ladder_pose:
        distance = (distance[0], distance[0])
        yaw = (0.0, 0.0)
        tilt = (0.0, 0.0)

    radius = curriculum.rung_radius
    if curriculum.randomize_rung_physics:
        radius = max(curriculum.rung_radius_range)

    return LadderConfig(
        max_rungs=curriculum.max_rungs,
        min_active_rungs=min_rungs,
        max_active_rungs=max_rungs,
        rung_radius=radius,
        spacing_min=spacing_min,
        spacing_max=spacing_max,
        ladder_distance_range=distance,
        ladder_yaw_range=yaw,
        ladder_tilt_range=tilt,
        rung_friction=curriculum.rung_friction,
    )


def curriculum_latch_cfg(curriculum: CurriculumConfig) -> LatchConfig:
    """Build a ``LatchConfig`` from curriculum latch parameters."""
    capture = curriculum.latch_capture_radius
    if curriculum.randomize_latch_capture:
        capture = max(curriculum.latch_capture_radius_range)
    return LatchConfig(
        capture_radius=capture,
        break_force=curriculum.latch_break_force,
    )


def with_curriculum_axis(
    curriculum: CurriculumConfig,
    *,
    randomize_rung_count: bool | None = None,
    randomize_spacing: bool | None = None,
    randomize_ladder_pose: bool | None = None,
    randomize_rung_physics: bool | None = None,
    randomize_initial_pose: bool | None = None,
    randomize_latch_capture: bool | None = None,
    initial_hand_attach_prob: float | None = None,
) -> CurriculumConfig:
    """Return a copy with selected curriculum axes enabled/disabled."""
    updates: dict[str, bool | float] = {}
    if randomize_rung_count is not None:
        updates["randomize_rung_count"] = randomize_rung_count
    if randomize_spacing is not None:
        updates["randomize_spacing"] = randomize_spacing
    if randomize_ladder_pose is not None:
        updates["randomize_ladder_pose"] = randomize_ladder_pose
    if randomize_rung_physics is not None:
        updates["randomize_rung_physics"] = randomize_rung_physics
    if randomize_initial_pose is not None:
        updates["randomize_initial_pose"] = randomize_initial_pose
    if randomize_latch_capture is not None:
        updates["randomize_latch_capture"] = randomize_latch_capture
    if initial_hand_attach_prob is not None:
        updates["initial_hand_attach_prob"] = initial_hand_attach_prob
    return replace(curriculum, **updates)
