"""Assisted contact-pose configuration for Stage 6 validation.

Stage 6 validates ``assisted_hold``: a deterministic IK pose, one-shot write,
force-seeded hand latches, and stiffened hold-env PD. This establishes a finite,
briefly supported ladder contact pose. Sustained four-contact standing with
training-parity gains is deferred to learnability experiments (Stage 9+).

Thresholds in this module are **frozen** for Stage 6 gate evidence. Do not drift
them without owner approval. Stricter diagnostic goals for future experiments are
recorded separately in ``FUTURE_DIAGNOSTIC_GOALS`` and are not Stage 6 blockers.
"""

from __future__ import annotations

from dataclasses import dataclass


# Future diagnostic targets (learnability / training-parity), not Stage 6 gates.
FUTURE_DIAGNOSTIC_GOALS: dict[str, float] = {
    "max_pelvis_drop_m": 0.10,
    "max_foot_slip_m": 0.05,
    "min_foot_contact_fraction": 0.80,
    "max_hand_attach_error_m": 0.02,
    "max_leg_saturation_fraction": 0.50,
}


@dataclass(frozen=True)
class HoldPoseConfig:
    """Frozen ``assisted_hold`` parameters and Stage 6 acceptance thresholds."""

    # Contact assignment on the fixed debug ladder (6 active rungs by default).
    # Feet on rung 1, hands two spacings up on rung 3: upright enough that
    # flat foot orientation is feasible (avoids the deep 1→2 crouch wedge).
    left_foot_rung_id: int = 1
    right_foot_rung_id: int = 1
    left_hand_rung_id: int = 3
    right_hand_rung_id: int = 3
    left_hand_site_id: int = 1
    right_hand_site_id: int = 3
    left_foot_y_m: float = -0.08
    right_foot_y_m: float = 0.08

    # G1 ``left_foot`` / ``right_foot`` sole sites sit at capsule bottoms
    # (+0.04 m body-X, −0.035 m body-Z from ``*_ankle_roll_link``).
    # Plant soles on the cylinder top-front (toward the robot).
    foot_approach_offset_m: float = 0.018
    # Sole targets above the cylinder surface so the ankle/heel volume clears.
    foot_clearance_m: float = 0.014
    # Extra root-Z after ankle plant if soles still sit below their targets.
    spawn_root_lift_m: float = 0.0
    # Pelvis seed / task distance in front of the ladder frame (−ladder X).
    pelvis_standoff_m: float = 0.28
    # Pelvis height blend: weight_foot * foot_h + weight_hand * hand_h + offset.
    pelvis_height_foot_weight: float = 0.85
    pelvis_height_hand_weight: float = 0.15
    pelvis_height_offset_m: float = 0.38
    # Post-IK ankle plant (exact set, then root realigned to sole targets).
    ankle_pitch_plant_rad: float = -0.10
    ankle_pitch_min_rad: float = -0.10
    ankle_roll_limit_rad: float = 0.0

    # IK solver.
    ik_dt: float = 0.02
    ik_max_iters: int = 320
    ik_damping: float = 1.0e-2
    ik_solver: str = "daqp"
    hand_position_cost: float = 10.0
    foot_position_cost: float = 10.0
    # Flat foot: ankle body axes aligned with ladder frame (+X toward ladder).
    foot_orientation_cost: float = 5.0
    pelvis_position_cost: float = 5.0
    torso_orientation_cost: float = 3.0
    posture_cost: float = 0.15

    # Stage-6-allowed PD stiffening vs stock G1 training actuators.
    pd_stiffness_scale: float = 8.0
    pd_damping_scale: float = 6.0

    # Hold simulation.
    hold_duration_s: float = 2.5
    settle_steps: int = 16

    # Stage 6 acceptance thresholds (frozen assisted_hold gates).
    # 0.135 m pelvis motion and ~0.071 m foot crawl do NOT establish stable
    # standing; they only show brief assisted support under stiffened hold PD.
    max_pelvis_drop_m: float = 0.16
    max_foot_slip_m: float = 0.18
    min_foot_contact_fraction: float = 0.40
    min_any_foot_contact_fraction: float = 0.80
    max_hand_attach_error_m: float = 0.04
    max_leg_saturation_fraction: float = 0.85
    # Sole-site-to-axis radial distance when planted on the cylinder top-front.
    foot_contact_axis_tol_m: float = 0.080

    # Optional Stage-6-allowed friction / radius bump for cylinder standing.
    rung_radius_m: float = 0.035
    rung_friction: tuple[float, float, float] = (2.2, 0.005, 0.0001)
    foot_friction: float = 2.0

    def __post_init__(self) -> None:
        if self.hold_duration_s < 2.0:
            raise ValueError("hold_duration_s must be at least 2.0 s for Stage 6.")
        if self.left_foot_rung_id < 0 or self.right_foot_rung_id < 0:
            raise ValueError("Foot rung ids must be non-negative.")
        if self.left_hand_rung_id < 0 or self.right_hand_rung_id < 0:
            raise ValueError("Hand rung ids must be non-negative.")
        if self.left_hand_site_id < 0 or self.right_hand_site_id < 0:
            raise ValueError("Hand site ids must be non-negative.")
        if self.max_pelvis_drop_m <= 0.0 or self.max_foot_slip_m <= 0.0:
            raise ValueError("Drop/slip thresholds must be positive.")
