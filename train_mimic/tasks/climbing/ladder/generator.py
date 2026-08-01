"""Build the fixed-topology ladder MJCF spec."""

from __future__ import annotations

import mujoco

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.robot import HAND_POINT_RADIUS
from train_mimic.tasks.climbing.ladder.geometry import create_rung_geometry

_RUNG_RGBA = (0.75, 0.45, 0.15, 1.0)
_RAIL_RGBA = (0.35, 0.35, 0.40, 1.0)
_SITE_RGBA = (0.10, 0.85, 0.95, 1.0)


def _site_positions_along_rung(length: float, count: int) -> list[float]:
    """Return Y offsets for attachment sites spanning the rung length."""
    if count < 2:
        raise ValueError("count must be at least 2.")
    half = length * 0.5
    if count == 2:
        return [-half, half]
    step = length / (count - 1)
    return [-half + i * step for i in range(count)]


def site_y_offsets(cfg: LadderConfig) -> tuple[float, ...]:
    """Stable ladder-frame Y offsets for attachment sites on each rung."""
    return tuple(_site_positions_along_rung(cfg.rung_length, cfg.sites_per_rung))


def latch_site_normal_offset(cfg: LadderConfig) -> float:
    """Ladder-frame −X offset from rung axis to latch target (hand-sphere centre).

    Robot approaches from −ladder X, so the resting contact pose is::

        p_target = p_rung_axis − (r_rung + r_hand + δ) * e_x_ladder

    Aligning the hand site with this target keeps the sphere outside the
    cylinder instead of fighting collision by pulling into the centreline.
    """
    return cfg.rung_radius + HAND_POINT_RADIUS + cfg.latch_site_clearance


def build_ladder_spec(cfg: LadderConfig) -> mujoco.MjSpec:
    """Build a fixed-topology ladder spec with one mocap frame, rails, and rungs.

    Rungs are collision geoms on the frame mocap body (not separate mocap bodies).
    MuJoCo only generates contacts for geoms parented directly on a mocap body;
    per-rung mocap bodies update visuals but do not collide.

    Attachment sites are offset from each rung axis toward the robot
    (``−ladder X``) by :func:`latch_site_normal_offset` so connect equalities
    do not contradict hand–rung collision.
    """
    geometry = create_rung_geometry(cfg.geometry_kind)
    spec = mujoco.MjSpec()

    frame = spec.worldbody.add_body(name="frame", mocap=True)
    rail_half_height = cfg.max_ladder_height * 0.5
    for side, y_offset in (("left", -cfg.rail_half_width), ("right", cfg.rail_half_width)):
        rail_geom = frame.add_geom(
            name=f"rail_{side}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=(cfg.rail_radius, rail_half_height, 0.0),
            pos=(0.0, y_offset, rail_half_height),
            rgba=_RAIL_RGBA,
            contype=1,
            conaffinity=1,
        )
        rail_geom.friction[:] = cfg.rung_friction

    site_y_offsets_list = site_y_offsets(cfg)
    site_x = -latch_site_normal_offset(cfg)

    for rung_id in range(cfg.max_rungs):
        rung_name = f"rung_{rung_id:02d}"
        geometry.add_to_body(
            frame,
            name=f"{rung_name}_geom",
            cfg=cfg,
            rgba=_RUNG_RGBA,
        )
        for site_idx, y_offset in enumerate(site_y_offsets_list):
            site = frame.add_site(
                name=f"{rung_name}_site_{site_idx:02d}",
                pos=(site_x, y_offset, 0.0),
                size=(0.008, 0.008, 0.008),
                rgba=_SITE_RGBA,
            )
            site.group = 1

    return spec


def expected_rung_geom_names(cfg: LadderConfig) -> tuple[str, ...]:
    """Stable rung collision-geom names independent of sampled heights."""
    return tuple(f"rung_{i:02d}_geom" for i in range(cfg.max_rungs))


def expected_site_names(cfg: LadderConfig, rung_id: int) -> tuple[str, ...]:
    """Stable attachment-site names for a rung index."""
    prefix = f"rung_{rung_id:02d}_site_"
    return tuple(f"{prefix}{i:02d}" for i in range(cfg.sites_per_rung))
