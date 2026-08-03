"""G1 robot configuration for General-Climbing-G1."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from teleopit.runtime.assets import UNITREE_G1_XML, missing_gmr_assets_message
from train_mimic.tasks.tracking.config.env import resolve_g1_training_xml

# Collision bit groups (MuJoCo: collide when contype_A & conaffinity_B).
ROBOT_BODY_CONTYPE = 1
HAND_POINT_CONTYPE = 2
HAND_POINT_CONAFFINITY = 4
LADDER_CONTYPE = 4
# Include LADDER_CONTYPE so mujoco_warp matches both filter directions vs ladder geoms.
ROBOT_BODY_CONAFFINITY = ROBOT_BODY_CONTYPE | LADDER_CONTYPE
LADDER_CONAFFINITY = ROBOT_BODY_CONTYPE | HAND_POINT_CONTYPE  # robot body + hand points

# Tighter than Stage-2 initial 25 mm sphere to reduce false-positive hand contacts.
HAND_POINT_RADIUS = 0.015
HAND_POINT_RGBA = (0.95, 0.25, 0.15, 0.9)

LEFT_WRIST_BODY = "left_wrist_yaw_link"
RIGHT_WRIST_BODY = "right_wrist_yaw_link"
LEFT_HAND_POINT = "left_hand_point"
RIGHT_HAND_POINT = "right_hand_point"
LEFT_HAND_POINT_SITE = "left_hand_point_site"
RIGHT_HAND_POINT_SITE = "right_hand_point_site"

HAND_POINT_OFFSETS: dict[str, tuple[float, float, float]] = {
    LEFT_WRIST_BODY: (0.18, -0.025, 0.0),
    RIGHT_WRIST_BODY: (0.18, 0.025, 0.0),
}

HAND_POINT_GEOM_NAMES = (LEFT_HAND_POINT, RIGHT_HAND_POINT)
HAND_NAMES = ("left", "right")
HAND_INDEX = {"left": 0, "right": 1}


def _get_g1_climbing_spec(robot_xml: str | Path | None = None) -> mujoco.MjSpec:
    xml_path = resolve_g1_training_xml(robot_xml)
    if not xml_path.is_file():
        raise FileNotFoundError(
            missing_gmr_assets_message(xml_path, label="G1 climbing MuJoCo XML")
        )
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)
    floor = spec.geom("floor")
    spec.delete(floor)

    for wrist_body, point_name, site_name in (
        (LEFT_WRIST_BODY, LEFT_HAND_POINT, LEFT_HAND_POINT_SITE),
        (RIGHT_WRIST_BODY, RIGHT_HAND_POINT, RIGHT_HAND_POINT_SITE),
    ):
        body = spec.body(wrist_body)
        offset = HAND_POINT_OFFSETS[wrist_body]
        body.add_geom(
            name=point_name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(HAND_POINT_RADIUS, 0.0, 0.0),
            pos=offset,
            rgba=HAND_POINT_RGBA,
            contype=HAND_POINT_CONTYPE,
            conaffinity=HAND_POINT_CONAFFINITY,
            condim=3,
            margin=0.0,
            gap=0.0,
        )
        body.add_site(
            name=site_name,
            pos=offset,
            size=(HAND_POINT_RADIUS * 0.5, HAND_POINT_RADIUS * 0.5, HAND_POINT_RADIUS * 0.5),
            rgba=HAND_POINT_RGBA,
        )
    return spec


def make_climbing_robot_cfg(robot_xml: str | Path | None = None) -> EntityCfg:
    """Canonical G1 training robot with point-hand collision geoms."""
    robot_cfg = get_g1_robot_cfg()
    robot_cfg = deepcopy(robot_cfg)
    xml_path = resolve_g1_training_xml(robot_xml)
    robot_cfg.spec_fn = partial(_get_g1_climbing_spec, xml_path)
    robot_cfg.collisions = (
        CollisionCfg(
            geom_names_expr=(r".*_collision$",),
            contype=ROBOT_BODY_CONTYPE,
            conaffinity=ROBOT_BODY_CONAFFINITY,
            condim=3,
            priority={r"^(left|right)_foot[1-7]_collision$": 1},
            friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
            disable_other_geoms=False,
        ),
    )
    return robot_cfg


def make_ladder_collision_cfg() -> CollisionCfg:
    """Collision groups so ladder rungs and rails hit robot body and hand points."""
    return CollisionCfg(
        geom_names_expr=(r".*_geom$", r"rail_.*"),
        contype=LADDER_CONTYPE,
        conaffinity=LADDER_CONAFFINITY,
        condim=3,
        disable_other_geoms=False,
    )


__all__ = [
    "G1_ACTION_SCALE",
    "HAND_INDEX",
    "HAND_NAMES",
    "HAND_POINT_CONAFFINITY",
    "HAND_POINT_CONTYPE",
    "HAND_POINT_GEOM_NAMES",
    "HAND_POINT_OFFSETS",
    "HAND_POINT_RADIUS",
    "LADDER_CONAFFINITY",
    "LADDER_CONTYPE",
    "LEFT_HAND_POINT",
    "LEFT_HAND_POINT_SITE",
    "LEFT_WRIST_BODY",
    "RIGHT_HAND_POINT",
    "RIGHT_HAND_POINT_SITE",
    "RIGHT_WRIST_BODY",
    "UNITREE_G1_XML",
    "make_climbing_robot_cfg",
    "make_ladder_collision_cfg",
]
