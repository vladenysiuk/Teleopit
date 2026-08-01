"""Debug visualization helpers for ladder scene inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from train_mimic.tasks.climbing.config.robot import (
    HAND_POINT_RADIUS,
    HAND_POINT_RGBA,
    LEFT_HAND_POINT,
    RIGHT_HAND_POINT,
)
from train_mimic.tasks.climbing.debug_probe import rung_mocap_pos_numpy
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

    from train_mimic.tasks.climbing.ladder.state import LadderRuntime


def _quat_apply_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a wxyz unit quaternion."""
    rot = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot, quat_wxyz)
    return rot.reshape(3, 3) @ vec


def attach_ladder_debug_visualizer(env: ManagerBasedRlEnv) -> None:
    """Attach an ``update_visualizers`` hook that highlights active rung sites."""

    def update_visualizers(visualizer: DebugVisualizer) -> None:
        from train_mimic.tasks.climbing.ladder.generator import expected_site_names
        from train_mimic.tasks.climbing.ladder.state import LadderRuntime

        runtime = LadderRuntime.get(env)
        if runtime.sample is None:
            return

        model = env.sim.mj_model
        prefix = runtime.topology.entity_prefix
        env_idx = visualizer.env_idx
        sample = runtime.sample

        for rung_id in range(runtime.cfg.max_rungs):
            if not bool(sample.active_mask[env_idx, rung_id].item()):
                continue

            rung_pos = np.asarray(
                sample.rung_pos_w[env_idx, rung_id].detach().cpu(),
                dtype=np.float64,
            )
            rung_quat = np.asarray(
                sample.rung_quat_w[env_idx, rung_id].detach().cpu(),
                dtype=np.float64,
            )

            body_name = f"{prefix}rung_{rung_id:02d}"
            for site_name in expected_site_names(runtime.cfg, rung_id):
                site_id = model.site(f"{prefix}{site_name}").id
                local = np.asarray(model.site_pos[site_id], dtype=np.float64)
                world = rung_pos + _quat_apply_wxyz(rung_quat, local)
                visualizer.add_sphere(
                    center=world,
                    radius=0.012,
                    color=(0.10, 0.85, 0.95, 0.85),
                    label=f"{body_name}_site",
                )

            visualizer.add_sphere(
                center=rung_pos,
                radius=runtime.cfg.rung_radius * 0.35,
                color=(1.0, 0.9, 0.2, 0.9),
                label=f"rung_{rung_id:02d}",
            )

    env.unwrapped.update_visualizers = update_visualizers  # type: ignore[attr-defined]


def attach_contact_debug_visualizer(env: ManagerBasedRlEnv) -> None:
    """Overlay hand points, contacted rungs, and contact locations."""

    def update_visualizers(visualizer: DebugVisualizer) -> None:
        from train_mimic.tasks.climbing.ladder.state import LadderRuntime

        runtime = LadderRuntime.get(env)
        env_idx = visualizer.env_idx
        robot = env.scene["robot"]

        for point_name in (LEFT_HAND_POINT, RIGHT_HAND_POINT):
            geom_ids, _ = robot.find_geoms([point_name], preserve_order=True)
            center = (
                robot.data.geom_pos_w[env_idx, geom_ids[0]].detach().cpu().numpy().astype(np.float64)
            )
            visualizer.add_sphere(
                center=center,
                radius=HAND_POINT_RADIUS,
                color=HAND_POINT_RGBA,
                label=point_name,
            )

        contacts = env.extras.get(ClimbContactState.EXTRA_KEY)
        if contacts is None or contacts.state is None:
            return

        state = contacts.state
        for hand_idx, color in ((0, (0.2, 0.9, 0.3, 0.95)), (1, (0.9, 0.3, 0.2, 0.95))):
            if not bool(state.hand_in_contact[env_idx, hand_idx].item()):
                continue
            rung_id = int(state.hand_rung_id[env_idx, hand_idx].item())
            rung_pos = rung_mocap_pos_numpy(env, rung_id, env_idx)
            visualizer.add_sphere(
                center=rung_pos,
                radius=runtime.cfg.rung_radius * 1.05,
                color=color,
                label=f"contact_rung_{rung_id:02d}",
            )
            contact_pos = (
                state.hand_contact_pos[env_idx, hand_idx].detach().cpu().numpy().astype(np.float64)
            )
            visualizer.add_sphere(
                center=contact_pos,
                radius=0.015,
                color=(1.0, 1.0, 0.2, 0.95),
                label=f"contact_point_{hand_idx}",
            )

    env.unwrapped.update_visualizers = update_visualizers  # type: ignore[attr-defined]
