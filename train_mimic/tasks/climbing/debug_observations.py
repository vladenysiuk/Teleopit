"""Debug helpers for climbing observation inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from train_mimic.tasks.climbing.ladder.relative_rungs import RelativeRungProvider
from train_mimic.tasks.climbing.ladder.state import _quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def print_observation_shapes(env: ManagerBasedRlEnv, *, header: str) -> None:
    """Print named observation group shapes from the live env."""
    obs = env.observation_manager.compute()
    print(header)
    for group_name, tensor in obs.items():
        if hasattr(tensor, "shape"):
            print(f"  {group_name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            continue
        if isinstance(tensor, dict):
            for term_name, term_value in tensor.items():
                print(
                    f"  {group_name}/{term_name}: shape={tuple(term_value.shape)} "
                    f"dtype={term_value.dtype}"
                )
            continue
        print(f"  {group_name}: type={type(tensor).__name__}")


def print_relative_rung_values(env: ManagerBasedRlEnv, *, env_idx: int = 0, header: str) -> None:
    """Print selected relative-rung geometry for one environment."""
    provider = RelativeRungProvider.get(env)
    print(header)
    for line in provider.format_debug_lines(env_idx):
        print(line)


def attach_observation_debug_visualizer(env: ManagerBasedRlEnv) -> None:
    """Highlight relative-rung endpoints transformed back to world frame."""

    def update_visualizers(visualizer: DebugVisualizer) -> None:
        provider = RelativeRungProvider.get(env)
        obs = provider.compute()
        robot = env.scene["robot"]
        torso_id = robot.body_names.index("torso_link")
        env_idx = visualizer.env_idx
        torso = np.asarray(
            robot.data.body_link_pos_w[env_idx, torso_id].detach().cpu(),
            dtype=np.float64,
        )
        torso_quat = robot.data.body_link_quat_w[env_idx, torso_id].detach().cpu()

        endpoints = obs.endpoints[env_idx].detach().cpu()
        valid = obs.valid_mask[env_idx].detach().cpu()
        for slot in range(endpoints.shape[0]):
            if not bool(valid[slot].item()):
                continue
            end_a_b = endpoints[slot, 0:3]
            end_b_b = endpoints[slot, 3:6]
            end_a_w = (
                _quat_apply(torso_quat.unsqueeze(0), end_a_b.unsqueeze(0)).squeeze(0).numpy()
                + torso
            )
            end_b_w = (
                _quat_apply(torso_quat.unsqueeze(0), end_b_b.unsqueeze(0)).squeeze(0).numpy()
                + torso
            )
            visualizer.add_sphere(
                center=end_a_w,
                radius=0.015,
                color=(0.2, 0.9, 0.4, 0.9),
                label=f"rung_slot_{slot:02d}_a",
            )
            visualizer.add_sphere(
                center=end_b_w,
                radius=0.015,
                color=(0.2, 0.6, 1.0, 0.9),
                label=f"rung_slot_{slot:02d}_b",
            )
            visualizer.add_sphere(
                center=torso,
                radius=0.012,
                color=(1.0, 1.0, 1.0, 0.8),
                label="torso_origin",
            )

    env.unwrapped.update_visualizers = update_visualizers  # type: ignore[attr-defined]
