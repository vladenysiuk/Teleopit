"""Shared climbing MDP helpers for rewards, terminations, and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.ladder.relative_rungs import (
    ladder_relative_height,
    ladder_upward_axis_w,
)
from train_mimic.tasks.climbing.ladder.state import LadderRuntime

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def body_pos_w(env: ManagerBasedRlEnv, *, body_name: str) -> torch.Tensor:
    """World-frame body position ``[B, 3]``."""
    robot = env.scene["robot"]
    body_id = robot.body_names.index(body_name)
    return robot.data.body_link_pos_w[:, body_id]


def body_height_w(env: ManagerBasedRlEnv, *, body_name: str) -> torch.Tensor:
    """World-frame body height ``[B]``."""
    return body_pos_w(env, body_name=body_name)[:, 2]


def body_ladder_relative_height(env: ManagerBasedRlEnv, *, body_name: str) -> torch.Tensor:
    """Body height along the ladder up axis ``[B]``."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    sample = runtime.sample
    pos_w = body_pos_w(env, body_name=body_name)
    upward = ladder_upward_axis_w(sample.frame_quat)
    return ladder_relative_height(pos_w, sample.frame_pos, upward)


def body_ladder_frame_distance_xy(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "pelvis",
) -> torch.Tensor:
    """Horizontal distance from body to ladder frame origin in ladder XY ``[B]``."""
    from train_mimic.tasks.climbing.ladder.relative_rungs import world_to_body

    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    sample = runtime.sample
    pos_w = body_pos_w(env, body_name=body_name)
    local = world_to_body(sample.frame_pos, sample.frame_quat, pos_w.unsqueeze(1)).squeeze(1)
    return torch.linalg.norm(local[:, :2], dim=-1)


def gather_active_rung_heights_l(
    env: ManagerBasedRlEnv,
    rung_ids: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Look up ladder-relative rung heights for ``rung_ids`` ``[N]``."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    sample = runtime.sample
    upward = ladder_upward_axis_w(sample.frame_quat)
    from train_mimic.tasks.climbing.ladder.relative_rungs import ladder_relative_heights_from_world

    rung_heights_l = ladder_relative_heights_from_world(
        sample.rung_pos_w,
        sample.frame_pos,
        upward,
    )
    if env_ids is None:
        batch = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
        return rung_heights_l[batch, rung_ids]
    return rung_heights_l[env_ids, rung_ids]


def top_active_rung_height_l(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Ladder-relative height of the highest active rung ``[B]``."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    sample = runtime.sample
    filled = sample.rung_heights.masked_fill(~sample.active_mask, float("-inf"))
    return filled.max(dim=-1).values


def top_active_rung_id(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Index of the highest active rung per env ``[B]`` int64."""
    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        raise RuntimeError("LadderRuntime has no sample.")
    sample = runtime.sample
    # Active rungs are packed from index 0, so the top id is num_active - 1.
    return (sample.num_active - 1).to(dtype=torch.int64).clamp_min(0)


def success_goal_height_l(
    env: ManagerBasedRlEnv,
    *,
    pelvis_clearance_below_top_l: float,
) -> torch.Tensor:
    """Pelvis ladder-relative success height for the sampled ladder ``[B]``."""
    return top_active_rung_height_l(env) - float(pelvis_clearance_below_top_l)
